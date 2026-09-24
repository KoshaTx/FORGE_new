"""Family-gradient concurrency with the unchanged ordered PCGrad combination."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from forge.model.compose_lipid_training import compose_lipid_forward_loss, trim_family_padding
from forge.model.reaction_program_transformer import project_family_gradients


def family_seed(seed: int, step: int, family: int) -> int:
    """Separate repeatable CUDA streams from thread scheduling and physical GPU indices."""
    from forge.core.hashing import sha256_json

    return int(str(sha256_json(dict(seed=seed, step=step, family=family)))[:15], 16)


def copy_replica_parameters(model, weights):
    """Preserve replica storage and strides when broadcasting the current parameters.

    Rebinding parameters to slices of a flat vector changes their alignment. On H100
    this caused otherwise identical family gradients to diverge over AdamW updates.
    """
    parameters = list(model.parameters())
    if weights.ndim != 1 or weights.numel() != sum(p.numel() for p in parameters):
        raise ValueError("Replica parameter vector has the wrong size")
    local = weights.to(parameters[0].device)
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(local[offset : offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()


def family_gradient(model, clean, config, node, bond, *, seed, device, weights=None):
    """One formal family's forward/backward; no optimizer update or cross-family averaging."""
    device = torch.device(device)
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        if weights is not None:
            copy_replica_parameters(model, weights)
        clean = {key: value.to(device) for key, value in clean.items()}
        clean = trim_family_padding(clean)
        generator = torch.Generator(device=device).manual_seed(seed)
        # Dropout and corruption have separate recorded per-family streams. CUDA state is
        # device-local, and only one worker ever writes a particular device's generator.
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.set_rng_state(generator.get_state())
        model.train()
        times = torch.rand(len(clean["nodes"]), generator=generator, device=device)
        loss, metrics = compose_lipid_forward_loss(
            model,
            clean,
            architecture=config["model"]["architecture"],
            node_marginal=node.to(device),
            bond_marginal=bond.to(device),
            times=times,
            generator=generator,
            semantic_weights=config["semantic_weights"],
            repeat_supervision="exact_fragment",
            objective=config["objective"],
        )
        params = [p for p in model.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        available = [value is not None for value in gradients]
        flat = torch.cat(
            [
                (value if value is not None else torch.zeros_like(param)).reshape(-1)
                for param, value in zip(params, gradients, strict=True)
            ]
        )
        destination = node.device
        return (
            flat.to(destination),
            available,
            loss.detach().to(destination),
            {
                key: value.detach().to(destination) if torch.is_tensor(value) else value
                for key, value in metrics.items()
            },
        )


class FamilyGradientTrainer:
    """Explicit family RNG policy, shared by the serial control and concurrent candidate.

    This policy differs from the legacy single generator's trajectory. Equality tests
    must compare the two paths with this same policy; it is not a legacy exact resume.
    """

    def __init__(self, model, optimizer, config, node, bond, *, seed: int, devices):
        if config["objective"]["gradient_balancing"] != "pcgrad":
            raise ValueError("Family-gradient execution requires the PCGrad objective")
        self.devices = tuple(torch.device(device) for device in devices)
        if not self.devices or (
            len(self.devices) > 1
            and (
                any(d.type != "cuda" for d in self.devices)
                or len(set(self.devices)) != len(self.devices)
            )
        ):
            raise ValueError("Concurrent gradients require distinct CUDA devices")
        self.model, self.optimizer, self.config = model, optimizer, config
        self.node, self.bond, self.seed = node, bond, seed
        self.models = [model] + [copy.deepcopy(model).to(device) for device in self.devices[1:]]
        self.executor = (
            ThreadPoolExecutor(max_workers=len(self.devices)) if len(self.devices) > 1 else None
        )

    def step(self, clean, step: int):
        self.optimizer.zero_grad(set_to_none=True)
        families = torch.unique(clean["family_states"], sorted=True).tolist()
        if len(families) < 2 or (self.executor is not None and len(families) != len(self.devices)):
            raise ValueError("Every concurrent device must receive one distinct family")
        groups = [
            {key: value[clean["family_states"] == family] for key, value in clean.items()}
            for family in families
        ]
        weights = (
            torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()
            if self.executor
            else None
        )
        futures, results = [], []
        for index, (family, group) in enumerate(zip(families, groups, strict=True)):
            target = index if self.executor else 0
            kwargs = dict(
                seed=family_seed(self.seed, step, family),
                device=self.devices[target],
                weights=weights if target else None,
            )
            args = (self.models[target], group, self.config, self.node, self.bond)
            if self.executor:
                futures.append(self.executor.submit(family_gradient, *args, **kwargs))
            else:
                results.append(family_gradient(*args, **kwargs))
        if futures:
            results = [future.result() for future in futures]
        flat = torch.stack([r[0] for r in results])
        diagnostic = project_family_gradients(
            flat,
            [r[1] for r in results],
            [p for p in self.model.parameters() if p.requires_grad],
            program_states=families,
            materialize_diagnostics=False,
        )
        loss = torch.stack([r[2] for r in results]).mean()
        metrics = {}
        for _, _, _, values in results:
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0) + value / len(families)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config["gradient_clip_norm"], error_if_nonfinite=True
        )
        self.optimizer.step()
        metrics.update(
            loss=loss, gradient_norm=norm, projected_conflicts=diagnostic["projected_conflicts"]
        )
        return {key: float(value) for key, value in metrics.items()}

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
