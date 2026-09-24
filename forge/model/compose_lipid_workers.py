"""Persistent family workers with stateless logical-shard random streams."""

import time
import traceback

import torch
import torch.multiprocessing as mp

from forge.model.compose_lipid_ipc import pack_batch, pack_result, unpack_batch, unpack_result
from forge.model.compose_lipid_parallel import copy_replica_parameters, family_gradient, family_seed
from forge.model.compose_lipid_sharded_loss import FamilyLoss, select
from forge.model.compose_lipid_training import trim_family_padding
from forge.model.ordered_relation_embedding import configure_relation_embedding
from forge.model.reaction_program_flow import noise_synthesis_program_batch
from forge.model.reaction_program_transformer import project_family_gradients
from forge.model.synthesis_program_training import (
    _synthesis_program_predict,
    build_synthesis_program_flow,
    synthesis_program_fixed_state_exact,
)


def configure(config):
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = config["runtime"].get(
        "fill_uninitialized_memory", True
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def add_results(parts):
    gradient, available, loss, metrics = parts[0]
    for g, a, part_loss, m in parts[1:]:
        gradient = gradient + g
        available = [left or right for left, right in zip(available, a, strict=True)]
        loss = loss + part_loss
        metrics = {key: value + m[key] for key, value in metrics.items()}
    return gradient, available, loss, metrics


def split_gradient(model, host, config, node, bond, *, seed, device, parts, divisions=2):
    clean = trim_family_padding({key: value.to(device) for key, value in host.items()})
    count = len(clean["nodes"])
    if divisions not in (2, 4) or count % divisions:
        raise ValueError("Family batch must divide into two or four logical shards")
    objective = FamilyLoss(clean, config["model"]["maximum_children"])
    generator = torch.Generator(device=device).manual_seed(seed)
    times = torch.rand(count, generator=generator, device=device)
    noisy = noise_synthesis_program_batch(clean, node, bond, times, generator)
    if not synthesis_program_fixed_state_exact(noisy, clean):
        raise ValueError("Noising changed fixed states")
    params = [p for p in model.parameters() if p.requires_grad]
    results = []
    for part in parts:
        region = slice(part * (count // divisions), (part + 1) * (count // divisions))
        local, state = select(clean, region), select(noisy, region)
        conditioned = dict(state)
        for key in ("parents", "closure_left", "closure_right"):
            conditioned[key] = local[key]
        paired_clean = {key: torch.cat((value, value)) for key, value in local.items()}
        paired_state = {key: torch.cat((value, conditioned[key])) for key, value in state.items()}
        # Stable logical shard identity, independent of physical GPU or execution order.
        dropout = torch.Generator(device=device).manual_seed(family_seed(seed, 0, part))
        if torch.device(device).type == "cuda":
            torch.cuda.set_rng_state(dropout.get_state(), device)
        else:
            torch.set_rng_state(dropout.get_state())
        model.train()
        output = _synthesis_program_predict(
            model, paired_clean, paired_state, torch.cat((times[region], times[region]))
        )
        half = count // divisions
        predictions = {key: value[:half] for key, value in output.items()}
        topology = {key: value[half:] for key, value in output.items()}
        loss, metrics = objective.loss(predictions, topology, region, config)
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        results.append(
            (
                torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for p, g in zip(params, gradients, strict=True)
                    ]
                ),
                [g is not None for g in gradients],
                loss.detach(),
                metrics,
            )
        )
    return add_results(results)


def calculate(
    model, batch, config, node, bond, seed, device, mode, part, *, parts=None, divisions=2
):
    if mode == "native":
        return family_gradient(model, batch, config, node, bond, seed=seed, device=device)
    return split_gradient(
        model,
        batch,
        config,
        node,
        bond,
        seed=seed,
        device=device,
        parts=parts if parts is not None else ((0, 1) if mode == "split_serial" else (part,)),
        divisions=divisions,
    )


def adaptive_family_tasks(groups, mode):
    """Assign four shards to the largest family and two to each other family."""
    if mode not in ("adaptive_serial", "adaptive_parallel"):
        raise ValueError(mode)
    if len(groups) != 3 or any(len(group["nodes"]) % 4 for group in groups):
        raise ValueError("Three family batches divisible by four are required")
    sizes = [int(group["node_mask"].sum(1).max()) for group in groups]
    heavy = max(range(3), key=lambda i: sizes[i])
    divisions = [4 if i == heavy else 2 for i in range(3)]
    if mode == "adaptive_serial":
        return [(i, tuple(range(n)), n) for i, n in enumerate(divisions)]
    # Workers start while dispatch is still in progress. Leave a short shard for
    # the supervising GPU and send the largest family to workers first.
    local = min(range(3), key=lambda i: sizes[i])
    tasks = [(local, (0,), divisions[local])]
    for family in sorted(range(3), key=lambda i: (-sizes[i], i)):
        for part in range(1 if family == local else 0, divisions[family]):
            tasks.append((family, (part,), divisions[family]))
    return tasks


def share_batch(pool, values):
    """Reuse one immutable host buffer per family until every worker has replied."""
    changed = False
    for dtype, value in values.items():
        if dtype not in pool or pool[dtype].numel() < value.numel():
            pool[dtype] = torch.empty(value.numel(), dtype=dtype).share_memory_()
            changed = True
        pool[dtype][: value.numel()].copy_(value)
    return changed


def worker(
    device, commands, replies, vocabulary, node_classes, config, state, node, bond, mode, part
):
    try:
        configure(config)
        torch.cuda.set_device(device)
        model = build_synthesis_program_flow(
            vocabulary=vocabulary,
            node_classes=node_classes,
            model_config=config["model"],
            device=device,
        )
        model.load_state_dict(state)
        configure_relation_embedding(model, config["runtime"].get("relation_embedding_backend"))
        node, bond = node.to(device), bond.to(device)
        buffers, weights, output = None, None, None
        replies.put(("ready", None))
        while True:
            command = commands.get()
            if command is None:
                return
            if "buffers" in command:
                buffers = command["buffers"]
            if "weights" in command:
                weights = command["weights"]
            started = time.perf_counter()
            batch = unpack_batch(
                (
                    {dtype: buffers[dtype][:size] for dtype, size in command["sizes"].items()},
                    command["layout"],
                )
            )
            copy_replica_parameters(model, weights)
            result = calculate(
                model,
                batch,
                config,
                node,
                bond,
                command["seed"],
                device,
                mode,
                part,
                parts=command.get("parts"),
                divisions=command.get("divisions", 2),
            )
            packed = pack_result(result)
            vector, available, size, layout = packed
            first = output is None
            if first:
                output = torch.empty_like(vector)
            if output.shape != vector.shape:
                raise ValueError("Gradient result schema changed")
            output.copy_(vector)
            torch.cuda.synchronize(device)
            reply = dict(
                available=available,
                size=size,
                layout=layout,
                seconds=time.perf_counter() - started,
                peak_bytes=torch.cuda.max_memory_allocated(device),
            )
            if first:
                reply["buffer"] = output
            replies.put(("result", reply))
            if commands.get() != "consumed":
                raise RuntimeError("Missing persistent-buffer acknowledgement")
            del result, packed, vector, batch, command
    except Exception:
        replies.put(("error", traceback.format_exc()))


class PersistentTrainer:
    def __init__(
        self, model, optimizer, config, node, bond, *, seed, vocabulary, node_classes, mode="native"
    ):
        if mode not in (
            "native",
            "split_serial",
            "split_parallel",
            "adaptive_serial",
            "adaptive_parallel",
        ):
            raise ValueError(mode)
        self.model, self.optimizer, self.config = model, optimizer, config
        configure_relation_embedding(model, config["runtime"].get("relation_embedding_backend"))
        self.node, self.bond, self.seed, self.mode = node, bond, seed, mode
        self.weights = torch.empty(sum(p.numel() for p in model.parameters()), device=node.device)
        self.workers = []
        self.last_phases = {}
        self.batch_buffers = [{}, {}, {}]
        context = mp.get_context("spawn")
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        try:
            for rank in range(
                1, 8 if mode == "adaptive_parallel" else (6 if mode == "split_parallel" else 3)
            ):
                commands, replies = context.Queue(maxsize=1), context.Queue(maxsize=1)
                process = context.Process(
                    target=worker,
                    args=(
                        f"cuda:{rank}",
                        commands,
                        replies,
                        vocabulary,
                        node_classes,
                        config,
                        state,
                        node.cpu(),
                        bond.cpu(),
                        mode,
                        rank // 3,
                    ),
                )
                process.start()
                self.workers.append(
                    dict(
                        process=process,
                        commands=commands,
                        replies=replies,
                        buffers={},
                        output=None,
                        rank=rank,
                        initialized=False,
                    )
                )
            for entry in self.workers:
                kind, value = entry["replies"].get(timeout=120)
                if kind != "ready":
                    raise RuntimeError(value)
        except BaseException:
            self.close()
            raise

    def step(self, clean, step):
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        families = torch.unique(clean["family_states"], sorted=True).tolist()
        if len(families) != 3 or clean["nodes"].device.type != "cpu":
            raise ValueError("Three host-resident families are required")
        groups = [
            {key: value[clean["family_states"] == f] for key, value in clean.items()}
            for f in families
        ]
        if self.mode.startswith("adaptive"):
            groups = [trim_family_padding(group) for group in groups]
        packed = [pack_batch(group) for group in groups]
        tasks = (
            adaptive_family_tasks(groups, self.mode) if self.mode.startswith("adaptive") else None
        )
        changed_groups = (
            [
                share_batch(pool, values)
                for pool, (values, _) in zip(self.batch_buffers, packed, strict=True)
            ]
            if tasks is not None
            else None
        )
        offset = 0
        with torch.no_grad():
            for parameter in self.model.parameters():
                self.weights[offset : offset + parameter.numel()].copy_(parameter.reshape(-1))
                offset += parameter.numel()
        torch.cuda.synchronize(self.node.device)
        for entry in self.workers:
            family = tasks[entry["rank"]][0] if tasks is not None else entry["rank"] % 3
            values, layout = packed[family]
            if tasks is not None:
                changed = changed_groups[family] or entry.get("buffer_family") != family
                entry["buffers"] = self.batch_buffers[family]
                entry["buffer_family"] = family
            else:
                changed = False
                for dtype, value in values.items():
                    if (
                        dtype not in entry["buffers"]
                        or entry["buffers"][dtype].numel() < value.numel()
                    ):
                        entry["buffers"][dtype] = torch.empty(
                            value.numel(), dtype=dtype
                        ).share_memory_()
                        changed = True
                    entry["buffers"][dtype][: value.numel()].copy_(value)
            command = dict(
                layout=layout,
                sizes={dtype: value.numel() for dtype, value in values.items()},
                seed=family_seed(self.seed, step, families[family]),
            )
            if tasks is not None:
                _, command["parts"], command["divisions"] = tasks[entry["rank"]]
            if changed:
                command["buffers"] = entry["buffers"]
            if not entry["initialized"]:
                command["weights"] = self.weights
                entry["initialized"] = True
            entry["commands"].put(command)
        dispatched = time.perf_counter()
        local_family = tasks[0][0] if tasks is not None else 0
        # Logical shard streams must not leak their physical placement into the
        # checkpoint RNG state. Six-GPU compatibility retains its original policy.
        with torch.random.fork_rng(devices=[self.node.device.index], enabled=tasks is not None):
            results = [
                calculate(
                    self.model,
                    groups[local_family],
                    self.config,
                    self.node,
                    self.bond,
                    family_seed(self.seed, step, families[local_family]),
                    self.node.device,
                    self.mode,
                    0,
                    parts=tasks[0][1] if tasks is not None else None,
                    divisions=tasks[0][2] if tasks is not None else 2,
                )
            ]
        local_done = time.perf_counter()
        worker_seconds, peaks = [], [torch.cuda.max_memory_allocated(self.node.device)]
        for entry in self.workers:
            kind, value = entry["replies"].get(timeout=120)
            if kind != "result":
                raise RuntimeError(value)
            if "buffer" in value:
                entry["output"] = value["buffer"]
            results.append(
                unpack_result(
                    (entry["output"], value["available"], value["size"], value["layout"]),
                    self.node.device,
                )
            )
            worker_seconds.append(value["seconds"])
            peaks.append(value["peak_bytes"])
            torch.cuda.synchronize(self.node.device)
            entry["commands"].put("consumed")
        if tasks is not None:
            results = [
                add_results(
                    [
                        result
                        for task, result in zip(tasks, results, strict=True)
                        if task[0] == family
                    ]
                )
                for family in range(3)
            ]
        elif self.mode == "split_parallel":
            results = [add_results([results[i], results[i + 3]]) for i in range(3)]
        gathered = time.perf_counter()
        diagnostic = project_family_gradients(
            torch.stack([r[0] for r in results]),
            [r[1] for r in results],
            [p for p in self.model.parameters() if p.requires_grad],
            program_states=families,
            materialize_diagnostics=False,
        )
        loss = torch.stack([r[2] for r in results]).mean()
        metrics = {}
        for _, _, _, values in results:
            for key, value in values.items():
                metrics[key] = metrics.get(key, 0) + value / 3
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config["gradient_clip_norm"], error_if_nonfinite=True
        )
        self.optimizer.step()
        metrics.update(
            loss=loss, gradient_norm=norm, projected_conflicts=diagnostic["projected_conflicts"]
        )
        output = {key: float(value) for key, value in metrics.items()}
        self.last_phases = dict(
            dispatch_seconds=dispatched - started,
            local_family_seconds=local_done - dispatched,
            gather_wait_seconds=gathered - local_done,
            projection_optimizer_seconds=time.perf_counter() - gathered,
            worker_seconds=worker_seconds,
            peak_bytes_by_device=peaks,
        )
        return output

    def close(self):
        for entry in self.workers:
            entry["output"] = None
        for entry in self.workers:
            process = entry["process"]
            if process.is_alive():
                try:
                    entry["commands"].put(None, timeout=1)
                except Exception:
                    process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            for queue in (entry["commands"], entry["replies"]):
                queue.cancel_join_thread()
                queue.close()
        self.workers.clear()
