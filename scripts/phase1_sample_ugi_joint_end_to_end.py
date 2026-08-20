#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.design.flow.ugi_chemistry_flow import TERMINAL_DECODER_MODES
from forge.design.sampling.ugi_joint_end_to_end_sampling import sample_ugi_joint_end_to_end

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--closure-checkpoint", type=Path, required=True)
    parser.add_argument("--matched-staged-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--maximum-adjacent-branch-runs",
        type=int,
        nargs=3,
        metavar=("HEAD", "ALDEHYDE", "ISOCYANIDE"),
        default=None,
        help="Optional maximum decoded-tree branch-component size for each Ugi role",
    )
    parser.add_argument("--qualified-reactions", type=Path)
    parser.add_argument("--evaluate-exact-l1-terminal-admission", action="store_true")
    parser.add_argument(
        "--terminal-decoder-mode",
        choices=TERMINAL_DECODER_MODES,
        default="argmax",
    )
    parser.add_argument("--terminal-decoder-seed", type=int)
    parser.add_argument("--terminal-temperature", type=float, default=1.0)
    parser.add_argument("--terminal-atom-temperature", type=float)
    parser.add_argument("--terminal-bond-temperature", type=float)
    parser.add_argument("--terminal-decoration-temperature", type=float)
    parser.add_argument("--program-limit", type=int)
    parser.add_argument("--program-offset", type=int, default=0)
    parser.add_argument(
        "--reference-comparison-mode",
        choices=("full", "deferred"),
        default="full",
    )
    parser.add_argument(
        "--terminal-atom-temperatures-by-origin",
        type=float,
        nargs=3,
        metavar=("AMINE", "ALDEHYDE", "ISOCYANIDE"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = sample_ugi_joint_end_to_end(
        REPO,
        args.output_dir,
        joint_checkpoint_path=args.joint_checkpoint,
        closure_checkpoint_path=args.closure_checkpoint,
        matched_staged_result_path=args.matched_staged_result,
        sample_steps=args.sample_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
        maximum_adjacent_branch_runs=args.maximum_adjacent_branch_runs,
        qualified_reactions_path=args.qualified_reactions,
        evaluate_exact_l1_terminal_admission=args.evaluate_exact_l1_terminal_admission,
        terminal_decoder_mode=args.terminal_decoder_mode,
        terminal_decoder_seed=args.terminal_decoder_seed,
        terminal_temperature=args.terminal_temperature,
        terminal_atom_temperature=args.terminal_atom_temperature,
        terminal_bond_temperature=args.terminal_bond_temperature,
        terminal_decoration_temperature=args.terminal_decoration_temperature,
        terminal_atom_temperatures_by_origin=args.terminal_atom_temperatures_by_origin,
        program_offset=args.program_offset,
        program_limit=args.program_limit,
        reference_comparison_mode=args.reference_comparison_mode,
    )
    statistics = {
        key: value for key, value in result["statistics"].items() if key != "valid_smiles"
    }
    print(json.dumps({"statistics": statistics, "render": result["render"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
