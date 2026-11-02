from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation import kappa_from_frames
from data import load_dataset
from dataset_audit import audit_dataset
from evaluation import ablation_study, calibration_study, evaluate_models
from openset import run_open_set_experiment
from reproducibility import experiment_record, write_json
from risk_coverage import simulate_risk_coverage

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "instance" / "experiments"


def _dataset(path: str | None):
    dataset_path = Path(path) if path else ROOT / "customer_support_tickets.csv"
    return load_dataset(dataset_path), dataset_path


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    frame, path = _dataset(args.dataset)
    report = audit_dataset(frame)
    output = Path(args.output) / "dataset_audit.json"
    write_json(output, experiment_record("dataset_audit", {"dataset": str(path)}, report, frame.attrs.get("dataset_fingerprint", {}), output))
    print(f"Wrote {output}")
    return report


def cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    frame, path = _dataset(args.dataset)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    results = {}
    for target in ("Ticket Type", "Ticket Priority"):
        results[target] = evaluate_models(
            frame,
            target=target,
            seeds=seeds,
            n_splits=args.folds,
            max_rows=args.max_rows,
        )
    output = Path(args.output) / "evaluation.json"
    write_json(
        output,
        experiment_record(
            "evaluation",
            {"dataset": str(path), "seeds": list(seeds), "folds": args.folds, "max_rows": args.max_rows},
            results,
            frame.attrs.get("dataset_fingerprint", {}),
            output,
        ),
    )
    print(f"Wrote {output}")
    return results


def cmd_ablation(args: argparse.Namespace) -> dict[str, Any]:
    frame, path = _dataset(args.dataset)
    result = {
        "ablation": ablation_study(frame, max_rows=args.max_rows),
        "calibration": calibration_study(frame, max_rows=min(args.max_rows or 500, 500)),
    }
    output = Path(args.output) / "ablation.json"
    write_json(output, experiment_record("ablation", {"dataset": str(path)}, result, frame.attrs.get("dataset_fingerprint", {}), output))
    print(f"Wrote {output}")
    return result


def cmd_openset(args: argparse.Namespace) -> dict[str, Any]:
    frame, path = _dataset(args.dataset)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    unknown = [item.strip() for item in args.unknown.split(",") if item.strip()] or None
    result = run_open_set_experiment(
        frame,
        seeds=seeds,
        unknown_classes=unknown,
        max_rows=args.max_rows,
    )
    output = Path(args.output) / "openset.json"
    write_json(output, experiment_record("openset", {"dataset": str(path), "unknown": unknown, "seeds": list(seeds)}, result, frame.attrs.get("dataset_fingerprint", {}), output))
    print(f"Wrote {output}")
    return result


def cmd_risk(args: argparse.Namespace) -> dict[str, Any]:
    frame, path = _dataset(args.dataset)
    result = simulate_risk_coverage(frame, target_coverage=args.coverage, max_rows=args.max_rows)
    output = Path(args.output) / "risk_coverage.json"
    write_json(output, experiment_record("risk_coverage", {"coverage": args.coverage}, result, frame.attrs.get("dataset_fingerprint", {}), output))
    print(f"Wrote {output}")
    return result


def cmd_kappa(args: argparse.Namespace) -> dict[str, Any]:
    frame_a = pd.read_csv(args.annotator_a)
    frame_b = pd.read_csv(args.annotator_b)
    result = kappa_from_frames(frame_a, frame_b)
    output = Path(args.output) / "kappa.json"
    write_json(output, result)
    print(f"Wrote {output}")
    return result


def cmd_reproduce(args: argparse.Namespace) -> None:
    cmd_audit(args)
    cmd_evaluate(args)
    cmd_ablation(args)
    cmd_openset(args)
    cmd_risk(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Opslyra reproducible experiment runner")
    parser.add_argument("--dataset", default=None, help="CSV path (default: customer_support_tickets.csv)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Directory for JSON experiment records")
    parser.add_argument("--max-rows", dest="max_rows", type=int, default=800)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--unknown", default="", help="Comma-separated labels to hold out for open-set")
    parser.add_argument("--coverage", type=float, default=0.6)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("evaluate")
    sub.add_parser("ablation")
    sub.add_parser("openset")
    sub.add_parser("risk-coverage")
    kappa = sub.add_parser("kappa")
    kappa.add_argument("annotator_a")
    kappa.add_argument("annotator_b")
    sub.add_parser("reproduce")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    commands = {
        "audit": cmd_audit,
        "evaluate": cmd_evaluate,
        "ablation": cmd_ablation,
        "openset": cmd_openset,
        "risk-coverage": cmd_risk,
        "kappa": cmd_kappa,
        "reproduce": cmd_reproduce,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
