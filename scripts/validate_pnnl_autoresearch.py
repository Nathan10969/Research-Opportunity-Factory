from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pnnl_pilot.autoresearch_validator import validate_autoresearch_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-a-report",
        type=Path,
        default=REPO_ROOT / "evidence/pnnl_gate_a/validation_report.json",
    )
    parser.add_argument(
        "--final-report",
        type=Path,
        default=REPO_ROOT / "evidence/pnnl_autoresearch/final_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / ".omx/specs/autoresearch-pnnl-innercv/result.json"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    gate_report = load_json(args.gate_a_report) or {"status": "missing"}
    final_report = load_json(args.final_report)
    result = validate_autoresearch_evidence(gate_report, final_report)
    result["gate_a_report"] = str(args.gate_a_report.resolve())
    result["final_report"] = str(args.final_report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
