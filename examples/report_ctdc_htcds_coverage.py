from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_mapping import (
    CTDCHTCDSPlusMapper,
    load_htcds_plus_schema,
    load_htcds_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report CTDC-to-HTCDS field coverage")
    parser.add_argument(
        "--ctdc-csv",
        type=Path,
        help="Optional local CTDC CSV; only its header is read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for the coverage report.",
    )
    args = parser.parse_args()

    base_schema = load_htcds_schema(
        ROOT / "HTCDS_standard" / "HTCDS Field Standards 2.0.xlsx"
    )
    schema = load_htcds_plus_schema(
        base_schema, ROOT / "HTCDS_standard" / "HTCDS+ Extensions.yaml"
    )
    mapper = CTDCHTCDSPlusMapper(schema)
    available_columns = None
    if args.ctdc_csv:
        available_columns = pd.read_csv(args.ctdc_csv, nrows=0).columns

    report = mapper.coverage_report(available_columns)
    print(report.to_string(index=False))
    print("\nSummary:")
    print(report["mapping_status"].value_counts().to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
