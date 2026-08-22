from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd

# 네 loader 파일 위치에 맞게 import가 안 되면 아래 import만 수정하면 됨.
# 예: from src.relbench_loader import load_relbench_bundle
from fdhg.data.relbench_loader import load_relbench_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out-dir", default="outputs/relbench_inspect")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"{args.dataset}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_relbench_bundle(args.dataset, args.task)

    print("=== Bundle ===")
    print("dataset:", bundle.dataset_name)
    print("task:", bundle.task_name)
    print("target_table:", bundle.target_table)

    print("\n=== Split tables ===")
    for name, df in [
        ("train", bundle.train_table),
        ("val", bundle.val_table),
        ("test", bundle.test_table),
    ]:
        if df is None:
            print(name, "None")
            continue
        print(f"\n[{name}] shape={df.shape}")
        print("columns:", list(df.columns))
        print(df.head())
        df.to_parquet(out_dir / f"target_{name}.parquet", index=False)

    print("\n=== DB tables ===")
    table_summary = []
    for name, df in bundle.tables.items():
        print(f"\n[{name}] shape={df.shape}")
        print("columns:", list(df.columns))
        print(df.head())
        table_summary.append({
            "table": name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": ",".join(map(str, df.columns)),
        })
        df.to_parquet(out_dir / f"table_{name}.parquet", index=False)

    pd.DataFrame(table_summary).to_csv(out_dir / "table_summary.csv", index=False)
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
