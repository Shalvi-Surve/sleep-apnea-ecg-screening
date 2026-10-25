from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COMPLEXITY = ROOT / "results" / "complexity"
OUT = ROOT / "results" / "model_comparison"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    rows = [
        {
            "model": "1D CNN",
            "parameters": 4801,
            "model_size_mb": 0.018314,
            "cpu_latency_ms": 0.378,
        },
        {
            "model": "Lightweight Transformer V1",
            "parameters": 70273,
            "model_size_mb": 0.268070,
            "cpu_latency_ms": 1.045,
        },
    ]

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / "final_model_complexity.csv",
        index=False,
    )

    print("=" * 70)
    print("FINAL MODEL COMPLEXITY")
    print("=" * 70)

    print(
        df.to_string(index=False)
    )

    print(
        "\nSaved:",
        OUT / "final_model_complexity.csv",
    )


if __name__ == "__main__":
    main()