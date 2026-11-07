from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw"

DATASET_NAME = "PolyAI/banking77"
DATASET_REVISION = "830c4f2be6949546b11fe83fbc50993348a2bccd"


def main():
    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
    )

    RAW_DATA.mkdir(parents=True, exist_ok=True)

    label_feature = dataset["train"].features["label"]

    for split_name in ("train", "test"):
        split = dataset[split_name]
        df = split.to_pandas()

        df["label"] = df["label"].map(label_feature.int2str)

        output_path = RAW_DATA / f"{split_name}.csv"
        df.to_csv(output_path, index=False)

        print(f"{split_name}: {len(df):,} examples")
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()