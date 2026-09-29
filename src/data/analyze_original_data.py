from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_PATH = ROOT / "data" / "raw" / "train.csv"

ORIGINAL_PATH = (
    ROOT
    / "data"
    / "external"
    / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
)

TARGET = "Will_Buy_EV"


def print_title(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():

    # =====================================================
    # 1. Load
    # =====================================================

    official = pd.read_csv(OFFICIAL_PATH)
    original = pd.read_csv(ORIGINAL_PATH)

    # =====================================================
    # Normalize target
    # =====================================================

    print("Official target values:")
    print(official[TARGET].value_counts(dropna=False))

    print("\nOriginal target values:")
    print(original[TARGET].value_counts(dropna=False))

    def normalize_target(series, dataset_name):

        # Already numeric
        if pd.api.types.is_numeric_dtype(series):
            return series.astype("int8")

        cleaned = (
            series
            .astype(str)
            .str.strip()
            .str.lower()
        )

        target_map = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

        mapped = cleaned.map(target_map)

        if mapped.isna().any():
            unknown = (
                series[
                    mapped.isna()
                ]
                .unique()
            )

            raise ValueError(
                f"Unknown {dataset_name} "
                f"target values: {unknown}"
            )

        return mapped.astype("int8")

    official[TARGET] = normalize_target(
        official[TARGET],
        "official",
    )

    original[TARGET] = normalize_target(
        original[TARGET],
        "original",
    )
    # =====================================================
    # 2. Columns
    # =====================================================

    print_title("OFFICIAL COLUMNS")

    for i, col in enumerate(official.columns, 1):
        print(f"{i:02d}. {col}")

    print_title("ORIGINAL COLUMNS")

    for i, col in enumerate(original.columns, 1):
        print(f"{i:02d}. {col}")

    official_cols = set(official.columns)
    original_cols = set(original.columns)

    print_title("COLUMN DIFFERENCES")

    print(
        "Only in official:",
        sorted(official_cols - original_cols),
    )

    print(
        "Only in original:",
        sorted(original_cols - official_cols),
    )

    common = [
        col
        for col in official.columns
        if col in original.columns
    ]

    print(
        "\nCommon columns:",
        len(common),
    )

    # =====================================================
    # 3. Dtypes
    # =====================================================

    print_title("DTYPE COMPARISON")

    rows = []

    for col in common:

        rows.append({
            "column": col,
            "official_dtype": str(official[col].dtype),
            "original_dtype": str(original[col].dtype),
        })

    dtype_df = pd.DataFrame(rows)

    print(
        dtype_df.to_string(index=False)
    )

    # =====================================================
    # 4. Missing values
    # =====================================================

    print_title("MISSING VALUES")

    rows = []

    for col in common:

        rows.append({
            "column": col,

            "official_missing":
                official[col].isna().sum(),

            "official_missing_pct":
                official[col].isna().mean(),

            "original_missing":
                original[col].isna().sum(),

            "original_missing_pct":
                original[col].isna().mean(),
        })

    missing_df = pd.DataFrame(rows)

    print(
        missing_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # =====================================================
    # 5. Target
    # =====================================================

    if (
        TARGET in official.columns
        and TARGET in original.columns
    ):

        print_title("TARGET DISTRIBUTION")

        official_rate = (
            official[TARGET].mean()
        )

        original_rate = (
            original[TARGET].mean()
        )

        print(
            f"Official positive rate : "
            f"{official_rate:.6f}"
        )

        print(
            f"Original positive rate : "
            f"{original_rate:.6f}"
        )

        print(
            f"Difference             : "
            f"{original_rate - official_rate:+.6f}"
        )

        print("\nOfficial:")
        print(
            official[TARGET]
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        print("\nOriginal:")
        print(
            original[TARGET]
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

    # =====================================================
    # 6. Numeric distribution
    # =====================================================

    print_title("NUMERIC DISTRIBUTION COMPARISON")

    numeric_cols = []

    for col in common:

        if col == TARGET:
            continue

        if (
            pd.api.types.is_numeric_dtype(
                official[col]
            )
            and
            pd.api.types.is_numeric_dtype(
                original[col]
            )
        ):
            numeric_cols.append(col)

    rows = []

    for col in numeric_cols:

        a = official[col]
        b = original[col]

        rows.append({

            "column": col,

            "official_mean":
                a.mean(),

            "original_mean":
                b.mean(),

            "mean_diff":
                b.mean() - a.mean(),

            "official_std":
                a.std(),

            "original_std":
                b.std(),

            "official_min":
                a.min(),

            "original_min":
                b.min(),

            "official_max":
                a.max(),

            "original_max":
                b.max(),

            "official_unique":
                a.nunique(),

            "original_unique":
                b.nunique(),
        })

    numeric_df = pd.DataFrame(rows)

    print(
        numeric_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # =====================================================
    # 7. Categorical values
    # =====================================================

    print_title("CATEGORICAL VALUE COMPARISON")

    categorical_cols = []

    for col in common:

        if col == TARGET:
            continue

        if (
            not pd.api.types.is_numeric_dtype(
                official[col]
            )
            or
            not pd.api.types.is_numeric_dtype(
                original[col]
            )
            or
            max(
                official[col].nunique(),
                original[col].nunique(),
            ) <= 20
        ):
            categorical_cols.append(col)

    for col in categorical_cols:

        print("\n" + "-" * 100)
        print(col)
        print("-" * 100)

        official_values = set(
            official[col]
            .dropna()
            .astype(str)
            .unique()
        )

        original_values = set(
            original[col]
            .dropna()
            .astype(str)
            .unique()
        )

        print(
            "Official unique:",
            len(official_values),
        )

        print(
            "Original unique:",
            len(original_values),
        )

        print(
            "Only official:",
            sorted(
                official_values
                - original_values
            ),
        )

        print(
            "Only original:",
            sorted(
                original_values
                - official_values
            ),
        )

    # =====================================================
    # 8. Duplicate / overlap detection
    # =====================================================

    print_title("EXACT ROW OVERLAP")

    feature_cols = [
        col
        for col in common
        if col != TARGET
    ]

    # -----------------------------------------------------
    # Normalize values before building row signatures
    # -----------------------------------------------------

    official_overlap = (
        official[feature_cols]
        .copy()
    )

    original_overlap = (
        original[feature_cols]
        .copy()
    )

    # Convert every column explicitly.
    # Missing values use the same marker.
    for col in feature_cols:
        official_overlap[col] = (
            official_overlap[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        original_overlap[col] = (
            original_overlap[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    # Build row signatures using numpy.
    # This avoids pandas/Arrow agg dtype issues.
    official_features = (
        official_overlap
        .apply(
            lambda row: "|".join(
                row.tolist()
            ),
            axis=1,
        )
    )

    original_features = (
        original_overlap
        .apply(
            lambda row: "|".join(
                row.tolist()
            ),
            axis=1,
        )
    )

    official_set = set(
        official_features.tolist()
    )

    overlap = (
        original_features
        .isin(official_set)
    )

    overlap_count = int(
        overlap.sum()
    )

    print(
        "Original rows appearing exactly "
        "in official train:",
        overlap_count,
    )

    print(
        "Original overlap rate:",
        f"{overlap.mean():.6f}",
    )
    # =====================================================
    # 9. Income structure
    # =====================================================

    if (
        "Annual_Income_USD"
        in common
    ):

        print_title("INCOME STRUCTURE")

        for name, df in [
            ("Official", official),
            ("Original", original),
        ]:

            income = (
                df["Annual_Income_USD"]
                .dropna()
                .round()
                .astype("int64")
                .abs()
            )

            print(
                f"\n{name}"
            )

            print(
                "Income unique:",
                income.nunique(),
            )

            print(
                "Last digit distribution:"
            )

            print(
                (
                    income % 10
                )
                .value_counts(
                    normalize=True
                )
                .sort_index()
                .round(4)
                .to_string()
            )

            print(
                "\nLast 2 digits "
                "unique:",
                (income % 100)
                .nunique(),
            )

            print(
                "Last 3 digits "
                "unique:",
                (income % 1000)
                .nunique(),
            )

    # =====================================================
    # 10. Final summary
    # =====================================================

    print_title("SUMMARY")

    print(
        f"Official rows : {len(official):,}"
    )

    print(
        f"Original rows : {len(original):,}"
    )

    print(
        f"Original / Official size ratio: "
        f"{len(original) / len(official):.4%}"
    )

    print(
        f"Common columns: {len(common)}"
    )

    print(
        f"Exact feature overlap rows: "
        f"{overlap_count:,}"
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()