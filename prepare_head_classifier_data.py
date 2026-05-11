import os
import pandas as pd

OUTPUT_DIR = "head_classifier_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# KLINICZNE CSV
# =========================

# XGB - bez spotmagów wszędzie
XGB_FILES = {
    "train": "clinical_probability_exports/xgb_without_spotmag_train_probabilities.csv",
    "val":   "clinical_probability_exports/xgb_without_spotmag_val_probabilities.csv",
    "test":  "clinical_probability_exports/xgb_without_spotmag_test_probabilities.csv",
}

# RF - train/val ze spotmagami, test bez spotmagów
RF_FILES = {
    "train": "clinical_probability_exports/rf_with_spotmag_train_probabilities.csv",
    "val":   "clinical_probability_exports/rf_with_spotmag_val_probabilities.csv",
    "test":  "clinical_probability_exports/rf_with_spotmag_test_probabilities_no_spotmag.csv",
}


# =========================
# NENCZAK -> XGB
# =========================

NENCZAK_MODELS = {
    "nenczak_resnet18": {
        "train": "logity_nenczak/probabilities_output/train_logits_18_with_probability.csv",
        "val":   "logity_nenczak/probabilities_output/val_logits_18_with_probability.csv",
        "test":  "logity_nenczak/probabilities_output/test_logits_18_with_probability.csv",
    },
    "nenczak_resnet50": {
        "train": "logity_nenczak/probabilities_output/train_logits_50_with_probability.csv",
        "val":   "logity_nenczak/probabilities_output/val_logits_50_with_probability.csv",
        "test":  "logity_nenczak/probabilities_output/test_logits_50_with_probability.csv",
    },
    "nenczak_convnext_base": {
        "train": "logity_nenczak/probabilities_output/train_logits_base_with_probability.csv",
        "val":   "logity_nenczak/probabilities_output/val_logits_base_with_probability.csv",
        "test":  "logity_nenczak/probabilities_output/test_logits_base_with_probability.csv",
    },
    "nenczak_convnext_tiny": {
        "train": "logity_nenczak/probabilities_output/train_logits_tiny_with_probability.csv",
        "val":   "logity_nenczak/probabilities_output/val_logits_tiny_with_probability.csv",
        "test":  "logity_nenczak/probabilities_output/test_logits_tiny_with_probability.csv",
    },
}


# =========================
# WILK REMOVED -> XGB
# =========================

WILK_REMOVED_MODELS = {
    "wilk_resnet18_removed": {
        "train": "logity_wilk/ZPB_final_results/resnet18/spotmags_removed/train_resnet18_final_spotmags_removed.csv",
        "val":   "logity_wilk/ZPB_final_results/resnet18/spotmags_removed/val_resnet18_final_spotmags_removed.csv",
        "test":  "logity_wilk/ZPB_final_results/resnet18/spotmags_removed/test_resnet18_final_spotmags_removed.csv",
    },
    "wilk_resnet50_removed": {
        "train": "logity_wilk/ZPB_final_results/resnet50/spotmags_removed/train_resnet50_final_spotmags_removed.csv",
        "val":   "logity_wilk/ZPB_final_results/resnet50/spotmags_removed/val_resnet50_final_spotmags_removed.csv",
        "test":  "logity_wilk/ZPB_final_results/resnet50/spotmags_removed/test_resnet50_final_spotmags_removed.csv",
    },
}


# =========================
# WILK CUT -> RF
# =========================

WILK_CUT_MODELS = {
    "wilk_resnet18_cut": {
        "train": "logity_wilk/ZPB_final_results/resnet18/spotmags_cut/train_resnet18_final_spotmags_cut.csv",
        "val":   "logity_wilk/ZPB_final_results/resnet18/spotmags_cut/val_resnet18_final_spotmags_cut.csv",
        "test":  "logity_wilk/ZPB_final_results/resnet18/spotmags_cut/test_resnet18_final_spotmags_cut.csv",
    },
    "wilk_resnet50_cut": {
        "train": "logity_wilk/ZPB_final_results/resnet50/spotmags_cut/train_resnet50_final_spotmags_cut.csv",
        "val":   "logity_wilk/ZPB_final_results/resnet50/spotmags_cut/val_resnet50_final_spotmags_cut.csv",
        "test":  "logity_wilk/ZPB_final_results/resnet50/spotmags_cut/test_resnet50_final_spotmags_cut.csv",
    },
}


def load_clinical_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "new_path" not in df.columns:
        raise ValueError(f"Kliniczny plik nie ma kolumny 'new_path': {csv_path}")

    if "probability" not in df.columns:
        raise ValueError(f"Kliniczny plik nie ma kolumny 'probability': {csv_path}")

    if "label_bin" in df.columns:
        label = pd.to_numeric(df["label_bin"], errors="coerce")
    elif "label" in df.columns:
        if df["label"].dtype == object:
            mapping = {"negative": 0, "suspicious": 1}
            label = df["label"].map(mapping)
        else:
            label = pd.to_numeric(df["label"], errors="coerce")
    else:
        raise ValueError(f"Kliniczny plik nie ma ani 'label_bin' ani 'label': {csv_path}")

    out = pd.DataFrame({
        "new_path": df["new_path"].astype(str),
        "label": label.astype(int),
        "clinical_probability": pd.to_numeric(df["probability"], errors="coerce"),
    })

    if out["clinical_probability"].isna().any():
        raise ValueError(f"Clinical probability zawiera NaN: {csv_path}")

    return out


def load_image_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "new_path" not in df.columns:
        if "path" in df.columns:
            df = df.rename(columns={"path": "new_path"})
        else:
            raise ValueError(f"Obrazowy plik nie ma ani 'new_path' ani 'path': {csv_path}")

    if "probability" not in df.columns:
        raise ValueError(f"Obrazowy plik nie ma kolumny 'probability': {csv_path}")

    if "label" not in df.columns:
        raise ValueError(f"Obrazowy plik nie ma kolumny 'label': {csv_path}")

    if df["label"].dtype == object:
        mapping = {"negative": 0, "suspicious": 1}
        label = df["label"].map(mapping)
    else:
        label = pd.to_numeric(df["label"], errors="coerce")

    out = pd.DataFrame({
        "new_path": df["new_path"].astype(str),
        "image_label": label.astype(int),
        "image_probability": pd.to_numeric(df["probability"], errors="coerce"),
    })

    if out["image_probability"].isna().any():
        raise ValueError(f"Image probability zawiera NaN: {csv_path}")

    if "logit" in df.columns:
        out["image_logit"] = pd.to_numeric(df["logit"], errors="coerce")

    return out


def merge_one_pair(
    clinical_csv: str,
    image_csv: str,
    split_name: str,
    clinical_source: str,
    image_source: str,
    output_csv: str,
):
    clin_df = load_clinical_csv(clinical_csv)
    img_df = load_image_csv(image_csv)

    merged = clin_df.merge(img_df, on="new_path", how="inner")

    if len(merged) == 0:
        raise ValueError(
            f"Po merge wyszło 0 rekordów:\n"
            f"clinical={clinical_csv}\n"
            f"image={image_csv}"
        )

    merged["label_match"] = (merged["label"] == merged["image_label"])
    n_mismatch = int((~merged["label_match"]).sum())

    if n_mismatch > 0:
        print(f"UWAGA: {n_mismatch} mismatch labeli w {output_csv}")

    merged["split"] = split_name
    merged["clinical_source"] = clinical_source
    merged["image_source"] = image_source

    final_cols = [
        "new_path",
        "label",
        "clinical_probability",
        "image_probability",
    ]

    if "image_logit" in merged.columns:
        final_cols.append("image_logit")

    final_cols += [
        "split",
        "clinical_source",
        "image_source",
        "label_match",
    ]

    final_df = merged[final_cols].copy()
    final_df.to_csv(output_csv, index=False)

    print(f"Zapisano: {output_csv} | n={len(final_df)} | mismatch={n_mismatch}")


def merge_group(image_models: dict, clinical_files: dict, clinical_source: str):
    for image_source, split_map in image_models.items():
        for split_name in ["train", "val", "test"]:
            clinical_csv = clinical_files[split_name]
            image_csv = split_map[split_name]

            output_filename = f"{image_source}__{clinical_source}__{split_name}.csv"
            output_csv = os.path.join(OUTPUT_DIR, output_filename)

            merge_one_pair(
                clinical_csv=clinical_csv,
                image_csv=image_csv,
                split_name=split_name,
                clinical_source=clinical_source,
                image_source=image_source,
                output_csv=output_csv,
            )


def main():
    # 1) Nenczak + XGB
    merge_group(
        image_models=NENCZAK_MODELS,
        clinical_files=XGB_FILES,
        clinical_source="xgb_without_spotmag",
    )

    # 2) Wilk removed + XGB
    merge_group(
        image_models=WILK_REMOVED_MODELS,
        clinical_files=XGB_FILES,
        clinical_source="xgb_without_spotmag",
    )

    # 3) Wilk cut + RF
    merge_group(
        image_models=WILK_CUT_MODELS,
        clinical_files=RF_FILES,
        clinical_source="rf_trainval_with_spotmag_test_without_spotmag",
    )

    print("\nGotowe.")
    print("Wszystkie pliki zapisane w:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()