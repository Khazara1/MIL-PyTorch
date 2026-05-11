import os
import pandas as pd
import torch
import torch.nn.functional as F

INPUT_DIR = "logity_nenczak/logits_output"
OUTPUT_DIR = "logity_nenczak/probabilities_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_logits_csv_to_probabilities(input_csv_path: str, output_csv_path: str):
    df = pd.read_csv(input_csv_path)

    required_cols = {"new_path", "logit", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn {missing} w pliku: {input_csv_path}")

    df["logit"] = pd.to_numeric(df["logit"], errors="coerce")
    if df["logit"].isna().any():
        n_bad = int(df["logit"].isna().sum())
        raise ValueError(
            f"Plik {input_csv_path} ma {n_bad} niepoprawnych wartości w kolumnie 'logit'"
        )

    logits_tensor = torch.tensor(df["logit"].values, dtype=torch.float32)
    probabilities_tensor = F.sigmoid(logits_tensor)
    df["probability"] = probabilities_tensor.cpu().numpy()

    df = df[["new_path", "logit", "probability", "label"]]

    df.to_csv(output_csv_path, index=False)
    print(f"Zapisano: {output_csv_path}")


def main():
    csv_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".csv")
    )

    if not csv_files:
        print(f"Nie znaleziono żadnych plików CSV w: {INPUT_DIR}")
        return

    for filename in csv_files:
        input_path = os.path.join(INPUT_DIR, filename)

        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_with_probability{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        convert_logits_csv_to_probabilities(input_path, output_path)

    print("\nGotowe.")
    print("Nowe pliki zapisane w:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()