#!/usr/bin/env python3
"""
Merge per-latent SAE CSVs into one long table.

Input:
    sae_latents/
        latent_0000.csv
        latent_0001.csv
        ...

Output:
    sae_latents_long.csv
"""

import os
import csv
import glob

# ----------------------------
# Config
# ----------------------------
LATENT_DIR = "../Data/sae_latents"
OUT_PATH   = "../Data/sae_latents_long.csv"

# ----------------------------
# Collect files
# ----------------------------
files = sorted(glob.glob(os.path.join(LATENT_DIR, "latent_*.csv")))

if not files:
    raise RuntimeError(f"No latent CSVs found in {LATENT_DIR}")

print(f"🔎 Found {len(files)} latent CSV files")

# ----------------------------
# Merge
# ----------------------------
rows_written = 0

with open(OUT_PATH, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)

    # unified header
    writer.writerow([
        "latent_id",
        "activation",
        "lemma",
        "present_phones",
        "past_phones",
    ])

    for path in files:
        latent_id = int(os.path.basename(path).split("_")[1].split(".")[0])

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                writer.writerow([
                    latent_id,
                    float(row["activation"]),
                    row["lemma"],
                    row["present_phones"],
                    row["past_phones"],
                ])
                rows_written += 1

print(f"✅ Wrote {rows_written:,} rows → {OUT_PATH}")
