#!/usr/bin/env python3
"""
Merge per-latent SAE CSVs into one long table
for ALL experiment configurations.

Input (per experiment):
    ../Data/sae_latents/{MODEL_TAG}/
        latent_0000.csv
        latent_0001.csv
        ...

Output (per experiment):
    ../Data/sae_latents_long/{MODEL_TAG}.csv
"""

import os
import csv
import glob
from typing import List

from rumelhart_mcclelland_extension import ExperimentConfig


# ============================================================
# Helper: derive MODEL_TAG exactly as in export script
# ============================================================
def model_tag_from_cfg(cfg: ExperimentConfig) -> str:
    base = f"L{cfg.num_hidden_layers}_H{cfg.hidden_size}"

    # SAE placement tag
    if cfg.sae_layer is None:
        sae_part = "SAE@final"
    else:
        sae_part = f"SAE@L{cfg.sae_layer + 1}"

    # Add top-k tag if present
    if getattr(cfg, "sae_top_k", None) is not None:
        sae_part += f"_K{cfg.sae_top_k}"

    return f"{base}_{sae_part}"



# ============================================================
# Merge one experiment
# ============================================================
def merge_one_experiment(
    cfg: ExperimentConfig,
    latent_root="../Data/sae_latents",
    out_root="../Data/sae_latents_long",
):
    model_tag = model_tag_from_cfg(cfg)

    latent_dir = os.path.join(latent_root, model_tag)
    out_path = os.path.join(out_root, f"{model_tag}.csv")

    if not os.path.isdir(latent_dir):
        print(f"⚠️  Skipping {model_tag}: no latent directory")
        return

    files = sorted(glob.glob(os.path.join(latent_dir, "latent_*.csv")))
    if not files:
        print(f"⚠️  Skipping {model_tag}: no latent CSVs")
        return

    os.makedirs(out_root, exist_ok=True)

    print(f"\n🔧 Merging SAE latents for {model_tag}")
    print(f"📁 Input dir : {latent_dir}")
    print(f"📄 Output    : {out_path}")
    print(f"🔎 Files     : {len(files)}")

    rows_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)

        writer.writerow([
            "latent_id",
            "activation",
            "lemma",
            "present_phones",
            "past_phones",
        ])

        for path in files:
            latent_id = int(
                os.path.basename(path)
                .replace("latent_", "")
                .replace(".csv", "")
            )

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

    print(f"✅ Wrote {rows_written:,} rows")


# ============================================================
# Experiment grid (MUST match export script)
# ============================================================
experiments: List[ExperimentConfig] = []

for L in [1, 2, 3, 4]:
    for num_sae_topk in [5, 10, 15, 20]:
        experiments.append(
            ExperimentConfig(
                hidden_size = 256,
                num_hidden_layers=L,
                use_sae=True,
                train_sae=True,
                finetune_with_sae=False,
                sae_layer = None,
                sae_top_k=num_sae_topk,
            )
        )

# SAE placements for 3-layer model
for sae_layer in [0, 1]:
    for num_hidden_l in [3, 4]:
        experiments.append(
            ExperimentConfig(
                hidden_size = 256,
                num_hidden_layers=num_hidden_l,
                use_sae=True,
                train_sae=True,
                finetune_with_sae=False,
                sae_layer=sae_layer,
                sae_top_k = 10,
            )
        )



# ============================================================
# Run
# ============================================================
for cfg in experiments:
    merge_one_experiment(cfg)
