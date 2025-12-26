#!/usr/bin/env python3
"""
Export SAE latent activations to per-latent CSV files.

Each latent gets its own CSV:
    latent_0000.csv
    latent_0001.csv
    ...

Each CSV contains verbs that activate that latent,
sorted by activation strength.

Usage:
    python export_sae_latents_to_csv.py \
        --out_dir sae_latents \
        --min_activation 0.0
"""

import os
import csv
import argparse
import torch

from typing import List, Tuple
from collections import defaultdict

def get_sae_hidden(hidden_states, sae_layer):
    if sae_layer is None:
        return hidden_states[-1]
    return hidden_states[sae_layer]

# --------------------------------------------------
# IMPORT YOUR MODEL CODE
# --------------------------------------------------
from rumelhart_mcclelland_extension import (
    PastTenseNet,
    SparseAutoencoder,
    encode,
    load_unimorph,
    load_cmudict,
    ExperimentConfig
)

def export_sae_latents_for_experiment(
    cfg,
    out_root="../Data/sae_latents",
    min_activation=0.0,
    cache_dir="../Data/data_cache",
    device=None,
):
    """
    Export SAE latent activations for a single ExperimentConfig.

    Creates:
        out_root/{MODEL_TAG}/latent_XXXX.csv
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Derive tags + paths
    # --------------------------------------------------
    BASE_MODEL_TAG = f"L{cfg.num_hidden_layers}_H{cfg.hidden_size}"

    if cfg.use_sae:
        if cfg.sae_layer is None:
            sae_part = "SAE@final"
        else:
            sae_part = f"SAE@L{cfg.sae_layer + 1}"

        if getattr(cfg, "sae_top_k", None) is not None:
            sae_part += f"_K{cfg.sae_top_k}"

        MODEL_TAG = f"{BASE_MODEL_TAG}_{sae_part}"
    else:
        raise ValueError("export_sae_latents_for_experiment requires USE_SAE=True")


    MODEL_PATH = f"../models/past_tense_net_{BASE_MODEL_TAG}.pt"
    SAE_PATH   = f"../models/sae_{MODEL_TAG}.pt"

    OUT_DIR = os.path.join(out_root, MODEL_TAG)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n🔧 Exporting SAE latents for {MODEL_TAG}")
    print(f"📄 Base model: {MODEL_PATH}")
    print(f"📄 SAE model : {SAE_PATH}")
    print(f"📁 Output    : {OUT_DIR}")

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------
    ckpt = torch.load(MODEL_PATH, map_location=device)

    inventory = ckpt["inventory"]
    phone2idx = ckpt["phone2idx"]

    model = PastTenseNet(
        inp=len(inventory) * ckpt["config"]["MAX_PHONES"],
        hid=ckpt["config"]["HIDDEN_SIZE"],
        out=len(inventory) * ckpt["config"]["MAX_PHONES"],
        num_hidden=ckpt["config"]["NUM_HIDDEN_LAYERS"],
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --------------------------------------------------
    # Load SAE
    # --------------------------------------------------
    sae_ckpt = torch.load(SAE_PATH, map_location=device)

    sae = SparseAutoencoder(
        input_dim=sae_ckpt["config"]["input_dim"],
        hidden_dim=sae_ckpt["config"]["hidden_dim"],
        top_k=sae_ckpt["config"]["top_k"],
    ).to(device)

    sae.load_state_dict(sae_ckpt["state_dict"])
    sae.eval()

    N_LATENTS = sae_ckpt["config"]["hidden_dim"]

    print(f"🧠 SAE latents: {N_LATENTS}")

    # --------------------------------------------------
    # Load verbs
    # --------------------------------------------------
    unimorph_path = os.path.join(cache_dir, "unimorph_eng.txt")
    cmudict_path  = os.path.join(cache_dir, "cmudict.dict")

    verb_pairs = load_unimorph(unimorph_path)
    cmu = load_cmudict(cmudict_path)

    examples = []
    for vp in verb_pairs:
        if vp.lemma in cmu and vp.past in cmu:
            examples.append((
                vp.lemma,
                cmu[vp.lemma][0],
                cmu[vp.past][0],
            ))

    print(f"🔎 Verbs loaded: {len(examples)}")

    # --------------------------------------------------
    # Collect activations
    # --------------------------------------------------
    from collections import defaultdict
    latent_rows = defaultdict(list)

    with torch.no_grad():
        for lemma, pres, past in examples:
            x = encode(pres, phone2idx).to(device)

            _, hidden_states = model(x.unsqueeze(0), return_all_hidden=True)
            h = get_sae_hidden(hidden_states, cfg.sae_layer)

            _, z = sae(h)
            z = z.squeeze(0)

            active = torch.nonzero(z > min_activation, as_tuple=False).squeeze(1)

            for i in active.tolist():
                latent_rows[i].append((
                    float(z[i]),
                    lemma,
                    " ".join(pres),
                    " ".join(past),
                ))

    # --------------------------------------------------
    # Write CSVs
    # --------------------------------------------------
    import csv

    written = 0
    for latent_id, rows in latent_rows.items():
        rows.sort(key=lambda x: x[0], reverse=True)

        out_path = os.path.join(OUT_DIR, f"latent_{latent_id:04d}.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "activation",
                "lemma",
                "present_phones",
                "past_phones",
            ])
            writer.writerows(rows)

        written += 1

    print(f"✅ Wrote {written} latent CSVs → {OUT_DIR}")


experiments = []

# Base SAE on final layer
for L in [1, 2, 3, 4]:
    for num_sae_topk in [5, 10, 15, 20]:
        experiments.append(
            ExperimentConfig(
                hidden_size = 256,
                num_hidden_layers=L,
                use_sae=True,
                train_sae=True,
                finetune_with_sae=False,
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

for cfg in experiments:
    export_sae_latents_for_experiment(
        cfg,
        out_root="../Data/sae_latents",
        min_activation=0.0,
    )
