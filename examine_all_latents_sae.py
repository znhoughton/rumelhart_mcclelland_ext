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

# --------------------------------------------------
# IMPORT YOUR MODEL CODE
# --------------------------------------------------
from rumelhart_mcclelland_extension import (
    PastTenseNet,
    SparseAutoencoder,
    encode,
    load_unimorph,
    load_cmudict,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = "./models/past_tense_net.pt"
SAE_PATH   = "./models/sae.pt"
CACHE_DIR  = "./data_cache"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Args
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="sae_latents")
parser.add_argument("--min_activation", type=float, default=0.0)
args = parser.parse_args()

OUT_DIR = args.out_dir
MIN_ACT = args.min_activation

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load past tense model
# --------------------------------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

inventory = ckpt["inventory"]
phone2idx = ckpt["phone2idx"]

model = PastTenseNet(
    inp=len(inventory) * ckpt["config"]["MAX_PHONES"],
    hid=ckpt["config"]["HIDDEN_SIZE"],
    out=len(inventory) * ckpt["config"]["MAX_PHONES"],
    num_hidden=ckpt["config"]["NUM_HIDDEN_LAYERS"],
).to(DEVICE)

model.load_state_dict(ckpt["state_dict"])
model.eval()

print(f"📦 Loaded past tense model")

# --------------------------------------------------
# Load SAE
# --------------------------------------------------
sae_ckpt = torch.load(SAE_PATH, map_location=DEVICE)

sae = SparseAutoencoder(
    input_dim=sae_ckpt["config"]["input_dim"],
    hidden_dim=sae_ckpt["config"]["hidden_dim"],
    top_k=sae_ckpt["config"]["top_k"],
).to(DEVICE)

sae.load_state_dict(sae_ckpt["state_dict"])
sae.eval()

N_LATENTS = sae_ckpt["config"]["hidden_dim"]

print(
    f"📦 Loaded SAE "
    f"(latents={N_LATENTS}, top_k={sae_ckpt['config']['top_k']})"
)

# --------------------------------------------------
# Load verbs
# --------------------------------------------------
unimorph_path = os.path.join(CACHE_DIR, "unimorph_eng.txt")
cmudict_path  = os.path.join(CACHE_DIR, "cmudict.dict")

verb_pairs = load_unimorph(unimorph_path)
cmu = load_cmudict(cmudict_path)

examples: List[Tuple[str, List[str], List[str]]] = []

for vp in verb_pairs:
    if vp.lemma in cmu and vp.past in cmu:
        examples.append((
            vp.lemma,
            cmu[vp.lemma][0],
            cmu[vp.past][0],
        ))

print(f"🔎 Loaded {len(examples)} verb pairs")

# --------------------------------------------------
# Collect activations
# --------------------------------------------------
latent_rows = defaultdict(list)

with torch.no_grad():
    for lemma, pres, past in examples:
        x = encode(pres, phone2idx).to(DEVICE)

        _, hidden = model(x.unsqueeze(0), return_hidden=True)
        _, z = sae(hidden)

        z = z.squeeze(0)

        for latent_id in torch.nonzero(z > MIN_ACT, as_tuple=False):
            i = latent_id.item()
            latent_rows[i].append((
                z[i].item(),
                lemma,
                " ".join(pres),
                " ".join(past),
            ))

# --------------------------------------------------
# Write CSVs
# --------------------------------------------------
for latent_id in range(N_LATENTS):
    rows = latent_rows.get(latent_id, [])

    if not rows:
        continue

    rows.sort(reverse=True, key=lambda x: x[0])

    out_path = os.path.join(OUT_DIR, f"latent_{latent_id:04d}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "activation",
            "lemma",
            "present_phones",
            "past_phones",
        ])
        for row in rows:
            writer.writerow(row)

print(f"\n✅ Exported {len(latent_rows)} latent CSV files → {OUT_DIR}")
