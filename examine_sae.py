#!/usr/bin/env python3
"""
Inspect which verbs activate a given SAE latent.

Usage:
    python inspect_sae_latent.py --latent_id 123 --top_n 50
"""

import argparse
import torch
import json
import os

from collections import defaultdict
from typing import List, Tuple

# ----------------------------
# Import your model definitions
# ----------------------------
from rumelhart_mcclelland_extension import (
    PastTenseNet,
    SparseAutoencoder,
    encode,
    decode,
    load_unimorph,
    load_cmudict,
)

# ----------------------------
# Paths (adjust if needed)
# ----------------------------
MODEL_PATH = "./models/past_tense_net.pt"
SAE_PATH   = "./models/sae.pt"
CACHE_DIR  = "./data_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--latent_id", type=int, required=True)
parser.add_argument("--top_n", type=int, default=50)
parser.add_argument("--min_activation", type=float, default=0.0)
args = parser.parse_args()

LATENT_ID = args.latent_id
TOP_N = args.top_n
MIN_ACT = args.min_activation

# ----------------------------
# Load past tense model
# ----------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

inventory = ckpt["inventory"]
phone2idx = ckpt["phone2idx"]
idx2phone = ckpt["idx2phone"]

model = PastTenseNet(
    inp=len(inventory) * ckpt["config"]["MAX_PHONES"],
    hid=ckpt["config"]["HIDDEN_SIZE"],
    out=len(inventory) * ckpt["config"]["MAX_PHONES"],
    num_hidden=ckpt["config"]["NUM_HIDDEN_LAYERS"],
).to(DEVICE)

model.load_state_dict(ckpt["state_dict"])
model.eval()

print(f"📦 Loaded past tense model from {MODEL_PATH}")

# ----------------------------
# Load SAE
# ----------------------------
sae_ckpt = torch.load(SAE_PATH, map_location=DEVICE)

sae = SparseAutoencoder(
    input_dim=sae_ckpt["config"]["input_dim"],
    hidden_dim=sae_ckpt["config"]["hidden_dim"],
    top_k=sae_ckpt["config"]["top_k"],
).to(DEVICE)

sae.load_state_dict(sae_ckpt["state_dict"])
sae.eval()

print(
    f"📦 Loaded SAE (hidden={sae_ckpt['config']['hidden_dim']}, "
    f"top_k={sae_ckpt['config']['top_k']})"
)

# ----------------------------
# Load verb data
# ----------------------------
unimorph_path = os.path.join(CACHE_DIR, "unimorph_eng.txt")
cmudict_path  = os.path.join(CACHE_DIR, "cmudict.dict")

verb_pairs = load_unimorph(unimorph_path)
cmu = load_cmudict(cmudict_path)

examples: List[Tuple[str, List[str], List[str]]] = []

for vp in verb_pairs:
    if vp.lemma in cmu and vp.past in cmu:
        pres = cmu[vp.lemma][0]
        past = cmu[vp.past][0]
        examples.append((vp.lemma, pres, past))

print(f"🔎 Loaded {len(examples)} verb pairs")

# ----------------------------
# Run all verbs through model + SAE
# ----------------------------
activations = []

with torch.no_grad():
    for lemma, pres_phones, past_phones in examples:
        x = encode(pres_phones, phone2idx).to(DEVICE)

        _, hidden = model(x.unsqueeze(0), return_hidden=True)
        _, z = sae(hidden)

        act = z[0, LATENT_ID].item()

        if act > MIN_ACT:
            activations.append((
                act,
                lemma,
                pres_phones,
                past_phones
            ))

# ----------------------------
# Sort and print
# ----------------------------
activations.sort(reverse=True, key=lambda x: x[0])

print(f"\n🔥 Latent {LATENT_ID} activated by {len(activations)} verbs")
print("=" * 80)

for act, lemma, pres, past in activations[:TOP_N]:
    print(
        f"{lemma:12s} | "
        f"in:  {' '.join(pres):18s} | "
        f"past: {' '.join(past):18s} | "
        f"act: {act:.4f}"
    )

if len(activations) > TOP_N:
    print(f"\n… ({len(activations) - TOP_N} more)")
