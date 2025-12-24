#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import List

from rumelhart_mcclelland_extension import (
    PastTenseNet,
    SparseAutoencoder,
    encode,
    decode,
    load_cmudict,
)

def report_latent_activations(z: torch.Tensor):
    for i in TARGET_LATENTS:
        val = z[0, i].item()
        print(f"      latent {i:4d} activation before: {val:.4f}")

# ----------------------------
# CONFIG
# ----------------------------
MODEL_TAG = "L1_H256"           # base model tag
SAE_TAG   = "L1_H256"           # or "L3_H256_SAE@L1"
SAE_LAYER = None                # must match how SAE was trained

MODEL_PATH = f"../models/past_tense_net_{MODEL_TAG}.pt"
SAE_PATH   = f"../models/sae_{SAE_TAG}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Latent intervention
TARGET_LATENTS = [282]           # indices of latents you want to manipulate
MODE = "zero"                   # "zero", "boost", "set"
BOOST_FACTOR = 2.0              # only used if MODE == "boost"
SET_VALUE = 5.0                 # only used if MODE == "set"

# Test words
WORDS = ["subjugate", "change", "long", "dredge", "lunge", "singe", "hinge", "range"]

# ----------------------------
# Load model
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

# ----------------------------
# CMUdict
# ----------------------------
cmu = load_cmudict("../Data/data_cache/cmudict.dict")

# ----------------------------
# Helper
# ----------------------------
def get_sae_hidden(hidden_states, sae_layer):
    if sae_layer is None:
        return hidden_states[-1]
    return hidden_states[sae_layer]

def intervene(z: torch.Tensor) -> torch.Tensor:
    z = z.clone()

    for i in TARGET_LATENTS:
        if MODE == "zero":
            z[:, i] = 0.0
        elif MODE == "boost":
            z[:, i] *= BOOST_FACTOR
        elif MODE == "set":
            z[:, i] = SET_VALUE
        else:
            raise ValueError(f"Unknown MODE={MODE}")

    return z

# ----------------------------
# Run intervention
# ----------------------------
print(f"\n🔬 Latent intervention: {MODE} on {TARGET_LATENTS}\n")

with torch.no_grad():
    for w in WORDS:
        if w not in cmu:
            continue

        pres = cmu[w][0]
        x = encode(pres, phone2idx).to(DEVICE)

        # ---- normal forward ----
        y_base, hidden_states = model(x.unsqueeze(0), return_all_hidden=True)
        pred_base = decode(y_base.squeeze(0).cpu(), idx2phone)

        # ---- SAE path ----
        h = get_sae_hidden(hidden_states, SAE_LAYER)
        recon, z = sae(h)

        print(f"\n{w}")
        report_latent_activations(z)

        # ---- intervene ----
        z_mod = intervene(z)

        # ---- report before → after ----
        for i in TARGET_LATENTS:
            before = z[0, i].item()
            after  = z_mod[0, i].item()
            print(f"      latent {i:4d}: {before:.4f} → {after:.4f}")

        h_mod = sae.decoder(z_mod)

        # ---- output layer ----
        y_mod = model.output(h_mod)
        pred_mod = decode(y_mod.squeeze(0).cpu(), idx2phone)

        print(f"  base: {' '.join(pred_base)}")
        print(f"  mod : {' '.join(pred_mod)}")
