#!/usr/bin/env python3

import torch
from rumelhart_mcclelland_extension import (
    PastTenseNet,
    SparseAutoencoder,
    encode,
    decode,
    load_cmudict,
)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_TAG = "L1_H256"
SAE_TAG   = "L1_H256"
SAE_LAYER = None

MODEL_PATH = f"../models/past_tense_net_{MODEL_TAG}.pt"
SAE_PATH   = f"../models/sae_{SAE_TAG}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD = "go"          # <-- single word here
MIN_ACT = 1e-4           # activation threshold

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

if WORD not in cmu:
    raise ValueError(f"{WORD} not found in CMUdict")

# ----------------------------
# Helpers
# ----------------------------
def get_sae_hidden(hidden_states, sae_layer):
    if sae_layer is None:
        return hidden_states[-1]
    return hidden_states[sae_layer]

# ----------------------------
# Run analysis
# ----------------------------
with torch.no_grad():
    pres = cmu[WORD][0]
    x = encode(pres, phone2idx).to(DEVICE)

    # ---- base forward ----
    y_base, hidden_states = model(x.unsqueeze(0), return_all_hidden=True)
    base_pred = decode(y_base.squeeze(0).cpu(), idx2phone)

    h = get_sae_hidden(hidden_states, SAE_LAYER)
    _, z = sae(h)

    z = z.squeeze(0)  # [n_latents]

    active_latents = torch.nonzero(z > MIN_ACT, as_tuple=False).squeeze(1)

    print(f"\n🔎 Word: {WORD}")
    print(f"Phones: {' '.join(pres)}")
    print(f"Base prediction: {' '.join(base_pred)}")
    print(f"Active latents (>{MIN_ACT}): {len(active_latents)}\n")

    for i in active_latents.tolist():
        act = z[i].item()

        # ---- ablate one latent ----
        z_mod = z.clone()
        z_mod[i] = -z[i]

        h_mod = sae.decoder(z_mod.unsqueeze(0))
        y_mod = model.output(h_mod)
        pred_mod = decode(y_mod.squeeze(0).cpu(), idx2phone)

        print(
            f"latent {i:4d} | act={act:6.3f} | "
            f"pred → {' '.join(pred_mod)}"
        )
