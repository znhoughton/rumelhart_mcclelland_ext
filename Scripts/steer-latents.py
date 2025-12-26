#!/usr/bin/env python3

import torch
from rumelhart_mcclelland_extension import (
    PastTenseNet,
    encode,
    decode,
    load_cmudict,
)

# ----------------------------
# CONFIG
# ----------------------------
MODEL_TAG = "L3_H256"   # adjust if needed
MODEL_PATH = f"../models/past_tense_net_{MODEL_TAG}.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORDS = ["go", "went", "be", "was", "have", "had"]

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
# CMUdict
# ----------------------------
cmu = load_cmudict("../Data/data_cache/cmudict.dict")

# ----------------------------
# Run base model only
# ----------------------------
print("\n🧪 BASE MODEL ONLY (NO SAE)\n")

with torch.no_grad():
    for w in WORDS:
        if w not in cmu:
            print(f"{w:8s} | not in CMUdict")
            continue

        pres = cmu[w][0]
        x = encode(pres, phone2idx).to(DEVICE)

        y = model(x.unsqueeze(0))
        pred = decode(y.squeeze(0).cpu(), idx2phone)

        print(f"{w:8s} | {' '.join(pres):10s} → {' '.join(pred)}")
