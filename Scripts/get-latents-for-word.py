#!/usr/bin/env python3
"""
Inspect SAE latents for words by specifying:
- num_hidden_layers
- sae_layer
- sae_top_k

Paths are auto-generated using your experiment naming convention.
"""

import os
import re
import csv
import torch
from typing import List, Dict
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PHONES = 10


###############################################################################
# SAME AS YOUR TRAINING SCRIPT — DON'T CHANGE
###############################################################################

def model_tag_from_cfg(num_hidden_layers, hidden_size, use_sae, sae_layer, sae_top_k):
    base = f"L{num_hidden_layers}_H{hidden_size}"
    if not use_sae:
        return base

    if sae_layer is None:
        sae_part = "SAE@final"
    else:
        sae_part = f"SAE@L{sae_layer + 1}"

    if sae_top_k is not None:
        sae_part += f"_K{sae_top_k}"

    return f"{base}_{sae_part}"


def encode(seq: List[str], phone2idx: Dict[str,int]) -> torch.Tensor:
    seq = seq[: MAX_PHONES - 1] + ["<EOS>"]
    seq = seq + ["_"] * (MAX_PHONES - len(seq))
    V = len(phone2idx)
    vec = torch.zeros(MAX_PHONES * V)
    for i,p in enumerate(seq):
        vec[i*V + phone2idx[p]] = 1.0
    return vec


def decode(vec, idx2phone):
    mat = vec.view(MAX_PHONES, len(idx2phone))
    out=[]
    for row in mat:
        ph = idx2phone[row.argmax().item()]
        if ph=="<EOS>": break
        if ph=="_": continue
        out.append(ph)
    return out


class PastTenseNet(torch.nn.Module):
    def __init__(self, inp,hid,out,num_hidden=1):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(inp,hid))
        for _ in range(num_hidden-1):
            self.hidden_layers.append(torch.nn.Linear(hid,hid))
        self.output = torch.nn.Linear(hid,out)

    def forward(self,x,return_all_hidden=False):
        h=x
        hidden=[]
        for layer in self.hidden_layers:
            h=torch.tanh(layer(h))
            hidden.append(h)
        y=self.output(h)
        if return_all_hidden:
            return y,hidden
        return y

class SparseAutoencoder(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,top_k=None):
        super().__init__()
        self.encoder=torch.nn.Linear(input_dim,hidden_dim)
        self.decoder=torch.nn.Linear(hidden_dim,input_dim)
        self.top_k=top_k

    def forward(self,x):

        # ---- ensure 2-D ----
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        z = torch.relu(self.encoder(x))   # [B,H]

        if self.top_k is not None:
            k = min(self.top_k, z.shape[1])
            vals, idx = torch.topk(z, k, dim=1)

            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, idx, vals)
            z = z_sparse

        xhat = self.decoder(z)

        if squeeze:
            xhat = xhat.squeeze(0)
            z = z.squeeze(0)

        return xhat, z



def get_sae_hidden(hidden_states, sae_layer):
    return hidden_states[-1] if sae_layer is None else hidden_states[sae_layer]


###############################################################################
# LOADING HELPERS
###############################################################################

def load_cmudict(path):
    STRESS_RE = re.compile(r"\d")
    cmu={}
    with open(path,encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"): continue
            parts=line.strip().split()
            if len(parts)<2: continue
            w=re.sub(r"\(\d+\)$","",parts[0]).lower()
            ph=[STRESS_RE.sub("",p) for p in parts[1:]]
            cmu.setdefault(w,[]).append(ph)
    return cmu


def read_train_types(path):
    rows=[]
    with open(path,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["lemma"],
                         row["present_phones"].split(),
                         row["past_phones"].split()))
    return rows


###############################################################################
# LOAD MODEL + SAE BASED ON CONFIG VALUES
###############################################################################

def load_all(num_hidden_layers, hidden_size, sae_layer, sae_top_k):

    tag = model_tag_from_cfg(
        num_hidden_layers,
        hidden_size,
        use_sae=True,
        sae_layer=sae_layer,
        sae_top_k=sae_top_k
    )

    base_tag = f"L{num_hidden_layers}_H{hidden_size}"

    BASE_MODEL_PATH = f"../models/past_tense_net_{base_tag}.pt"
    SAE_PATH        = f"../models/sae_{tag}.pt"

    print(f"\nLoading BASE model: {BASE_MODEL_PATH}")
    print(f"Loading SAE model : {SAE_PATH}")

    ckpt = torch.load(BASE_MODEL_PATH, map_location=DEVICE)

    inventory = ckpt["inventory"]
    phone2idx = ckpt["phone2idx"]
    idx2phone = ckpt["idx2phone"]

    model = PastTenseNet(
        inp=MAX_PHONES*len(inventory),
        hid=hidden_size,
        out=MAX_PHONES*len(inventory),
        num_hidden=num_hidden_layers,
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sae_ckpt = torch.load(SAE_PATH, map_location=DEVICE)

    sae = SparseAutoencoder(
        input_dim=sae_ckpt["config"]["input_dim"],
        hidden_dim=sae_ckpt["config"]["hidden_dim"],
        top_k=sae_ckpt["config"]["top_k"]
    ).to(DEVICE)

    sae.load_state_dict(sae_ckpt["state_dict"])
    sae.eval()

    return model, sae, phone2idx, idx2phone


###############################################################################
# CORE ANALYSIS
###############################################################################

def get_word_latents(word, model, sae, phone2idx, idx2phone, cmu, sae_layer):

    pres = cmu[word][0]
    x = encode(pres, phone2idx).to(DEVICE)

    with torch.no_grad():
        y, hidden_states = model(x, return_all_hidden=True)
        h = get_sae_hidden(hidden_states, sae_layer)
        _, z = sae(h)
        pred = decode(y.cpu(), idx2phone)

    active = (z > 0).nonzero(as_tuple=True)[0].cpu().tolist()


    return pres, pred, z.cpu(), active



def build_latent_index(train_rows, model, sae, phone2idx, idx2phone, sae_layer):
    idx = defaultdict(list)
    with torch.no_grad():
        for lemma, pres, _ in train_rows:
            x = encode(pres, phone2idx).to(DEVICE)
            _, h_states = model(x, return_all_hidden=True)
            h = get_sae_hidden(h_states, sae_layer)
            _, z = sae(h)
            active = (z > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            for lat in active:
                idx[lat].append(lemma)
    return idx


def show_word(word, *args, **kw):
    pres,pred,z,active=get_word_latents(word,*args,**kw)

    print("\n====================================")
    print(f"WORD: {word}")
    print("====================================")
    print("Input :", " ".join(pres))
    print("Pred  :", " ".join(pred))
    print("\nActive latents:", active)

    print("\nlatent | activation")
    print("-------------------")
    for i in sorted(active):
        print(f"{i:6d} | {z[i]:.6f}")


###############################################################################
# MAIN
###############################################################################

if __name__=="__main__":

    # 🔧 CHOOSE WHICH SAE MODEL YOU WANT
    NUM_HIDDEN_LAYERS = 1
    HIDDEN_SIZE = 1024
    SAE_LAYER = None     # None = final layer, or 0 / 1
    SAE_TOP_K = 5

    TRAIN_PATH = "../Data/datasets/fixed_train_sample_SEED0_N80000.csv"
    CMU_PATH   = "../Data/data_cache/cmudict.dict"

    cmu = load_cmudict(CMU_PATH)
    train_rows = read_train_types(TRAIN_PATH)

    model, sae, phone2idx, idx2phone = load_all(
        NUM_HIDDEN_LAYERS,
        HIDDEN_SIZE,
        SAE_LAYER,
        SAE_TOP_K
    )

    latent_index = build_latent_index(
        train_rows,
        model,
        sae,
        phone2idx,
        idx2phone,
        SAE_LAYER
    )

    # 🔍 Inspect a word
    word = "rise"
    show_word(word, model, sae, phone2idx, idx2phone, cmu, SAE_LAYER)

    print("\n======= LATENT NEIGHBORHOODS =======\n")
    _,_,_,active = get_word_latents(word, model, sae, phone2idx, idx2phone, cmu, SAE_LAYER)

    for lat in active:
        neighbors = sorted(set(latent_index[lat]))
        print(f"latent {lat} → {len(neighbors)} words")
        print(", ".join(neighbors[:40]))
        print()
