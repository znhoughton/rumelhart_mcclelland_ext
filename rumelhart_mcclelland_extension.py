#!/usr/bin/env python3
"""
English Past Tense Learner
- ARPABET input/output (CMUdict)
- Large-scale verb list (UniMorph)
- Token-frequency exposure via Google Books 1-gram corpus (.gz streaming)
- 1-hidden-layer feedforward neural network
"""

from __future__ import annotations

import os
import re
import math
import random
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import gzip
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import json


# ============================================================
# Configuration
# ============================================================
CACHE_DIR = "./data_cache"
UNIMORPH_URL = "https://raw.githubusercontent.com/unimorph/eng/master/eng"
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
UNIGRAM_CACHE_PATH = os.path.join(CACHE_DIR, "google_1gram_cache.json")


# Google Books Ngrams 1-gram corpus (2012 snapshot)
UNIGRAM_BASE = "http://storage.googleapis.com/books/ngrams/books/"
UNIGRAM_PREFIX = "googlebooks-eng-all-1gram-20120701-"

MAX_PHONES = 10
HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 1  
EPOCHS = 2000
LR = 0.01
TEST_RATIO = 0.2
MAX_REPS = 40
SEED = 0
# ----------------------------
# Saving / loading
# ----------------------------
SAVE_MODEL = False
LOAD_MODEL = True
MODEL_PATH = "./models/past_tense_net.pt"

# ----------------------------
# Sparse Autoencoder (optional)
# ----------------------------
USE_SAE = True
TRAIN_SAE = True
FINETUNE_WITH_SAE = False

SAE_HIDDEN_SIZE = 1024
SAE_L1 = 1e-3
SAE_EPOCHS = 200
SAE_LR = 1e-3
SAE_PATH = "./models/sae.pt"
SAE_TOP_K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Utilities
# ============================================================
def download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"↓ downloading {url}")
        urllib.request.urlretrieve(url, path)

def exact_match(pred: List[str], gold: List[str]) -> bool:
    return pred == gold


def edit_distance(a: List[str], b: List[str]) -> int:
    # simple Levenshtein distance
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def past_tense_suffix_type(phones: List[str]) -> str:
    if len(phones) >= 2 and phones[-2:] == ["IH", "D"]:
        return "IH-D"
    if phones and phones[-1] == "T":
        return "T"
    if phones and phones[-1] == "D":
        return "D"
    return "OTHER"

VOWELS = {
    "AA","AE","AH","AO","AW","AY",
    "EH","ER","EY","IH","IY",
    "OW","OY","UH","UW"
}

def phones_match(p_pres: str, p_past: str) -> bool:
    # Allow IH in *gold past* to match any vowel in present (your rule)
    if p_past == "IH" and p_pres in VOWELS:
        return True
    return p_pres == p_past

def shared_stem_boundary(pres: List[str], past: List[str]) -> int:
    """Longest prefix match length under phones_match()."""
    i = 0
    while i < len(pres) and i < len(past) and phones_match(pres[i], past[i]):
        i += 1
    return i

def morphologically_correct(pred: List[str], pres: List[str], gold: List[str]) -> bool:
    """
    Your new rule:

    1) If pres == gold (zero-change), pred must match exactly.
    2) Else find shared stem between pres and gold (IH in gold matches any pres vowel).
       - If no shared stem -> irregular -> pred must match gold exactly.
    3) Else take gold suffix after divergence and require pred suffix after same boundary matches it.
    """
    # 1) zero-change
    if pres == gold:
        return pred == gold

    # 2) shared stem?
    stem_end = shared_stem_boundary(pres, gold)
    if stem_end == 0:
        return pred == gold  # irregular

    # 3) suffix must match (same boundary)
    gold_suffix = gold[stem_end:]
    pred_suffix = pred[stem_end:]
    return pred_suffix == gold_suffix


# ============================================================
# UniMorph
# ============================================================
@dataclass
class VerbPair:
    lemma: str
    past: str


def load_unimorph(path: str) -> List[VerbPair]:
    pairs = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) >= 3:
                lemma, form, tag = parts[:3]
                if tag == "V;PST" and lemma.isalpha() and form.isalpha():
                    pairs.add((lemma.lower(), form.lower()))
    return [VerbPair(*p) for p in sorted(pairs)]

def load_unigram_cache() -> Dict[str, int]:
    if os.path.exists(UNIGRAM_CACHE_PATH):
        with open(UNIGRAM_CACHE_PATH, "r", encoding="utf-8") as f:
            print(f"📦 Loaded unigram cache from {UNIGRAM_CACHE_PATH}")
            return json.load(f)
    return {}


def save_unigram_cache(cache: Dict[str, int]):
    os.makedirs(os.path.dirname(UNIGRAM_CACHE_PATH), exist_ok=True)
    with open(UNIGRAM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"💾 Saved unigram cache ({len(cache)} entries)")


# ============================================================
# CMUdict
# ============================================================
STRESS_RE = re.compile(r"\d")


def load_cmudict(path: str) -> Dict[str, List[List[str]]]:
    cmu: Dict[str, List[List[str]]] = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
                phones = [STRESS_RE.sub("", p) for p in parts[1:]]
                cmu.setdefault(word, []).append(phones)
    return cmu


# ============================================================
# Google Books 1-gram streaming (bulk counts)
# ============================================================
_alpha_re = re.compile(r"[^a-z]")

def _clean_ngram_token(s: str) -> str:
    # match your binomials approach: keep only lowercase a-z
    return _alpha_re.sub("", s.lower())

def get_bulk_unigram_counts(words: Set[str]) -> Dict[str, int]:
    cache = load_unigram_cache()

    words = {w.lower() for w in words if w and w[0].isalpha()}
    missing = words - cache.keys()

    if not missing:
        print("✅ All unigram counts found in cache.")
        return cache

    counts: Dict[str, int] = {w: 0 for w in missing}

    # group *missing* words by first letter
    groups: Dict[str, List[str]] = defaultdict(list)
    for w in missing:
        groups[w[0]].append(w)

    print(f"\n📚 Need unigram counts for {len(missing)} new words")
    print(f"🔤 Prefix groups to process: {sorted(groups.keys())}")

    for prefix, targets in groups.items():
        fname = f"{UNIGRAM_PREFIX}{prefix}.gz"
        url = UNIGRAM_BASE + fname
        target_set = set(targets)

        print(f"\n📂 Fetching 1-gram file for prefix '{prefix}': {fname}")
        print(f"   Words to find in this file: {len(target_set)}")

        try:
            with requests.get(url, stream=True, timeout=None) as r:
                r.raise_for_status()
                f = gzip.GzipFile(fileobj=r.raw)

                for rawline in tqdm(
                    f, desc=f"   📄 Scanning {prefix}.gz", unit="lines", mininterval=0.5
                ):
                    try:
                        line = rawline.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                    parts = line.split("\t")
                    if len(parts) != 4:
                        continue

                    ngram, year, match_count, _ = parts
                    cleaned = _clean_ngram_token(ngram)
                    if cleaned in target_set:
                        try:
                            counts[cleaned] += int(match_count)
                        except ValueError:
                            pass

        except Exception as e:
            print(f"⚠️ Error retrieving file for prefix '{prefix}': {e}")

    cache.update(counts)
    save_unigram_cache(cache)
    print("\n✅ Done. Unigram counts cached for all requested words.")
    return cache




# ============================================================
# Encoding
# ============================================================
def encode(seq: List[str], phone2idx: Dict[str, int]) -> torch.Tensor:
    seq = seq[:MAX_PHONES] + ["_"] * max(0, MAX_PHONES - len(seq))
    vec = torch.zeros(MAX_PHONES * len(phone2idx))
    for i, p in enumerate(seq):
        vec[i * len(phone2idx) + phone2idx.get(p, phone2idx["_"])] = 1.0
    return vec


def decode(vec: torch.Tensor, idx2phone: Dict[int, str]) -> List[str]:
    mat = vec.view(MAX_PHONES, len(idx2phone))
    return [idx2phone[i] for i in mat.argmax(dim=1).tolist() if idx2phone[i] != "_"]


# ============================================================
# Model
# ============================================================
class PastTenseNet(nn.Module):
    def __init__(self, inp, hid, out, num_hidden=1):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(inp, hid))
        for _ in range(num_hidden - 1):
            self.hidden_layers.append(nn.Linear(hid, hid))

        self.output = nn.Linear(hid, out)

    def forward(self, x, return_hidden=False):
        h = x
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))

        if return_hidden:
            return self.output(h), h
        return self.output(h)

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, top_k=None):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.top_k = top_k

    def forward(self, x):
        # Encoder
        z = torch.relu(self.encoder(x))

        # --- TOP-K SPARSITY ---
        if self.top_k is not None:
            # z: [batch, hidden_dim]
            k = min(self.top_k, z.shape[1])
            values, indices = torch.topk(z, k, dim=1)

            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, indices, values)
            z = z_sparse

        # Decoder
        x_hat = self.decoder(z)
        return x_hat, z




# ============================================================
# Main
# ============================================================
def main():
    unimorph_path = os.path.join(CACHE_DIR, "unimorph_eng.txt")
    cmudict_path = os.path.join(CACHE_DIR, "cmudict.dict")

    download(UNIMORPH_URL, unimorph_path)
    download(CMUDICT_URL, cmudict_path)

    verb_pairs = load_unimorph(unimorph_path)
    cmu = load_cmudict(cmudict_path)

    # Build examples: (lemma_str, past_str, pres_phones, past_phones)
    examples: List[Tuple[str, str, List[str], List[str]]] = []
    for vp in verb_pairs:
        if vp.lemma in cmu and vp.past in cmu:
            pres_phones = cmu[vp.lemma][0]
            past_phones = cmu[vp.past][0]
            if len(pres_phones) <= MAX_PHONES and len(past_phones) <= MAX_PHONES:
                examples.append((vp.lemma, vp.past, pres_phones, past_phones))

    print(f"Total usable verb pairs: {len(examples)}")

    # Train/test split (by example; if you want lemma-split, we can do that too)
    random.shuffle(examples)
    split1 = int(0.7 * len(examples))
    split2 = int(0.85 * len(examples))

    train_raw = examples[:split1]
    val_raw   = examples[split1:split2]
    test      = examples[split2:]

    print(
        f"Train pairs: {len(train_raw)} | "
        f"Val pairs: {len(val_raw)} | "
        f"Test pairs: {len(test)}"
    )

    # ---------------------------
    # BULK unigram counts for all past forms in training
    # ---------------------------
    past_forms = {past_word for (_lemma, past_word, _p, _q) in train_raw}

    # Quick sanity check words
    sanity = {"blinked", "walked", "went", "saw", "took"} & past_forms
    print(f"\nSanity words present in train set: {sorted(sanity)}")

    unigram_counts = get_bulk_unigram_counts(past_forms)

    # Print sanity frequencies
    print("\nSanity unigram counts:")
    for w in ["blinked", "walked", "went", "saw", "took"]:
        if w in unigram_counts:
            print(f"  {w:10s} -> {unigram_counts[w]}")

    # ---------------------------
    # Expand training tokens by log frequency
    # ---------------------------
    train: List[Tuple[str, List[str], List[str]]] = []
    print("\nExpanding training tokens by log(freq+1)...")
    for i, (lemma, past_word, pres_phones, past_phones) in enumerate(train_raw, start=1):
        freq = unigram_counts.get(past_word, 0)
        reps = max(1, min(MAX_REPS, int(round(math.log(freq + 1)))))
        train.extend([(lemma, pres_phones, past_phones)] * reps)

        if i % 50 == 0 or i == len(train_raw):
            print(
                f"  [{i:4d}/{len(train_raw)}] "
                f"past='{past_word:15s}' "
                f"freq={float(freq):10.2f} "
                f"reps={reps:2d}"
            )

    print(f"\nExpanded training tokens: {len(train)}")

    # ---------------------------
    # Build phone inventory from training data (phones, not strings)
    # ---------------------------
    phone_set = set()
    for _lemma, pres_phones, past_phones in train:
        phone_set.update(pres_phones)
        phone_set.update(past_phones)
    phone_set.add("_")

    inventory = sorted(phone_set)
    phone2idx = {p: i for i, p in enumerate(inventory)}
    idx2phone = {i: p for p, i in phone2idx.items()}

    X_val = torch.stack(
        [encode(p, phone2idx) for _, _, p, _ in val_raw]
    ).to(DEVICE)

    Y_val = torch.stack(
        [encode(q, phone2idx) for _, _, _, q in val_raw]
    ).to(DEVICE)

    X = torch.stack([encode(p, phone2idx) for _, p, _ in train]).to(DEVICE)
    Y = torch.stack([encode(q, phone2idx) for _, _, q in train]).to(DEVICE)

    # ---------------------------
    # Build model
    # ---------------------------
    model = PastTenseNet(
        inp=MAX_PHONES * len(inventory),
        hid=HIDDEN_SIZE,
        out=MAX_PHONES * len(inventory),
        num_hidden=NUM_HIDDEN_LAYERS,
    ).to(DEVICE)

    # ---------------------------
    # Optionally load model
    # ---------------------------
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"📦 Loaded past tense model from {MODEL_PATH}")

    # ---------------------------
    # Train model (unless loaded)
    # ---------------------------
    opt_model = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_history: List[float] = []

    window = 200
    min_avg_improvement = 1e-5
    min_epochs_before_check = window

    if LOAD_MODEL:
        print("⏩ Skipping past tense training (using loaded model).")
    else:
        print("\nTraining with windowed early stopping...")

        for epoch in range(EPOCHS):
            # ---- train ----
            model.train()
            opt_model.zero_grad()
            train_loss = ((model(X) - Y) ** 2).mean()
            train_loss.backward()
            opt_model.step()

            # ---- validate ----
            model.eval()
            with torch.no_grad():
                val_loss = ((model(X_val) - Y_val) ** 2).mean()

            val_val = float(val_loss.item())
            val_history.append(val_val)

            # ---- track best model ----
            if val_val < best_val_loss:
                best_val_loss = val_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"train {train_loss.item():.4f} | "
                    f"val {val_val:.4f}"
                )

            # ---- windowed stopping ----
            if epoch + 1 >= min_epochs_before_check:
                recent = val_history[-window:]
                improvements = [recent[i - 1] - recent[i] for i in range(1, len(recent))]
                avg_improvement = sum(improvements) / len(improvements)

                if avg_improvement < min_avg_improvement:
                    print(
                        f"\n⏹ Early stopping at epoch {epoch} | "
                        f"avg improvement {avg_improvement:.6f}"
                    )
                    break

        # restore best model
        model.load_state_dict(best_state)


    # ---------------------------
    # Save model (optional)
    # ---------------------------
    if SAVE_MODEL:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "inventory": inventory,
                "phone2idx": phone2idx,
                "idx2phone": idx2phone,
                "config": {
                    "MAX_PHONES": MAX_PHONES,
                    "HIDDEN_SIZE": HIDDEN_SIZE,
                    "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
                },
            },
            MODEL_PATH,
        )
        print(f"💾 Saved past tense model → {MODEL_PATH}")

    # ---------------------------
    # SAE: train/load (optional) AFTER model is trained/loaded
    # ---------------------------
    sae = None
    if USE_SAE:
        # Create SAE instance (needed for both train and load)
        sae = SparseAutoencoder(
            input_dim=HIDDEN_SIZE,
            hidden_dim=SAE_HIDDEN_SIZE,
            top_k=SAE_TOP_K,
        ).to(DEVICE)


        if os.path.exists(SAE_PATH) and not TRAIN_SAE:
            ckpt = torch.load(SAE_PATH, map_location=DEVICE)

            sae = SparseAutoencoder(
                input_dim=ckpt["config"]["input_dim"],
                hidden_dim=ckpt["config"]["hidden_dim"],
                top_k=ckpt["config"]["top_k"],
            ).to(DEVICE)

            sae.load_state_dict(ckpt["state_dict"])
            print(
                f"📦 Loaded SAE from {SAE_PATH} "
                f"(hidden={ckpt['config']['hidden_dim']}, top_k={ckpt['config']['top_k']})"
            )


        if TRAIN_SAE:
            print("\n🧠 Training Sparse Autoencoder...")

            model.eval()
            with torch.no_grad():
                _, hidden_train = model(X, return_hidden=True)

            # IMPORTANT: SAE input dim must match hidden_train dim
            if sae.encoder.in_features != hidden_train.shape[1]:
                sae = SparseAutoencoder(
                    input_dim=hidden_train.shape[1],
                    hidden_dim=SAE_HIDDEN_SIZE,
                    top_k=SAE_TOP_K,
                ).to(DEVICE)


            opt_sae = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

            for epoch in range(SAE_EPOCHS):
                sae.train()
                opt_sae.zero_grad()

                recon, z = sae(hidden_train)
                recon_loss = ((recon - hidden_train) ** 2).mean()
                sparsity_loss = z.abs().mean()
                if SAE_TOP_K is not None:
                    loss = recon_loss
                else:
                    loss = recon_loss + SAE_L1 * sparsity_loss


                loss.backward()
                opt_sae.step()

                if epoch % 50 == 0:
                    print(
                        f"SAE Epoch {epoch:4d} | "
                        f"recon {recon_loss.item():.4f} | "
                        f"sparsity {sparsity_loss.item():.4f}"
                    )

            os.makedirs(os.path.dirname(SAE_PATH), exist_ok=True)
            torch.save(
                {
                    "state_dict": sae.state_dict(),
                    "config": {
                        "input_dim": sae.encoder.in_features,
                        "hidden_dim": SAE_HIDDEN_SIZE,
                        "top_k": SAE_TOP_K,
                    },
                },
                SAE_PATH,
            )
            print(f"💾 Saved SAE → {SAE_PATH}")


        # ---------------------------
        # Fine-tune model with SAE constraint (optional)
        # ---------------------------
        if FINETUNE_WITH_SAE:
            if sae is None:
                raise RuntimeError("FINETUNE_WITH_SAE=True but SAE was not created/loaded.")

            print("\n🔁 Fine-tuning past tense model with SAE constraint...")

            sae.eval()
            for p in sae.parameters():
                p.requires_grad = False

            opt_ft = torch.optim.Adam(model.parameters(), lr=LR * 0.1)

            FT_EPOCHS = 300
            SAE_PENALTY = 1e-3

            for epoch in range(FT_EPOCHS):
                model.train()
                opt_ft.zero_grad()

                y_hat, h = model(X, return_hidden=True)
                _, z = sae(h)

                task_loss = ((y_hat - Y) ** 2).mean()
                sparsity_penalty = z.abs().mean()
                loss = task_loss + SAE_PENALTY * sparsity_penalty

                loss.backward()
                opt_ft.step()

                if epoch % 50 == 0:
                    print(
                        f"FT Epoch {epoch:4d} | "
                        f"task {task_loss.item():.4f} | "
                        f"sparsity {sparsity_penalty.item():.4f}"
                    )

            # Optional: re-save fine-tuned model
            if SAVE_MODEL:
                ft_path = MODEL_PATH.replace(".pt", "_ft_sae.pt")
                torch.save({"state_dict": model.state_dict()}, ft_path)
                print(f"💾 Saved SAE-finetuned model → {ft_path}")



    # ---------------------------
    # Evaluation
    # ---------------------------
    print("\n=== HELD-OUT VERBS (ACCURACY) ===")

    n = 0
    exact = 0
    morph_ok = 0
    edit_dists = []

    wrong_examples = []
    exact_wrong_but_morph_ok = []

    model.eval()
    with torch.no_grad():
        for lemma, past_word, pres_phones, past_phones in test:
            pred = decode(
                model(encode(pres_phones, phone2idx).to(DEVICE)).cpu(),
                idx2phone,
            )

            n += 1

            is_exact = (pred == past_phones)
            is_morph = morphologically_correct(pred, pres_phones, past_phones)

            if is_exact:
                exact += 1
            else:
                wrong_examples.append((lemma, pres_phones, past_phones, pred))
                if is_morph:
                    exact_wrong_but_morph_ok.append((lemma, pres_phones, past_phones, pred))

            if is_morph:
                morph_ok += 1


            edit_dists.append(edit_distance(pred, past_phones))

    exact_acc = exact / n
    morph_acc = morph_ok / n
    mean_edit = sum(edit_dists) / len(edit_dists)

    print(f"Exact match accuracy:          {exact_acc:.3f}")
    print(f"Morphologically-correct accuracy: {morph_acc:.3f}")
    print(f"Mean edit distance:           {mean_edit:.3f}")

    print("\n=== SAMPLE HELD-OUT ERRORS (EXACT) ===")
    for lemma, pres, gold, pred in wrong_examples[:10]:
        print(
            f"{lemma:12s} | "
            f"in:   {' '.join(pres):18s} | "
            f"gold: {' '.join(gold):18s} | "
            f"pred: {' '.join(pred)}"
        )

    print("\n=== EXACT WRONG BUT MORPHOLOGICALLY OK (≤1 stem edit) ===")
    for lemma, pres, gold, pred in exact_wrong_but_morph_ok[:10]:
        print(
            f"{lemma:12s} | "
            f"in:   {' '.join(pres):18s} | "
            f"gold: {' '.join(gold):18s} | "
            f"pred: {' '.join(pred)}"
        )



    print("\n=== NONCE VERBS (MORPHOLOGICAL ACCURACY) ===")

    nonce = {
        "splim": ["S", "P", "L", "IH", "M"],
        "blick": ["B", "L", "IH", "K"],
        "norp":  ["N", "AO", "R", "P"],
    }

    suffix_counts = defaultdict(int)

    with torch.no_grad():
        for w, phones in nonce.items():
            pred = decode(
                model(encode(phones, phone2idx).to(DEVICE)).cpu(),
                idx2phone,
            )
            suffix = past_tense_suffix_type(pred)
            suffix_counts[suffix] += 1

            print(f"{w:10s} -> {' '.join(pred)} [{suffix}]")


if __name__ == "__main__":
    main()
