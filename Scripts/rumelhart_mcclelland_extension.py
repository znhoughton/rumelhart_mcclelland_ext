#!/usr/bin/env python3
"""
English Past Tense Learner
- ARPABET input/output (CMUdict)
- Large-scale verb list (UniMorph)
- Token-frequency exposure via Google Books 1-gram corpus (.gz streaming)
- 1-hidden-layer feedforward neural network
"""

from __future__ import annotations
import csv
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

from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    hidden_size: int
    num_hidden_layers: int
    use_sae: bool
    train_sae: bool
    finetune_with_sae: bool
    sae_layer: int | None

N_SAMPLES = 80000
# ============================================================
# Configuration
# ============================================================
CACHE_DIR = "../Data/data_cache"
UNIMORPH_URL = "https://raw.githubusercontent.com/unimorph/eng/master/eng"
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
UNIGRAM_CACHE_PATH = os.path.join(CACHE_DIR, "google_1gram_cache.json")


# Google Books Ngrams 1-gram corpus (2012 snapshot)
UNIGRAM_BASE = "http://storage.googleapis.com/books/ngrams/books/"
UNIGRAM_PREFIX = "googlebooks-eng-all-1gram-20120701-"


LOG_WEIGHT = False

MAX_PHONES = 10

EPOCHS = 2000
LR = 0.001
TEST_RATIO = 0.2
MAX_REPS = None
SEED = 0


EARLY_STOPPING = True
ES_PATIENCE = 500        # epochs without improvement
ES_MIN_DELTA = 1e-5      # minimum improvement
ES_MIN_EPOCHS = 750      # don't stop before this

SAE_EARLY_STOPPING = True
SAE_ES_PATIENCE   = 500      # epochs without improvement
SAE_ES_MIN_DELTA  = 1e-5     # absolute improvement threshold
SAE_ES_MIN_EPOCHS = 750      # don't stop too early
SAE_HIDDEN_SIZE = 512
SAE_L1 = 1e-3
SAE_EPOCHS = 40000
SAE_LR = 1e-3
SAE_TOP_K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Utilities
# ============================================================
def cross_entropy_phone_loss(logits, targets, vocab_size, criterion):
    """
    logits:  [batch, MAX_PHONES * vocab_size]
    targets: [batch, MAX_PHONES * vocab_size] (one-hot)
    """
    B = logits.size(0)

    logits = logits.view(B, MAX_PHONES, vocab_size)
    targets = targets.view(B, MAX_PHONES, vocab_size)

    target_ids = targets.argmax(dim=2)  # [B, MAX_PHONES]

    logits = logits.view(B * MAX_PHONES, vocab_size)
    target_ids = target_ids.view(B * MAX_PHONES)

    return criterion(logits, target_ids)

def count_stem_mismatches(pres: List[str], past: List[str]) -> int:
    """
    Counts mismatches in the shared stem region,
    using phones_match() for tolerance.
    """
    mismatches = 0
    L = min(len(pres), len(past))
    for i in range(L):
        if not phones_match(pres[i], past[i]):
            mismatches += 1
    return mismatches


def is_irregular_gold(pres: List[str], gold: List[str]) -> bool:
    """
    Gold-only definition of irregularity.

    Regular if:
      - Past ends in D / T / IH-D
      - AND stem differs by at most 1 weak phonological change

    Irregular otherwise.
    """
    suffix = past_tense_suffix_type(gold)

    # If no regular suffix → irregular
    if suffix == "OTHER":
        return True

    # Count phonological stem mismatches
    mismatches = count_stem_mismatches(pres, gold)

    # Allow up to 1 weak change
    return mismatches > 1




def write_irregular_results(
    path: str,
    rows: list[dict],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"💾 Appended {len(rows)} irregular rows → {path}")


def eval_irregular_split(
    split_name: str,
    split_data,
    *,
    model,
    phone2idx,
    idx2phone,
    model_tag: str,
):
    rows = []

    model.eval()
    with torch.no_grad():
        for lemma, past_word, pres_phones, past_phones in split_data:

            if not is_irregular_gold(pres_phones, past_phones):
                continue

            pred = decode(
                model(encode(pres_phones, phone2idx).to(DEVICE)).cpu(),
                idx2phone,
            )

            rows.append({
                "model_tag": model_tag,
                "split": split_name,
                "lemma": lemma,
                "past_word": past_word,  # handy to keep
                "present_phones": " ".join(pres_phones),
                "gold_past_phones": " ".join(past_phones),
                "pred_past_phones": " ".join(pred),
                "exact": int(pred == past_phones),
                "morph_ok": int(morphologically_correct(pred, pres_phones, past_phones)),
                "edit_distance": edit_distance(pred, past_phones),
            })

    return rows



def download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"↓ downloading {url}")
        urllib.request.urlretrieve(url, path)

def exact_match(pred: List[str], gold: List[str]) -> bool:
    return pred == gold

def explicit_go_test(model, phone2idx, idx2phone, cmu):
    print("\n=== 🔎 EXPLICIT TEST: GO → WENT ===")

    if "go" not in cmu or "went" not in cmu:
        print("⚠️ 'go' or 'went' not found in CMUdict")
        return

    pres = cmu["go"][0]
    gold = cmu["went"][0]

    with torch.no_grad():
        pred = decode(
            model(encode(pres, phone2idx).to(DEVICE)).cpu(),
            idx2phone,
        )

    print(f"Input phones : {' '.join(pres)}")
    print(f"Gold past    : {' '.join(gold)}")
    print(f"Model output : {' '.join(pred)}")

    if pred == gold:
        print("✅ CORRECT irregular mapping")
    else:
        print("❌ INCORRECT")

def model_tag_from_cfg(cfg: ExperimentConfig) -> str:
    base = f"L{cfg.num_hidden_layers}_H{cfg.hidden_size}"
    if cfg.use_sae:
        if cfg.sae_layer is None:
            return f"{base}_SAE@final"
        else:
            return f"{base}_SAE@L{cfg.sae_layer + 1}"
    return base

RESULTS_DIR = "../Data/model_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(RESULTS_DIR, "test_summary.csv")

def append_test_summary(row: dict):
    write_header = not os.path.exists(SUMMARY_PATH)

    with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_tag",
                "hidden_size",
                "num_hidden_layers",
                "use_sae",
                "sae_layer",
                "exact_acc",
                "morph_acc",
                "mean_edit_distance",
                "n_test_items",
            ],
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


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



def write_split_csv(path, examples):
    """
    examples: List of (lemma, past_word, pres_phones, past_phones)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lemma",
            "past",
            "present_phones",
            "past_phones",
        ])

        for lemma, past, pres_phones, past_phones in examples:
            writer.writerow([
                lemma,
                past,
                " ".join(pres_phones),
                " ".join(past_phones),
            ])

    print(f"💾 Wrote {len(examples)} rows → {path}")

def write_sampled_train_csv(path, rows):
    """
    rows: List of (lemma, present_phones, past_phones)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lemma",
            "present_phones",
            "past_phones",
        ])
        for lemma, pres, past in rows:
            writer.writerow([
                lemma,
                " ".join(pres),
                " ".join(past),
            ])

    print(f"💾 Wrote fixed sampled training set → {path} ({len(rows)} rows)")


def read_sampled_train_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                row["lemma"],
                row["present_phones"].split(),
                row["past_phones"].split(),
            ))

    print(f"📦 Loaded fixed sampled training set → {path} ({len(rows)} rows)")
    return rows

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
    seq = seq[: MAX_PHONES - 1]          # reserve space for EOS
    seq = seq + ["<EOS>"]                # append EOS
    seq = seq + ["_"] * (MAX_PHONES - len(seq))

    vec = torch.zeros(MAX_PHONES * len(phone2idx))
    for i, p in enumerate(seq):
        vec[i * len(phone2idx) + phone2idx[p]] = 1.0

    return vec



def decode(vec: torch.Tensor, idx2phone: Dict[int, str]) -> List[str]:
    mat = vec.view(MAX_PHONES, len(idx2phone))
    out = []

    for row in mat:
        phone = idx2phone[row.argmax().item()]
        if phone == "<EOS>":
            break
        if phone == "_":
            continue
        out.append(phone)

    return out




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

    def forward(self, x, return_hidden=False, return_all_hidden=False):
        h = x
        hidden_states = []

        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
            hidden_states.append(h)

        y = self.output(h)

        if return_all_hidden:
            return y, hidden_states
        if return_hidden:
            return y, h
        return y

def get_sae_hidden(hidden_states, sae_layer):
    """
    hidden_states: list of tensors, one per hidden layer
    sae_layer:
        None → final hidden layer
        int  → specific layer index
    """
    if sae_layer is None:
        return hidden_states[-1]

    if sae_layer < 0 or sae_layer >= len(hidden_states):
        raise ValueError(
            f"Invalid SAE_LAYER={sae_layer}; "
            f"model has {len(hidden_states)} hidden layers"
        )

    return hidden_states[sae_layer]


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

# ============================================================
# Main
# ============================================================
def run_experiment(cfg: ExperimentConfig):
    HIDDEN_SIZE = cfg.hidden_size
     # ---------------------------
    # Per-experiment derived config
    # ---------------------------
    NUM_HIDDEN_LAYERS = cfg.num_hidden_layers
    USE_SAE = cfg.use_sae
    TRAIN_SAE = cfg.train_sae
    FINETUNE_WITH_SAE = cfg.finetune_with_sae
    SAE_LAYER = cfg.sae_layer

    # Safety check for SAE placement
    if SAE_LAYER is not None and SAE_LAYER >= NUM_HIDDEN_LAYERS:
        raise ValueError(
            f"SAE_LAYER={SAE_LAYER} but model has NUM_HIDDEN_LAYERS={NUM_HIDDEN_LAYERS}"
        )

    # Decide save/load behavior for this run
    # (you can tweak these defaults)
    SAVE_MODEL = True

    # Tags + paths (must be per-experiment to avoid overwriting)
    BASE_MODEL_TAG = f"L{NUM_HIDDEN_LAYERS}_H{HIDDEN_SIZE}"

    if USE_SAE:
        if SAE_LAYER is None:
            MODEL_TAG = f"{BASE_MODEL_TAG}_SAE@final"
        else:
            MODEL_TAG = f"{BASE_MODEL_TAG}_SAE@L{SAE_LAYER+1}"

        
    else:
        MODEL_TAG = BASE_MODEL_TAG

    MODEL_PATH = f"../models/past_tense_net_{MODEL_TAG}.pt"
    SAE_PATH   = f"../models/sae_{MODEL_TAG}.pt"

    # IMPORTANT: choose whether the fixed train sample is shared across runs.
    # If you want it shared across ALL experiments, do NOT include MODEL_TAG here.
    FIXED_TRAIN_SAMPLE_PATH = (
        f"../Data/datasets/fixed_train_sample_SEED{SEED}_N{N_SAMPLES}.csv"
    )



    
    split1 = int(0.7 * len(examples))
    split2 = int(0.85 * len(examples))

    train_raw = examples[:split1]
    val_raw   = examples[split1:split2]
    test      = examples[split2:]

    # --------------------------------------------------
    # Save dataset splits
    # --------------------------------------------------
    DATASET_DIR = "../Data/datasets"

    write_split_csv(
        f"{DATASET_DIR}/train_{MODEL_TAG}.csv",
        train_raw
    )

    write_split_csv(
        f"{DATASET_DIR}/val_{MODEL_TAG}.csv",
        val_raw
    )

    write_split_csv(
        f"{DATASET_DIR}/test_{MODEL_TAG}.csv",
        test
    )

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
    # Fixed sampled training set (shared across models)
    # ---------------------------
    if os.path.exists(FIXED_TRAIN_SAMPLE_PATH):
        train = read_sampled_train_csv(FIXED_TRAIN_SAMPLE_PATH)

    else:
        print("\nSampling training tokens from raw-frequency distribution...")

        # Build frequency weights
        items = []
        weights = []

        for lemma, past_word, pres_phones, past_phones in train_raw:
            freq = unigram_counts.get(past_word, 0)
            if LOG_WEIGHT:
                weight = math.log(freq) if freq > 1 else 1
            else:
                weight = max(1, freq)
            items.append((lemma, pres_phones, past_phones))
            weights.append(weight)

        # IMPORTANT: isolate RNG so sampling is reproducible
        rng = random.Random(SEED)

        sample_indices = rng.choices(
            range(len(items)),
            weights=weights,
            k=N_SAMPLES
        )

        train = [items[i] for i in sample_indices]

        write_sampled_train_csv(FIXED_TRAIN_SAMPLE_PATH, train)

    print(f"📊 Training tokens used: {len(train)}")


    print(f"Sampled training set size: {len(train)}")


    print(f"\nExpanded training tokens: {len(train)}")

    # ---------------------------
    # Build phone inventory from training data (phones, not strings)
    # ---------------------------
    phone_set = set()
    for _lemma, pres_phones, past_phones in train:
        phone_set.update(pres_phones)
        phone_set.update(past_phones)
    phone_set.update({"_", "<EOS>"})


    inventory = sorted(phone_set)
    phone2idx = {p: i for i, p in enumerate(inventory)}
    idx2phone = {i: p for p, i in phone2idx.items()}

    PAD_IDX = phone2idx["_"]
    EOS_IDX = phone2idx["<EOS>"]

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX
    ).to(DEVICE)


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

    # --------------------------------------------------
    # Load base model if this experiment depends on it
    # --------------------------------------------------
    if USE_SAE:
        BASE_MODEL_PATH = f"../models/past_tense_net_L{NUM_HIDDEN_LAYERS}_H{HIDDEN_SIZE}.pt"

        if not os.path.exists(BASE_MODEL_PATH):
            raise RuntimeError(
                f"Base model required for SAE but not found: {BASE_MODEL_PATH}"
            )

        ckpt = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"📦 Loaded base model → {BASE_MODEL_PATH}")


    # ---------------------------
    # Train model (unless loaded)
    # ---------------------------
    opt_model = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_model,
        mode="min",
        factor=0.5,        # halve the LR
        patience=50,       # wait 50 epochs without improvement
        threshold=1e-4,    # minimum relative improvement
        min_lr=1e-5       # don't collapse to zero
    )


    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    val_history: List[float] = []

    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if not USE_SAE:
        print("Training base model")
        for epoch in range(EPOCHS):
            # ---- train ----
            model.train()
            opt_model.zero_grad()
            logits = model(X)
            train_loss = cross_entropy_phone_loss(
                logits, Y, len(inventory), criterion
            )
            train_loss.backward()
            opt_model.step()

            # ---- validate ----
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = cross_entropy_phone_loss(
                    val_logits, Y_val, len(inventory), criterion
                )

            val_val = val_loss.item()
            scheduler.step(val_val)

            # ---- early stopping bookkeeping ----
            if val_val < best_val_loss - ES_MIN_DELTA:
                best_val_loss = val_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # ---- logging ----
            if epoch % 100 == 0:
                lr = opt_model.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d} | "
                    f"train {train_loss.item():.4f} | "
                    f"val {val_val:.4f} | "
                    f"lr {lr:.6f} | "
                    f"no_improve {epochs_no_improve}"
                )
                explicit_go_test(model, phone2idx, idx2phone, cmu)

            # ---- optional stopping ----
            if (
                EARLY_STOPPING
                and epoch >= ES_MIN_EPOCHS
                and epochs_no_improve >= ES_PATIENCE
            ):
                print(
                    f"\n🛑 Early stopping at epoch {epoch} "
                    f"(best val={best_val_loss:.4f})"
                )
                break
    else:
        print("Skipping base training")
    # ==============================
    # AFTER LOOP
    # ==============================
    if not USE_SAE:
        model.load_state_dict(best_state)
        print(f"\n✅ Restored best model (val={best_val_loss:.4f})")
        explicit_go_test(model, phone2idx, idx2phone, cmu)


    # ---------------------------
    # Save model (optional)
    # ---------------------------
    if SAVE_MODEL and not USE_SAE:
        BASE_MODEL_PATH = f"../models/past_tense_net_L{NUM_HIDDEN_LAYERS}_H{HIDDEN_SIZE}.pt"
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
            BASE_MODEL_PATH,
        )
        print(f"💾 Saved BASE model → {BASE_MODEL_PATH}")


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

            # ---------------------------------------------------------
            # Build SAE training inputs as TYPES (no frequency expansion)
            # ---------------------------------------------------------
            # Use the un-expanded training set (train_raw) so each lemma/past pair appears once.
            # Optionally dedupe in case train_raw has duplicates.
            seen = set()
            sae_pres_phones = []
            for (lemma, past_word, pres_phones, past_phones) in train_raw:
                key = (lemma, past_word)  # or (lemma,) if you want lemma-types only
                if key in seen:
                    continue
                seen.add(key)
                sae_pres_phones.append(pres_phones)

            print(f"🧠 SAE training on verb TYPES: {len(sae_pres_phones)} (no repetition)")

            X_sae = torch.stack([encode(p, phone2idx) for p in sae_pres_phones]).to(DEVICE)

            model.eval()
            with torch.no_grad():
                _, hidden_states = model(X_sae, return_all_hidden=True)
                hidden_train = get_sae_hidden(hidden_states, SAE_LAYER)


            # IMPORTANT: SAE input dim must match hidden_train dim
            if sae.encoder.in_features != hidden_train.shape[1]:
                sae = SparseAutoencoder(
                    input_dim=hidden_train.shape[1],
                    hidden_dim=SAE_HIDDEN_SIZE,
                    top_k=SAE_TOP_K,
                ).to(DEVICE)


            opt_sae = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

            best_recon = float("inf")
            best_state = None
            epochs_no_improve = 0

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

                recon_val = recon_loss.item()

                # ---- early stopping bookkeeping ----
                if recon_val < best_recon - SAE_ES_MIN_DELTA:
                    best_recon = recon_val
                    best_state = {k: v.detach().clone() for k, v in sae.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # ---- logging ----
                if epoch % 50 == 0:
                    print(
                        f"SAE Epoch {epoch:4d} | "
                        f"recon {recon_val:.6f} | "
                        f"sparsity {sparsity_loss.item():.6f} | "
                        f"no_improve {epochs_no_improve}"
                    )

                # ---- stopping condition ----
                if (
                    SAE_EARLY_STOPPING
                    and epoch >= SAE_ES_MIN_EPOCHS
                    and epochs_no_improve >= SAE_ES_PATIENCE
                ):
                    print(
                        f"\n🛑 SAE early stopping at epoch {epoch} "
                        f"(best recon={best_recon:.6f})"
                    )
                    break

            # Restore best SAE
            if best_state is not None:
                sae.load_state_dict(best_state)
                print(f"✅ Restored best SAE (recon={best_recon:.6f})")


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

            FT_EPOCHS = 1000
            SAE_PENALTY = 1e-3

            for epoch in range(FT_EPOCHS):
                model.train()
                opt_ft.zero_grad()

                y_hat, hidden_states = model(X, return_all_hidden=True)
                h_sae = get_sae_hidden(hidden_states, SAE_LAYER)
                _, z = sae(h_sae)


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
    # Evaluation + logging
    # ---------------------------
    print("\n=== HELD-OUT VERBS (ACCURACY) ===")
    wrong_examples = []
    exact_wrong_but_morph_ok = []

    model_tag = model_tag_from_cfg(cfg)
    preds_path = os.path.join(RESULTS_DIR, f"test_preds_{model_tag}.csv")

    rows = []

    n = 0
    exact = 0
    morph_ok = 0
    edit_dists = []

    model.eval()
    with torch.no_grad():
        for lemma, past_word, pres_phones, past_phones in test:
            pred = decode(
                model(encode(pres_phones, phone2idx).to(DEVICE)).cpu(),
                idx2phone,
            )

            ed = edit_distance(pred, past_phones)
            is_exact = pred == past_phones
            is_morph = morphologically_correct(pred, pres_phones, past_phones)

            rows.append({
                "lemma": lemma,
                "present_phones": " ".join(pres_phones),
                "gold_past_phones": " ".join(past_phones),
                "pred_past_phones": " ".join(pred),
                "exact": int(is_exact),
                "morph_ok": int(is_morph),
                "edit_distance": ed,
            })

            n += 1
            exact += int(is_exact)
            morph_ok += int(is_morph)
            edit_dists.append(ed)

    exact_acc = exact / n
    morph_acc = morph_ok / n
    mean_edit = sum(edit_dists) / n

    print(f"Exact match accuracy:              {exact_acc:.3f}")
    print(f"Morphologically-correct accuracy: {morph_acc:.3f}")
    print(f"Mean edit distance:               {mean_edit:.3f}")


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

    # Write detailed test predictions
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"💾 Wrote test predictions → {preds_path}")

    append_test_summary({
        "model_tag": model_tag,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "use_sae": cfg.use_sae,
        "sae_layer": cfg.sae_layer,
        "exact_acc": exact_acc,
        "morph_acc": morph_acc,
        "mean_edit_distance": mean_edit,
        "n_test_items": n,
    })

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


    IRREG_PATH = os.path.join(RESULTS_DIR, "irregular_verb_results.csv")

    irregular_rows = []
    irregular_rows += eval_irregular_split(
        "train", train_raw,
        model=model, phone2idx=phone2idx, idx2phone=idx2phone, model_tag=model_tag
    )
    irregular_rows += eval_irregular_split(
        "val", val_raw,
        model=model, phone2idx=phone2idx, idx2phone=idx2phone, model_tag=model_tag
    )
    irregular_rows += eval_irregular_split(
        "test", test,
        model=model, phone2idx=phone2idx, idx2phone=idx2phone, model_tag=model_tag
    )

    if irregular_rows:
        write_irregular_results(IRREG_PATH, irregular_rows)


if __name__ == "__main__":

    experiments: list[ExperimentConfig] = []

    # --------------------------------------------------
    # Base models: 1, 2, 3 hidden layers
    # --------------------------------------------------
    for L in [1, 2, 3,4]:
        experiments.append(
            ExperimentConfig(
                hidden_size = 256,
                num_hidden_layers=L,
                use_sae=False,
                train_sae=False,
                finetune_with_sae=False,
                sae_layer=None,
            )
        )

    # --------------------------------------------------
    # SAE finetuning on all depths (default = final layer)
    # --------------------------------------------------
    for L in [1, 2, 3, 4]:
        experiments.append(
            ExperimentConfig(
                hidden_size = 256,
                num_hidden_layers=L,
                use_sae=True,
                train_sae=True,
                finetune_with_sae=False,
                sae_layer=None,
            )
        )

    # --------------------------------------------------
    # Extra SAE placements for 3-layer model
    # --------------------------------------------------
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
                )
            )

    # --------------------------------------------------
    # Run serially
    # --------------------------------------------------
    for cfg in experiments:
        print("\n" + "=" * 80)
        print(f"🚀 Running experiment: {cfg}")
        print("=" * 80 + "\n")

        run_experiment(cfg)
