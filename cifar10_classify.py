#!/usr/bin/env python3

import os
import io
import base64
import random
import time
import argparse
import json
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

PROMPTS = {
    "baseline": """
You are an image classifier for CIFAR-10. The input images are small (32x32) and low detail.
Only respond with exactly one of these labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
If unsure, choose the most likely. Do not include punctuation or extra words.
""".strip(),
    "rules": """
Act as a strict CIFAR-10 grader. RULES:
1) Output only one token from this set: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
2) Prefer coarse shape and context over texture (e.g., airplane has wings + fuselage; ship sits on water; frog is green with squat body).
3) If it's a car-like road vehicle say "automobile"; if it has a cargo bed or looks heavy-duty say "truck"; if on water say "ship".
4) For animals: beak + wings = "bird"; whiskers + ears + feline face = "cat"; antlers/hooves = "deer"; canine snout = "dog"; squat green amphibian = "frog"; long face + mane = "horse".
5) No extra text. Exactly one of the ten labels.
""".strip(),
}

USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else.
""".strip()

def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.

    Uses OpenAI-style content parts with an image_url Data URL.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                last_err = RuntimeError(f"API error {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            last_err = e

        if attempt < max_retries:
            time.sleep(retry_backoff ** attempt)

    raise last_err if last_err else RuntimeError("Unknown API error")

def normalize_label(text: str) -> str:
    """Map model reply to a valid CIFAR-10 class if possible (simple heuristic)."""
    t = text.lower().strip()
    # exact match first
    if t in CLASSES:
        return t
    # loose matching: pick first class name contained in output
    for c in CLASSES:
        if c in t:
            return c
    # fallback: unknown (will count as incorrect)
    return "__unknown__"

def stratified_sample_cifar10(root: str, seed: int, per_class: int) -> List[Tuple[Image.Image, int]]:
    """
    Download CIFAR-10 (train split) and return a list of (PIL_image, target) pairs:
    exactly per_class per class.
    """
    ds = CIFAR10(root=root, train=True, download=True)
    # Build indices per class
    per_class_idx: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class_idx[label].append(idx)

    # Sample with fixed seed
    random.seed(seed)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class_idx[label], per_class)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 via ai.sooners.us (gemma3:4b)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--per-class", type=int, default=10, help="Samples per class")
    parser.add_argument("--limit", type=int, default=100, help="Total images to classify (<= 10*per-class)")
    parser.add_argument("--data-root", type=str, default="./data", help="Where to cache CIFAR-10")
    parser.add_argument("--system-prompt", type=str, default="baseline", help="Prompt key or raw string")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model")
    parser.add_argument("--out-prefix", type=str, default="run", help="Prefix for output artifact filenames")
    args = parser.parse_args()

    system_prompt = PROMPTS.get(args.system_prompt, args.system_prompt)

    print(f"Using model={MODEL}, base_url={BASE_URL}")
    print(f"Prompt key='{args.system_prompt}' -> {len(system_prompt)} chars")

    print("Preparing CIFAR-10 sample...")
    samples = stratified_sample_cifar10(root=args.data_root, seed=args.seed, per_class=args.per_class)
    if args.limit < len(samples):
        samples = samples[:args.limit]

    y_true: List[int] = []
    y_pred: List[int] = []
    bad: List[Dict] = []

    print(f"Classifying {len(samples)} images...")
    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)
        try:
            reply = post_chat_completion_image(
                image_data_url=data_url,
                system_prompt=system_prompt,
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"[{i}/{len(samples)}] API error: {e}")
            pred_label = "__error__"
            pred_idx = -1
        else:
            pred_label = normalize_label(reply)
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1

        true_label = CLASSES[tgt]

        if pred_idx == -1:
            pred_idx = (tgt + 1) % 10  # always wrong but stable

            bad.append({
                "i": i,
                "true": true_label,
                "raw_reply": reply,
            })

        y_true.append(tgt)
        y_pred.append(pred_idx)

        print(f"[{i:03d}/{len(samples)}] true={true_label:>10s} | pred={CLASSES[pred_idx]:>10s} | raw='{reply}'")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy over {len(samples)} images: {acc*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    plt.figure(figsize=(7.5, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"CIFAR-10 Confusion Matrix ({MODEL} via ai.sooners.us)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CLASSES)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.tight_layout()
    cm_path = f"{args.out_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    print(f"Saved {cm_path}")

    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred), start=1):
        rows.append({
            "index": i,
            "true_index": yt,
            "true_label": CLASSES[yt],
            "pred_index": yp,
            "pred_label": CLASSES[yp],
        })
    csv_path = f"{args.out_prefix}_predictions.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    # Save misclassifications
    bad_path = f"{args.out_prefix}_misclassifications.jsonl"
    with open(bad_path, "w") as f:
        for row in bad:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(bad)} misclassification rows to {bad_path}")

    manifest = {
        "model": MODEL,
        "base_url": BASE_URL,
        "seed": args.seed,
        "per_class": args.per_class,
        "limit": len(samples),
        "system_prompt_key": args.system_prompt if args.system_prompt in PROMPTS else None,
        "temperature": args.temperature,
        "accuracy": acc,
        "outputs": {
            "confusion_matrix_png": cm_path,
            "predictions_csv": csv_path,
            "misclassifications_jsonl": bad_path,
        },
    }
    with open(f"{args.out_prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {args.out_prefix}_manifest.json")

if __name__ == "__main__":
    main()
