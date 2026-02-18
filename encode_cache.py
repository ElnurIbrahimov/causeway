"""
Encode SCM datasets through Mistral and cache to disk.
No training — just encoding. Download the cache files and stop the pod.

Usage:
    python encode_cache.py --model mistralai/Mistral-7B-v0.3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.3")
    p.add_argument("--num_samples", type=int, default=50000)
    p.add_argument("--batch_encode_size", type=int, default=32)
    args = p.parse_args()

    model_short = args.model.split("/")[-1]

    for domain in ["deployment", "clinical"]:
        cache_path = f"cache_{domain}_{model_short}_{args.num_samples}_v2.pt"

        if os.path.exists(cache_path):
            print(f"SKIP: {cache_path} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"  Encoding {domain} domain with {args.model}")
        print(f"  -> {cache_path}")
        print(f"{'='*60}\n")

        if domain == "clinical":
            from environments.text_clinical import TextClinicalDataset as DS
        else:
            from environments.text_scm import TextSCMDataset as DS

        DS(
            model_name=args.model,
            num_samples=args.num_samples,
            batch_encode_size=args.batch_encode_size,
            device="cuda",
            cache_path=cache_path,
        )

        print(f"DONE: {cache_path}")

    print("\n\nAll caches saved. Download them and stop the pod:")
    for domain in ["deployment", "clinical"]:
        f = f"cache_{domain}_{model_short}_{args.num_samples}_v2.pt"
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / 1e6
            print(f"  {f}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
