"""
Generate 30 valid counterfactual and invariant pairs for each supported type:
- Type 1: Mirror Angle Flip (30 pairs)
- Type 2: Cosmetic Control (30 pairs)
- Type 3: Obstacle Insertion (30 pairs)
- Type 4: Irrelevant Obstacle (30 pairs)

Total: 120 pairs (240 scenes), completely verified by the VisibilityOracle.
"""

import time
from pathlib import Path

from mirrorlab.generation.counterfactual_factory import CounterfactualFactory


def main():
    out_dir = Path("data/counterfactual_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Generating 120 Counterfactual Pairs (30 per type)")
    print(f"Target Output Directory: {out_dir.resolve()}")
    print("==================================================")

    factory = CounterfactualFactory(
        canvas_width=1000.0,
        canvas_height=1000.0,
        image_size=1024,
    )

    t0 = time.time()
    bundles = factory.generate_dataset(
        num_pairs_per_type=30,
        output_dir=out_dir,
        start_seed=40000,
    )
    elapsed = time.time() - t0

    print(
        f"\nSuccessfully generated {len(bundles)} counterfactual pairs in {elapsed:.2f}s!"
    )
    print("Manifests generated:")
    print(f"  - {out_dir / 'pairs.jsonl'}")
    print(f"  - {out_dir / 'tasks.jsonl'}")
    print(f"  - {out_dir / 'ground_truth.jsonl'}")

    type_counts = {}
    for b in bundles:
        pt = b.pair_type.value
        type_counts[pt] = type_counts.get(pt, 0) + 1

    print("\nSummary per Pair Type:")
    for pt, count in type_counts.items():
        print(f"  {pt}: {count} pairs")


if __name__ == "__main__":
    main()
