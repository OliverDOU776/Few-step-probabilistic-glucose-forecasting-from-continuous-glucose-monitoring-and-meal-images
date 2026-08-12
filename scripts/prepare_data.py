#!/usr/bin/env python
"""Prepare datasets, normalization statistics, windows, and CLIP embeddings.

Generates the release-time data diagnostics:
- Dataset cards for CGMacros + Weinstock
- Normalization JSON files + histogram plots
- Window sanity visualizations (Task A + Task B)
- CLIP embedding cache for meal photos
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from glucoflow.data.adapters import CGMacrosAdapter, DatasetOutput, GlucoBenchAdapter
from glucoflow.data.normalization import ZScoreStats, compute_zscore_stats
from glucoflow.evaluation.splits import subject_disjoint_split
from glucoflow.evaluation.windows import extract_windows, extract_meal_windows


# ========== Step 2.1: Dataset Adapters + Cards ==========

def generate_dataset_card(output: DatasetOutput, save_dir: Path):
    """Generate a dataset card markdown file."""
    cgm = output.cgm
    n_subjects = cgm["subject_id"].nunique()
    n_rows = len(cgm)

    # Per-subject stats
    per_subject = cgm.groupby("subject_id").agg(
        n_rows=("glucose_mgdl", "count"),
        min_ts=("timestamp", "min"),
        max_ts=("timestamp", "max"),
        missingness=("glucose_mgdl", lambda x: x.isna().mean()),
    )
    per_subject["days"] = (per_subject["max_ts"] - per_subject["min_ts"]).dt.total_seconds() / 86400

    total_days = per_subject["days"].sum()
    mean_days = per_subject["days"].mean()
    mean_missingness = per_subject["missingness"].mean()

    # Sampling interval distribution
    intervals = []
    for sid, group in cgm.groupby("subject_id"):
        ts = group["timestamp"].sort_values()
        diffs = ts.diff().dt.total_seconds().dropna()
        intervals.extend(diffs.values[:1000])  # sample first 1000 per subject
    intervals = np.array(intervals)
    interval_median = np.median(intervals)
    interval_mode = float(pd.Series(intervals).mode().iloc[0])

    # Glucose stats
    gl = cgm["glucose_mgdl"].dropna()
    gl_min, gl_max = gl.min(), gl.max()
    gl_mean, gl_std = gl.mean(), gl.std()

    card_path = save_dir / f"{output.name}.md"
    with open(card_path, "w") as f:
        f.write(f"# Dataset Card: {output.name}\n\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| Subjects | {n_subjects} |\n")
        f.write(f"| Total rows | {n_rows:,} |\n")
        f.write(f"| Total days | {total_days:.1f} |\n")
        f.write(f"| Mean days/subject | {mean_days:.1f} |\n")
        f.write(f"| Sampling interval (median) | {interval_median:.0f} s |\n")
        f.write(f"| Sampling interval (mode) | {interval_mode:.0f} s |\n")
        f.write(f"| Glucose range | {gl_min:.0f} – {gl_max:.0f} mg/dL |\n")
        f.write(f"| Glucose mean ± std | {gl_mean:.1f} ± {gl_std:.1f} mg/dL |\n")
        f.write(f"| Mean missingness | {mean_missingness:.4f} ({mean_missingness*100:.2f}%) |\n")

        if output.meals is not None:
            n_meals = len(output.meals)
            n_with_photos = (output.meals["image_path"].str.len() > 0).sum()
            f.write(f"| Total meals | {n_meals:,} |\n")
            f.write(f"| Meals with photos | {n_with_photos} ({n_with_photos/n_meals*100:.1f}%) |\n")

        f.write(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    print(f"  Dataset card saved: {card_path}")
    return card_path


# ========== Step 2.2: Normalization ==========

def run_normalization(output, save_dir_cards, save_dir_figs):
    """Compute normalization stats, save JSON, and plot histograms."""
    cgm = output.cgm
    # Subject-disjoint split to get train stats
    split = subject_disjoint_split(cgm, train_frac=0.7, val_frac=0.15, seed=42)
    train_cgm = split["train"]["cgm"]

    stats = compute_zscore_stats(train_cgm)

    # Save JSON
    json_path = save_dir_cards / f"{output.name}_norm.json"
    with open(json_path, "w") as f:
        json.dump({"mean": stats.mean, "std": stats.std}, f, indent=2)
    print(f"  Normalization stats: μ={stats.mean:.2f}, σ={stats.std:.2f} → {json_path}")

    # Verify train stats
    train_normalized = stats.normalize(train_cgm["glucose_mgdl"].dropna().values)
    print(f"  Train normalized: mean={train_normalized.mean():.6f}, std={train_normalized.std():.6f}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    raw_vals = cgm["glucose_mgdl"].dropna().values
    axes[0].hist(raw_vals, bins=80, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(stats.mean, color="red", linestyle="--", label=f"μ={stats.mean:.1f}")
    axes[0].set_xlabel("Glucose (mg/dL)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{output.name} — Raw Distribution")
    axes[0].legend()

    norm_vals = stats.normalize(raw_vals)
    axes[1].hist(norm_vals, bins=80, color="darkorange", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", label="μ=0")
    axes[1].set_xlabel("Normalized Glucose")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{output.name} — Normalized (z-score)")
    axes[1].legend()

    plt.tight_layout()
    fig_path = save_dir_figs / f"{output.name}_norm.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Histogram saved: {fig_path}")

    return stats


# ========== Step 2.3: Windowing Visualization ==========

def visualize_task_a_windows(output, stats, save_dir, n_samples=10):
    """Visualize Task A sliding windows."""
    cgm = output.cgm
    interval = output.sampling_interval_sec

    windows = extract_windows(
        cgm, history_minutes=120,
        forecast_minutes_list=[30, 60, 120],
        sampling_interval_sec=interval,
    )
    if len(windows) == 0:
        print(f"  WARNING: No Task A windows for {output.name}")
        return

    rng = np.random.RandomState(42)
    indices = rng.choice(len(windows), size=min(n_samples, len(windows)), replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        w = windows[idx]
        ax = axes[i]

        h_len = len(w["history"])
        f_len = len(w["future_120"])
        total_len = h_len + f_len

        t_hist = np.arange(h_len)
        t_fut = np.arange(h_len, total_len)

        ax.plot(t_hist, w["history"], color="steelblue", lw=1.2, label="History")
        ax.plot(t_fut, w["future_120"], color="coral", lw=1.2, label="Future 120m")
        ax.axvline(h_len, color="gray", linestyle="--", alpha=0.5)

        ax.set_title(f"Win {idx}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
        ax.set_ylabel("mg/dL" if i % 5 == 0 else "")

    plt.suptitle(f"Task A: Sliding Windows — {output.name}", fontsize=13)
    plt.tight_layout()
    fig_path = save_dir / f"{output.name}_task_a_samples.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Task A visualization ({len(windows)} total windows): {fig_path}")
    return len(windows)


def visualize_task_b_windows(output, stats, save_dir, n_samples=10):
    """Visualize Task B meal-anchored windows (CGMacros only)."""
    if output.meals is None:
        print(f"  Skipping Task B (no meals) for {output.name}")
        return

    cgm = output.cgm
    meals = output.meals
    interval = output.sampling_interval_sec

    windows = extract_meal_windows(
        cgm, meals,
        pre_minutes=120, post_minutes=180,
        sampling_interval_sec=interval,
    )
    if len(windows) == 0:
        print(f"  WARNING: No Task B windows for {output.name}")
        return

    rng = np.random.RandomState(42)
    indices = rng.choice(len(windows), size=min(n_samples, len(windows)), replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=False)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        w = windows[idx]
        ax = axes[i]

        h_len = len(w["history"])
        f_len = len(w["future"])

        t_hist = np.arange(-h_len, 0)
        t_fut = np.arange(0, f_len)

        ax.plot(t_hist, w["history"], color="steelblue", lw=1.2, label="Pre-meal")
        ax.plot(t_fut, w["future"], color="coral", lw=1.2, label="Post-meal")
        ax.axvline(0, color="red", linestyle="-", alpha=0.7, lw=2, label="Meal")

        meal_info = w["meal_info"]
        carbs = meal_info.get("carbs_g", "?")
        ax.set_title(f"Meal {idx} ({carbs:.0f}g carbs)\niAUC₂ₕ={w['iauc_2h']:.0f}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
        ax.set_ylabel("mg/dL" if i % 5 == 0 else "")
        ax.set_xlabel("steps from meal")

    plt.suptitle(f"Task B: Meal-Anchored PPGR Windows — {output.name}\n(H_pre=120min, H_post=180min)", fontsize=12)
    plt.tight_layout()
    fig_path = save_dir / f"{output.name}_task_b_samples.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Task B visualization ({len(windows)} total windows): {fig_path}")

    # Report shapes
    sample_w = windows[0]
    print(f"  Shapes: history={sample_w['history'].shape}, future={sample_w['future'].shape}")
    print(f"  iAUC_2h range: {min(w['iauc_2h'] for w in windows):.1f} – {max(w['iauc_2h'] for w in windows):.1f}")
    return len(windows)


# ========== Step 2.4: CLIP Embedding Cache ==========

def cache_clip_embeddings(output, cache_dir: Path):
    """Precompute CLIP embeddings for all meal photos."""
    if output.meals is None:
        print(f"  Skipping CLIP cache (no meals) for {output.name}")
        return 0, 0

    meals = output.meals
    photo_meals = meals[meals["image_path"].str.len() > 0].copy()
    if photo_meals.empty:
        print(f"  No meal photos to embed for {output.name}")
        return 0, 0

    cache_dir.mkdir(parents=True, exist_ok=True)

    import open_clip
    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    n_embedded = 0
    n_skipped = 0

    for i, (_, row) in enumerate(photo_meals.iterrows()):
        img_path = Path(row["image_path"])
        # Use subject_id + meal_timestamp as unique ID
        meal_id = f"{row['subject_id']}_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d_%H%M%S')}"
        out_path = cache_dir / f"{meal_id}.npy"

        if not img_path.exists():
            n_skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model.encode_image(img_tensor)
                embedding = embedding.cpu().numpy().squeeze()

            np.save(out_path, embedding)
            n_embedded += 1
        except Exception as e:
            print(f"  WARNING: Failed to embed {img_path}: {e}")
            n_skipped += 1

        if (i + 1) % 200 == 0:
            print(f"    Progress: {i+1}/{len(photo_meals)} photos embedded")

    print(f"  CLIP cache: {n_embedded} embedded, {n_skipped} skipped → {cache_dir}")
    return n_embedded, n_skipped


def verify_clip_cache(cache_dir: Path, n_check: int = 20):
    """Verify CLIP cache determinism by checking checksums."""
    npy_files = sorted(cache_dir.glob("*.npy"))
    if len(npy_files) == 0:
        print("  No cache files to verify")
        return False

    rng = np.random.RandomState(42)
    check_files = rng.choice(npy_files, size=min(n_check, len(npy_files)), replace=False)

    checksums = {}
    for f in check_files:
        data = np.load(f)
        h = hashlib.sha256(data.tobytes()).hexdigest()
        checksums[f.name] = h

    print(f"  Verified {len(checksums)} checksums (sample: {list(checksums.values())[0][:16]}...)")
    return checksums


# ========== Main ==========

def main():
    print("=" * 60)
    print("GlucoFlow Data Preparation")
    print("=" * 60)

    cards_dir = ROOT / "artifacts" / "dataset_cards"
    figs_dir = ROOT / "artifacts" / "figures" / "problem_proofs"
    window_dir = ROOT / "artifacts" / "qualitative" / "window_sanity"
    cache_dir = ROOT / "artifacts" / "cache" / "clip_embeddings" / "cgmacros"

    cards_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 2.1: Load datasets ---
    print("\n--- Step 2.1: Dataset Adapters ---")

    cgmacros = CGMacrosAdapter(ROOT / "data" / "raw" / "cgmacros").load()
    print(f"CGMacros: {cgmacros.cgm['subject_id'].nunique()} subjects, {len(cgmacros.cgm):,} rows")
    generate_dataset_card(cgmacros, cards_dir)

    weinstock = GlucoBenchAdapter(
        ROOT / "external" / "GlucoBench" / "raw_data", "weinstock"
    ).load()
    print(f"Weinstock: {weinstock.cgm['subject_id'].nunique()} subjects, {len(weinstock.cgm):,} rows")
    generate_dataset_card(weinstock, cards_dir)

    # --- Step 2.2: Normalization ---
    print("\n--- Step 2.2: Normalization ---")

    for ds in [cgmacros, weinstock]:
        print(f"\n  {ds.name}:")
        run_normalization(ds, cards_dir, figs_dir)

    # --- Step 2.3: Windowing ---
    print("\n--- Step 2.3: Windowing Visualization ---")

    for ds in [cgmacros, weinstock]:
        print(f"\n  {ds.name} — Task A:")
        visualize_task_a_windows(ds, None, window_dir)

    print(f"\n  CGMacros — Task B:")
    visualize_task_b_windows(cgmacros, None, window_dir)

    # --- Step 2.4: CLIP Caching ---
    print("\n--- Step 2.4: CLIP Embedding Cache ---")
    n_embedded, n_skipped = cache_clip_embeddings(cgmacros, cache_dir)

    if n_embedded > 0:
        print("\n  Verifying cache determinism...")
        checksums = verify_clip_cache(cache_dir)

        # Report disk size
        total_bytes = sum(f.stat().st_size for f in cache_dir.glob("*.npy"))
        print(f"  Cache disk size: {total_bytes / 1024 / 1024:.1f} MB")
        print(f"  Total embeddings: {len(list(cache_dir.glob('*.npy')))}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
