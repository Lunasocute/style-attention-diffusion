import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from transformers import AutoImageProcessor, AutoModel
from config import Config


def get_extractors():
    """
    Initialize feature extractors:
      - DINOv2: used as a style descriptor
      - CLIP-Vision: used as a content descriptor
    """
    # DINOv2 (style features)
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(Config.DEVICE).eval()

    # CLIP (content features)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        Config.CLIP_IMAGE_MODEL
    ).to(Config.DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained(Config.CLIP_IMAGE_MODEL)

    return dino_processor, dino_model, clip_model, clip_processor


def extract_features_for_metadata(df_meta: pd.DataFrame):
    """
    Extract features for all images listed in df_meta.

    For each image:
      - style_embeddings: DINOv2 feature (mean-pooled over tokens)
      - content_embeddings: CLIP image_embeds

    Returns:
      style_embeddings  : np.ndarray of shape (N, D_style)
      content_embeddings: np.ndarray of shape (N, D_content)

    NOTE: The order of embeddings matches df_meta.index.
    """
    dino_processor, dino_model, clip_model, clip_processor = get_extractors()
    
    style_feats = []
    content_feats = []

    print("Extracting DINOv2 (style) and CLIP (content) features for all images...")
    paths = df_meta["file_path"].tolist()

    with torch.no_grad():
        for path in tqdm(paths):
            try:
                img = Image.open(path).convert("RGB")
                
                # --- Style features (DINOv2) ---
                dino_inputs = dino_processor(images=img, return_tensors="pt").to(Config.DEVICE)
                dino_out = dino_model(**dino_inputs)
                style_emb = dino_out.last_hidden_state.mean(dim=1)  # [1, D_style]
                style_feats.append(style_emb.squeeze(0).cpu().numpy())
                
                # --- Content features (CLIP) ---
                inputs = clip_processor(images=img, return_tensors="pt").to(Config.DEVICE)
                clip_emb = clip_model(**inputs).image_embeds  # [1, D_content]
                content_feats.append(clip_emb.squeeze(0).cpu().numpy())
            
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # On error, fall back to zero vectors (same dim as common configs)
                style_feats.append(np.zeros(768))
                content_feats.append(np.zeros(768))

    style_embeddings = np.stack(style_feats, axis=0)
    content_embeddings = np.stack(content_feats, axis=0)
    return style_embeddings, content_embeddings


def build_group_based_pairs_and_refs(df_meta, content_embeddings, out_dir):
    """
    Build style/target pairs and test references within each group_id.

    - df_meta must contain: [file_path, artist, genre, style, group_id]
    - content_embeddings: np.ndarray (N, D), aligned with df_meta.index

    For each group_id:
      - Split images into: a. test references (held-out images); b.train images (for pairing)
      - For train images, create pairs (target_image, style_image) where style_image is 
        chosen from the same group but with lower content similarity (based on CLIP embeddings).

    Outputs:
      - TEST_REFS_FILE: CSV of reference images
      - PAIRS_FILE    : CSV of (target, style) training pairs
    """
    os.makedirs(out_dir, exist_ok=True)

    min_imgs = getattr(Config, "MIN_IMAGES_PER_GROUP", 5)
    test_frac = getattr(Config, "TEST_REF_FRACTION_PER_GROUP", 0.2)
    sim_th = getattr(Config, "CONTENT_SIMILARITY_THRESHOLD", 0.8)

    all_pairs = []
    all_refs = []

    grouped = df_meta.groupby("group_id")

    print("[Pairing] Building pairs and test refs per group_id ...")
    for group_id, g in tqdm(grouped, desc="Groups"):
        if len(g) < min_imgs:
            continue

        # Shuffle images inside the group for a random split
        g = g.sample(frac=1.0, random_state=42)

        idxs = g.index.to_list()  # indices align with embeddings
        n = len(idxs)

        n_test = max(1, int(test_frac * n))
        test_idxs = idxs[:n_test]
        train_idxs = idxs[n_test:]

        # Collect test reference images
        for idx in test_idxs:
            row = df_meta.loc[idx]
            all_refs.append({
                "file_path": row["file_path"],
                "artist": row["artist"],
                "genre": row["genre"],
                "style": row["style"],
                "group_id": row["group_id"],
            })

        # Need at least 2 images to form a pair
        if len(train_idxs) < 2:
            continue

        # Local lookup: global idx -> row
        train_rows = {idx: df_meta.loc[idx] for idx in train_idxs}

        # For each target, pick a style image from the same group with low content similarity
        for idx_t in train_idxs:
            row_t = train_rows[idx_t]
            t_vec = content_embeddings[idx_t]
            norm_t = np.linalg.norm(t_vec)
            if norm_t == 0:
                continue

            # Candidate style indices: other images from the same group
            cand_idxs = [i for i in train_idxs if i != idx_t]
            if not cand_idxs:
                continue

            # Randomly sample up to n_cand candidates
            n_cand = min(10, len(cand_idxs))
            cand_sample = np.random.choice(cand_idxs, size=n_cand, replace=False)

            chosen_ref = None
            for idx_r in cand_sample:
                r_vec = content_embeddings[idx_r]
                norm_r = np.linalg.norm(r_vec)
                if norm_r == 0:
                    continue
                sim = float(np.dot(t_vec, r_vec) / (norm_t * norm_r))
                if sim < sim_th:
                    chosen_ref = idx_r
                    break

            if chosen_ref is None:
                chosen_ref = int(cand_sample[0])     # just use the first sampled candidate

            row_r = train_rows[chosen_ref]
            all_pairs.append({
                "target_image": row_t["file_path"],
                "style_image": row_r["file_path"],
                "artist": row_t["artist"],
                "genre": row_t["genre"],
                "style": row_t["style"],
                "group_id": row_t["group_id"],
            })

    # Save test references
    refs_df = pd.DataFrame(all_refs).drop_duplicates(subset=["file_path"])
    refs_path = getattr(Config, "TEST_REFS_FILE", os.path.join(out_dir, "test_refs.csv"))
    refs_df.to_csv(refs_path, index=False)
    print(f"[Refs] Saved {len(refs_df)} test references to: {refs_path}")

    # Save train pairs
    pairs_df = pd.DataFrame(all_pairs)
    pairs_path = Config.PAIRS_FILE
    pairs_df.to_csv(pairs_path, index=False)
    print(f"[Pairs] Saved {len(pairs_df)} train pairs to: {pairs_path}")


def run_pca_per_group(df_meta, style_embeddings, out_dir, max_groups=5, min_points=3):
    """
    Run PCA over style embeddings for several group_id values and plot them.

    For each selected group:
      - Take its style embeddings
      - Run PCA to 2D
      - Save a single-color scatter plot

    This is mainly for visually checking the distribution of style features
    within each group.
    """
    os.makedirs(out_dir, exist_ok=True)

    grouped = df_meta.groupby("group_id")
    # Sort by group size (descending) and prioritize larger groups
    group_sizes = grouped.size().sort_values(ascending=False)

    if max_groups is None:
        chosen_group_ids = group_sizes.index
    else:
        chosen_group_ids = group_sizes.index[:max_groups]

    print(f"[Group PCA] Will analyze groups: {list(chosen_group_ids)}")

    for gid in chosen_group_ids:
        g = grouped.get_group(gid)
        idxs = g.index.to_list()
        if len(idxs) < min_points:
            continue  # skip very small groups

        X = style_embeddings[idxs]  # [num_imgs_in_group, D]

        print(f"[Group PCA] Group '{gid}' size={len(idxs)}")
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)

        plt.figure(figsize=(5, 4))
        plt.scatter(X2[:, 0],X2[:, 1],c="royalblue",s=8,alpha=0.7,)
        plt.title(f"Group PCA (single color):\n{gid}", fontsize=9)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        safe_gid = gid.replace(" ", "_").replace("|", "_")
        img_path = os.path.join(out_dir, f"group_{safe_gid}_pca.png")
        plt.savefig(img_path, dpi=200)
        plt.close()
        print(f"  - Saved PCA plot to {img_path}")


def main():
    os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)

    # Load metadata
    meta_path = Config.METADATA_CSV
    df = pd.read_csv(meta_path)
    print(f"Loaded metadata from {meta_path}, total rows = {len(df)}")

    # Build group_id = "artist | genre | style"
    df["artist"] = df["artist"].astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip()
    df["style"] = df["style"].astype(str).str.strip()
    df["group_id"] = df["artist"] + " | " + df["genre"] + " | " + df["style"]

    # Filter out very small groups
    min_imgs = getattr(Config, "MIN_IMAGES_PER_GROUP", 5)
    group_counts = df["group_id"].value_counts()
    valid_groups = group_counts[group_counts >= min_imgs].index
    df = df[df["group_id"].isin(valid_groups)].reset_index(drop=True)
    print(
        f"After filtering small groups (<{min_imgs}), "
        f"remaining images = {len(df)}, groups = {df['group_id'].nunique()}"
    )

    # Extract DINOv2 / CLIP features (once globally)
    style_embs, content_embs = extract_features_for_metadata(df)

    # Build train pairs + test refs per group
    build_group_based_pairs_and_refs(df_meta=df, content_embeddings=content_embs, out_dir=Config.PROCESSED_DATA_PATH,)

    # Run PCA visualization for a few groups
    run_pca_per_group(
        df_meta=df,
        style_embeddings=style_embs,
        out_dir=os.path.join(Config.PROCESSED_DATA_PATH, "group_pca"),
        max_groups=7,   # increase or set to None to see more groups
        min_points=3,
    )


if __name__ == "__main__":
    main()