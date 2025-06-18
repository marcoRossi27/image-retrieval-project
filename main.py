# /main.py
import os
import torch
import random
import numpy as np
import open_clip
import argparse
import shutil
from collections import defaultdict
from PIL import Image
from torchvision.datasets import Food101
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Project-specific Imports ---
# Import all the modules we have created
import config
from src.data.loader import make_loaders
from src.training.model import CLIPWithMLP
from src.training.trainer import train_model
from src.inference.evaluate import extract_embeddings, compute_metrics

def prepare_food101_split(output_dir, num_classes, max_per_class, seed):
    """
    Downloads and splits the Food101 dataset into the required structure.
    This function is called only if the default data directory is not found.
    """
    print(f"INFO: Default directory '{output_dir}' not found.")
    print("      Running automatic preparation of the Food101 dataset...")

    training_dir = os.path.join(output_dir, 'training')
    gallery_dir = os.path.join(output_dir, 'test', 'gallery')
    query_dir = os.path.join(output_dir, 'test', 'query')

    for d in [training_dir, gallery_dir, query_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print("      - Downloading Food101 (this may take a while)...")
    ds = Food101(root="data_cache", download=True, split="train")

    all_labels = sorted(list(set(ds._labels)))
    random.seed(seed)
    selected_labels = sorted(random.sample(all_labels, num_classes))

    images_per_class = defaultdict(list)
    for idx, (path, label) in enumerate(zip(ds._image_files, ds._labels)):
        if label in selected_labels:
            images_per_class[label].append((idx, path))

    print(f"      - Splitting {num_classes} classes into train/gallery/query sets...")
    for label in tqdm(selected_labels, desc="      - Processing classes"):
        items = images_per_class[label][:max_per_class]

        train_items, test_items = train_test_split(items, test_size=0.3, random_state=seed)
        gallery_items, query_items = train_test_split(test_items, test_size=0.33, random_state=seed)

        subsets = {
            os.path.join(training_dir, str(label)): train_items,
            os.path.join(gallery_dir, str(label)): gallery_items,
            os.path.join(query_dir, str(label)): query_items
        }

        for class_dir, subset_items in subsets.items():
            os.makedirs(class_dir, exist_ok=True)
            for idx, img_path in subset_items:
                try:
                    Image.open(img_path).convert("RGB").save(os.path.join(class_dir, f"{idx}.jpg"))
                except Exception as e:
                    print(f"Warning: could not process image {img_path}. Error: {e}")

    print(f"✅ Food101 data is ready in '{output_dir}'.")


def get_data_dir():
    """
    Determines the data directory to use.
    Priority 1: Command-line argument (--data_dir).
    Priority 2: Default directory from config.py. If it doesn't exist, it's created.
    """
    parser = argparse.ArgumentParser(description="Run Image Retrieval Training and Evaluation.")
    parser.add_argument(
        '-d', '--data_dir',
        type=str,
        default=None,
        help=f"Path to the root data directory. If not specified, uses Food101 in the default folder '{config.DEFAULT_DATA_DIR}'."
    )
    args = parser.parse_args()

    if args.data_dir:
        print(f"▶️  Using custom dataset from: {args.data_dir}")
        return args.data_dir
    else:
        root_dir = config.DEFAULT_DATA_DIR
        print(f"▶️  No custom path provided. Using default dataset (Food101) in: {root_dir}")
        if not os.path.isdir(root_dir):
            prepare_food101_split(
                output_dir=root_dir,
                num_classes=config.FOOD101_NUM_CLASSES,
                max_per_class=config.FOOD101_MAX_PER_CLASS,
                seed=config.RANDOM_SEED
            )
        return root_dir


def main():
    """
    Main function to orchestrate the entire pipeline.
    """
    # 1. Initial Setup
    torch.manual_seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if config.DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    root_data_dir = get_data_dir()

    # 2. Model and Loaders Creation
    print("🔧 Creating model and data loaders...")
    clip_model, _, preprocess_fn = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED_WEIGHTS
    )

    train_loader, query_loader, gallery_loader, num_classes, _, _ = make_loaders(root_data_dir, config, preprocess_fn)
    print(f"   - Found {num_classes} training classes.")

    model = CLIPWithMLP(
        clip_base=clip_model,
        embed_dim=config.EMBED_DIM,
        num_classes=num_classes,
        unfreeze_layers=config.UNFREEZE_LAYERS
    )

    # 3. Training
    train_model(model, train_loader, config)

    # 4. Evaluation
    print("\n📊 Starting final evaluation...")
    gallery_embeddings, gallery_labels = extract_embeddings(model, gallery_loader, config.DEVICE)
    query_embeddings, query_labels = extract_embeddings(model, query_loader, config.DEVICE)

    metrics = compute_metrics(
        query_embeddings.numpy(), gallery_embeddings.numpy(),
        query_labels, gallery_labels, config.TOP_K_VALUES
    )

    print("\n--- Final Validation Results ---")
    for key, value in metrics.items():
        print(f"{key}: {value*100:.2f}%")
    print("--------------------------------")


if __name__ == "__main__":
    main()
