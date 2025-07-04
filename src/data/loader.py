# /src/data/loader.py
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

def make_loaders(root_dir, config, preprocess_fn):
    """Create DataLoaders for training, query and gallery.

    Parameters
    ----------
    root_dir : str
        Root directory containing the ``training`` and ``test`` folders.
    config : module
        Configuration object with at least ``BATCH_SIZE`` defined.
    preprocess_fn : Callable
        Base preprocessing transform to apply to all images.

    Returns
    -------
    tuple
        ``(train_loader, query_loader, gallery_loader, num_classes,
        gallery_dataset.imgs, query_dataset.imgs)``
    """

    # Paths to the required folders
    training_dir = os.path.join(root_dir, "training")
    gallery_dir = os.path.join(root_dir, "test", "gallery")
    query_dir = os.path.join(root_dir, "test", "query")

    # Ensure the directories exist
    for d in (training_dir, gallery_dir, query_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required directory not found: '{d}'")

    # Determine the number of classes from the training folder structure
    class_names = sorted(
        [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
    )
    num_classes = len(class_names)

    if num_classes == 0:
        raise ValueError(f"No class subfolders found in '{training_dir}'")

    # Augmentation pipeline for the training set
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            preprocess_fn,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )

    # Build datasets
    train_dataset = ImageFolder(training_dir, transform=train_transform)
    gallery_dataset = ImageFolder(gallery_dir, transform=preprocess_fn)
    query_dataset = ImageFolder(query_dir, transform=preprocess_fn)

    # Sampler to ensure balanced classes in each batch
    train_sampler = MPerClassSampler(
        train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
    )

    return (
        train_loader,
        query_loader,
        gallery_loader,
        num_classes,
        gallery_dataset.imgs,
        query_dataset.imgs,
    )
