# /src/data/loader.py
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

def make_loaders(root_dir, config, preprocess_fn):
    """
    Creates the DataLoaders for training, query, and gallery from a root_dir.
    """
    training_dir = os.path.join(root_dir, 'training')
    gallery_dir = os.path.join(root_dir, 'test', 'gallery')
    query_dir = os.path.join(root_dir, 'test', 'query')

    # Check if the required directories exist
    for d in [training_dir, gallery_dir, query_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required directory not found: '{d}'")

    class_names = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])
    num_classes = len(class_names)

    if num_classes == 0:
        raise ValueError(f"No class subfolders found in '{training_dir}'")

    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        preprocess_fn,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    train_dataset = ImageFolder(training_dir, transform=train_transform)
    # The MPerClassSampler is crucial for metric learning.
    # It ensures each batch has a fixed number of samples from a fixed number of classes.
    train_sampler = MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))
