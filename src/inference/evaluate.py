# /src/inference/evaluate.py
import torch
import numpy as np
from tqdm import tqdm

def extract_embeddings(model, loader, device):
    """
    Extracts embeddings and labels for a given dataset using the trained model.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the dataset.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing a tensor of all embeddings and a numpy array of all labels.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting Embeddings"):
            images = images.to(device)
            # We only need the embedding for evaluation, not the logits
            embeddings, _ = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.numpy())

    return torch.vstack(all_embeddings), np.array(all_labels)


def compute_metrics(query_embeddings, gallery_embeddings, query_labels, gallery_labels, top_k_list):
    """
    Computes retrieval metrics like Top-K accuracy and mAP@k.

    Args:
        query_embeddings (np.array): Embeddings for the query set.
        gallery_embeddings (np.array): Embeddings for the gallery set.
        query_labels (np.array): Labels for the query set.
        gallery_labels (np.array): Labels for the gallery set.
        top_k_list (list): A list of integers for k (e.g., [1, 5, 10]).

    Returns:
        dict: A dictionary with the computed metrics.
    """
    # Calculate cosine similarity (embeddings are L2 normalized)
    similarities = query_embeddings @ gallery_embeddings.T
    # Get the indices of the most similar gallery images for each query
    sorted_indices = np.argsort(-similarities, axis=1)

    results = {}
    num_queries = len(query_labels)

    for k in top_k_list:
        top_k_indices = sorted_indices[:, :k]

        # Top-K Accuracy: is the correct label present in the top K results?
        correct_predictions = [
            query_labels[i] in gallery_labels[top_k_indices[i]] for i in range(num_queries)
        ]
        accuracy = np.mean(correct_predictions)

        # Mean Average Precision (mAP) @ k
        aps = []
        for i in range(num_queries):
            query_label = query_labels[i]
            retrieved_labels = gallery_labels[top_k_indices[i]]

            # A mask indicating which of the retrieved items are relevant (correct)
            relevant_mask = (retrieved_labels == query_label)
            if relevant_mask.sum() == 0:
                aps.append(0.0)
                continue

            # Calculate precision at each position
            precision_at_k = np.cumsum(relevant_mask) / (np.arange(k) + 1)
            # Average precision is the sum of precisions at relevant positions, divided by total relevant items
            ap = (precision_at_k * relevant_mask).sum() / relevant_mask.sum()
            aps.append(ap)

        map_at_k = np.mean(aps)

        results[f"Top-{k} Accuracy"] = accuracy
        results[f"mAP@{k}"] = map_at_k

    return results
