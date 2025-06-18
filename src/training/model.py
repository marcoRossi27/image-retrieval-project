# /src/training/model.py
import torch.nn as nn
import torch.nn.functional as F

class CLIPWithMLP(nn.Module):
    """
    This model uses a CLIP visual backbone and adds an MLP head on top
    for feature projection, plus a classifier for the multi-task loss.
    """
    def __init__(self, clip_base, embed_dim, num_classes, unfreeze_layers):
        super().__init__()
        self.clip = clip_base

        # Freeze the backbone, except for the last N transformer blocks
        total_blocks = len(self.clip.visual.transformer.resblocks)
        for name, param in self.clip.visual.named_parameters():
            param.requires_grad = False
            if "resblocks" in name:
                try:
                    block_idx = int(name.split('.')[2])
                    if block_idx >= total_blocks - unfreeze_layers:
                        param.requires_grad = True
                except (ValueError, IndexError):
                    # Handle cases where parsing the block index might fail
                    # For now, we assume standard naming and pass
                    pass

        # MLP head for metric learning
        self.head = nn.Sequential(
            nn.Linear(self.clip.visual.output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1024, embed_dim)
        )

        # Classifier for the classification loss component
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Extract features from the CLIP backbone
        # Use .float() to ensure compatibility with AMP (Automatic Mixed Precision)
        features = self.clip.encode_image(x).float()

        # Project features through the MLP head to get the final embedding
        # L2 normalization is crucial for cosine-based similarity losses
        embedding = F.normalize(self.head(features), p=2, dim=-1)

        # Get logits for the classification loss
        logits = self.classifier(embedding)

        return embedding, logits
