# /src/training/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_metric_learning.losses import ProxyAnchorLoss, TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from tqdm import tqdm

def train_model(model, train_loader, config):
    """
    Handles the complete training loop for the model.
    """
    model.to(config.DEVICE)

    # --- Loss Functions ---
    # Main metric learning loss
    proxy_loss_fn = ProxyAnchorLoss(
        num_classes=model.classifier.out_features,
        embedding_size=config.EMBED_DIM,
        margin=config.PROXY_MARGIN,
        alpha=config.PROXY_ALPHA
    ).to(config.DEVICE)

    # Classification loss for regularization and stability
    ce_loss_fn = nn.CrossEntropyLoss()

    # Hard negative mining setup
    triplet_miner = TripletMarginMiner(margin=config.TRIPLET_MARGIN, type_of_triplets="hard")
    triplet_loss_fn = TripletMarginLoss(margin=config.TRIPLET_MARGIN)

    # --- Optimizer & Scheduler ---
    # Use different learning rates for the backbone and the new head
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.clip.visual.parameters() if p.requires_grad], 'lr': config.LR_BACKBONE},
        {'params': list(model.head.parameters()) + list(model.classifier.parameters()), 'lr': config.LR_BASE},
        {'params': proxy_loss_fn.parameters(), 'lr': config.LR_BASE} # Proxies are learnable parameters
    ], weight_decay=config.WEIGHT_DECAY)

    # Scheduler for learning rate warmup and annealing
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * config.WARMUP_EPOCHS, T_mult=1)

    # GradScaler for Automatic Mixed Precision (AMP) training
    scaler = GradScaler()

    print("\n🚀 Starting Main Training Loop...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}", unit="batch")
        for images, labels in progress_bar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                embeddings, logits = model(images)

                # Calculate all loss components
                loss_proxy = proxy_loss_fn(embeddings, labels)
                hard_triplets = triplet_miner(embeddings, labels)
                loss_triplet = triplet_loss_fn(embeddings, labels, hard_triplets)
                loss_ce = ce_loss_fn(logits, labels)

                # Combine losses
                loss = loss_proxy + loss_triplet + (config.CE_WEIGHT * loss_ce)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

    print("\n🔧 Starting Final Head Fine-Tuning...")
    # Freeze the backbone completely
    for p in model.clip.parameters(): 
        p.requires_grad = False

    # Create a new optimizer for the head only with a smaller learning rate
    optimizer2 = torch.optim.AdamW(
        list(model.head.parameters()) + list(model.classifier.parameters()) + list(proxy_loss_fn.parameters()), 
        lr=config.LR_BASE * 0.1, 
        weight_decay=config.WEIGHT_DECAY
    )
    scaler2 = GradScaler()

    # Run one final fine-tuning pass
    for images, labels in tqdm(train_loader, desc="Final FT"):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer2.zero_grad()
        with autocast():
            emb, logits = model(images)
            # Recalculate loss
            loss_proxy = proxy_loss_fn(emb, labels)
            hard_triplets = triplet_miner(emb, labels)
            loss_triplet = triplet_loss_fn(emb, labels, hard_triplets)
            loss_ce = ce_loss_fn(logits, labels)
            loss = loss_proxy + loss_triplet + (config.CE_WEIGHT * loss_ce)
        scaler2.scale(loss).backward()
        scaler2.step(optimizer2)
        scaler2.update()

    print("✅ Training complete.")
