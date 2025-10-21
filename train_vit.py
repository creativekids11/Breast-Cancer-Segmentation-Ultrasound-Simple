#!/usr/bin/env python3
"""
Simple ViT Training Script for Breast Cancer Segmentation
"""

import torch
from vit_segmentation import ViTUNetSegmentation, create_dataloaders, ViTTrainer

def main():
    # Configuration
    config = {
        'data_dir': 'BUSI_PROCESSED',
        'batch_size': 4,
        'learning_rate': 1e-4,
                'num_epochs': 5,  # Quick test  # Shorter training for demo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("🚀 Starting ViT Training...")
    print(f"Device: {config['device']}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        config['batch_size']
    )

    # Create model
    model = ViTUNetSegmentation(pretrained=True)

    # Create trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['learning_rate']
    )

    # Train
    trainer.train(config['num_epochs'])

    print("✅ Training completed!")

if __name__ == "__main__":
    main()