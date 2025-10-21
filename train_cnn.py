#!/usr/bin/env python3
"""
Simple CNN Training Script for Breast Cancer Segmentation
"""

from cnn_segmentation import SimpleUNet, create_dataloaders, CNNTrainer

def main():
    # Configuration
    config = {
        'data_dir': 'BUSI_PROCESSED',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 20,  # Shorter training for demo
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    }

    print("🚀 Starting CNN Training...")
    print(f"Device: {config['device']}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        config['batch_size']
    )

    # Create model
    model = SimpleUNet()

    # Create trainer
    trainer = CNNTrainer(
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