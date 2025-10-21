#!/usr/bin/env python3
"""
Simple CNN for Breast Cancer Ultrasound Segmentation
A lightweight, efficient CNN-based approach for medical image segmentation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import json
from typing import Tuple, Dict, List, Optional
from collections import Counter


class SimpleUNet(nn.Module):
    """
    Simple U-Net architecture for medical image segmentation
    Encoder-decoder with skip connections
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output
        self.final = nn.Conv2d(64, out_channels, 1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Upsampling
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 224x224 -> 224x224
        p1 = self.pool(e1)  # 224x224 -> 112x112

        e2 = self.enc2(p1)  # 112x112 -> 112x112
        p2 = self.pool(e2)  # 112x112 -> 56x56

        e3 = self.enc3(p2)  # 56x56 -> 56x56
        p3 = self.pool(e3)  # 56x56 -> 28x28

        e4 = self.enc4(p3)  # 28x28 -> 28x28
        p4 = self.pool(e4)  # 28x28 -> 14x14

        # Bottleneck
        b = self.bottleneck(p4)  # 14x14 -> 14x14

        # Decoder with skip connections
        d4 = self.up4(b)  # 14x14 -> 28x28
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # 28x28 -> 56x56
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # 56x56 -> 112x112
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # 112x112 -> 224x224
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)

        # Final output
        out = self.final(d1)
        return out


class BUSIDataset(Dataset):
    """
    BUSI Dataset for CNN segmentation
    Handles loading and preprocessing of ultrasound images and masks
    """

    def __init__(self, data_dir: str, image_size: int = 224, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Find all images and masks
        self.samples = self._load_samples()
        
        # Calculate class weights for balanced sampling
        self.class_weights = self._calculate_class_weights()
        
        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ])
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"Class distribution: {dict(Counter([s['category'] for s in self.samples]))}")
        print(f"Class weights: {self.class_weights}")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """Load all image-mask pairs"""
        samples = []
        
        # Check both raw and processed directories
        for images_dir in [self.data_dir / "IMAGES", self.data_dir]:
            if images_dir.exists():
                # Handle different directory structures
                subdirs_to_check = []
                
                # Check if we have train/val/test structure
                if (images_dir / "train").exists():
                    for split in ["train", "val", "test"]:
                        split_dir = images_dir / split
                        if split_dir.exists():
                            subdirs_to_check.append((split_dir, split))
                else:
                    # Original structure: direct category folders
                    for category in ["benign", "malignant", "normal"]:
                        cat_dir = images_dir / category
                        if cat_dir.exists():
                            subdirs_to_check.append((cat_dir, None))
                
                # Process each directory
                for img_dir, split in subdirs_to_check:
                    for img_file in img_dir.glob("*.png"):
                        # Extract category from filename
                        if img_file.stem.startswith("benign"):
                            category = "benign"
                        elif img_file.stem.startswith("malignant"):
                            category = "malignant"
                        elif img_file.stem.startswith("normal"):
                            category = "normal"
                        else:
                            continue  # Skip files that don't match expected pattern
                        
                        # Look for corresponding mask in MASKS directory
                        mask_dir = self.data_dir / "MASKS"
                        if split:
                            mask_dir = mask_dir / split
                        
                        # Try different mask naming patterns
                        mask_patterns = [
                            f"{category} {img_file.stem.split(' ', 1)[1]}_mask.png" if ' ' in img_file.stem else None,
                            img_file.name.replace(".png", "_mask.png"),
                            img_file.stem + "_mask.png"
                        ]
                        
                        mask_file = None
                        for pattern in mask_patterns:
                            if pattern:
                                potential_mask = mask_dir / pattern
                                if potential_mask.exists():
                                    mask_file = potential_mask
                                    break
                        
                        if mask_file and mask_file.exists():
                            samples.append({
                                'image': str(img_file),
                                'mask': str(mask_file),
                                'category': category
                            })
        
        return samples
    
    def _calculate_class_weights(self) -> Dict[str, float]:
        """Calculate class weights for balanced sampling"""
        # Count samples per class
        class_counts = Counter([sample['category'] for sample in self.samples])
        total_samples = len(self.samples)
        
        # Calculate weights as inverse of class frequency
        class_weights = {}
        for category, count in class_counts.items():
            class_weights[category] = total_samples / (len(class_counts) * count)
        
        return class_weights
    
    def get_sample_weights(self) -> List[float]:
        """Get weights for each sample for WeightedRandomSampler"""
        return [self.class_weights[sample['category']] for sample in self.samples]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        image = Image.open(sample['image']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')

        # Apply augmentations if training
        if self.augment:
            # Apply same random augmentation to both image and mask
            seed = torch.randint(0, 2**32, size=(1,)).item()

            torch.manual_seed(seed)
            image = self.augment_transform(image)

            torch.manual_seed(seed)
            # Convert mask to tensor temporarily for augmentation
            mask_tensor = transforms.ToTensor()(mask)
            mask_tensor = transforms.RandomHorizontalFlip(p=0.5)(mask_tensor)
            mask_tensor = transforms.RandomRotation(degrees=10)(mask_tensor)
            mask = transforms.ToPILImage()(mask_tensor)

        # Apply final transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Ensure mask is binary
        mask = (mask > 0.5).float()

        return {
            'image': image,
            'mask': mask,
            'category': sample['category'],
            'path': sample['image']
        }


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for segmentation"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE loss
        bce_loss = self.bce(pred, target)

        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class CNNTrainer:
    """
    Trainer class for CNN segmentation model
    Handles training, validation, and model saving
    """

    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': []
        }

    def calculate_dice_score(self, pred, target):
        """Calculate Dice score"""
        pred = torch.sigmoid(pred) > 0.5
        target = target > 0.5

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean().item()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = self.calculate_dice_score(outputs, masks)

                total_loss += loss.item()
                total_dice += dice
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return avg_loss, avg_dice

    def train(self, num_epochs, save_dir="checkpoints"):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        best_dice = 0.0

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_dice = self.validate()

            # Update scheduler
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f}")
            print(f"LR: {current_lr:.2e}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_loss': val_loss
                }, os.path.join(save_dir, 'best_cnn_model.pth'))
                print(f"🎉 New best model saved! Dice: {best_dice:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'final_dice': val_dice
        }, os.path.join(save_dir, 'final_cnn_model.pth'))

        print(f"\n🏁 Training completed!")
        print(f"Best Dice Score: {best_dice:.4f}")

        return self.history


def create_dataloaders(data_dir: str, batch_size: int = 8, val_split: float = 0.2, use_weighted_sampling: bool = True):
    """Create train and validation dataloaders with optional weighted sampling"""
    
    # Load full dataset
    full_dataset = BUSIDataset(data_dir, augment=True)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create weighted sampler for training if requested
    train_sampler = None
    if use_weighted_sampling:
        # Get weights for the training subset
        train_indices = train_dataset.indices
        train_weights = [full_dataset.get_sample_weights()[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Using weighted sampling for training with {len(train_weights)} samples")
    
    # Create val dataset without augmentation
    val_base_dataset = BUSIDataset(data_dir, augment=False)
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_base_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # Use weighted sampler if available
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        num_workers=0, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation")
    if use_weighted_sampling:
        print("✅ Weighted sampling enabled for balanced class distribution")
    
    return train_loader, val_loader
def main():
    """Main training function"""

    # Configuration
    config = {
        'data_dir': 'BUSI_PROCESSED',
        'batch_size': 4,  # Smaller batch size for stability
        'learning_rate': 1e-4,
        'num_epochs': 30,  # Shorter training for demo
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("🏥 CNN Breast Cancer Segmentation Training")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        config['batch_size'],
        use_weighted_sampling=True  # Enable weighted sampling for class balance
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

    # Train model
    history = trainer.train(config['num_epochs'])

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_dice'])
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ Training completed successfully!")
    print("📊 Training history saved to 'training_history.png'")
    print("💾 Best model saved to 'checkpoints/best_cnn_model.pth'")


if __name__ == "__main__":
    main()