#!/usr/bin/env python3
"""
Vision Transformer (ViT) for Breast Cancer Ultrasound Segmentation
A simplified, high-performance approach using transformers for medical image segmentation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
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

class ViTUNetSegmentation(nn.Module):
    """
    Vision Transformer with U-Net decoder for medical image segmentation
    Combines ViT encoder with convolutional decoder for precise segmentation
    """

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()

        # Load pretrained ViT backbone
        if pretrained:
            try:
                weights = ViT_B_16_Weights.IMAGENET1K_V1
                self.vit = vit_b_16(weights=weights)
            except:
                print("Warning: Could not load pretrained weights, using random initialization")
                self.vit = vit_b_16(weights=None)
        else:
            self.vit = vit_b_16(weights=None)

        # Remove classification head
        self.vit.heads = nn.Identity()

        # ViT outputs 768-dimensional features
        vit_dim = 768

        # Decoder layers for upsampling
        self.decoder = nn.ModuleDict({
            'conv1': nn.Conv2d(vit_dim, 512, 3, padding=1),
            'bn1': nn.BatchNorm2d(512),
            'conv2': nn.Conv2d(512, 256, 3, padding=1),
            'bn2': nn.BatchNorm2d(256),
            'conv3': nn.Conv2d(256, 128, 3, padding=1),
            'bn3': nn.BatchNorm2d(128),
            'conv4': nn.Conv2d(128, 64, 3, padding=1),
            'bn4': nn.BatchNorm2d(64),
            'final': nn.Conv2d(64, num_classes, 1)
        })

        # Initialize decoder weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize decoder weights"""
        for module in self.decoder.values():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input shape: [B, 3, 224, 224]
        batch_size = x.shape[0]

        # Use ViT as feature extractor - run full forward pass
        with torch.no_grad():  # Freeze ViT backbone during training
            # Get the encoded sequence from ViT
            encoded_seq = self.vit(x)  # This gives classification output, but we want intermediate features

            # Instead, let's manually extract patch embeddings
            # Patch embedding
            x_patched = self.vit.conv_proj(x)  # [B, 768, 14, 14]
            x_patched = x_patched.flatten(2).transpose(1, 2)  # [B, 196, 768]

            # Add positional embedding (need to handle this carefully)
            if hasattr(self.vit, 'encoder'):
                # Get positional embedding from encoder
                pos_embed = self.vit.encoder.pos_embedding
                # Add class token
                cls_token = self.vit.class_token.expand(batch_size, -1, -1)
                x_full = torch.cat((cls_token, x_patched), dim=1)  # [B, 197, 768]
                x_full = x_full + pos_embed

                # Pass through encoder
                encoded = self.vit.encoder(x_full)  # [B, 197, 768]
            else:
                # Fallback: just use patch embeddings without positional encoding
                encoded = x_patched.unsqueeze(1)  # [B, 1, 196, 768] - dummy dimension
                encoded = encoded.squeeze(1)  # [B, 196, 768]

        # For now, if we have the full encoded sequence, remove class token
        if encoded.shape[1] == 197:
            patch_features = encoded[:, 1:]  # [B, 196, 768]
        else:
            patch_features = encoded  # [B, 196, 768]

        # Reshape to spatial grid: [B, 196, 768] -> [B, 14, 14, 768] -> [B, 768, 14, 14]
        num_patches = patch_features.shape[1]
        h = w = int(np.sqrt(num_patches))  # Should be 14
        spatial_features = patch_features.reshape(batch_size, h, w, -1)
        x = spatial_features.permute(0, 3, 1, 2)  # [B, 768, 14, 14]

        # Decoder with progressive upsampling
        # Stage 1: 14x14 -> 28x28
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.decoder['bn1'](self.decoder['conv1'](x)))

        # Stage 2: 28x28 -> 56x56
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.decoder['bn2'](self.decoder['conv2'](x)))

        # Stage 3: 56x56 -> 112x112
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.decoder['bn3'](self.decoder['conv3'](x)))

        # Stage 4: 112x112 -> 224x224
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.decoder['bn4'](self.decoder['conv4'](x)))

        # Final prediction
        x = self.decoder['final'](x)

        return x


class BUSIDataset(Dataset):
    """
    BUSI Dataset for Vision Transformer segmentation
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
        
        # Transforms for ViT (ImageNet normalization)
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


class ViTTrainer:
    """
    Trainer class for ViT segmentation model
    Handles training, validation, and model saving
    """
    
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
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
            self.scheduler.step()
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
                }, os.path.join(save_dir, 'best_vit_model.pth'))
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
        }, os.path.join(save_dir, 'final_vit_model.pth'))
        
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
        'batch_size': 4,  # Smaller batch size for ViT
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🏥 ViT Breast Cancer Segmentation Training")
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
    model = ViTUNetSegmentation(pretrained=True)
    
    # Create trainer
    trainer = ViTTrainer(
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
    print("💾 Best model saved to 'checkpoints/best_vit_model.pth'")


if __name__ == "__main__":
    main()