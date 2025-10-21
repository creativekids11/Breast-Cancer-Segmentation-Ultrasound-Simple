#!/usr/bin/env python3
"""
Test script to verify weighted sampling functionality
"""

from cnn_segmentation import BUSIDataset, create_dataloaders
from collections import Counter
import torch

def test_weighted_sampling():
    """Test that weighted sampling balances the classes"""

    print("Testing Weighted Sampling for Class Balance")
    print("=" * 50)

    # Create dataset
    dataset = BUSIDataset('BUSI_PROCESSED', augment=False)

    # Show original distribution
    original_counts = Counter([s['category'] for s in dataset.samples])
    print(f"Original class distribution: {dict(original_counts)}")

    # Show calculated weights
    print(f"Calculated class weights: {dataset.class_weights}")

    # Create dataloaders with weighted sampling
    train_loader, val_loader = create_dataloaders(
        'BUSI_PROCESSED',
        batch_size=8,
        val_split=0.2,
        use_weighted_sampling=True
    )

    # Sample batches and count class distribution
    sampled_counts = Counter()
    num_batches = min(10, len(train_loader))  # Sample first 10 batches

    print(f"\nSampling {num_batches} batches with weighted sampling...")

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        categories = batch['category']
        batch_counts = Counter(categories)
        sampled_counts.update(batch_counts)

    print(f"Sampled class distribution: {dict(sampled_counts)}")

    # Calculate balance improvement
    total_original = sum(original_counts.values())
    total_sampled = sum(sampled_counts.values())

    print("\nBalance Analysis:")
    print("Class     | Original % | Sampled %  | Improvement")
    print("-" * 55)

    for category in ['benign', 'malignant', 'normal']:
        orig_pct = original_counts.get(category, 0) / total_original * 100
        sampled_pct = sampled_counts.get(category, 0) / total_sampled * 100
        improvement = sampled_pct - orig_pct
        print(f"{category:8} | {orig_pct:9.1f}% | {sampled_pct:9.1f}% | {improvement:+8.1f}%")

    print("\n✅ Weighted sampling test completed!")
    print("If 'Improvement' values are positive for underrepresented classes,")
    print("weighted sampling is working correctly.")

if __name__ == "__main__":
    test_weighted_sampling()