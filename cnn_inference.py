#!/usr/bin/env python3
"""
CNN Inference for Breast Cancer Ultrasound Segmentation
Simple and efficient inference pipeline for CNN-based segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as transforms

# Import our CNN model
from cnn_segmentation import SimpleUNet


class CNNInference:
    """
    Inference class for CNN segmentation model
    Handles loading, preprocessing, and prediction
    """

    def __init__(self, checkpoint_path: str = "checkpoints/best_cnn_model.pth", device: str = "auto"):
        """
        Initialize CNN inference

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model(checkpoint_path)

        # Image preprocessing transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("✅ CNN model loaded successfully!")

    def _load_model(self, checkpoint_path: str) -> SimpleUNet:
        """Load trained model from checkpoint"""

        # Create model
        model = SimpleUNet()

        # Try to load checkpoint
        if Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("✅ Model loaded from checkpoint")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("Using untrained model (results will be random)")

        model.to(self.device)
        model.eval()

        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference

        Args:
            image_path: Path to input image

        Returns:
            Tuple of (processed_tensor, original_image_array)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # Apply transform
        processed = self.transform(image).unsqueeze(0)  # Add batch dimension

        return processed, original_image

    def predict(self, image_path: str, threshold: float = 0.5) -> Dict:
        """
        Run inference on a single image

        Args:
            image_path: Path to input image
            threshold: Probability threshold for binary mask

        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        input_tensor, original_image = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.model(input_tensor)

            # Apply sigmoid and threshold
            probabilities = torch.sigmoid(output)
            mask = (probabilities > threshold).float()

            # Convert to numpy
            probabilities = probabilities.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()

        # Resize predictions back to original size
        original_h, original_w = original_image.shape[:2]
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        probabilities = cv2.resize(probabilities, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Calculate statistics
        tumor_pixels = np.sum(mask)
        total_pixels = mask.size
        tumor_percentage = (tumor_pixels / total_pixels) * 100

        # Confidence scores
        tumor_probabilities = probabilities[mask == 1]
        mean_confidence = np.mean(tumor_probabilities) if len(tumor_probabilities) > 0 else 0
        max_confidence = np.max(tumor_probabilities) if len(tumor_probabilities) > 0 else 0

        return {
            'mask': mask,
            'probabilities': probabilities,
            'original_image': original_image,
            'original_size': (original_h, original_w),
            'tumor_pixels': int(tumor_pixels),
            'total_pixels': total_pixels,
            'tumor_percentage': tumor_percentage,
            'mean_confidence': float(mean_confidence),
            'max_confidence': float(max_confidence),
            'image_path': image_path
        }

    def predict_batch(self, image_paths: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Run inference on multiple images

        Args:
            image_paths: List of paths to input images
            threshold: Probability threshold for binary masks

        Returns:
            List of prediction dictionaries
        """
        results = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.predict(image_path, threshold)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })

        return results

    def visualize_result(self, result: Dict, save_path: Optional[str] = None):
        """
        Visualize prediction result

        Args:
            result: Prediction result dictionary
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Segmentation mask
        axes[1].imshow(result['mask'], cmap='gray')
        axes[1].set_title(f'Segmentation Mask\n{result["tumor_percentage"]:.1f}% tumor')
        axes[1].axis('off')

        # Probability map
        prob_plot = axes[2].imshow(result['probabilities'], cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f'Probability Map\nMean conf: {result["mean_confidence"]:.3f}')
        axes[2].axis('off')
        plt.colorbar(prob_plot, ax=axes[2], shrink=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

    def save_results(self, results: List[Dict], output_dir: str = "inference_results"):
        """
        Save batch inference results

        Args:
            results: List of prediction results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save individual results
        for i, result in enumerate(results):
            if 'error' in result:
                continue

            base_name = Path(result['image_path']).stem

            # Save mask
            mask_path = output_path / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_path), (result['mask'] * 255).astype(np.uint8))

            # Save probabilities
            prob_path = output_path / f"{base_name}_probabilities.npy"
            np.save(prob_path, result['probabilities'])

            # Save visualization
            vis_path = output_path / f"{base_name}_visualization.png"
            self.visualize_result(result, str(vis_path))

        # Save summary JSON
        summary = []
        for result in results:
            if 'error' not in result:
                summary.append({
                    'image_path': result['image_path'],
                    'tumor_percentage': result['tumor_percentage'],
                    'mean_confidence': result['mean_confidence'],
                    'max_confidence': result['max_confidence']
                })

        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Results saved to: {output_path}")
        print(f"📊 Summary saved to: {summary_path}")


def main():
    """Command-line interface for CNN inference"""

    parser = argparse.ArgumentParser(description="CNN Breast Cancer Segmentation Inference")
    parser.add_argument("--image", "-i", help="Path to single image")
    parser.add_argument("--directory", "-d", help="Path to directory containing images")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/best_cnn_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--output", "-o", default="inference_results",
                       help="Output directory for results")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Probability threshold for segmentation")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Show visualizations")

    args = parser.parse_args()

    # Initialize inference
    inference = CNNInference(args.checkpoint)

    if args.image:
        # Single image inference
        print(f"Processing single image: {args.image}")
        result = inference.predict(args.image, args.threshold)

        print("Results:")
        print(f"  Tumor area: {result['tumor_percentage']:.1f}%")
        print(f"  Mean confidence: {result['mean_confidence']:.3f}")
        print(f"  Max confidence: {result['max_confidence']:.3f}")

        if args.visualize:
            inference.visualize_result(result)

    elif args.directory:
        # Batch inference
        image_dir = Path(args.directory)
        if not image_dir.exists():
            print(f"❌ Directory not found: {args.directory}")
            return

        # Find all images
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f"**/{ext}")))

        if not image_paths:
            print(f"❌ No images found in: {args.directory}")
            return

        print(f"Found {len(image_paths)} images")

        # Run inference
        results = inference.predict_batch([str(p) for p in image_paths], args.threshold)

        # Save results
        inference.save_results(results, args.output)

        # Print summary
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            avg_tumor = np.mean([r['tumor_percentage'] for r in valid_results])
            avg_conf = np.mean([r['mean_confidence'] for r in valid_results])
            print("\nBatch Summary:")
            print(f"  Images processed: {len(valid_results)}")
            print(f"  Average tumor area: {avg_tumor:.1f}%")
            print(f"  Average confidence: {avg_conf:.3f}")

    else:
        print("❌ Please specify either --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()