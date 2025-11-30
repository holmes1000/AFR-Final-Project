import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms

class ImageSegmenter:
    def __init__(self):
        print("Loading DeepLabV3 model (this may take a moment)...")
        
        # Load pre-trained DeepLabV3 with ResNet50 backbone
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        )
        self.model.eval()
        
        # Image preprocessing (required by the model)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # COCO class names (DeepLabV3 is trained on COCO)
        # Class 7 is 'car' in COCO
        self.class_names = {
            0: 'background', 7: 'car', 2: 'bicycle', 6: 'bus',
            14: 'motorbike', 1: 'person', 4: 'airplane'
        }
        
        self.car_class_id = 7
        
        print("Model loaded successfully!")
    
    def segment(self, image):
        """
        Perform semantic segmentation on an image
        
        Args:
            image: RGB numpy array (H, W, 3)
        
        Returns:
            segmentation_mask: (H, W) array with class IDs
            car_mask: (H, W) boolean array where True = car pixel
        """
        # Preprocess
        input_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # Shape: (num_classes, H, W)
        
        # Get class with highest probability for each pixel
        segmentation_mask = output.argmax(0).numpy()
        
        # Extract car mask
        car_mask = (segmentation_mask == self.car_class_id)
        
        return segmentation_mask, car_mask
    
    def visualize(self, image, car_mask, frame_idx=0):
        """Visualize the car segmentation result"""
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Car mask
        axes[1].imshow(car_mask, cmap='gray')
        axes[1].set_title(f"Car Mask ({car_mask.sum()} pixels)")
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[car_mask] = [255, 255, 0]  # Yellow for cars
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        axes[2].imshow(blended)
        axes[2].set_title("Car Segmentation Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"image_segmentation_frame_{frame_idx}.png", dpi=150)
        plt.show()
        
        return car_mask


if __name__ == "__main__":
    from data_loader import KITTIDataLoader
    
    # Load data
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    
    # Initialize segmenter
    segmenter = ImageSegmenter()
    
    # Test on frame 0
    frame_idx = 0
    image = loader.load_image(frame_idx)
    
    print(f"\nSegmenting frame {frame_idx}...")
    seg_mask, car_mask = segmenter.segment(image)
    
    print(f"Image shape: {image.shape}")
    print(f"Segmentation mask shape: {seg_mask.shape}")
    print(f"Number of car pixels: {car_mask.sum()}")
    print(f"Unique classes detected: {np.unique(seg_mask)}")
    
    # Visualize
    segmenter.visualize(image, car_mask, frame_idx)