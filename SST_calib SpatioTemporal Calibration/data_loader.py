import numpy as np
import cv2
import os

class KITTIDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.image_path = os.path.join(base_path, "image_02", "data")
        self.velo_path = os.path.join(base_path, "velodyne_points", "data")
        
        # Get list of frames
        self.frames = sorted([f.split('.')[0] for f in os.listdir(self.image_path)])
        print(f"Loaded dataset with {len(self.frames)} frames")
    
    def load_image(self, frame_idx):
        """Load RGB image for a given frame index"""
        filename = f"{self.frames[frame_idx]}.png"
        img_file = os.path.join(self.image_path, filename)
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image
    
    def load_velodyne(self, frame_idx):
        """Load LIDAR point cloud for a given frame index
        
        Velodyne points are stored as (x, y, z, reflectance)
        Returns: Nx4 numpy array
        """
        filename = f"{self.frames[frame_idx]}.bin"
        velo_file = os.path.join(self.velo_path, filename)
        
        # KITTI velodyne data is stored as float32 binary
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        return points


# Test it out
if __name__ == "__main__":
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    
    # Load frame 0
    image = loader.load_image(0)
    points = loader.load_velodyne(0)
    
    print(f"\nImage shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    print(f"\nPoint cloud shape: {points.shape}")
    print(f"Point cloud dtype: {points.dtype}")
    print(f"Sample points (first 5):")
    print(points[:5])