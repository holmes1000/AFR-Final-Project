"""
Step 4.2: Point Cloud Semantic Segmentation

This module provides point cloud segmentation for the SST-Calib pipeline.

Two approaches are implemented:
1. GeometricSegmentor: Simple geometric filtering (baseline, no ML)
2. SqueezeSegV3Segmentor: Deep learning based (requires pretrained model)

For the calibration task, we primarily need to identify "car" points.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path


class GeometricSegmentor:
    """
    Simple geometric-based point cloud segmentation.
    
    Uses height filtering and clustering to identify potential vehicle points.
    This is a baseline approach that works without any trained model.
    
    Assumptions for vehicle detection:
    - Vehicles are typically 0.3m to 2.5m above ground
    - Ground plane is roughly at z ≈ -1.7m (LIDAR height on KITTI vehicle)
    - Vehicles are within reasonable distance (5m to 50m for good visibility)
    """
    
    def __init__(self, 
                 ground_height=-1.7,
                 min_height_above_ground=0.3,
                 max_height_above_ground=2.5,
                 min_distance=3.0,
                 max_distance=50.0):
        """
        Initialize geometric segmentor.
        
        Args:
            ground_height: estimated ground plane z-coordinate in velodyne frame
            min_height_above_ground: minimum height for vehicle points
            max_height_above_ground: maximum height for vehicle points  
            min_distance: minimum distance from sensor
            max_distance: maximum distance from sensor
        """
        self.ground_height = ground_height
        self.min_height = ground_height + min_height_above_ground
        self.max_height = ground_height + max_height_above_ground
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        print(f"GeometricSegmentor initialized:")
        print(f"  Ground height: {ground_height}m")
        print(f"  Vehicle height range: [{self.min_height:.1f}, {self.max_height:.1f}]m")
        print(f"  Distance range: [{min_distance}, {max_distance}]m")
    
    def segment(self, point_cloud):
        """
        Segment point cloud to identify potential vehicle points.
        
        Args:
            point_cloud: (N, 4) array [x, y, z, reflectance]
            
        Returns:
            labels: (N,) array with class labels (0=other, 1=vehicle)
            vehicle_mask: (N,) boolean mask for vehicle points
        """
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        
        # Calculate distance from sensor
        distance = np.sqrt(x**2 + y**2)
        
        # Height filtering (vehicles are above ground but not too high)
        height_mask = (z > self.min_height) & (z < self.max_height)
        
        # Distance filtering
        distance_mask = (distance > self.min_distance) & (distance < self.max_distance)
        
        # Forward-facing points only (camera FOV)
        forward_mask = x > 0
        
        # Combine masks
        vehicle_mask = height_mask & distance_mask & forward_mask
        
        # Create labels
        labels = np.zeros(len(point_cloud), dtype=np.int32)
        labels[vehicle_mask] = 1
        
        return labels, vehicle_mask
    
    def get_vehicle_points(self, point_cloud):
        """
        Get only the vehicle points.
        
        Args:
            point_cloud: (N, 4) array
            
        Returns:
            vehicle_points: (M, 4) array of vehicle points
            vehicle_indices: indices of vehicle points in original array
        """
        labels, vehicle_mask = self.segment(point_cloud)
        vehicle_points = point_cloud[vehicle_mask]
        vehicle_indices = np.where(vehicle_mask)[0]
        
        return vehicle_points, vehicle_indices


class RangeImageConverter:
    """
    Converts 3D point cloud to 2D range image format.
    
    This is required for SqueezeSegV3 and similar architectures.
    The range image represents the scene as a 2D projection based on
    the spherical coordinates of each point.
    """
    
    def __init__(self, 
                 height=64,           # Number of vertical beams (Velodyne HDL-64E)
                 width=2048,          # Horizontal resolution
                 fov_up=3.0,          # Field of view up (degrees)
                 fov_down=-25.0):     # Field of view down (degrees)
        """
        Initialize range image converter.
        
        Args:
            height: vertical resolution (number of laser beams)
            width: horizontal resolution
            fov_up: upper field of view limit in degrees
            fov_down: lower field of view limit in degrees
        """
        self.height = height
        self.width = width
        self.fov_up = fov_up / 180.0 * np.pi      # Convert to radians
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_total = abs(self.fov_down) + abs(self.fov_up)
        
    def convert(self, point_cloud):
        """
        Convert 3D point cloud to range image.
        
        Args:
            point_cloud: (N, 4) array [x, y, z, reflectance]
            
        Returns:
            range_image: (5, H, W) array with channels [x, y, z, intensity, range]
            point_to_pixel: (N,) array mapping each point to pixel index (-1 if not mapped)
            pixel_to_point: (H, W) array mapping each pixel to point index (-1 if empty)
        """
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        intensity = point_cloud[:, 3]
        
        # Calculate range (distance)
        range_vals = np.sqrt(x**2 + y**2 + z**2)
        
        # Calculate angles
        # Azimuth (yaw): angle in x-y plane from x-axis
        azimuth = np.arctan2(y, x)
        
        # Elevation (pitch): angle from x-y plane
        elevation = np.arcsin(z / (range_vals + 1e-8))
        
        # Map to image coordinates
        # u: horizontal pixel (0 to width-1)
        u = 0.5 * (1.0 - azimuth / np.pi) * self.width
        u = np.floor(u).astype(np.int32)
        u = np.clip(u, 0, self.width - 1)
        
        # v: vertical pixel (0 to height-1)
        v = (1.0 - (elevation - self.fov_down) / self.fov_total) * self.height
        v = np.floor(v).astype(np.int32)
        v = np.clip(v, 0, self.height - 1)
        
        # Initialize range image
        range_image = np.zeros((5, self.height, self.width), dtype=np.float32)
        
        # Initialize mapping arrays
        point_to_pixel = np.full(len(point_cloud), -1, dtype=np.int32)
        pixel_to_point = np.full((self.height, self.width), -1, dtype=np.int32)
        
        # Fill range image (use closest point for each pixel)
        depth_buffer = np.full((self.height, self.width), np.inf)
        
        for i in range(len(point_cloud)):
            vi, ui = v[i], u[i]
            
            if range_vals[i] < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = range_vals[i]
                
                range_image[0, vi, ui] = x[i]
                range_image[1, vi, ui] = y[i]
                range_image[2, vi, ui] = z[i]
                range_image[3, vi, ui] = intensity[i]
                range_image[4, vi, ui] = range_vals[i]
                
                # Update previous point mapping
                prev_point = pixel_to_point[vi, ui]
                if prev_point >= 0:
                    point_to_pixel[prev_point] = -1
                
                point_to_pixel[i] = vi * self.width + ui
                pixel_to_point[vi, ui] = i
        
        return range_image, point_to_pixel, pixel_to_point


class SqueezeSegV3Segmentor:
    """
    SqueezeSegV3-based point cloud segmentation.
    
    NOTE: This requires downloading the pretrained model weights.
    For now, this is a placeholder that shows the interface.
    
    To use the real SqueezeSegV3:
    1. Clone: https://github.com/chenfengxu714/SqueezeSegV3
    2. Download pretrained weights for KITTI
    3. Update the model path below
    """
    
    # Class mapping for SemanticKITTI (what SqueezeSegV3 is trained on)
    SEMANTIC_KITTI_CLASSES = {
        0: 'unlabeled',
        1: 'car',
        2: 'bicycle', 
        3: 'motorcycle',
        4: 'truck',
        5: 'other-vehicle',
        6: 'person',
        7: 'bicyclist',
        8: 'motorcyclist',
        9: 'road',
        10: 'parking',
        11: 'sidewalk',
        12: 'other-ground',
        13: 'building',
        14: 'fence',
        15: 'vegetation',
        16: 'trunk',
        17: 'terrain',
        18: 'pole',
        19: 'traffic-sign'
    }
    
    VEHICLE_CLASS_IDS = [1, 2, 3, 4, 5]  # car, bicycle, motorcycle, truck, other-vehicle
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize SqueezeSegV3 segmentor.
        
        Args:
            model_path: path to pretrained weights
            device: torch device
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.range_converter = RangeImageConverter()
        self.model = None
        self.model_loaded = False
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print("SqueezeSegV3 model not loaded. Using geometric fallback.")
            print("To use SqueezeSegV3, download weights from:")
            print("  https://github.com/chenfengxu714/SqueezeSegV3")
            self.fallback = GeometricSegmentor()
    
    def _load_model(self, model_path):
        """Load pretrained SqueezeSegV3 model."""
        # Placeholder - implement actual model loading here
        print(f"Loading SqueezeSegV3 from {model_path}")
        self.model_loaded = True
    
    def segment(self, point_cloud):
        """
        Segment point cloud using SqueezeSegV3.
        
        Args:
            point_cloud: (N, 4) array
            
        Returns:
            labels: (N,) array with class labels
            vehicle_mask: (N,) boolean mask for vehicle points
        """
        if not self.model_loaded:
            return self.fallback.segment(point_cloud)
        
        # Convert to range image
        range_image, point_to_pixel, pixel_to_point = self.range_converter.convert(point_cloud)
        
        # Run model inference
        # ... (implement actual inference here)
        
        # Map predictions back to points
        # ... 
        
        # Placeholder return
        return self.fallback.segment(point_cloud)


class ProjectionSegmentor:
    """
    Segments point cloud by projecting to image and using image segmentation.
    
    This approach:
    1. Projects 3D points onto the image plane
    2. Looks up the semantic label from the image segmentation mask
    3. Assigns that label to the 3D point
    
    This is actually what SST-Calib uses - transferring labels via projection!
    """
    
    def __init__(self, image_segmentor, data_loader):
        """
        Initialize projection-based segmentor.
        
        Args:
            image_segmentor: ImageSegmentor instance
            data_loader: KITTIDataLoader instance
        """
        self.image_segmentor = image_segmentor
        self.data_loader = data_loader
        
    def segment(self, point_cloud, image, R=None, t=None):
        """
        Segment point cloud by projecting to image.
        
        Args:
            point_cloud: (N, 4) array
            image: RGB image
            R: rotation matrix (uses ground truth if None)
            t: translation vector (uses ground truth if None)
            
        Returns:
            labels: (N,) array with class labels (0=other, 1=target class)
            vehicle_mask: (N,) boolean mask
            points_2d: (M, 2) projected coordinates for valid points
            valid_mask: (N,) boolean mask for points that project into image
        """
        # Get image segmentation
        full_mask, binary_mask = self.image_segmentor.segment(image)
        
        # Project points to image
        points_2d, points_3d, valid_mask = self.data_loader.project_velodyne_to_image(
            point_cloud, R=R, t=t
        )
        
        # Initialize labels
        labels = np.zeros(len(point_cloud), dtype=np.int32)
        
        # For valid projected points, look up label in image mask
        points_2d_int = points_2d.astype(np.int32)
        u_coords = points_2d_int[:, 0]
        v_coords = points_2d_int[:, 1]
        
        # Get labels from binary mask
        projected_labels = binary_mask[v_coords, u_coords]
        
        # Assign labels back to 3D points
        valid_indices = np.where(valid_mask)[0]
        labels[valid_indices] = projected_labels.astype(np.int32)
        
        vehicle_mask = labels == 1
        
        return labels, vehicle_mask, points_2d, valid_mask


def visualize_point_cloud_segmentation(point_cloud, labels, title="Point Cloud Segmentation"):
    """
    Visualize segmented point cloud in bird's eye view.
    
    Args:
        point_cloud: (N, 4) array
        labels: (N,) array of labels
        title: plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bird's eye view - all points
    ax = axes[0]
    scatter = ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1], 
        c=point_cloud[:, 2],
        cmap='viridis',
        s=0.5,
        alpha=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Height (m)')
    ax.set_xlabel('X (forward, m)')
    ax.set_ylabel('Y (left, m)')
    ax.set_title('All Points (colored by height)')
    ax.set_xlim(0, 80)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    
    # Bird's eye view - segmented
    ax = axes[1]
    
    # Plot non-vehicle points in gray
    non_vehicle = labels == 0
    ax.scatter(
        point_cloud[non_vehicle, 0],
        point_cloud[non_vehicle, 1],
        c='lightgray',
        s=0.5,
        alpha=0.3,
        label='Other'
    )
    
    # Plot vehicle points in red
    vehicle = labels == 1
    ax.scatter(
        point_cloud[vehicle, 0],
        point_cloud[vehicle, 1],
        c='red',
        s=2,
        alpha=0.8,
        label='Vehicle'
    )
    
    ax.set_xlabel('X (forward, m)')
    ax.set_ylabel('Y (left, m)')
    ax.set_title(f'Segmented Points ({np.sum(vehicle)} vehicle points)')
    ax.set_xlim(0, 80)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def test_pointcloud_segmentation():
    """Test point cloud segmentation methods."""
    import sys
    sys.path.append('src')
    from data_loader import KITTIDataLoader
    from image_segmentation import ImageSegmentor
    
    # =====================================================
    # CONFIGURE THESE PATHS FOR YOUR SYSTEM
    # =====================================================
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    DATE = "2011_09_26"
    DRIVE = "0005"
    # =====================================================
    
    # Load data
    print("Loading data...")
    loader = KITTIDataLoader(BASE_PATH, DATE, DRIVE)
    
    # Test frame
    frame_idx = 0
    image, point_cloud = loader.load_frame_pair(frame_idx)
    
    print(f"\nFrame {frame_idx}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Point cloud shape: {point_cloud.shape}")
    
    # ===== Test 1: Geometric Segmentor =====
    print("\n" + "="*60)
    print("Test 1: Geometric Segmentor")
    print("="*60)
    
    geo_segmentor = GeometricSegmentor()
    geo_labels, geo_mask = geo_segmentor.segment(point_cloud)
    
    print(f"Vehicle points (geometric): {np.sum(geo_mask)} / {len(point_cloud)}")
    
    fig = visualize_point_cloud_segmentation(
        point_cloud, geo_labels, 
        title=f"Geometric Segmentation - Frame {frame_idx}"
    )
    plt.savefig('outputs/pc_seg_geometric.png', dpi=150)
    plt.show()
    
    # ===== Test 2: Projection-based Segmentor =====
    print("\n" + "="*60)
    print("Test 2: Projection-based Segmentor (using image segmentation)")
    print("="*60)
    
    img_segmentor = ImageSegmentor(target_classes=['car', 'bus'])
    proj_segmentor = ProjectionSegmentor(img_segmentor, loader)
    
    proj_labels, proj_mask, points_2d, valid_mask = proj_segmentor.segment(
        point_cloud, image
    )
    
    print(f"Vehicle points (projection): {np.sum(proj_mask)} / {len(point_cloud)}")
    print(f"Points in camera FOV: {np.sum(valid_mask)}")
    
    fig = visualize_point_cloud_segmentation(
        point_cloud, proj_labels,
        title=f"Projection-based Segmentation - Frame {frame_idx}"
    )
    plt.savefig('outputs/pc_seg_projection.png', dpi=150)
    plt.show()
    
    # ===== Visualize projection with segmentation overlay =====
    print("\n" + "="*60)
    print("Visualizing projection with segmentation")
    print("="*60)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    
    # Show image
    ax.imshow(image)
    
    # Get image segmentation mask
    _, binary_mask = img_segmentor.segment(image)
    
    # Overlay image mask
    mask_overlay = np.zeros((*binary_mask.shape, 4))
    mask_overlay[binary_mask] = [1, 1, 0, 0.3]  # Yellow
    ax.imshow(mask_overlay)
    
    # Plot projected points colored by their label
    valid_points = point_cloud[valid_mask]
    valid_labels = proj_labels[valid_mask]
    
    # Non-vehicle points
    non_veh = valid_labels == 0
    ax.scatter(
        points_2d[non_veh, 0],
        points_2d[non_veh, 1],
        c='cyan',
        s=1,
        alpha=0.3,
        label='Other points'
    )
    
    # Vehicle points
    veh = valid_labels == 1
    ax.scatter(
        points_2d[veh, 0],
        points_2d[veh, 1],
        c='red',
        s=3,
        alpha=0.8,
        label='Vehicle points'
    )
    
    ax.set_title(f'Frame {frame_idx}: Image Segmentation + Projected Point Labels')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/pc_seg_overlay.png', dpi=150)
    plt.show()
    
    print("\nSegmentation tests complete!")
    print("Saved visualizations to outputs/")


if __name__ == "__main__":
    test_pointcloud_segmentation()