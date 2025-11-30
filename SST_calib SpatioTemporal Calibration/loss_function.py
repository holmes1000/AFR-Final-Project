import numpy as np
from scipy.spatial import KDTree

class BiDirectionalLoss:
    """
    Implements the bi-directional semantic alignment loss from the paper.
    
    Two components:
    1. Point-to-Pixel (Eq. 3): For each projected point, find nearest car pixel
    2. Pixel-to-Point (Eq. 4): For each car pixel, find nearest projected point
    """
    
    def __init__(self, downsample_ratio=0.02):
        """
        Args:
            downsample_ratio: Fraction of pixels to use for pixel-to-point loss
                             (paper uses 2%)
        """
        self.downsample_ratio = downsample_ratio
    
    def compute_loss(self, projected_points, car_pixel_coords, weight_p2i=1.0, weight_i2p=1.0):
        """
        Compute bi-directional semantic alignment loss
        
        Args:
            projected_points: Nx2 array of projected car point coordinates (u, v)
            car_pixel_coords: Mx2 array of car pixel coordinates (u, v)
            weight_p2i: Weight for point-to-pixel loss
            weight_i2p: Weight for pixel-to-point loss
            
        Returns:
            total_loss: Combined bi-directional loss
            loss_p2i: Point-to-pixel loss
            loss_i2p: Pixel-to-point loss
        """
        if len(projected_points) == 0 or len(car_pixel_coords) == 0:
            return float('inf'), float('inf'), float('inf')
        
        # Build KD-tree for fast nearest neighbor search
        pixel_tree = KDTree(car_pixel_coords)
        point_tree = KDTree(projected_points)
        
        # --- Point-to-Pixel Loss (Eq. 3) ---
        # For each projected point, find distance to nearest car pixel
        distances_p2i, _ = pixel_tree.query(projected_points)
        loss_p2i = np.mean(distances_p2i ** 2)  # Squared Euclidean distance
        
        # --- Pixel-to-Point Loss (Eq. 4) ---
        # Downsample pixels (paper uses 2%)
        num_samples = max(1, int(len(car_pixel_coords) * self.downsample_ratio))
        sample_indices = np.random.choice(len(car_pixel_coords), num_samples, replace=False)
        sampled_pixels = car_pixel_coords[sample_indices]
        
        # For each sampled pixel, find distance to nearest projected point
        distances_i2p, _ = point_tree.query(sampled_pixels)
        loss_i2p = np.mean(distances_i2p ** 2)
        
        # --- Bi-directional Loss (Eq. 5) ---
        # Normalize pixel-to-point loss by ratio of points to pixels
        np_ni_ratio = len(projected_points) / num_samples
        total_loss = weight_p2i * loss_p2i + weight_i2p * np_ni_ratio * loss_i2p
        
        return total_loss, loss_p2i, loss_i2p


def get_car_pixel_coordinates(car_mask):
    """
    Extract (u, v) coordinates of all car pixels from segmentation mask
    
    Args:
        car_mask: HxW boolean array
        
    Returns:
        Mx2 array of (u, v) coordinates
    """
    # np.where returns (rows, cols) = (v, u)
    v_coords, u_coords = np.where(car_mask)
    return np.column_stack([u_coords, v_coords])


# Test the loss function
if __name__ == "__main__":
    from data_loader import KITTIDataLoader
    from calibration import Calibration
    from segmentation import ImageSegmenter
    
    # Load everything
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib = Calibration("dataset/2011_09_26")
    img_segmenter = ImageSegmenter()
    
    # Load frame
    frame_idx = 0
    image = loader.load_image(frame_idx)
    points = loader.load_velodyne(frame_idx)
    
    # Get image segmentation
    _, car_mask = img_segmenter.segment(image)
    car_pixel_coords = get_car_pixel_coordinates(car_mask)
    print(f"Car pixels: {len(car_pixel_coords)}")
    
    # Project points and filter to car points in view
    projected, depths = calib.project_velo_to_image(points)
    img_h, img_w = image.shape[:2]
    
    # Filter to valid points in image
    valid = (
        (depths > 0) &
        (projected[:, 0] >= 0) &
        (projected[:, 0] < img_w) &
        (projected[:, 1] >= 0) &
        (projected[:, 1] < img_h)
    )
    projected_valid = projected[valid]
    
    # Filter to points that land on car pixels (approximately)
    # For now, just use all valid points - we'll refine this
    projected_car = projected_valid  # Simplified for testing
    print(f"Projected points in view: {len(projected_car)}")
    
    # Compute loss
    loss_fn = BiDirectionalLoss(downsample_ratio=0.02)
    total_loss, loss_p2i, loss_i2p = loss_fn.compute_loss(
        projected_car, car_pixel_coords
    )
    
    print(f"\n--- Loss with CORRECT calibration ---")
    print(f"Point-to-Pixel Loss: {loss_p2i:.4f}")
    print(f"Pixel-to-Point Loss: {loss_i2p:.4f}")
    print(f"Total Bi-directional Loss: {total_loss:.4f}")