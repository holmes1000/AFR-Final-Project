import numpy as np
from data_loader import KITTIDataLoader
from calibration import Calibration
from segmentation import ImageSegmenter
from loss_function import BiDirectionalLoss, get_car_pixel_coordinates

def identify_car_points(points, car_mask_image, calib_gt, img_h, img_w):
    """
    Identify which points are car points using GROUND TRUTH calibration.
    This is done once and the indices are reused for all calibration tests.
    
    Returns:
        car_point_indices: Indices of points that are cars
        car_points_3d: The 3D coordinates of car points
    """
    projected, depths = calib_gt.project_velo_to_image(points)
    
    valid = (
        (depths > 0) &
        (projected[:, 0] >= 0) &
        (projected[:, 0] < img_w) &
        (projected[:, 1] >= 0) &
        (projected[:, 1] < img_h)
    )
    
    car_point_indices = []
    
    for i in range(len(points)):
        if valid[i]:
            u, v = int(projected[i, 0]), int(projected[i, 1])
            if car_mask_image[v, u]:
                car_point_indices.append(i)
    
    car_point_indices = np.array(car_point_indices)
    car_points_3d = points[car_point_indices]
    
    return car_point_indices, car_points_3d


def project_car_points(car_points_3d, calib, img_h, img_w):
    """
    Project pre-identified car points using a given calibration.
    Filter to points that remain in view.
    
    Returns:
        projected_coords: Nx2 array of (u, v) coordinates
    """
    projected, depths = calib.project_velo_to_image(car_points_3d)
    
    # Filter to points in front of camera and within image
    valid = (
        (depths > 0) &
        (projected[:, 0] >= 0) &
        (projected[:, 0] < img_w) &
        (projected[:, 1] >= 0) &
        (projected[:, 1] < img_h)
    )
    
    return projected[valid]


if __name__ == "__main__":
    # Load everything
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib_gt = Calibration("dataset/2011_09_26")  # Ground truth
    img_segmenter = ImageSegmenter()
    
    # Load frame
    frame_idx = 0
    image = loader.load_image(frame_idx)
    points = loader.load_velodyne(frame_idx)
    img_h, img_w = image.shape[:2]
    
    # Get image segmentation
    _, car_mask = img_segmenter.segment(image)
    car_pixel_coords = get_car_pixel_coordinates(car_mask)
    print(f"Car pixels: {len(car_pixel_coords)}")
    
    # STEP 1: Identify car points using GROUND TRUTH (done once!)
    car_indices, car_points_3d = identify_car_points(
        points, car_mask, calib_gt, img_h, img_w
    )
    print(f"Identified {len(car_points_3d)} car points in 3D")
    
    # STEP 2: Test with CORRECT calibration
    car_projected_correct = project_car_points(car_points_3d, calib_gt, img_h, img_w)
    
    loss_fn = BiDirectionalLoss(downsample_ratio=0.02)
    total_correct, p2i_correct, i2p_correct = loss_fn.compute_loss(
        car_projected_correct, car_pixel_coords
    )
    
    print(f"\n--- Loss with CORRECT calibration ---")
    print(f"Projected points: {len(car_projected_correct)}")
    print(f"Point-to-Pixel Loss: {p2i_correct:.4f}")
    print(f"Pixel-to-Point Loss: {i2p_correct:.4f}")
    print(f"Total Loss: {total_correct:.4f}")
    
    # STEP 3: Test with WRONG calibration
    print("\n" + "="*50)
    print("Testing with MISCALIBRATED parameters...")
    print("="*50)
    
    from visualize_miscalib import apply_calibration_noise
    
    calib_noisy = apply_calibration_noise(
        calib_gt,
        rotation_noise_deg=5.0,
        translation_noise_m=0.05
    )
    
    # Project the SAME car points with noisy calibration
    car_projected_noisy = project_car_points(car_points_3d, calib_noisy, img_h, img_w)
    
    total_noisy, p2i_noisy, i2p_noisy = loss_fn.compute_loss(
        car_projected_noisy, car_pixel_coords
    )
    
    print(f"\n--- Loss with WRONG calibration ---")
    print(f"Projected points: {len(car_projected_noisy)}")
    print(f"Point-to-Pixel Loss: {p2i_noisy:.4f}")
    print(f"Pixel-to-Point Loss: {i2p_noisy:.4f}")
    print(f"Total Loss: {total_noisy:.4f}")
    
    # Compare
    print(f"\n--- COMPARISON ---")
    print(f"Correct calibration loss: {total_correct:.4f}")
    print(f"Wrong calibration loss:   {total_noisy:.4f}")
    if total_noisy > total_correct:
        print(f"✓ Loss INCREASED by {total_noisy/total_correct:.2f}x (expected!)")
    else:
        print(f"✗ Loss decreased - something is still wrong")