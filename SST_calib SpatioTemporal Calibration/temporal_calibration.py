# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import KDTree

# from data_loader import KITTIDataLoader
# from calibration import Calibration
# from segmentation import ImageSegmenter
# from loss_function import get_car_pixel_coordinates
# from test_loss_c import identify_car_points


# class FinalTemporalCalibrator:
#     """
#     Final corrected temporal calibrator.
    
#     Key fix: Measure baseline offset first, then subtract it.
#     """
    
#     def __init__(self, calib, car_pixel_coords, img_shape):
#         self.calib = calib
#         self.car_pixel_coords = car_pixel_coords
#         self.img_h, self.img_w = img_shape
        
#         self.pixel_tree = KDTree(car_pixel_coords)
        
#         num_samples = max(1, int(len(car_pixel_coords) * 0.02))
#         np.random.seed(42)
#         sample_indices = np.random.choice(len(car_pixel_coords), num_samples, replace=False)
#         self.sampled_pixels = car_pixel_coords[sample_indices]
    
#     def project_points(self, points):
#         """Project points to image"""
#         projected, depths = self.calib.project_velo_to_image(points)
        
#         valid = (
#             (depths > 0) &
#             (projected[:, 0] >= 0) &
#             (projected[:, 0] < self.img_w) &
#             (projected[:, 1] >= 0) &
#             (projected[:, 1] < self.img_h)
#         )
        
#         return projected[valid], valid
    
#     def compute_loss(self, projected):
#         """Compute alignment loss"""
#         if len(projected) < 10:
#             return 1e10
        
#         distances_p2i, _ = self.pixel_tree.query(projected)
#         loss_p2i = np.mean(distances_p2i ** 2)
        
#         point_tree = KDTree(projected)
#         distances_i2p, _ = point_tree.query(self.sampled_pixels)
#         loss_i2p = np.mean(distances_i2p ** 2)
        
#         np_ni_ratio = len(projected) / len(self.sampled_pixels)
#         total_loss = loss_p2i + 5.0 * np_ni_ratio * loss_i2p
        
#         return total_loss
    
#     def find_optimal_delay(self, car_points, velocity_lidar, search_range=(-0.3, 0.3)):
#         """
#         Find the delay that minimizes alignment loss.
        
#         The compensation formula:
#         compensated_points = original_points + velocity × δ
        
#         We search for δ that gives best alignment.
#         """
#         delays = np.linspace(search_range[0], search_range[1], 121)  # Finer resolution
#         losses = []
        
#         for delta in delays:
#             # Apply compensation
#             compensated = car_points.copy()
#             compensated[:, :3] = car_points[:, :3] + velocity_lidar * delta
            
#             projected, _ = self.project_points(compensated)
#             loss = self.compute_loss(projected)
#             losses.append(loss)
        
#         losses = np.array(losses)
#         best_idx = np.argmin(losses)
        
#         return delays[best_idx], losses[best_idx], delays, losses


# def run_final_temporal_test():
#     """
#     Final temporal calibration test with baseline correction.
#     """
#     print("="*70)
#     print("FINAL TEMPORAL CALIBRATION TEST")
#     print("="*70)
    
#     # Load data
#     loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
#     calib = Calibration("dataset/2011_09_26")
#     img_segmenter = ImageSegmenter()
    
#     frame_idx = 50
    
#     points = loader.load_velodyne(frame_idx)
#     image = loader.load_image(frame_idx)
#     img_h, img_w = image.shape[:2]
    
#     _, car_mask = img_segmenter.segment(image)
#     car_pixel_coords = get_car_pixel_coordinates(car_mask)
#     _, car_points_3d = identify_car_points(points, car_mask, calib, img_h, img_w)
    
#     print(f"Car pixels: {len(car_pixel_coords)}, Car points: {len(car_points_3d)}")
    
#     # Ego velocity in LIDAR frame (X=forward)
#     velocity_lidar = np.array([5.0, 0.0, 0.0])
#     print(f"Simulated velocity: {velocity_lidar} m/s")
    
#     calibrator = FinalTemporalCalibrator(
#         calib=calib,
#         car_pixel_coords=car_pixel_coords,
#         img_shape=(img_h, img_w)
#     )
    
#     # STEP 1: Find baseline offset with NO synthetic delay
#     print("\n" + "-"*50)
#     print("STEP 1: Finding baseline offset (no delay applied)")
#     print("-"*50)
    
#     baseline_delta, baseline_loss, _, _ = calibrator.find_optimal_delay(
#         car_points_3d, velocity_lidar
#     )
#     print(f"Baseline optimal delta: {baseline_delta * 1000:.1f}ms")
#     print(f"This offset will be subtracted from all estimates")
    
#     # STEP 2: Test with synthetic delays
#     print("\n" + "-"*50)
#     print("STEP 2: Testing with synthetic delays")
#     print("-"*50)
    
#     test_delays_ms = [50, 100, 150, 200]
#     results = []
    
#     for true_delay_ms in test_delays_ms:
#         true_delay_sec = true_delay_ms / 1000.0
        
#         # Apply synthetic delay by shifting points
#         # When LIDAR is behind camera by δ, points appear shifted by -v×δ
#         shifted_points = car_points_3d.copy()
#         shifted_points[:, :3] = car_points_3d[:, :3] - velocity_lidar * true_delay_sec
        
#         # Find optimal compensation
#         raw_estimate, loss, delays, losses = calibrator.find_optimal_delay(
#             shifted_points, velocity_lidar
#         )
        
#         # Subtract baseline to get corrected estimate
#         corrected_estimate = raw_estimate - baseline_delta
#         error_ms = abs(true_delay_sec - corrected_estimate) * 1000
        
#         print(f"\nTrue: {true_delay_ms}ms, Raw estimate: {raw_estimate*1000:.1f}ms, "
#               f"Corrected: {corrected_estimate*1000:.1f}ms, Error: {error_ms:.1f}ms")
        
#         results.append({
#             'true_ms': true_delay_ms,
#             'raw_ms': raw_estimate * 1000,
#             'corrected_ms': corrected_estimate * 1000,
#             'error_ms': error_ms,
#             'delays': delays,
#             'losses': losses
#         })
    
#     # Visualization
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     axes = axes.flatten()
    
#     for i, result in enumerate(results):
#         ax = axes[i]
#         ax.plot(result['delays'] * 1000, result['losses'], 'b-', linewidth=2)
#         ax.axvline(x=result['true_ms'] + baseline_delta*1000, color='orange', linestyle='-.',
#                    linewidth=2, label=f"True+baseline: {result['true_ms'] + baseline_delta*1000:.0f}ms")
#         ax.axvline(x=result['raw_ms'], color='green', linestyle=':',
#                    linewidth=2, label=f"Raw est: {result['raw_ms']:.0f}ms")
#         ax.set_xlabel('Time delay (ms)')
#         ax.set_ylabel('Loss')
#         ax.set_title(f"True={result['true_ms']}ms → Corrected={result['corrected_ms']:.0f}ms (Err={result['error_ms']:.0f}ms)")
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)
    
#     plt.suptitle(f'Temporal Calibration (baseline={baseline_delta*1000:.0f}ms subtracted)', fontsize=14)
#     plt.tight_layout()
#     plt.savefig('temporal_calibration_final.png', dpi=150)
#     plt.show()
    
#     # Summary
#     print("\n" + "="*70)
#     print("FINAL SUMMARY")
#     print("="*70)
#     print(f"Baseline offset: {baseline_delta * 1000:.1f}ms (subtracted from all estimates)\n")
#     print(f"{'True Delay':<12} {'Corrected Est':<15} {'Error':<10}")
#     print("-"*40)
#     for r in results:
#         print(f"{r['true_ms']}ms          {r['corrected_ms']:.1f}ms            {r['error_ms']:.1f}ms")
    
#     avg_error = np.mean([r['error_ms'] for r in results])
#     print("-"*40)
#     print(f"Average error: {avg_error:.1f}ms")
#     print(f"Paper reports: 3.4ms average error")
    
#     if avg_error < 15:
#         print("\n✓ Temporal calibration is working well!")
#     elif avg_error < 30:
#         print("\n~ Temporal calibration is working reasonably.")
#     else:
#         print("\n✗ There may still be issues.")
    
#     return results, baseline_delta


# if __name__ == "__main__":
#     run_final_temporal_test()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from data_loader import KITTIDataLoader
from calibration import Calibration
from segmentation import ImageSegmenter
from loss_function import get_car_pixel_coordinates
from test_loss_correct import identify_car_points
from visual_odometry import VisualOdometry # Import VO

class CorrectedTemporalCalibrator:
    """
    Temporal calibrator that uses Visual Odometry for velocity estimation
    and performs grid search for time delay delta.
    """
    
    def __init__(self, calib, car_pixel_coords, img_shape):
        self.calib = calib
        self.car_pixel_coords = car_pixel_coords
        self.img_h, self.img_w = img_shape
        
        # Pre-build KDTree for fast loss computation
        self.pixel_tree = KDTree(car_pixel_coords)
        
        # Downsample for i2p loss (Equation 4)
        num_samples = max(1, int(len(car_pixel_coords) * 0.02))
        np.random.seed(42)
        sample_indices = np.random.choice(len(car_pixel_coords), num_samples, replace=False)
        self.sampled_pixels = car_pixel_coords[sample_indices]
    
    def project_compensated(self, points_3d, velocity_cam, delta_t):
        """
        Implements Equation (6) from SST-Calib paper.
        Projects points with temporal compensation applied in CAMERA frame.
        
        P' = K * [ R*P_lidar + t + v_cam * delta ]
        """
        # 1. Transform LiDAR points to Camera Frame (Standard Extrinsics)
        # P_cam = R * P_lidar + t
        # Note: project_velo_to_image usually handles the whole chain. 
        # We break it down to insert the time term.
        
        # Homogeneous coordinates
        N = len(points_3d)
        pts_3d_hom = np.hstack([points_3d[:, :3], np.ones((N, 1))])
        
        # Apply Extrinsics (LiDAR -> Camera)
        pts_cam = (self.calib.Tr_velo_to_cam @ pts_3d_hom.T).T # Nx4
        pts_cam = pts_cam[:, :3] # Nx3 (x,y,z) in cam frame
        
        # 2. Apply Temporal Compensation (Equation 6)
        # P_cam_new = P_cam + v_cam * delta
        pts_cam_compensated = pts_cam + (velocity_cam * delta_t)
        
        # 3. Apply Rectification (R0_rect) - Specific to KITTI
        pts_cam_comp_hom = np.hstack([pts_cam_compensated, np.ones((N, 1))])
        pts_rect = (self.calib.R0_rect_4x4 @ pts_cam_comp_hom.T).T
        
        # 4. Project to Image Plane (Intrinsics P2)
        pts_2d_hom = (self.calib.P2 @ pts_rect.T).T
        
        # 5. Normalize
        depths = pts_2d_hom[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = pts_2d_hom[:, 0] / depths
            v = pts_2d_hom[:, 1] / depths
        
        projected = np.stack([u, v], axis=1)
        
        # Filter valid
        valid = (
            (depths > 0) &
            (projected[:, 0] >= 0) &
            (projected[:, 0] < self.img_w) &
            (projected[:, 1] >= 0) &
            (projected[:, 1] < self.img_h)
        )
        
        return projected[valid]

    def compute_loss(self, projected):
        """Standard Bi-Directional Loss"""
        if len(projected) < 10:
            return 1e10
            
        # Point-to-Pixel
        distances_p2i, _ = self.pixel_tree.query(projected)
        loss_p2i = np.mean(distances_p2i ** 2)
        
        # Pixel-to-Point
        point_tree = KDTree(projected)
        distances_i2p, _ = point_tree.query(self.sampled_pixels)
        loss_i2p = np.mean(distances_i2p ** 2)
        
        # Weighted Sum (Using 5.0 as in your code, or dynamically as in paper)
        np_ni_ratio = len(projected) / len(self.sampled_pixels)
        total_loss = loss_p2i + 5.0 * np_ni_ratio * loss_i2p
        
        return total_loss

    def grid_search_delay(self, car_points_3d, velocity_cam, search_range=(-0.2, 0.2), steps=41):
        """
        Grid search for optimal delta_t using Camera-frame velocity.
        """
        delays = np.linspace(search_range[0], search_range[1], steps)
        losses = []
        
        for delta in delays:
            proj = self.project_compensated(car_points_3d, velocity_cam, delta)
            loss = self.compute_loss(proj)
            losses.append(loss)
            
        best_idx = np.argmin(losses)
        return delays[best_idx], losses[best_idx], delays, losses

def run_corrected_temporal_test():
    print("="*70)
    print("CORRECTED TEMPORAL CALIBRATION TEST (With Visual Odometry)")
    print("="*70)
    
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib = Calibration("dataset/2011_09_26")
    img_segmenter = ImageSegmenter()
    
    # Use Visual Odometry class
    K = calib.P2[:3, :3]
    vo = VisualOdometry(K)
    
    # Select a frame where the car is actually moving!
    frame_idx = 50 
    
    # Load Frame T and T-1 for VO
    image_curr = loader.load_image(frame_idx)
    image_prev = loader.load_image(frame_idx - 1)
    points = loader.load_velodyne(frame_idx)
    img_h, img_w = image_curr.shape[:2]
    
    # 1. Estimate REAL velocity using VO
    # Note: estimate_velocity_between_frames returns v in Camera Frame
    velocity_cam, valid_vo = vo.estimate_velocity_between_frames(image_prev, image_curr)
    
    if not valid_vo or np.linalg.norm(velocity_cam) < 0.5:
        print(f"WARNING: Velocity too low ({np.linalg.norm(velocity_cam):.2f} m/s) or VO failed.")
        print("Temporal calibration requires a moving vehicle!")
        # For simulation purposes only, we might enforce a velocity if VO fails 
        # (but in real life we would skip this frame)
        # velocity_cam = np.array([0, 0, 5.0]) # Forward in Camera frame (Z-axis)
    
    print(f"Estimated Ego-Velocity (Cam Frame): {velocity_cam} m/s")
    
    # 2. Segment and Identify Car Points
    _, car_mask = img_segmenter.segment(image_curr)
    car_pixel_coords = get_car_pixel_coordinates(car_mask)
    _, car_points_3d = identify_car_points(points, car_mask, calib, img_h, img_w)
    
    calibrator = CorrectedTemporalCalibrator(calib, car_pixel_coords, (img_h, img_w))
    
    # 3. Synthetic Test Loop
    test_delays_ms = [0, 50, 100, -50]
    
    for true_delay_ms in test_delays_ms:
        true_delay_sec = true_delay_ms / 1000.0
        
        # CREATE Synthetic Delay:
        # If LiDAR is DELAYED by delta (delta > 0), the points are captured later.
        # The car has moved forward. The world moves backward relative to sensor.
        # P_delayed = P_true - v * delta
        # We need to shift points in the direction of motion to simulate delay.
        # Since we only have velocity in Cam frame, we project v_cam -> v_lidar
        
        # Transform v_cam back to v_lidar for simulation shifting
        R_cam_to_velo = calib.R_velo.T 
        # v_lidar = R^T * v_cam (Approximate, ignoring rectification for simulation)
        v_lidar_sim = R_cam_to_velo @ velocity_cam
        
        shifted_points = car_points_3d.copy()
        # Apply synthetic shift
        shifted_points[:, :3] = car_points_3d[:, :3] - (v_lidar_sim * true_delay_sec)
        
        # RECOVER Delay
        est_delay, loss, _, _ = calibrator.grid_search_delay(
            shifted_points, velocity_cam, search_range=(-0.2, 0.2)
        )
        
        print(f"True: {true_delay_ms:>4}ms | Est: {est_delay*1000:>5.1f}ms | Err: {abs(true_delay_ms - est_delay*1000):.1f}ms")

if __name__ == "__main__":
    run_corrected_temporal_test()