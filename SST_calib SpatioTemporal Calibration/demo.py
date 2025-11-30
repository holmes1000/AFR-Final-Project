import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy

# Project imports
from data_loader import KITTIDataLoader
from calibration import Calibration
from segmentation import ImageSegmenter
from visual_odometry import VisualOdometry
from loss_function import get_car_pixel_coordinates
from test_loss_correct import identify_car_points
from visualize_miscalib import apply_calibration_noise

# UPDATED: Import the new Joint Optimizer
from optimizer import JointCalibrationOptimizer

def run_full_sst_calib_demo():
    """
    SST-Calib Demonstration:
    1. Static Spatial Calibration (Initialization)
    2. Joint Spatial-Temporal Calibration (Refinement)
    """
    print("="*70)
    print("SST-CALIB: Simultaneous Spatial-Temporal Calibration Demo")
    print("="*70)
    
    # 1. Setup System
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib_gt = Calibration("dataset/2011_09_26")
    img_segmenter = ImageSegmenter()
    
    # Initialize Visual Odometry with Camera Intrinsics
    K = calib_gt.P2[:3, :3]
    vo = VisualOdometry(K)
    
    img_h, img_w = 375, 1242 # Standard KITTI size
    
    #=========================================
    # PHASE 1: STATIC SPATIAL CALIBRATION
    #=========================================
    print("\n" + "="*70)
    print("PHASE 1: STATIC SPATIAL CALIBRATION (Initialization)")
    print("="*70)
    print("Goal: Get a rough geometric estimate from a static frame.")
    
    # Use Frame 0 (Assuming car is stationary or slow)
    frame_static = 0
    image_static = loader.load_image(frame_static)
    points_static = loader.load_velodyne(frame_static)
    
    # A. Segmentation
    print("Segmenting static frame...")
    _, car_mask_static = img_segmenter.segment(image_static)
    car_pixels_static = get_car_pixel_coordinates(car_mask_static)
    
    # B. Identify 3D Car Points (Using GT for labeling - Simulation "Cheat")
    # In a real pipeline, you would use SqueezeSegV3 on the point cloud directly.
    _, car_points_static = identify_car_points(points_static, car_mask_static, calib_gt, img_h, img_w)
    print(f"  > Found {len(car_pixels_static)} car pixels and {len(car_points_static)} car points.")
    
    # C. Create Miscalibration (The "Unknown" State)
    print("\nApplying synthetic miscalibration...")
    np.random.seed(123)
    calib_noisy = apply_calibration_noise(calib_gt, rotation_noise_deg=5.0, translation_noise_m=0.10)
    
    # Calculate Initial Error
    R_gt = calib_gt.Tr_velo_to_cam[:3, :3]
    T_gt = calib_gt.Tr_velo_to_cam[:3, 3]
    
    R_init = calib_noisy.Tr_velo_to_cam[:3, :3]
    T_init = calib_noisy.Tr_velo_to_cam[:3, 3]
    
    err_rot_init = np.mean(np.abs(Rotation.from_matrix(R_init @ R_gt.T).as_euler('xyz', degrees=True)))
    err_trans_init = np.mean(np.abs((T_init - T_gt) * 100))
    print(f"  > Initial Error: AEAD={err_rot_init:.2f}°, ATD={err_trans_init:.2f}cm")
    
    # D. Run Optimization (Static Mode)
    optimizer_static = JointCalibrationOptimizer(
        calib_init=calib_noisy,
        car_points_3d=car_points_static,
        car_pixel_coords=car_pixels_static,
        img_shape=(img_h, img_w),
        velocity_camera_frame=None  # Indicates Static Mode
    )
    
    T_static_result, _ = optimizer_static.optimize_staged()
    
    # Update calibration object with result
    calib_stage1 = copy.deepcopy(calib_noisy)
    calib_stage1.Tr_velo_to_cam = T_static_result
    
    #=========================================
    # PHASE 2: JOINT SPATIAL-TEMPORAL CALIB
    #=========================================
    print("\n" + "="*70)
    print("PHASE 2: JOINT SPATIAL-TEMPORAL CALIBRATION")
    print("="*70)
    
    # Settings for Simulation
    frame_idx = 63
    # CHANGE THIS to test different delays (e.g., 0.135, 0.085)
    true_delay_sec = 0.129 
    
    print(f"Simulating synchronization error:")
    print(f"  > Frame: {frame_idx}")
    print(f"  > True Time Delay: {true_delay_sec*1000:.1f}ms")
    
    # 1. Load Synchronized Data (Frame 50 / Frame 50)
    points_sync = loader.load_velodyne(frame_idx)
    image_sync = loader.load_image(frame_idx)
    image_prev = loader.load_image(frame_idx - 1) # For VO
    
    # 2. Estimate Velocity (Visual Odometry)
    print("\nEstimating Ego-Velocity...")
    # velocity_cam is in Camera Frame (x=right, y=down, z=forward)
    velocity_cam, valid_vo = vo.estimate_velocity_between_frames(image_prev, image_sync)
    
    # Velocity Sanity Check
    speed = np.linalg.norm(velocity_cam)
    if not valid_vo or speed < 1.0:
        print(f"  > WARNING: Speed is too low ({speed:.2f} m/s) for temporal calibration!")
        print("  > Forcing fallback velocity: [0, 0, 8.0]")
        velocity_cam = np.array([0.0, 0.0, 8.0])
    else:
        print(f"  > Velocity (Cam Frame): [{velocity_cam[0]:.2f}, {velocity_cam[1]:.2f}, {velocity_cam[2]:.2f}] m/s")

    # 3. Segment Image
    print("Segmenting image...")
    _, car_mask = img_segmenter.segment(image_sync)
    car_pixels = get_car_pixel_coordinates(car_mask)
    
    # 4. Identify Car Points on SYNCHRONIZED Data (CRITICAL FIX)
    # We find which points are cars *before* we break the alignment with the time shift.
    print("Identifying car points...")
    _, car_points_sync_subset = identify_car_points(points_sync, car_mask, calib_gt, img_h, img_w)
    print(f"  > Found {len(car_points_sync_subset)} valid car points.")

    if len(car_points_sync_subset) < 50:
        print("  > ERROR: Too few car points found! Try a different frame.")
        return
    
    # Convert velocity to LiDAR frame for shifting
    # v_lidar = R_cam_to_velo * v_cam
    R_cam_to_velo = calib_gt.Tr_velo_to_cam[:3, :3].T
    v_lidar = R_cam_to_velo @ velocity_cam
    
    car_points_dyn = car_points_sync_subset.copy()
    car_points_dyn[:, :3] = car_points_sync_subset[:, :3] - (v_lidar * true_delay_sec)
    
    print(f"  > Applied {true_delay_sec*1000:.1f}ms shift to input points.")    

    # 5. Run Joint Optimization
    optimizer_joint = JointCalibrationOptimizer(
        calib_init=calib_stage1,   # Start from Phase 1 result
        car_points_3d=car_points_dyn,
        car_pixel_coords=car_pixels,
        img_shape=(img_h, img_w),
        velocity_camera_frame=velocity_cam,
        lambda_trans=0.01,  # Lowered from 10.0 to fix "stuck" translation
        lambda_rot=100.0
    )
    
    # --- CRITICAL: GRID SEARCH INITIALIZATION ---
    # Because 42ms is far from 135ms, gradient descent might get stuck.
    # We scan to find a good starting point.
    print("\nRunning Grid Search Initialization...")
    init_delay = optimizer_joint.grid_search_delay(search_range=(-0.3, 0.3), steps=61)
    
    # Set initial guess
    # [roll, pitch, yaw, x, y, z, time_delay]
    initial_params = np.zeros(7)
    initial_params[6] = init_delay
    
    # Run Optimizer
    T_final_matrix, estimated_delay = optimizer_joint.optimize_staged(initial_guess=initial_params)
    
    #=========================================
    # RESULTS ANALYSIS
    #=========================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Spatial Metrics
    R_final = T_final_matrix[:3, :3]
    T_final = T_final_matrix[:3, 3]
    
    err_rot_final = np.mean(np.abs(Rotation.from_matrix(R_final @ R_gt.T).as_euler('xyz', degrees=True)))
    err_trans_final = np.mean(np.abs((T_final - T_gt) * 100))
    
    # Temporal Metrics
    estimated_delay_ms = estimated_delay 
    err_temporal = abs(true_delay_sec - estimated_delay_ms)
    
    print(f"SPATIAL CALIBRATION:")
    print(f"  > Rotation Error (AEAD):    {err_rot_init:.2f}° -> {err_rot_final:.2f}°")
    print(f"  > Translation Error (ATD):  {err_trans_init:.2f}cm -> {err_trans_final:.2f}cm")
    
    print(f"\nTEMPORAL CALIBRATION:")
    print(f"  > True Delay:      {true_delay_sec:.3f}s")
    print(f"  > Estimated Delay: {estimated_delay_ms:.3f}s")
    print(f"  > Absolute Error:  {err_temporal:.3f}s")

    #=========================================
    # VISUALIZATION
    #=========================================
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Helper for plotting
    def plot_proj(ax, calib_obj, points, title, color, delay=0.0, velocity=None):
        # 1. Create a temp optimizer instance just to access the projection logic
        # We pass the velocity so it can compute the shift: v * delay
        temp_opt = JointCalibrationOptimizer(
            calib_obj, points, car_pixels, (img_h, img_w), velocity_camera_frame=velocity
        )
        
        # 2. Get Transformation Components
        R = calib_obj.Tr_velo_to_cam[:3, :3]
        T = calib_obj.Tr_velo_to_cam[:3, 3]
        
        # 3. Calculate Motion Shift (The Correction)
        # Shift = v_cam * delta_t
        motion_shift = velocity * delay if velocity is not None else np.zeros(3)
        
        # 4. Project using the Joint Model
        # (Projects 3D points -> Adds Motion Shift -> Rectifies -> Projects to 2D)
        proj = temp_opt.project_points_joint(R, T, motion_shift)
        
        # 5. Plot
        ax.imshow(image_sync) # Background is the synchronized image
        if len(proj) > 0:
            ax.scatter(proj[:, 0], proj[:, 1], c=color, s=2, alpha=0.6)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # --- Plot 1: Before Temporal Calibration ---
    # We use the Spatial result from Phase 1, but Delay = 0.
    # Since the points `car_points_dyn` are synthetically shifted, 
    # they should appear misaligned (sliding off the car) if we don't account for time.
    plot_proj(
        axes[0], 
        calib_stage1, 
        car_points_dyn, 
        f"Input Data (No Correction)\nPoints shifted by {true_delay_sec*1000:.0f}ms", 
        'red', 
        delay=0.0, 
        velocity=velocity_cam
    )
    
    # --- Plot 2: After Joint Calibration ---
    # We use the Final optimized geometry and the Estimated Delay.
    calib_final = copy.deepcopy(calib_gt)
    calib_final.Tr_velo_to_cam = T_final_matrix
    
    plot_proj(
        axes[1], 
        calib_final, 
        car_points_dyn, 
        f"Optimized Result\nEst. Delay: {estimated_delay*1000:.1f}ms\n(Err: {err_temporal:.1f}ms)", 
        'green', 
        delay=estimated_delay, 
        velocity=velocity_cam
    )
    
    # --- Plot 3: Ground Truth ---
    # We use GT geometry and the True Delay.
    # This proves that if we knew the correct time, the points would match perfectly.
    plot_proj(
        axes[2], 
        calib_gt, 
        car_points_dyn, 
        f"Ground Truth\nTrue Delay: {true_delay_sec*1000:.0f}ms", 
        'blue', 
        delay=true_delay_sec, 
        velocity=velocity_cam
    )
    
    plt.tight_layout()
    plt.savefig('sst_calib_joint_results.png', dpi=150)
    print("Visualization saved to 'sst_calib_joint_results.png'")
    plt.show()

if __name__ == "__main__":
    run_full_sst_calib_demo()