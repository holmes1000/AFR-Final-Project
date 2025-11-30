import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy
import time

# Project imports
from data_loader import KITTIDataLoader
from calibration import Calibration
from segmentation import ImageSegmenter
from visual_odometry import VisualOdometry
from loss_function import get_car_pixel_coordinates
from test_loss_correct import identify_car_points
from visualize_miscalib import apply_calibration_noise
from optimizer import JointCalibrationOptimizer

def run_online_simulation():
    print("="*70)
    print("SST-CALIB: ONLINE CALIBRATION LOOP (154 Frames) - AGGRESSIVE TRACKING")
    print("="*70)
    
    # 1. SETUP
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib_gt = Calibration("dataset/2011_09_26")
    img_segmenter = ImageSegmenter()
    
    K = calib_gt.P2[:3, :3]
    vo = VisualOdometry(K)
    img_h, img_w = 375, 1242
    
    # --- SIMULATION SETTINGS ---
    # Simplified Test: Single constant delay to verify tracking stability
    TRUE_DELAY_SEC = 0.100       
    SMOOTHING_FACTOR = 0.5       # Higher smoothing to respond faster
    MIN_VELOCITY = 2.0           
    
    # 2. INITIALIZATION
    print("\nInitializing with noisy calibration...")
    np.random.seed(42)
    current_calib = apply_calibration_noise(calib_gt, rotation_noise_deg=5.0, translation_noise_m=0.10)
    current_params = np.zeros(7) 
    
    history_rot_err = []
    history_trans_err = []
    history_time_err = []
    frames_processed = []
    
    # 3. MAIN LOOP
    print("\nStarting Online Loop...")
    start_time = time.time()
    
    # Range 5 to 154
    for frame_idx in range(5, 50):
        print(f"\n--- Frame {frame_idx} ---")
        
        # A. Load Data
        try:
            image_curr = loader.load_image(frame_idx)
            image_prev = loader.load_image(frame_idx - 1)
            points_curr = loader.load_velodyne(frame_idx)
        except IndexError:
            print("End of stream.")
            break

        # B. Estimate Velocity
        velocity_cam, valid_vo = vo.estimate_velocity_between_frames(image_prev, image_curr)
        speed = np.linalg.norm(velocity_cam)
        
        if not valid_vo or speed < MIN_VELOCITY:
            print(f"Skipping: Low speed ({speed:.1f} m/s).")
            # Log current state (drift) but don't update
            frames_processed.append(frame_idx)
            # (Append logic omitted for brevity, usually you just append last known error)
            continue
            
        print(f"Velocity: {speed:.1f} m/s")

        # C. Detect Cars (Ghost Car Logic: Unshifted First)
        _, car_mask = img_segmenter.segment(image_curr)
        car_pixels = get_car_pixel_coordinates(car_mask)
        _, car_points_sync = identify_car_points(points_curr, car_mask, calib_gt, img_h, img_w)
        
        # [TUNING] Filter out frames with too little signal
        if len(car_points_sync) < 200: # Increased threshold for stability
            print(f"Skipping: Too few car points ({len(car_points_sync)}).")
            continue

        # D. Apply Synthetic Delay
        # Shift the identified points by the TRUE delay
        v_cam_unrect = calib_gt.R0_rect_4x4[:3, :3].T @ velocity_cam
        R_cam_to_velo = calib_gt.Tr_velo_to_cam[:3, :3].T
        v_lidar = R_cam_to_velo @ v_cam_unrect
        
        car_points = car_points_sync.copy()
        car_points[:, :3] = car_points_sync[:, :3] - (v_lidar * TRUE_DELAY_SEC)
        
        # E. Online Optimization Setup
        optimizer = JointCalibrationOptimizer(
            calib_init=current_calib,
            car_points_3d=car_points,
            car_pixel_coords=car_pixels,
            img_shape=(img_h, img_w),
            velocity_camera_frame=velocity_cam,
            # [TUNING] Increase translation regularization to lock geometry and isolate time
            lambda_trans=0.01, 
            lambda_rot=100.0
        )
        
        # F. Bootstrap Initialization (Force on Frame 5)
# F. Bootstrap Initialization (Run Once at Frame 5)
        if frame_idx == 5:
            print("  > BOOTSTRAP INITIATED...")
            
            # STEP 1: Rough Spatial Alignment (Fix the 5-degree error first)
            # We assume Delay=0 for a second to fix Rotation/Translation
            print("    1. Fixing Geometry (Spatial-Only)...")
            # Create a static optimizer (no velocity) just to lock geometry
            opt_spatial = JointCalibrationOptimizer(
                calib_init=current_calib,
                car_points_3d=car_points,
                car_pixel_coords=car_pixels,
                img_shape=(img_h, img_w),
                velocity_camera_frame=None, # Force Static Mode
                lambda_trans=0.1, lambda_rot=100.0
            )
            # Optimize only 6 params (Geometry)
            T_static, _ = opt_spatial.optimize_staged()
            
            # Update our current best guess with this better geometry
            current_calib.Tr_velo_to_cam = T_static
            optimizer = JointCalibrationOptimizer(
                calib_init=current_calib, # Now passing the FIXED calibration
                car_points_3d=car_points,
                car_pixel_coords=car_pixels,
                img_shape=(img_h, img_w),
                velocity_camera_frame=velocity_cam,
                lambda_trans=100.0,
                lambda_rot=100.0
            )
            # STEP 2: Now run Grid Search for Time
            # (Now that geometry is close, the Grid Search will actually find the car)
            print("    2. Searching for Time Delay...")
            
            # Re-initialize joint optimizer with the NEW geometry
            optimizer.calib_init = current_calib 
            
            init_delay = optimizer.grid_search_delay(search_range=(-0.4, 0.4), steps=81)
            print(f"    > Found Best Delay: {init_delay*1000:.1f}ms")
            
            current_params[6] = init_delay
        
        # G. Optimization Step (Aggressive Tracking)
        from scipy.optimize import minimize
        
        # [TUNING] Increased maxiter from 5 to 50 to force convergence
        res = minimize(
            optimizer.objective, 
            current_params, 
            method='Powell',
            options={'maxiter': 50, 'disp': False} 
        )
        
        new_params = res.x
        R_new, T_new, _, _, _ = optimizer.get_transform_and_motion(new_params)
        
        # 2. Reconstruct the 4x4 Matrix manually
        T_new_matrix = np.eye(4)
        T_new_matrix[:3, :3] = R_new
        T_new_matrix[:3, 3] = T_new        
        delay_new = new_params[6]
        delay_new = np.clip(delay_new, -0.2, 0.2)
        
        # H. Smoothing
        current_calib.Tr_velo_to_cam = T_new_matrix 
        prev_delay = current_params[6]
        smoothed_delay = (1 - SMOOTHING_FACTOR) * prev_delay + SMOOTHING_FACTOR * delay_new
        
        current_params = np.zeros(7)
        current_params[6] = smoothed_delay
        
        
        # I. Calculate Errors
        R_curr = current_calib.Tr_velo_to_cam[:3, :3]
        T_curr = current_calib.Tr_velo_to_cam[:3, 3] # Fixed index from 2 to 3 (vector)
        R_gt = calib_gt.Tr_velo_to_cam[:3, :3]
        T_gt = calib_gt.Tr_velo_to_cam[:3, 3] # Fixed index from 2 to 3
        
        rot_err = np.mean(np.abs(Rotation.from_matrix(R_curr @ R_gt.T).as_euler('xyz', degrees=True)))
        trans_err = np.mean(np.abs((T_curr - T_gt) * 100))
        time_err = abs(TRUE_DELAY_SEC - smoothed_delay) * 1000
        
        print(f"Errors -> Rot: {rot_err:.2f}°, Trans: {trans_err:.2f}cm, Time: {time_err:.1f}ms")
        
        history_rot_err.append(rot_err)
        history_trans_err.append(trans_err)
        history_time_err.append(time_err)
        frames_processed.append(frame_idx)

    total_time = time.time() - start_time
    print(f"\nProcessed {len(frames_processed)} frames in {total_time:.1f}s")
    
    # 4. VISUALIZE
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(frames_processed, history_rot_err, 'r-')
    plt.title("Rotation Error (deg)")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(frames_processed, history_trans_err, 'g-')
    plt.title("Translation Error (cm)")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(frames_processed, history_time_err, 'b-')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title("Time Delay Error (ms)")
    plt.xlabel("Frame Index")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('online_calibration_results.png')
    print("Results saved to 'online_calibration_results.png'")
    plt.show()

if __name__ == "__main__":
    run_online_simulation()