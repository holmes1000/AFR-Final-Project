import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import copy
import time

class JointCalibrationOptimizer:
    """
    Optimizer for Simultaneous Spatial-Temporal Parameter Calibration (SST-Calib).
    
    Implements the joint optimization of 7 parameters:
    - 6 Geometry: [roll, pitch, yaw, tx, ty, tz]
    - 1 Temporal: [time_delay]
    
    Paper Reference: Equations (6), (8), (9)
    """
    
    def __init__(self, calib_init, car_points_3d, car_pixel_coords, img_shape,
                 velocity_camera_frame=None, 
                 lambda_trans=1.0, lambda_rot=100.0):
        """
        Args:
            velocity_camera_frame: (3,) array estimating ego-velocity in camera frame (v_k)
        """
        self.calib_init = calib_init
        self.car_points_3d = car_points_3d
        self.car_pixel_coords = car_pixel_coords
        self.img_h, self.img_w = img_shape
        
        # Velocity v_k from Visual Odometry (Eq 6). 
        # If None (Static Calibration), assumes zero velocity.
        if velocity_camera_frame is None:
            self.velocity = np.zeros(3)
            self.is_static = True
        else:
            self.velocity = velocity_camera_frame
            self.is_static = False

        # Hyperparameters from paper (approximate, tuned in original code)
        self.lambda_trans = lambda_trans
        self.lambda_rot = lambda_rot
        
        # Initial guesses
        self.R_init = calib_init.Tr_velo_to_cam[:3, :3].copy()
        self.T_init = calib_init.Tr_velo_to_cam[:3, 2].copy()
        
        # Extract projection matrices for manual projection (speedup + motion comp)
        self.P2 = calib_init.P2
        self.R0_rect = calib_init.R0_rect_4x4
        
        # Pre-sample pixels for i2p loss (Equation 4)
        # Paper suggests downsampling to ~2% for efficiency
        num_samples = max(1, int(len(car_pixel_coords) * 0.02))
        np.random.seed(42)
        sample_indices = np.random.choice(len(car_pixel_coords), num_samples, replace=False)
        self.sampled_pixels = car_pixel_coords[sample_indices]
        
        self.pixel_tree = KDTree(car_pixel_coords)
        
        # Optimization state
        self.iteration = 0
        self.current_weight_i2p = 1.0 # w_l
        self.best_loss = float('inf')
        self.best_params = None

    def grid_search_delay(self, search_range=(-0.2, 0.2), steps=21):
        """
        Performs a coarse search for time delay to initialize the optimizer.
        Holds R and T constant, varies only delta.
        """
        print(f"\nRunning Grid Search for Delay ({search_range[0]}s to {search_range[1]}s)...")
        
        delays = np.linspace(search_range[0], search_range[1], steps)
        best_loss = float('inf')
        best_delta = 0.0
        
        # Cache current params
        R_current = self.R_init
        T_current = self.T_init
        
        for delta in delays:
            # Calculate motion shift for this delta
            motion_shift = self.velocity * delta
            
            # Project
            projected = self.project_points_joint(R_current, T_current, motion_shift)
            
            if len(projected) < 10:
                continue
                
            # Compute Simple Loss (Point-to-Pixel only for speed)
            distances_p2i, _ = self.pixel_tree.query(projected)
            loss = np.mean(distances_p2i ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_delta = delta
                
        print(f"  > Grid Search Best Delta: {best_delta*1000:.1f}ms (Loss: {best_loss:.2f})")
        return best_delta

    def unpack_params(self, params):
        """Unpack flat parameter array into components"""
        # params: [roll, pitch, yaw, tx, ty, tz, delta_t]
        d_roll, d_pitch, d_yaw = params[0:3]
        d_tx, d_ty, d_tz = params[3:6]
        
        # If static, force time delay to 0
        if self.is_static or len(params) < 7:
            delta_t = 0.0
        else:
            delta_t = params[6]
            
        return (d_roll, d_pitch, d_yaw), (d_tx, d_ty, d_tz), delta_t

    def get_transform_and_motion(self, params):
        (rot, trans, delta_t) = self.unpack_params(params)
        
        # 1. Geometry: Calculate new Extrinsics (R, t)
        # R_new = R_delta * R_init
        R_delta = Rotation.from_euler('xyz', rot, degrees=True).as_matrix()
        R_new = R_delta @ self.R_init
        
        # T_new = T_init + T_delta
        T_new = self.T_init + np.array(trans)
        
        # 2. Temporal: Calculate Motion Shift (Eq 6)
        # Shift = v_k * delta_t (in Camera Frame)
        motion_shift = self.velocity * delta_t
        
        return R_new, T_new, motion_shift, R_delta, np.array(trans)

    def project_points_joint(self, R, T, motion_shift):
        """
        Custom projection pipeline implementing Equation (6):
        p_img = K * [ R*p_lidar + t + v*delta ]
        """
        # 1. Transform LiDAR to Camera Frame
        # P_cam = R * P_lidar + T
        pts_3d = self.car_points_3d[:, :3] # Nx3
        pts_cam = (R @ pts_3d.T).T + T # Nx3
        
        R_rect_3x3 = self.R0_rect[:3, :3]
        pts_cam_rect = (R_rect_3x3 @ pts_cam.T).T
        # 2. Apply Motion Compensation (Temporal Term)
        # P_cam' = P_cam + v * delta
        pts_cam_compensated = pts_cam_rect + motion_shift
        
        # 3. Apply Rectification (specific to KITTI)
        # P_rect = R0_rect * P_cam'
        # Homogeneous coord for matrix mult
        N = len(pts_cam_compensated)
        pts_hom_rect = np.hstack((pts_cam_compensated, np.ones((N, 1))))
        # pts_rect = (self.R0_rect @ pts_hom.T).T # Nx4
        
        # 4. Project to Image Plane
        # P_uv = P2 * P_rect
        pts_2d_hom = (self.P2 @ pts_hom_rect.T).T # Nx3
        
        # 5. Normalize (u/z, v/z)
        depths = pts_2d_hom[:, 2]
        # Avoid division by zero
        depths[depths == 0] = 1e-5
        
        u = pts_2d_hom[:, 0] / depths
        v = pts_2d_hom[:, 1] / depths
        
        projected = np.stack((u, v), axis=1)
        
        # 6. Filter valid points
        valid = (
            (depths > 0) &
            (projected[:, 0] >= 0) &
            (projected[:, 0] < self.img_w) &
            (projected[:, 1] >= 0) &
            (projected[:, 1] < self.img_h)
        )
        
        return projected[valid]

    def compute_regularization(self, R_delta, T_delta):
        """Equation (9): Keep estimation close to initial guess"""
        # Convert translation to cm for reasonable scaling vs rotation
        T_delta_cm = T_delta# * 100
        trans_reg = self.lambda_trans * np.sum(T_delta_cm ** 2)
        rot_reg = self.lambda_rot * np.sum((R_delta - np.eye(3)) ** 2)
        return trans_reg + rot_reg

    def objective(self, params):
        R_new, T_new, motion_shift, R_delta, T_delta = self.get_transform_and_motion(params)
        
        # Project with temporal compensation
        projected = self.project_points_joint(R_new, T_new, motion_shift)
        
        if len(projected) < 10:
            return 1e10
        
        # --- Bi-Directional Loss (Eq 3, 4, 5) ---
        
        # 1. Point-to-Pixel (p2i)
        distances_p2i, _ = self.pixel_tree.query(projected)
        loss_p2i = np.mean(distances_p2i ** 2)
        
        # 2. Pixel-to-Point (i2p)
        # Query nearest projected point for each sampled pixel
        point_tree = KDTree(projected)
        distances_i2p, _ = point_tree.query(self.sampled_pixels)
        loss_i2p = np.mean(distances_i2p ** 2)
        
        # Combined Loss
        np_ni_ratio = len(projected) / len(self.sampled_pixels)
        alignment_loss = loss_p2i + self.current_weight_i2p * np_ni_ratio * loss_i2p
        
        # Regularization (Eq 8)
        reg_loss = self.compute_regularization(R_delta, T_delta)
        
        total_loss = alignment_loss + reg_loss
        
        # Track best
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_params = params.copy()
            
        return total_loss

    def optimize_staged(self, initial_guess=None):
        """
        Runs the multi-stage optimization process described in the paper.
        Weights (w_l) decay: 20 -> 1 -> 0.02
        """
        print(f"\nStarting {'Static' if self.is_static else 'Joint Spatial-Temporal'} Optimization...")
        
        # Initialize parameters: 6 for static, 7 for joint
        num_params = 6 if self.is_static else 7
        if initial_guess is not None and len(initial_guess) == num_params:
            current_params = initial_guess
        else:
            current_params = np.zeros(num_params)
        
        start_time = time.time()
        bounds = [
            (-10, 10), (-10, 10), (-10, 10),  # Rot (deg)
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), # Trans (m)
            (-0.3, 0.3)  # Time (s) <--- KEEPS IT SAFE
        ]
        bounds = bounds[:num_params]
        # Define stages (weight_i2p values)
        stages = [
            (20.0, 5, 'Powell'), # Coarse alignment
            (1.0,  5, 'Powell'), # Balanced
            (0.02, 5, 'Powell')  # Fine refinement
        ]
        
        for i, (w_l, iters, method) in enumerate(stages):
            print(f"\n--- Stage {i+1}: w_l = {w_l} ---")
            self.current_weight_i2p = w_l
            self.best_loss = float('inf') # Reset best loss for new stage
            
            res = minimize(self.objective, current_params, method=method, bounds=bounds,
                           options={'maxiter': iters, 'disp': False})
            
            # Update params for next stage
            current_params = self.best_params if self.best_params is not None else res.x
            
            (rot, trans, dt) = self.unpack_params(current_params)
            print(f"  Best Loss: {self.best_loss:.2f}")
            print(f"  Current Delta: Rot={rot}, Trans={trans}, Time={dt*1000:.1f}ms")

        elapsed = time.time() - start_time
        print(f"\nOptimization Complete ({elapsed:.2f}s)")
        
        # Final Result extraction
        final_params = self.best_params
        R_final, T_final, motion_shift, _, _ = self.get_transform_and_motion(final_params)
        
        # Construct 4x4 Matrix
        final_transform = np.eye(4)
        final_transform[:3, :3] = R_final
        final_transform[:3, 3] = T_final
        
        _, _, final_delay = self.unpack_params(final_params)
        
        return final_transform, final_delay