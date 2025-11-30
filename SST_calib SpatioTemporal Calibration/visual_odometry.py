import numpy as np
import cv2

class VisualOdometry:
    """
    Estimate ego-vehicle velocity from consecutive camera frames.
    Uses sparse optical flow (Lucas-Kanade) as described in the paper.
    """
    
    def __init__(self, camera_matrix):
        """
        Args:
            camera_matrix: 3x3 intrinsic matrix K
        """
        self.K = camera_matrix
        
        # Feature detector parameters
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def estimate_motion(self, img1, img2):
        """
        Estimate camera motion between two frames.
        
        Args:
            img1: First RGB image (earlier frame)
            img2: Second RGB image (later frame)
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector (unit scale)
            velocity: Estimated velocity direction (unit vector)
            num_inliers: Number of inlier matches
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Detect features in first image
        pts1 = cv2.goodFeaturesToTrack(gray1, **self.feature_params)
        
        if pts1 is None or len(pts1) < 8:
            print("Warning: Not enough features detected")
            return None, None, None, 0
        
        # Track features to second image using optical flow
        pts2, status, err = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, pts1, None, **self.lk_params
        )
        
        # Keep only good matches
        good_mask = status.flatten() == 1
        pts1_good = pts1[good_mask].reshape(-1, 2)
        pts2_good = pts2[good_mask].reshape(-1, 2)
        
        if len(pts1_good) < 8:
            print("Warning: Not enough good matches")
            return None, None, None, 0
        
        # Estimate Essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            pts1_good, pts2_good, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            print("Warning: Could not estimate Essential matrix")
            return None, None, None, 0
        
        # Recover rotation and translation from Essential matrix
        num_inliers, R, t, mask_pose = cv2.recoverPose(
            E, pts1_good, pts2_good, self.K
        )
        
        # t is unit vector, represents direction of motion
        velocity = t.flatten()
        
        return R, t, velocity, num_inliers
    
    def estimate_velocity_between_frames(self, img1, img2, time_between_frames=0.1):
        """
        Estimate velocity vector between two frames.
        
        Note: Visual odometry gives direction but not absolute scale.
        We assume a nominal speed or use IMU/GPS for scale.
        
        Args:
            img1: First image
            img2: Second image  
            time_between_frames: Time in seconds (KITTI is 10Hz = 0.1s)
            
        Returns:
            velocity: 3D velocity vector in camera frame
            valid: Whether estimation succeeded
        """
        R, t, direction, num_inliers = self.estimate_motion(img1, img2)
        
        if direction is None:
            return np.zeros(3), False
        
        # The translation from visual odometry is unit scale
        # For KITTI urban driving, typical speed is 10-50 km/h
        # Let's assume ~30 km/h = 8.3 m/s as nominal
        # In 0.1s, that's about 0.83m displacement
        
        assumed_displacement = 0.5  # meters per frame (conservative estimate)
        
        velocity = direction * (assumed_displacement / time_between_frames)
        
        return velocity, True


# Test visual odometry
if __name__ == "__main__":
    from data_loader import KITTIDataLoader
    from calibration import Calibration
    
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib = Calibration("dataset/2011_09_26")
    
    # Extract intrinsic matrix from P2
    # P2 = K @ [R|t], for rectified images R=I, so K is first 3x3 of P2
    K = calib.P2[:3, :3]
    print(f"Camera intrinsic matrix K:\n{K}")
    
    # Initialize visual odometry
    vo = VisualOdometry(K)
    
    # Test on consecutive frames
    print("\nTesting visual odometry on consecutive frames...")
    
    for i in range(0, 10):
        img1 = loader.load_image(i)
        img2 = loader.load_image(i + 1)
        
        R, t, direction, num_inliers = vo.estimate_motion(img1, img2)
        
        if direction is not None:
            print(f"Frame {i} -> {i+1}: "
                  f"direction=[{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}], "
                  f"inliers={num_inliers}")
        else:
            print(f"Frame {i} -> {i+1}: Failed to estimate motion")