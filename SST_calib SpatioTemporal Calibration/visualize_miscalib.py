import numpy as np
import matplotlib.pyplot as plt
from data_loader import KITTIDataLoader
from calibration import Calibration

def apply_calibration_noise(calib, rotation_noise_deg=5.0, translation_noise_m=0.1):
    """
    Create a miscalibrated version by adding noise to extrinsics
    
    Args:
        calib: Original calibration object
        rotation_noise_deg: Noise in degrees for each axis
        translation_noise_m: Noise in meters for each axis
    
    Returns:
        Modified calibration object
    """
    import copy
    noisy_calib = copy.deepcopy(calib)
    
    # Add rotation noise (convert degrees to radians)
    noise_rad = np.radians(rotation_noise_deg)
    roll_noise = np.random.uniform(-noise_rad, noise_rad)
    pitch_noise = np.random.uniform(-noise_rad, noise_rad)
    yaw_noise = np.random.uniform(-noise_rad, noise_rad)
    
    # Create rotation matrices for each axis
    def rot_x(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    
    def rot_y(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    
    def rot_z(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    
    # Combined noise rotation
    R_noise = rot_z(yaw_noise) @ rot_y(pitch_noise) @ rot_x(roll_noise)
    
    # Add translation noise
    T_noise = np.random.uniform(-translation_noise_m, translation_noise_m, size=(3, 1))
    
    # Apply noise to original calibration
    R_new = R_noise @ noisy_calib.R_velo
    T_new = noisy_calib.T_velo + T_noise
    
    # Update the transformation matrix
    noisy_calib.Tr_velo_to_cam[:3, :3] = R_new
    noisy_calib.Tr_velo_to_cam[:3, 3:4] = T_new
    
    print(f"Added noise:")
    print(f"  Rotation: roll={np.degrees(roll_noise):.2f}°, pitch={np.degrees(pitch_noise):.2f}°, yaw={np.degrees(yaw_noise):.2f}°")
    print(f"  Translation: x={T_noise[0,0]*100:.1f}cm, y={T_noise[1,0]*100:.1f}cm, z={T_noise[2,0]*100:.1f}cm")
    
    return noisy_calib


def project_and_filter(points, calib, img_h, img_w):
    """Project points and filter to valid image region"""
    projected, depths = calib.project_velo_to_image(points)
    
    valid = (
        (depths > 0) &
        (projected[:, 0] >= 0) &
        (projected[:, 0] < img_w) &
        (projected[:, 1] >= 0) &
        (projected[:, 1] < img_h)
    )
    
    return projected[valid], depths[valid]


def visualize_comparison(image, points, calib_correct, calib_noisy, frame_idx=0):
    """Compare correct vs miscalibrated projection"""
    
    img_h, img_w = image.shape[:2]
    
    # Project with both calibrations
    proj_correct, depth_correct = project_and_filter(points, calib_correct, img_h, img_w)
    proj_noisy, depth_noisy = project_and_filter(points, calib_noisy, img_h, img_w)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Correct calibration
    axes[0].imshow(image)
    axes[0].scatter(proj_correct[:, 0], proj_correct[:, 1], 
                    c=depth_correct, cmap='jet_r', s=1, alpha=0.7)
    axes[0].set_title("Correct Calibration (Ground Truth)")
    axes[0].axis('off')
    
    # Right: Miscalibrated
    axes[1].imshow(image)
    axes[1].scatter(proj_noisy[:, 0], proj_noisy[:, 1], 
                    c=depth_noisy, cmap='jet_r', s=1, alpha=0.7)
    axes[1].set_title("Miscalibrated (With Noise)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"miscalibration_comparison_frame_{frame_idx}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Load data
    loader = KITTIDataLoader("dataset/2011_09_26/2011_09_26_drive_0005_sync")
    calib_correct = Calibration("dataset/2011_09_26")
    
    # Create noisy calibration (similar to paper: ±10° rotation, ±10cm translation)
    print("\nCreating miscalibrated version...")
    calib_noisy = apply_calibration_noise(
        calib_correct, 
        rotation_noise_deg=10.0,  # ±10 degrees
        translation_noise_m=0.10  # ±10 cm
    )
    
    # Visualize
    frame_idx = 0
    image = loader.load_image(frame_idx)
    points = loader.load_velodyne(frame_idx)
    
    print(f"\nComparing projections...")
    visualize_comparison(image, points, calib_correct, calib_noisy, frame_idx)