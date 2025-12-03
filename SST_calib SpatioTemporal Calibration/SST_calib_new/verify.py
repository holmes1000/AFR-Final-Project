"""
Step 2: Verify KITTI data is correctly structured and readable.
Run this first to ensure your data is valid before proceeding.
"""

import os
import struct
import numpy as np
from pathlib import Path


def check_folder_exists(path, name):
    """Check if a folder exists and print status."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    return exists


def check_file_exists(path, name):
    """Check if a file exists and print status."""
    exists = os.path.isfile(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    return exists


def count_files_in_folder(folder_path, extension):
    """Count files with given extension in a folder."""
    if not os.path.isdir(folder_path):
        return 0
    
    # KITTI stores data in a 'data' subfolder
    data_folder = os.path.join(folder_path, "data")
    if os.path.isdir(data_folder):
        folder_path = data_folder
    
    count = len([f for f in os.listdir(folder_path) if f.endswith(extension)])
    return count


def read_sample_velodyne(bin_path):
    """Read a sample velodyne point cloud to verify format."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)  # x, y, z, reflectance
    return point_cloud


def read_calibration_file(calib_path):
    """Read a KITTI calibration file."""
    calib_data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                calib_data[key.strip()] = value.strip()
    return calib_data


def verify_kitti_dataset(base_path, date, drive):
    """
    Verify KITTI dataset structure and contents.
    
    Args:
        base_path: Path to 'dataset' folder
        date: Date string like '2011_09_26'
        drive: Drive string like '0005'
    """
    print("=" * 60)
    print("KITTI Dataset Verification")
    print("=" * 60)
    
    # Build paths
    date_folder = os.path.join(base_path, date)
    drive_folder = os.path.join(date_folder, f"{date}_drive_{drive}_sync")
    
    print(f"\nBase path: {base_path}")
    print(f"Date: {date}")
    print(f"Drive: {drive}")
    
    # Check main folders
    print("\n1. Checking folder structure...")
    all_ok = True
    
    all_ok &= check_folder_exists(date_folder, "Date folder")
    all_ok &= check_folder_exists(drive_folder, "Drive folder")
    all_ok &= check_folder_exists(os.path.join(drive_folder, "image_02"), "image_02 (color left)")
    all_ok &= check_folder_exists(os.path.join(drive_folder, "velodyne_points"), "velodyne_points")
    
    # Check calibration files
    print("\n2. Checking calibration files...")
    calib_cam_path = os.path.join(date_folder, "calib_cam_to_cam.txt")
    calib_velo_path = os.path.join(date_folder, "calib_velo_to_cam.txt")
    calib_imu_path = os.path.join(date_folder, "calib_imu_to_velo.txt")
    
    all_ok &= check_file_exists(calib_cam_path, "calib_cam_to_cam.txt")
    all_ok &= check_file_exists(calib_velo_path, "calib_velo_to_cam.txt")
    all_ok &= check_file_exists(calib_imu_path, "calib_imu_to_velo.txt")
    
    # Count frames
    print("\n3. Counting frames...")
    image_folder = os.path.join(drive_folder, "image_02", "data")
    velo_folder = os.path.join(drive_folder, "velodyne_points", "data")
    
    n_images = count_files_in_folder(os.path.join(drive_folder, "image_02"), ".png")
    n_velodyne = count_files_in_folder(os.path.join(drive_folder, "velodyne_points"), ".bin")
    
    print(f"  Images (image_02): {n_images} frames")
    print(f"  Velodyne scans: {n_velodyne} frames")
    
    if n_images != n_velodyne:
        print(f"  ⚠ Warning: Frame count mismatch!")
    else:
        print(f"  ✓ Frame counts match")
    
    # Read sample data
    print("\n4. Reading sample data...")
    
    # Sample image
    if os.path.isdir(image_folder):
        sample_images = sorted(os.listdir(image_folder))
        if sample_images:
            sample_image_path = os.path.join(image_folder, sample_images[0])
            from PIL import Image
            img = Image.open(sample_image_path)
            print(f"  ✓ Sample image: {img.size[0]}x{img.size[1]} pixels, mode={img.mode}")
    
    # Sample velodyne
    if os.path.isdir(velo_folder):
        sample_velo = sorted(os.listdir(velo_folder))
        if sample_velo:
            sample_velo_path = os.path.join(velo_folder, sample_velo[0])
            pc = read_sample_velodyne(sample_velo_path)
            print(f"  ✓ Sample point cloud: {pc.shape[0]} points, shape={pc.shape}")
            print(f"    X range: [{pc[:, 0].min():.2f}, {pc[:, 0].max():.2f}]")
            print(f"    Y range: [{pc[:, 1].min():.2f}, {pc[:, 1].max():.2f}]")
            print(f"    Z range: [{pc[:, 2].min():.2f}, {pc[:, 2].max():.2f}]")
    
    # Read calibration
    print("\n5. Reading calibration files...")
    
    if os.path.isfile(calib_velo_path):
        velo_calib = read_calibration_file(calib_velo_path)
        print(f"  ✓ calib_velo_to_cam.txt keys: {list(velo_calib.keys())}")
        
        # Parse R and T (rotation and translation from velodyne to camera 0)
        if 'R' in velo_calib and 'T' in velo_calib:
            R = np.array([float(x) for x in velo_calib['R'].split()]).reshape(3, 3)
            T = np.array([float(x) for x in velo_calib['T'].split()]).reshape(3, 1)
            print(f"  ✓ Rotation matrix R:\n{R}")
            print(f"  ✓ Translation vector T: {T.flatten()}")
    
    if os.path.isfile(calib_cam_path):
        cam_calib = read_calibration_file(calib_cam_path)
        # Get projection matrix for camera 2
        if 'P_rect_02' in cam_calib:
            P2 = np.array([float(x) for x in cam_calib['P_rect_02'].split()]).reshape(3, 4)
            print(f"  ✓ Camera 2 projection matrix P_rect_02:\n{P2}")
            # Extract intrinsic matrix K
            K = P2[:3, :3]
            print(f"  ✓ Intrinsic matrix K:\n{K}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok and n_images > 0 and n_velodyne > 0:
        print("✓ Dataset verification PASSED")
        print(f"  Ready to process {min(n_images, n_velodyne)} frames")
    else:
        print("✗ Dataset verification FAILED")
        print("  Please check the errors above")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    # =====================================================
    # CONFIGURE THESE PATHS FOR YOUR SYSTEM
    # =====================================================
    
    # Path to your 'dataset' folder (parent of date folder)
    BASE_PATH = r"D:\Coding\SST_calib SpatioTemporal Calibration\dataset"  # <-- CHANGE THIS
    
    # Date and drive from your folder names
    DATE = "2011_09_26"
    DRIVE = "0005"
    
    # =====================================================
    
    # Run verification
    verify_kitti_dataset(BASE_PATH, DATE, DRIVE)