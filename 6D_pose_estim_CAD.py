#!/usr/bin/env python3
"""
Pose estimation using 3D model (.ply file) with object detection
"""

import os
import json
import numpy as np
import cv2
import math
from plyfile import PlyData
import open3d as o3d

def load_camera_intrinsics(json_path):
    """Load camera intrinsics from JSON file"""
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    
    camera_id = list(camera_data.keys())[0]
    cam_K = camera_data[camera_id]['cam_K']
    K = np.array(cam_K).reshape(3, 3)
    return K

def load_ply_model(ply_path):
    """Load 3D model from PLY file"""
    try:
        # Try using open3d first
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not mesh.has_vertices():
            raise ValueError("No vertices found in mesh")
        
        # Get vertices and faces
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        print(f"Loaded PLY model with {len(vertices)} vertices and {len(faces)} faces")
        return vertices, faces, mesh
        
    except Exception as e:
        print(f"Open3D failed to load PLY: {e}")
        try:
            # Fallback to plyfile
            plydata = PlyData.read(ply_path)
            vertices = np.array([[x, y, z] for x, y, z in plydata['vertex']])
            faces = np.array([[f[0], f[1], f[2]] for f in plydata['face']])
            
            print(f"Loaded PLY model with {len(vertices)} vertices and {len(faces)} faces")
            return vertices, faces, None
            
        except Exception as e2:
            print(f"Plyfile also failed: {e2}")
            return None, None, None

def detect_object_improved(image):
    """Improved object detection focusing on the actual object - using the working method from pose_with_rotation.py"""
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Method 1: Edge-based detection
    edges = cv2.Canny(gray, 50, 150)
    
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 30, 200])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    
    kernel = np.ones((5,5), np.uint8)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    valid_contours = []
    image_area = image.shape[0] * image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Too small
            continue
        if area > image_area * 0.8:  # Too large (likely the frame)
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter by aspect ratio (objects are usually not extremely elongated)
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
            
        # Filter by size relative to image
        if w < image.shape[1] * 0.05 or h < image.shape[0] * 0.05:  # Too small
            continue
        if w > image.shape[1] * 0.9 or h > image.shape[0] * 0.9:  # Too large
            continue
            
        valid_contours.append(contour)
    
    # If no valid contours found, try alternative method
    if len(valid_contours) == 0:
        print("No valid contours found, trying alternative detection...")
        # Try using edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < image_area * 0.3:  # Reasonable size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:
                    valid_contours.append(contour)
    
    return valid_contours

def estimate_rotation_from_contour(contour):
    """Estimate rotation from contour shape and orientation"""
    # Fit ellipse to contour
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        
        # Convert angle to radians and normalize
        angle_rad = math.radians(angle)
        
        # Get major and minor axes
        major_axis, minor_axis = axes
        
        yaw = angle_rad
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        if aspect_ratio > 1.5:
            pitch = 0.1  # Slight pitch
        elif aspect_ratio < 0.7:
            pitch = -0.1  # Slight negative pitch
        else:
            pitch = 0.0  # No pitch
        
        # Roll is typically small for flat objects
        roll = 0.0
        
    else:
        # Fallback: use bounding rectangle orientation
        x, y, w, h = cv2.boundingRect(contour)
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
    
    return roll, pitch, yaw

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Convert Roll-Pitch-Yaw angles to rotation matrix"""
    # Rotation around X-axis (roll)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    
    # Rotation around Y-axis (pitch)
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    
    # Rotation around Z-axis (yaw)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def estimate_pose_with_3d_model(image, contours, K, vertices, object_dimensions):
    """Estimate pose using 3D model and object dimensions"""
    poses = []
    
    # Object dimensions in meters
    object_height, object_width = object_dimensions
    
    for i, contour in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate centroid using moments (more accurate than bounding box center)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # Fallback to bounding box center
            center_x = x + w // 2
            center_y = y + h // 2
        fx, fy = K[0, 0], K[1, 1]
        
        if h > w:  # Use height for depth estimation
            depth = (object_height * fy) / h
        else:  # Use width for depth estimation
            depth = (object_width * fx) / w
        
        # Convert to 3D coordinates
        cx, cy = K[0, 2], K[1, 2]
        
        # Back-project to 3D
        X = (center_x - cx) * depth / fx
        Y = (center_y - cy) * depth / fy
        Z = depth
        
        # Estimate rotation from contour
        roll, pitch, yaw = estimate_rotation_from_contour(contour)
        
        # Convert RPY to rotation matrix
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        
        # Create pose
        pose = {
            'translation': [X, Y, Z],
            'rotation_matrix': R.tolist(),
            'rpy': [roll, pitch, yaw],  # Roll, Pitch, Yaw in radians
            'rpy_degrees': [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],  # In degrees
            'bbox': [x, y, w, h],
            'center': [center_x, center_y],
            'contour': contour,  # Keep as numpy array for visualization
            'area': cv2.contourArea(contour),
            'depth_estimate': depth,
            'object_dimensions_used': object_dimensions
        }
        
        poses.append(pose)
    
    return poses

def draw_pose_visualization(image, poses, K):
    """Draw visualization with pose information"""
    vis_image = image.copy()
    
    for i, pose in enumerate(poses):
        # Get data
        x, y, w, h = pose['bbox']
        center_x, center_y = pose['center']
        translation = pose['translation']
        rpy_degrees = pose['rpy_degrees']
        area = pose['area']
        depth = pose['depth_estimate']
        
        # Draw contour
        cv2.drawContours(vis_image, [pose['contour']], -1, (0, 255, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw centroid
        cv2.circle(vis_image, (center_x, center_y), 8, (0, 255, 0), -1)
        cv2.circle(vis_image, (center_x, center_y), 10, (255, 255, 255), 2)
        
        # Draw coordinate axes (rotated based on estimated rotation)
        axis_length = 30
        roll, pitch, yaw = pose['rpy']
        
        # Apply rotation to axis directions
        R = pose['rotation_matrix']
        
        # X-axis (red) - apply rotation
        x_axis = np.array([axis_length, 0, 0])
        x_axis_rotated = R @ x_axis
        end_x = int(center_x + x_axis_rotated[0])
        end_y = int(center_y + x_axis_rotated[1])
        cv2.line(vis_image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(vis_image, "X", (end_x + 5, end_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Y-axis (green) - apply rotation
        y_axis = np.array([0, -axis_length, 0])
        y_axis_rotated = R @ y_axis
        end_x = int(center_x + y_axis_rotated[0])
        end_y = int(center_y + y_axis_rotated[1])
        cv2.line(vis_image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(vis_image, "Y", (end_x + 5, end_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Z-axis (blue) - pointing towards camera
        cv2.circle(vis_image, (center_x, center_y), 4, (255, 0, 0), -1)
        cv2.putText(vis_image, "Z", (center_x + 5, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw information
        info_text = f"Object {i+1}"
        cv2.putText(vis_image, info_text, (x, y - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw translation values
        tx, ty, tz = translation
        trans_text = f"T: ({tx:.3f}, {ty:.3f}, {tz:.3f})"
        cv2.putText(vis_image, trans_text, (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw RPY values (in degrees)
        roll_deg, pitch_deg, yaw_deg = rpy_degrees
        rpy_text = f"RPY: ({roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°)"
        cv2.putText(vis_image, rpy_text, (x, y + h + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw centroid coordinates
        center_text = f"Centroid: ({center_x}, {center_y})"
        cv2.putText(vis_image, center_text, (x, y + h + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw area and depth info
        area_text = f"Area: {area:.0f}, Depth: {depth:.2f}m"
        cv2.putText(vis_image, area_text, (x, y + h + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image

def main():
    """Main function"""
    print("=== Pose Estimation with 3D Model ===")
    
    # Configuration
    image_path = "data/my_test_images/2025-07-09-115536.jpg"
    camera_json_path = "data/my_test_images/scene_camera.json"
    ply_model_path = "data/my_models/bolt_nut.ply"  # Your PLY file
    output_dir = "results"
    
    # Object dimensions in meters (height, width)
    object_dimensions = (0.075, 0.03)  # 7.5cm height, 3cm width
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load camera intrinsics
    print("Loading camera intrinsics...")
    K = load_camera_intrinsics(camera_json_path)
    
    # Load 3D model
    print("Loading 3D model...")
    vertices, faces, mesh = load_ply_model(ply_model_path)
    if vertices is None:
        print("Failed to load 3D model. Exiting.")
        return
    
    # Load image
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    print(f"Image shape: {image.shape}")
    
    # Detect objects using the working method
    print("Detecting objects...")
    contours = detect_object_improved(image)
    print(f"Found {len(contours)} valid objects")
    
    if len(contours) == 0:
        print("No objects detected. Trying to create debug visualization...")
        # Create debug visualization to see what's happening
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Show different detection methods
        cv2.imwrite(os.path.join(output_dir, 'debug_gray.png'), gray)
        cv2.imwrite(os.path.join(output_dir, 'debug_hsv.png'), hsv)
        
        # Show edge detection
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(os.path.join(output_dir, 'debug_edges.png'), edges)
        
        # Show color mask
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 30, 200])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        cv2.imwrite(os.path.join(output_dir, 'debug_mask.png'), gray_mask)
        
        print("Debug images saved. Check the debug_*.png files to see what the detection methods are finding.")
        return
    
    # Estimate poses with 3D model
    print("Estimating poses with 3D model...")
    poses = estimate_pose_with_3d_model(image, contours, K, vertices, object_dimensions)
    print(f"Estimated {len(poses)} poses")
    
    # Print detailed results
    for i, pose in enumerate(poses):
        print(f"\nObject {i+1}:")
        print(f"  Translation: {pose['translation']}")
        print(f"  RPY (radians): {pose['rpy']}")
        print(f"  RPY (degrees): {pose['rpy_degrees']}")
        print(f"  Centroid: {pose['center']}")
        print(f"  Area: {pose['area']:.0f}")
        print(f"  Depth: {pose['depth_estimate']:.2f}m")
        print(f"  Object dimensions used: {pose['object_dimensions_used']}")
    
    # Create visualization
    print("Creating visualization...")
    vis_image = draw_pose_visualization(image, poses, K)
    
    # Save visualization
    vis_path = os.path.join(output_dir, 'pose_with_3d_model_visualization.png')
    cv2.imwrite(vis_path, vis_image)
    print(f"Visualization saved to: {vis_path}")
    
    # Save results to JSON
    # Convert contours to lists for JSON serialization
    json_poses = []
    for pose in poses:
        pose_copy = pose.copy()
        if isinstance(pose_copy['contour'], np.ndarray):
            pose_copy['contour'] = pose_copy['contour'].tolist()
        json_poses.append(pose_copy)

    results = {
        'image_path': image_path,
        'ply_model_path': ply_model_path,
        'camera_intrinsics': K.tolist(),
        'object_dimensions': object_dimensions,
        'poses': json_poses,
        'num_objects_detected': len(contours)
    }
    
    results_path = os.path.join(output_dir, 'pose_with_3d_model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Display visualization
    print("Displaying visualization...")
    print("Press any key to close the window")
    
    # Resize image for better display
    height, width = vis_image.shape[:2]
    scale = min(1.0, 800 / width, 600 / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(vis_image, (new_width, new_height))
    
    cv2.imshow("Pose Estimation with 3D Model", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
