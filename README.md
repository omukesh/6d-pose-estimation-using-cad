# 6D Pose Estimation with 3D CAD Models

This project provides a static approach to estimate 6D pose (position and orientation) of objects using 3D CAD models in PLY format. The system uses computer vision techniques to detect objects in images and estimate their pose relative to the camera.

## Overview

The system performs the following steps:
1. Loads camera intrinsics from a JSON file
2. Loads a 3D model in PLY format
3. Detects objects in the input image using contour detection
4. Estimates 6D pose (translation and rotation) for each detected object
5. Generates visualizations and saves results

## Features

- 3D model loading from PLY files using Open3D and plyfile libraries
- Object detection using contour-based methods
- 6D pose estimation with translation and rotation (Roll-Pitch-Yaw)
- Camera intrinsics support for accurate depth estimation
- Visualization with coordinate axes and pose information
- Results export in JSON format

## Project Structure

```
6d_pose_estimation_project/
├── 6D_pose_estim_CAD.py          # Main pose estimation script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/
│   ├── my_test_images/           # Test images and camera data
│   │   ├── *.jpg                 # Test images
│   │   └── scene_camera.json     # Camera intrinsics
│   └── my_models/                # 3D CAD models
│       └── bolt_nut.ply          # Example PLY model
└── results/                      # Output directory
    ├── pose_with_3d_model_visualization.png
    └── pose_with_3d_model_results.json
```

## Requirements

- Python 3.7 or higher
- OpenCV for image processing
- NumPy for numerical computations
- Open3D for 3D model loading
- plyfile for PLY format support

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place your test images in `data/my_test_images/`
   - Add your 3D model (PLY format) in `data/my_models/`
   - Ensure you have a `scene_camera.json` file with camera intrinsics

2. Update the configuration in `6D_pose_estim_CAD.py`:
   - Set the correct image path
   - Set the correct PLY model path
   - Adjust object dimensions (height, width in meters)

3. Run the pose estimation:
   ```bash
   python 6D_pose_estim_CAD.py
   ```

## Input Data Format

### Camera Intrinsics (scene_camera.json)
```json
{
  "0": {
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "cam_t_w2c": [0, 0, 0]
  }
}
```

### 3D Model
- PLY format with vertices and faces
- Units should be in meters
- Origin should be at the object center

## Output

The system generates:
1. **Visualization image**: Shows detected objects with coordinate axes and pose information
2. **JSON results**: Contains detailed pose data including:
   - Translation (X, Y, Z in meters)
   - Rotation matrix
   - Roll-Pitch-Yaw angles (radians and degrees)
   - Bounding box coordinates
   - Object centroid
   - Estimated depth

## Configuration

Key parameters in the script:
- `object_dimensions`: Physical dimensions of the object (height, width in meters)
- `image_path`: Path to the input image
- `ply_model_path`: Path to the 3D PLY model
- `camera_json_path`: Path to camera intrinsics file

## Limitations

- Uses contour-based detection which works best with high contrast objects
- Requires accurate object dimensions for depth estimation
- Assumes objects are roughly planar or have known dimensions
- Performance depends on image quality and object visibility

## Troubleshooting

- If no objects are detected, check the debug images in the results folder
- Ensure camera intrinsics are correct for your camera
- Verify object dimensions match the actual physical object
- Check that the PLY model is properly formatted and loaded

## Dependencies

- numpy: Numerical computing
- opencv-python: Computer vision and image processing
- plyfile: PLY format file reading
- open3d: 3D model processing and visualization 