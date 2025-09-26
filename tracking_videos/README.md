# Tracking Videos in High Order Interactions of Cockroaches

## Overview
The `tracking_videos` folder contains scripts and methods for tracking the movements of *Blaptica dubia* cockroaches in video recordings. The tracking process involves preprocessing video frames, detecting objects, and analyzing their motion over time. The workflow leverages the `skimage` library and clustering algorithms to ensure accurate tracking and identity maintenance.

## Tracking Workflow

### 1. Searching Process
The first step involves preparing the video data for analysis:
- **Resolution Enhancement:** Upscales videos to 1080x1920 for improved detail.
- **Contrast Adjustment:** Applies local histogram equalization to enhance visibility.
- **Edge Detection:** Uses the Canny edge detection algorithm to define object boundaries.
- **Profiling Methods:** Evaluates two approaches:
  - **PF:** Histogram equalization followed by profiling.
  - **FP:** Profiling followed by histogram equalization (yields better results).
- **Object Tracking:** Extracts cockroach positions using grayscale level analysis with `skimage.measure.regionprops`.

### 2. Clustering Process
After identifying the tracked objects, clustering techniques are applied:
- **Velocity Analysis:** Examines velocity distributions for better object grouping.
- **K-Means Clustering:** Groups detected objects based on a distance threshold to maintain identity.
- **Boundary Updates (Optional):** Refines object boundaries where necessary.

### 3. Smoothing Process
To improve tracking consistency, the following steps are applied:
- **Identity Correction:** Detects and resolves ID swaps during tracking.
- **Frame Filtering:** Removes incorrect or inconsistent frames.
- **Interpolation:** Fills gaps in tracking data caused by dropped frames.
- **Orientation Updates (Under Review):** Adjusts orientation calculations to better represent movement.

## Notepad Application

Finally, a Tkinter-based application for visualizing cockroach tracking data is built. Allows users to load CSV files, display tracking data, edit positions, and navigate through frames using keyboard shortcuts. with a user-friendly interface.

For the deployment of the application it is recommended to create a new folder in the same folder where the MP4 video is allocated, and to transform MP4 video frames as PNG frames with `ffmpeg` in the following command:

```ffmpeg -i video_name.mp4 -vf "select=not(mod(n\,3))" -vsync vfr -frame_pts 1 -q:v 10 "video_name/frame_%06d.png"
```

### Features
- Open `.txt` or `.csv` files.
- Save text to `.txt` or `.csv` files.
- Basic text editing.
- Keyboard shortcuts for efficiency.

### Keyboard Shortcuts
| Shortcut                 | Function                  |
|--------------------------|---------------------------|
| `<Shift>`                | Enable Edit Positions     |
| `Q`                      | Rotate anticlockwise ID 0 |
| `W`                      | Rotate anticlockwise ID 1 |
| `E`                      | Rotate anticlockwise ID 2 |
| `R`                      | Rotate clockwise ID 3     |
| `A`                      | Rotate clockwise ID 0     |
| `S`                      | Rotate clockwise ID 1     |
| `D`                      | Rotate clockwise ID 2     |
| `F`                      | Rotate clockwise ID 3     |
| `<Right>`                | Move next frame           |
| `<Left>`                 | Move previous frame       |
| `X`                      | Swap ID 0-1               |
| `C`                      | Swap ID 0-2               |
| `V`                      | Swap ID 0-3               |
| `B`                      | Swap ID 1-2               |
| `N`                      | Swap ID 1-3               |
| `M`                      | Swap ID 2-3               |

### Installation & Usage
1. Ensure you have Python installed (version 3.x recommended).
2. Run the script using:
   ```sh
   python ./modules/generate_tracking_widget.py
   ```
3. Use the menu or keyboard shortcuts to interact with the app.

### Dependencies
- `tkinter` (comes pre-installed with Python)

### License
This project is released under the MIT License.

## References
For more details, refer to the following resources:
- [Local Histogram Equalization](https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_local_equalize.html)
- [Canny Edge Detection](https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html)
- [Region Properties in skimage](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

---
This tracking method allows efficient and reliable analysis of cockroach movement patterns, supporting higher-order interaction studies.
