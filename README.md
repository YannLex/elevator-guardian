# Elevator Guardian

Elevator Guardian is a Python-based MVP designed to monitor escalators and detect if a child is riding alone. If a child is detected without an accompanying adult, the system sends a signal to stop the escalator.

## Features
- **YOLOv11 Integration**: Uses the latest YOLOv11 model for person detection.
- **Age Estimation**: Utilizes a pre-trained `age_net` model to classify individuals into age groups.
- **Real-Time Video Processing**: Processes video files or live feeds to detect children and adults.
- **Stop Signal Simulation**: Simulates sending a stop signal when a child is detected riding alone.

## Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `opencv-python`
  - `ultralytics`
  - `numpy`

## Installation
1. Clone this repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install opencv-python ultralytics numpy
   ```
3. Ensure the following pre-trained model files are in the project directory:
   - `yolov8n.pt`
   - `deploy_age.prototxt`
   - `age_net.caffemodel`
   - `deploy_face.prototxt`
   - `res10_300x300_ssd_iter_140000.caffemodel`

## Usage
1. Place a video file in the project directory.
2. Run the `main.py` script:
   ```bash
   python main.py
   ```
3. Provide the path to the video file when prompted.
4. The system will process the video and display the results in a window. If a child is detected riding alone, the system will print a message and stop further detection.

## File Structure
- `main.py`: Main script for video processing and detection.
- `yolov8n.pt`: YOLOv11 model file for person detection.
- `deploy_age.prototxt` and `age_net.caffemodel`: Files for age estimation.
- `deploy_face.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`: Files for face detection.

## Project Files
- `main.py`: Main script for video processing and detection.
- `yolov8n.pt`: YOLOv11 model file for person detection.
- `deploy_age.prototxt` and `age_net.caffemodel`: Files for age estimation.
- `deploy_face.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`: Files for face detection.
- `LICENSE`: MIT License for the project.
- `README.md`: Documentation for the project.

## Updates
- **2025-05-05**: Added MIT License and updated README to reflect the current project structure and features.

## Notes
- The age estimation is based on face detection and may require clear visibility of faces for accurate results.
- Adjust the thresholds and parameters in the script to suit your specific use case and video resolution.

