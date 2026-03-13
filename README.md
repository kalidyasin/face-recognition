# Face & Body Tracking in Rust

A real-time face detection, facial landmark tracing, and human pose estimation application built using Rust and YOLOv8 models.

## Features

- **Real-time Face Detection**: Detects faces using YOLOv8n-face.
- **Facial Landmark Tracing**: Traces 5 key landmarks (eyes, nose, mouth corners) on detected faces.
- **Body Tracking (Pose Estimation)**: Tracks 17 body keypoints (skeleton) using YOLOv8n-pose.
- **Multiple Modes**: Supports real-time webcam streaming and single-image processing.
- **Cross-Platform AI**: Powered by ONNX Runtime (`ort`) for high-performance CPU/GPU inference.

## Prerequisites

### Linux (Ubuntu/Debian)
Install the required system libraries for camera access and windowing:
```bash
sudo apt-get update
sudo apt-get install -y libfontconfig1-dev libasound2-dev libv4l-dev libx11-dev libwayland-dev
```

### Windows/macOS
Ensure you have a working C++ compiler installed (Visual Studio Build Tools on Windows or Xcode on macOS).

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kalidyasin/face-recognition.git
   cd face-recognition
   ```

2. **Build the project**:
   ```bash
   cargo build --release
   ```

## Usage

### 1. Real-time Webcam Tracking
Launch the application using your default webcam:
```bash
cargo run --release -- realtime
```
- Press **ESC** to exit the application.
- Use `--camera-index <N>` to specify a different camera if you have multiple devices.

### 2. Single Image Processing
Process a photo and save the results to `output.jpg`:
```bash
cargo run --release -- image path/to/your/photo.jpg
```

## How It Works

- **Inference Engine**: The project uses the `ort` crate, a Rust wrapper for ONNX Runtime, to run YOLOv8 models.
- **Models**: It automatically downloads the official YOLOv8n-face and YOLOv8n-pose models on the first run.
- **Camera Access**: Uses the `nokhwa` crate for cross-platform webcam support.
- **Visualization**: Uses `imageproc` for drawing bounding boxes and keypoints, and `minifb` for high-performance window display.

## Troubleshooting

- **First Run Delay**: On the first run, the application may take a minute to download the models and initialize the ONNX Runtime libraries.
- **No Cameras Found**: Ensure your webcam is connected and recognized by your OS.
- **Graphical Environment**: This app requires a windowing system (X11 or Wayland on Linux). It will not run in a pure CLI/SSH environment without X-forwarding.

## License
MIT
