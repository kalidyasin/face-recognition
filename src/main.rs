use anyhow::Result;
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, Rgb};
use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::path::Path;
use std::time::Instant;

mod models;
mod utils;

use models::YoloModel;
use utils::{draw_bbox, nms, preprocess_yolo};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Real-time face and body tracking from webcam
    Realtime {
        #[arg(long, default_value = "0")]
        camera_index: u32,
    },
    /// Process a single image
    Image {
        path: String,
    },
}

const FACE_MODEL_URL: &str = "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx";
const POSE_MODEL_URL: &str = "https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov8n-pose.onnx";

fn main() -> Result<()> {
    // Initialize ORT (optional for default CPU)
    // ort::init().commit()?;

    let cli = Cli::parse();

    // Check and download models
    let face_model_path = "yolov8n-face.onnx";
    let pose_model_path = "yolov8n-pose.onnx";
    check_download(FACE_MODEL_URL, face_model_path)?;
    check_download(POSE_MODEL_URL, pose_model_path)?;

    println!("Initializing Face Model from {}...", face_model_path);
    let mut face_model = YoloModel::new(face_model_path)?;
    println!("Face Model loaded.");

    println!("Initializing Pose Model from {}...", pose_model_path);
    let mut pose_model = YoloModel::new(pose_model_path)?;
    println!("Pose Model loaded.");

    match cli.command {
        Commands::Realtime { camera_index } => {
            run_realtime(camera_index, &mut face_model, &mut pose_model)?;
        }
        Commands::Image { path } => {
            run_image(&path, &mut face_model, &mut pose_model)?;
        }
    }

    Ok(())
}

fn check_download(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Downloading {}...", path);
        let resp = reqwest::blocking::get(url)?;
        let bytes = resp.bytes()?;
        std::fs::write(path, bytes)?;
        println!("Downloaded {}.", path);
    }
    Ok(())
}

fn run_realtime(index: u32, face_model: &mut YoloModel, pose_model: &mut YoloModel) -> Result<()> {
    println!("Checking for cameras...");
    let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
    if cameras.is_empty() {
        return Err(anyhow::anyhow!("No cameras found! Make sure your webcam is connected."));
    }
    println!("Found {} camera(s). Using index {}.", cameras.len(), index);
    for (i, cam) in cameras.iter().enumerate() {
        println!("  {}: {}", i, cam.human_name());
    }

    let index = CameraIndex::Index(index);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested)?;
    
    println!("Opening camera stream...");
    camera.open_stream()?;
    println!("Camera stream opened at {}x{}", camera.resolution().width_x, camera.resolution().height_y);

    let width = camera.resolution().width_x;
    let height = camera.resolution().height_y;
    
    println!("Creating window ({}x{})...", width, height);
    let mut window = Window::new(
        "Face & Body Tracking - ESC to exit",
        width as usize,
        height as usize,
        WindowOptions::default(),
    ).map_err(|e| anyhow::anyhow!("Failed to create window: {}. Are you in a graphical environment (X11/Wayland)?", e))?;
    println!("Window created.");
    
    // Limit to ~30 FPS
    window.limit_update_rate(Some(std::time::Duration::from_micros(33300)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let mut rgb_image = decoded; 
        
        let display_buffer = process_frame(&mut rgb_image, face_model, pose_model)?;

        window.update_with_buffer(&display_buffer, width as usize, height as usize)?;
    }
    Ok(())
}

fn run_image(path: &str, face_model: &mut YoloModel, pose_model: &mut YoloModel) -> Result<()> {
    let img = image::open(path)?;
    let mut rgb_image = img.to_rgb8();
    
    // Process
    let _ = process_frame(&mut rgb_image, face_model, pose_model)?;
    
    // Save output
    rgb_image.save("output.jpg")?;
    println!("Saved result to output.jpg");
    Ok(())
}

fn process_frame(
    img: &mut image::RgbImage,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel
) -> Result<Vec<u32>> {
    let (w, h) = img.dimensions();
    
    // 1. Preprocess
    let (input, scale, dx, dy) = utils::preprocess_yolo(&DynamicImage::ImageRgb8(img.clone()), (640, 640));

    // 2. Inference
    let face_dets = face_model.detect(input.clone(), 0.4)?;
    let pose_dets = pose_model.detect(input, 0.4)?;

    // 3. Post-process & Draw (Face)
    let mut face_boxes = Vec::new();
    for (i, det) in face_dets.iter().enumerate() {
        face_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_faces = utils::nms(&face_boxes, 0.45); // IoU threshold

    for &idx in &keep_faces {
        let det = &face_dets[idx];
        
        // Scale back coordinates
        let x = (det.bbox.0 - dx) / scale;
        let y = (det.bbox.1 - dy) / scale;
        let w_box = det.bbox.2 / scale;
        let h_box = det.bbox.3 / scale;
        
        utils::draw_bbox(img, (x, y, w_box, h_box), Rgb([0, 255, 0]), Some("Face"));

        for (i, (kx, ky, _kc)) in det.keypoints.iter().enumerate() {
            let rx = (kx - dx) / scale;
            let ry = (ky - dy) / scale;
            
            let color = match i {
                0 | 1 => Rgb([0, 255, 255]), // Eyes (Cyan)
                2 => Rgb([255, 255, 0]),     // Nose (Yellow)
                3 | 4 => Rgb([255, 0, 255]), // Mouth (Magenta)
                _ => Rgb([255, 0, 0]),       // Others
            };
            
            imageproc::drawing::draw_filled_circle_mut(img, (rx as i32, ry as i32), 3, color);
        }

        // Draw mouth line with a slightly thicker or more prominent visual
        if det.keypoints.len() >= 5 {
            let m1 = &det.keypoints[3];
            let m2 = &det.keypoints[4];
            let x1 = (m1.0 - dx) / scale;
            let y1 = (m1.1 - dy) / scale;
            let x2 = (m2.0 - dx) / scale;
            let y2 = (m2.1 - dy) / scale;
            
            // Draw a few lines to make it thicker
            imageproc::drawing::draw_line_segment_mut(img, (x1, y1), (x2, y2), Rgb([255, 0, 255]));
            imageproc::drawing::draw_line_segment_mut(img, (x1, y1 + 1.0), (x2, y2 + 1.0), Rgb([255, 0, 255]));
        }
    }

    // 4. Post-process & Draw (Pose)
    let mut pose_boxes = Vec::new();
    for (i, det) in pose_dets.iter().enumerate() {
        pose_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_pose = utils::nms(&pose_boxes, 0.5);

    for &idx in &keep_pose {
        let det = &pose_dets[idx];
        // Draw Skeleton (simplified, just points for now)
        for (kx, ky, kc) in &det.keypoints {
            if *kc > 0.5 {
                let rx = (kx - dx) / scale;
                let ry = (ky - dy) / scale;
                imageproc::drawing::draw_filled_circle_mut(img, (rx as i32, ry as i32), 3, Rgb([0, 0, 255]));
            }
        }
    }

    // Convert to u32 buffer for minifb
    let mut buffer = Vec::with_capacity((w * h) as usize);
    for pixel in img.pixels() {
        let r = pixel[0] as u32;
        let g = pixel[1] as u32;
        let b = pixel[2] as u32;
        buffer.push((r << 16) | (g << 8) | b);
    }
    
    Ok(buffer)
}
