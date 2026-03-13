use anyhow::Result;
use clap::{Parser, Subcommand};
use image::{DynamicImage, GenericImageView, Rgb};
use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use rayon::prelude::*;
use std::path::Path;

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
    Image { path: String },
}

const FACE_MODEL_URL: &str = "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx";
const POSE_MODEL_URL: &str = "https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov8n-pose.onnx";

fn main() -> Result<()> {
    let cli = Cli::parse();

    let face_model_path = "yolov8n-face.onnx";
    let pose_model_path = "yolov8n-pose.onnx";
    check_download(FACE_MODEL_URL, face_model_path)?;
    check_download(POSE_MODEL_URL, pose_model_path)?;

    println!("Initializing Face Model...");
    let mut face_model = YoloModel::new(face_model_path)?;
    println!("Initializing Pose Model...");
    let mut pose_model = YoloModel::new(pose_model_path)?;

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
    }
    Ok(())
}

fn run_realtime(index: u32, face_model: &mut YoloModel, pose_model: &mut YoloModel) -> Result<()> {
    let index = CameraIndex::Index(index);
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;

    let width = camera.resolution().width_x;
    let height = camera.resolution().height_y;

    let mut window = Window::new(
        "Face & Body Tracker - ESC to exit",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )?;

    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        let mut rgb_image = decoded;

        let (display_buffer, num_faces, num_pose) =
            process_frame(&mut rgb_image, face_model, pose_model)?;

        window.set_title(&format!(
            "Face & Body Tracker | Faces: {} | Bodies: {}",
            num_faces, num_pose
        ));
        window.update_with_buffer(&display_buffer, width as usize, height as usize)?;
    }
    Ok(())
}

fn run_image(path: &str, face_model: &mut YoloModel, pose_model: &mut YoloModel) -> Result<()> {
    let img = image::open(path)?;
    let mut rgb_image = img.to_rgb8();
    let _ = process_frame(&mut rgb_image, face_model, pose_model)?;
    rgb_image.save("output.jpg")?;
    println!("Saved result to output.jpg");
    Ok(())
}

fn process_frame(
    img: &mut image::RgbImage,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
) -> Result<(Vec<u32>, usize, usize)> {
    let (w, h) = img.dimensions();
    let (input, scale, dx, dy) =
        utils::preprocess_yolo(&DynamicImage::ImageRgb8(img.clone()), (640, 640));

    let face_dets = face_model.detect(input.clone(), 0.3)?;
    let pose_dets = pose_model.detect(input, 0.3)?;

    // Face Post-processing
    let mut face_boxes = Vec::new();
    for (i, det) in face_dets.iter().enumerate() {
        face_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_faces = utils::nms(&face_boxes, 0.45);

    for &idx in &keep_faces {
        let det = &face_dets[idx];
        let x = (det.bbox.0 - dx) / scale;
        let y = (det.bbox.1 - dy) / scale;
        let w_box = det.bbox.2 / scale;
        let h_box = det.bbox.3 / scale;

        utils::draw_bbox(img, (x, y, w_box, h_box), Rgb([0, 255, 0]), None);

        for (i, (kx, ky, _kc)) in det.keypoints.iter().enumerate() {
            let rx = (kx - dx) / scale;
            let ry = (ky - dy) / scale;
            let color = match i {
                0 | 1 => Rgb([0, 255, 255]), // Eyes
                2 => Rgb([255, 255, 0]),     // Nose
                3 | 4 => Rgb([255, 0, 255]), // Mouth
                _ => Rgb([255, 0, 0]),
            };
            imageproc::drawing::draw_filled_circle_mut(img, (rx as i32, ry as i32), 3, color);
        }

        if det.keypoints.len() >= 5 {
            let m1 = &det.keypoints[3];
            let m2 = &det.keypoints[4];
            imageproc::drawing::draw_line_segment_mut(
                img,
                ((m1.0 - dx) / scale, (m1.1 - dy) / scale),
                ((m2.0 - dx) / scale, (m2.1 - dy) / scale),
                Rgb([255, 0, 255]),
            );
        }
    }

    // Pose Post-processing
    let mut pose_boxes = Vec::new();
    for (i, det) in pose_dets.iter().enumerate() {
        pose_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_pose = utils::nms(&pose_boxes, 0.45);

    for &idx in &keep_pose {
        let det = &pose_dets[idx];
        draw_skeleton(img, &det.keypoints, dx, dy, scale);
    }

    let raw_pixels = img.as_raw();
    let buffer: Vec<u32> = raw_pixels
        .par_chunks_exact(3)
        .map(|p| {
            let r = p[0] as u32;
            let g = p[1] as u32;
            let b = p[2] as u32;
            (r << 16) | (g << 8) | b
        })
        .collect();

    Ok((buffer, keep_faces.len(), keep_pose.len()))
}

const SKELETON: [(usize, usize); 12] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
];

fn draw_skeleton(
    img: &mut image::RgbImage,
    kpts: &Vec<(f32, f32, f32)>,
    dx: f32,
    dy: f32,
    scale: f32,
) {
    for &(i, j) in &SKELETON {
        if i < kpts.len() && j < kpts.len() {
            let (x1, y1, c1) = kpts[i];
            let (x2, y2, c2) = kpts[j];
            if c1 > 0.4 && c2 > 0.4 {
                imageproc::drawing::draw_line_segment_mut(
                    img,
                    ((x1 - dx) / scale, (y1 - dy) / scale),
                    ((x2 - dx) / scale, (y2 - dy) / scale),
                    Rgb([0, 100, 255]),
                );
            }
        }
    }
    // Draw joints, skip 0-4 (Face points from pose model) to avoid duplicates
    for (idx, (kx, ky, kc)) in kpts.iter().enumerate() {
        if *kc > 0.4 && idx > 4 {
            imageproc::drawing::draw_filled_circle_mut(
                img,
                (((kx - dx) / scale) as i32, ((ky - dy) / scale) as i32),
                2,
                Rgb([255, 255, 255]),
            );
        }
    }
}
