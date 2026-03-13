use anyhow::Result;
use clap::{Parser, Subcommand};
use image::{DynamicImage, Rgb, RgbImage};
use minifb::{Key, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use ndarray::Array1;
use std::time::{SystemTime, UNIX_EPOCH};

mod models;
mod utils;

use models::{YoloModel, FaceRecognizer, Detection};
use utils::preprocess_yolo;

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
const RECOG_MODEL_URL: &str = "https://huggingface.co/LPDoctor/insightface/resolve/main/models/antelopev2/glintr100.onnx?download=true";

struct KnownPerson {
    name: String,
    embeddings: Vec<Array1<f32>>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let face_model_path = "yolov8n-face.onnx";
    let pose_model_path = "yolov8n-pose.onnx";
    let recog_model_path = "arcface.onnx";

    check_download(FACE_MODEL_URL, face_model_path)?;
    check_download(POSE_MODEL_URL, pose_model_path)?;
    check_download(RECOG_MODEL_URL, recog_model_path)?;

    println!("Initializing AI Models...");
    let mut face_model = YoloModel::new(face_model_path)?;
    let mut pose_model = YoloModel::new(pose_model_path)?;
    let mut recog_model = FaceRecognizer::new(recog_model_path)?;

    println!("Loading known faces from 'known_faces/' subdirectories...");
    let mut known_people = load_known_people(&mut face_model, &mut recog_model)?;
    println!("Loaded {} known people.", known_people.len());

    match cli.command {
        Commands::Realtime { camera_index } => {
            run_realtime(camera_index, &mut face_model, &mut pose_model, &mut recog_model, &mut known_people)?;
        }
        Commands::Image { path } => {
            run_image(&path, &mut face_model, &mut pose_model, &mut recog_model, &known_people)?;
        }
    }

    Ok(())
}

fn check_download(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Downloading {} (this may take a while)...", path);
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build()?;
        let mut response = client.get(url).send()?;
        if !response.status().is_success() { return Err(anyhow::anyhow!("Download failed: {}", response.status())); }
        let mut file = std::fs::File::create(path)?;
        std::io::copy(&mut response, &mut file)?;
        println!("  - Finished. Size: {} MB", std::fs::metadata(path)?.len() / 1024 / 1024);
    }
    Ok(())
}

fn load_known_people(face_model: &mut YoloModel, recog_model: &mut FaceRecognizer) -> Result<Vec<KnownPerson>> {
    let base_dir = Path::new("known_faces");
    if !base_dir.exists() {
        std::fs::create_dir_all(base_dir)?;
        println!("Created 'known_faces/' folder. Usage: known_faces/<Name>/<image.jpg>");
        return Ok(vec![]);
    }

    let mut people = Vec::new();
    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let mut embeddings = Vec::new();
            println!("  Loading {}...", name);
            
            for img_entry in std::fs::read_dir(&path)? {
                let img_path = img_entry?.path();
                if img_path.is_file() && (img_path.extension().map_or(false, |e| e=="jpg" || e=="png" || e=="jpeg")) {
                    if let Ok(img) = image::open(&img_path) {
                        let rgb = img.to_rgb8();
                        let (input, scale, dx, dy) = preprocess_yolo(&DynamicImage::ImageRgb8(rgb.clone()), (640, 640));
                        if let Ok(dets) = face_model.detect(input, 0.5) {
                            if let Some(det) = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()) {
                                let x = (((det.bbox.0 - dx) / scale).max(0.0) as u32).min(rgb.width()-1);
                                let y = (((det.bbox.1 - dy) / scale).max(0.0) as u32).min(rgb.height()-1);
                                let w = (det.bbox.2 / scale) as u32; let h = (det.bbox.3 / scale) as u32;
                                let w = w.min(rgb.width() - x); let h = h.min(rgb.height() - y);
                                if w > 0 && h > 0 {
                                    let face_crop = image::imageops::crop_imm(&rgb, x, y, w, h).to_image();
                                    if let Ok(emb) = recog_model.get_embedding(&face_crop) {
                                        embeddings.push(emb);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !embeddings.is_empty() {
                people.push(KnownPerson { name, embeddings });
            }
        }
    }
    Ok(people)
}

fn run_realtime(index: u32, face_model: &mut YoloModel, pose_model: &mut YoloModel, recog_model: &mut FaceRecognizer, known: &mut Vec<KnownPerson>) -> Result<()> {
    let index = CameraIndex::Index(index);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;

    let width = camera.resolution().width_x;
    let height = camera.resolution().height_y;
    
    let mut window = Window::new("Face & Body Tracking - [S] to Save Face", width as usize, height as usize, WindowOptions::default())?;
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut last_processed_frame: Option<image::RgbImage> = None;
    let mut last_detections: Vec<Detection> = Vec::new();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.frame()?;
        let mut rgb_image = frame.decode_image::<RgbFormat>()?;
        
        // Enrollment Key
        if window.is_key_pressed(Key::S, minifb::KeyRepeat::No) {
            if let Some(det) = last_detections.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()) {
                if let Some(ref last_img) = last_processed_frame {
                    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                    let name = format!("Person_{}", ts);
                    let folder = PathBuf::from("known_faces").join(&name);
                    std::fs::create_dir_all(&folder)?;
                    
                    let (scale, dx, dy) = utils::get_scale_params(last_img.width(), last_img.height(), 640, 640);
                    let x = (((det.bbox.0 - dx) / scale).max(0.0) as u32).min(last_img.width()-1);
                    let y = (((det.bbox.1 - dy) / scale).max(0.0) as u32).min(last_img.height()-1);
                    let w = ((det.bbox.2 / scale) as u32).min(last_img.width() - x);
                    let h = ((det.bbox.3 / scale) as u32).min(last_img.height() - y);
                    
                    let face_crop = image::imageops::crop_imm(last_img, x, y, w, h).to_image();
                    face_crop.save(folder.join("face.jpg"))?;
                    
                    if let Ok(emb) = recog_model.get_embedding(&face_crop) {
                        known.push(KnownPerson { name: name.clone(), embeddings: vec![emb] });
                        println!("Enrolled new person: {}", name);
                    }
                }
            }
        }

        let (display_buffer, num_faces, _, current_dets) = process_frame(&mut rgb_image, face_model, pose_model, recog_model, known)?;
        last_processed_frame = Some(rgb_image.clone());
        last_detections = current_dets;

        window.update_with_buffer(&display_buffer, width as usize, height as usize)?;
    }
    Ok(())
}

fn run_image(path: &str, face_model: &mut YoloModel, pose_model: &mut YoloModel, recog_model: &mut FaceRecognizer, known: &[KnownPerson]) -> Result<()> {
    let img = image::open(path)?;
    let mut rgb_image = img.to_rgb8();
    let (display_buffer, _, _, _) = process_frame(&mut rgb_image, face_model, pose_model, recog_model, known)?;
    // (Optional: Convert buffer back or just save rgb_image since process_frame modifies it)
    rgb_image.save("output.jpg")?;
    println!("Saved result to output.jpg");
    Ok(())
}

fn process_frame(
    img: &mut image::RgbImage,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
    recog_model: &mut FaceRecognizer,
    known: &[KnownPerson]
) -> Result<(Vec<u32>, usize, usize, Vec<Detection>)> {
    let (input, scale, dx, dy) = utils::preprocess_yolo(&DynamicImage::ImageRgb8(img.clone()), (640, 640));

    let face_dets = face_model.detect(input.clone(), 0.4)?;
    let pose_dets = pose_model.detect(input, 0.4)?;

    let mut face_boxes = Vec::new();
    for (i, det) in face_dets.iter().enumerate() {
        face_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_faces = utils::nms(&face_boxes, 0.45);

    let mut processed_face_dets = Vec::new();

    for &idx in &keep_faces {
        let det = &face_dets[idx];
        processed_face_dets.push(det.clone());
        let x_raw = ((det.bbox.0 - dx) / scale).max(0.0);
        let y_raw = ((det.bbox.1 - dy) / scale).max(0.0);
        let w_box = det.bbox.2 / scale;
        let h_box = det.bbox.3 / scale;
        
        let mut name = "Unknown".to_string();
        let x = (x_raw as u32).min(img.width()-1);
        let y = (y_raw as u32).min(img.height()-1);
        let w = (w_box as u32).min(img.width() - x);
        let h = (h_box as u32).min(img.height() - y);

        if w > 0 && h > 0 && !known.is_empty() {
            let face_crop = image::imageops::crop_imm(img, x, y, w, h).to_image();
            if let Ok(emb) = recog_model.get_embedding(&face_crop) {
                let mut best_score = 0.0;
                for person in known {
                    for person_emb in &person.embeddings {
                        let score = models::cosine_similarity(&emb, person_emb);
                        if score > best_score {
                            best_score = score;
                            if score > 0.45 { name = format!("{} ({:.2})", person.name, score); }
                        }
                    }
                }
            }
        }

        utils::draw_bbox(img, (x_raw, y_raw, w_box, h_box), Rgb([0, 255, 0]), Some(&name));
        
        for (i, (kx, ky, _kc)) in det.keypoints.iter().enumerate() {
            let color = match i { 0|1 => Rgb([0,255,255]), 2 => Rgb([255,255,0]), 3|4 => Rgb([255,0,255]), _ => Rgb([255,0,0]) };
            imageproc::drawing::draw_filled_circle_mut(img, (((kx - dx)/scale) as i32, ((ky - dy)/scale) as i32), 3, color);
        }
        
        if det.keypoints.len() >= 5 {
            let m1 = &det.keypoints[3]; let m2 = &det.keypoints[4];
            imageproc::drawing::draw_line_segment_mut(img, ((m1.0 - dx)/scale, (m1.1 - dy)/scale), ((m2.0 - dx)/scale, (m2.1 - dy)/scale), Rgb([255,0,255]));
        }
    }

    let mut pose_boxes = Vec::new();
    for (i, det) in pose_dets.iter().enumerate() {
        pose_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_pose = utils::nms(&pose_boxes, 0.45);
    for &idx in &keep_pose {
        draw_skeleton(img, &pose_dets[idx].keypoints, dx, dy, scale);
    }

    let buffer: Vec<u32> = img.as_raw().par_chunks_exact(3).map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)).collect();
    Ok((buffer, keep_faces.len(), keep_pose.len(), processed_face_dets))
}

const SKELETON: [(usize, usize); 12] = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),(8,10)];
fn draw_skeleton(img: &mut RgbImage, kpts: &Vec<(f32,f32,f32)>, dx: f32, dy: f32, scale: f32) {
    for &(i,j) in &SKELETON {
        if i < kpts.len() && j < kpts.len() {
            let (x1,y1,c1) = kpts[i]; let (x2,y2,c2) = kpts[j];
            if c1 > 0.4 && c2 > 0.4 {
                imageproc::drawing::draw_line_segment_mut(img, ((x1-dx)/scale, (y1-dy)/scale), ((x2-dx)/scale, (y2-dy)/scale), Rgb([0,100,255]));
            }
        }
    }
}
