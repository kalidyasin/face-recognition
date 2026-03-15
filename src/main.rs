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
use std::time::{UNIX_EPOCH, Instant, Duration, SystemTime};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod models;
mod utils;
mod database;

use models::{YoloModel, FaceRecognizer, Detection};
use database::FaceDb;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Realtime {
        #[arg(long, default_value = "0")]
        camera_index: u32,
    },
    Image { path: String },
}

const FACE_MODEL_URL: &str = "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx";
const POSE_MODEL_URL: &str = "https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov8n-pose.onnx";
const RECOG_MODEL_URL: &str = "https://huggingface.co/LPDoctor/insightface/resolve/main/models/antelopev2/glintr100.onnx?download=true";

#[derive(Clone)]
struct KnownPerson {
    name: String,
    embeddings: Vec<Array1<f32>>,
}

struct AiState {
    face_dets: Vec<Detection>,
    pose_dets: Vec<Detection>,
    recognitions: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let face_model_path = "yolov8n-face.onnx";
    let pose_model_path = "yolov8n-pose.onnx";
    let recog_model_path = "arcface.onnx";

    check_download(FACE_MODEL_URL, face_model_path)?;
    check_download(POSE_MODEL_URL, pose_model_path)?;
    check_download(RECOG_MODEL_URL, recog_model_path)?;

    let db = Arc::new(FaceDb::open()?);
    let mut face_model = YoloModel::new(face_model_path)?;
    let mut pose_model = YoloModel::new(pose_model_path)?;
    let mut recog_model = FaceRecognizer::new(recog_model_path)?;

    println!("Loading database...");
    let known_people = load_known_people(&mut recog_model, &db)?;
    println!("Total People: {}", known_people.len());

    match cli.command {
        Commands::Realtime { camera_index } => {
            run_realtime(camera_index, face_model, pose_model, recog_model, known_people)?;
        }
        Commands::Image { path } => {
            run_image(&path, &mut face_model, &mut pose_model, &mut recog_model, &known_people)?;
        }
    }

    Ok(())
}

fn check_download(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Downloading {}...", path);
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(600)).build()?;
        let mut response = client.get(url).send()?;
        if !response.status().is_success() { return Err(anyhow::anyhow!("Download failed: {}", response.status())); }
        let mut file = std::fs::File::create(path)?;
        std::io::copy(&mut response, &mut file)?;
        println!("  - Finished. Size: {} MB", std::fs::metadata(path)?.len() / 1024 / 1024);
    }
    Ok(())
}

fn load_known_people(recog_model: &mut FaceRecognizer, db: &FaceDb) -> Result<Vec<KnownPerson>> {
    let base_dir = Path::new("known_faces");
    if !base_dir.exists() { std::fs::create_dir_all(base_dir)?; return Ok(vec![]); }

    let mut people_map: HashMap<String, Vec<Array1<f32>>> = HashMap::new();

    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        let mut process_file = |p: PathBuf, name: String| -> Result<()> {
            let mtime = std::fs::metadata(&p)?.modified()?.duration_since(UNIX_EPOCH)?.as_secs() as i64;
            
            if let Some(emb) = db.get_cached_embedding(&p, mtime)? {
                people_map.entry(name).or_default().push(emb);
            } else if let Ok(img) = image::open(&p) {
                let emb = recog_model.get_embedding(&img.to_rgb8())?;
                db.save_embedding(&p, mtime, &emb)?;
                people_map.entry(name).or_default().push(emb);
            }
            Ok(())
        };

        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            for img_entry in std::fs::read_dir(&path)? {
                let p = img_entry?.path();
                if is_image(&p) { let _ = process_file(p, name.clone()); }
            }
        } else if is_image(&path) {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let _ = process_file(path, name);
        }
    }
    Ok(people_map.into_iter().map(|(name, embeddings)| KnownPerson { name, embeddings }).collect())
}

fn is_image(path: &Path) -> bool {
    path.is_file() && (path.extension().map_or(false, |e| e=="jpg" || e=="png" || e=="jpeg"))
}

fn run_realtime(index: u32, mut face_model: YoloModel, mut pose_model: YoloModel, mut recog_model: FaceRecognizer, known: Vec<KnownPerson>) -> Result<()> {
    let index = CameraIndex::Index(index);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;

    let (width, height) = (camera.resolution().width_x, camera.resolution().height_y);
    let mut window = Window::new("AI Tracker - Debugging Scores", width as usize, height as usize, WindowOptions::default())?;
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let ai_state = Arc::new(Mutex::new(AiState { face_dets: vec![], pose_dets: vec![], recognitions: vec![] }));
    let camera_frame = Arc::new(Mutex::new(RgbImage::new(1, 1)));
    let running = Arc::new(Mutex::new(true));

    let ai_state_thread = ai_state.clone();
    let camera_frame_thread = camera_frame.clone();
    let running_thread = running.clone();
    std::thread::spawn(move || {
        while *running_thread.lock().unwrap() {
            let frame = camera_frame_thread.lock().unwrap().clone();
            if frame.width() < 10 { std::thread::sleep(Duration::from_millis(10)); continue; }

            if let Ok((face_raw, pose_raw, recognitions)) = analyze_frame_fast(&frame, &mut face_model, &mut pose_model, &mut recog_model, &known) {
                let mut state = ai_state_thread.lock().unwrap();
                state.face_dets = face_raw;
                state.pose_dets = pose_raw;
                state.recognitions = recognitions;
            }
        }
    });

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.frame()?;
        let rgb_image = frame.decode_image::<RgbFormat>()?;
        *camera_frame.lock().unwrap() = rgb_image.clone();

        let mut display_img = rgb_image;
        let state = ai_state.lock().unwrap();
        
        let (display_buffer, _, _) = draw_overlay(&mut display_img, &state.face_dets, &state.pose_dets, &state.recognitions)?;
        window.update_with_buffer(&display_buffer, width as usize, height as usize)?;
    }
    
    *running.lock().unwrap() = false;
    Ok(())
}

fn analyze_frame_fast(
    img: &image::RgbImage,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
    recog_model: &mut FaceRecognizer,
    known: &[KnownPerson]
) -> Result<(Vec<Detection>, Vec<Detection>, Vec<String>)> {
    let (input, scale, dx, dy) = utils::preprocess_yolo(&DynamicImage::ImageRgb8(img.clone()), (640, 640));
    let face_raw_dets = face_model.detect(input.clone(), 0.4)?;
    let pose_raw_dets = pose_model.detect(input, 0.4)?;

    let mut face_boxes = Vec::new();
    for (i, det) in face_raw_dets.iter().enumerate() { face_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i)); }
    let keep_faces = utils::nms(&face_boxes, 0.45);
    
    let mut face_dets = Vec::new();
    let mut recognitions = Vec::new();

    for &idx in &keep_faces {
        let det = &face_raw_dets[idx];
        face_dets.push(det.clone());
        
        let mut name = "Unknown".to_string();
        let x = (((det.bbox.0 - dx) / scale).max(0.0) as u32).min(img.width()-1);
        let y = (((det.bbox.1 - dy) / scale).max(0.0) as u32).min(img.height()-1);
        let w = ((det.bbox.2 / scale) as u32).min(img.width() - x);
        let h = ((det.bbox.3 / scale) as u32).min(img.height() - y);

        if w > 10 && h > 10 {
            let face_crop = image::imageops::crop_imm(img, x, y, w, h).to_image();
            if let Ok(emb) = recog_model.get_embedding(&face_crop) {
                let mut best_score = -1.0;
                let mut best_name = "Unknown".to_string();
                
                for person in known {
                    for person_emb in &person.embeddings {
                        let score = models::cosine_similarity(&emb, person_emb);
                        if score > best_score { 
                            best_score = score;
                            best_name = person.name.clone();
                        }
                    }
                }

                if best_score > 0.45 {
                    name = format!("{} ({:.2})", best_name, best_score);
                } else if best_score > 0.30 {
                    name = format!("{}? ({:.2})", best_name, best_score);
                } else {
                    name = "Unknown".to_string();
                }

                // Log the winning match to terminal
                if best_score > 0.25 {
                    println!("Winner: {} [{:.4}]", best_name, best_score);
                }
            }
        }
        recognitions.push(name);
    }

    let mut pose_boxes = Vec::new();
    for (i, det) in pose_raw_dets.iter().enumerate() { pose_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i)); }
    let keep_pose = utils::nms(&pose_boxes, 0.45);
    let mut pose_dets = Vec::new();
    for &idx in &keep_pose { pose_dets.push(pose_raw_dets[idx].clone()); }

    Ok((face_dets, pose_dets, recognitions))
}

fn draw_overlay(img: &mut RgbImage, face_dets: &Vec<Detection>, pose_dets: &Vec<Detection>, recognitions: &Vec<String>) -> Result<(Vec<u32>, usize, usize)> {
    let (scale, dx, dy) = utils::get_scale_params(img.width(), img.height(), 640, 640);
    for (idx, det) in face_dets.iter().enumerate() {
        let x = ((det.bbox.0 - dx) / scale).max(0.0);
        let y = ((det.bbox.1 - dy) / scale).max(0.0);
        let w = det.bbox.2 / scale; let h = det.bbox.3 / scale;
        utils::draw_bbox(img, (x, y, w, h), Rgb([0, 255, 0]), recognitions.get(idx).map(|s| s.as_str()));
        for (i, (kx, ky, _kc)) in det.keypoints.iter().enumerate() {
            let color = match i { 0|1 => Rgb([0,255,255]), 2 => Rgb([255,255,0]), 3|4 => Rgb([255,0,255]), _ => Rgb([255,0,0]) };
            imageproc::drawing::draw_filled_circle_mut(img, (((kx - dx)/scale) as i32, ((ky - dy)/scale) as i32), 3, color);
        }
    }
    for det in pose_dets { draw_skeleton(img, &det.keypoints, dx, dy, scale); }
    let buffer: Vec<u32> = img.as_raw().par_chunks_exact(3).map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)).collect();
    Ok((buffer, face_dets.len(), pose_dets.len()))
}

fn draw_skeleton(img: &mut RgbImage, kpts: &Vec<(f32,f32,f32)>, dx: f32, dy: f32, scale: f32) {
    let skeleton = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),(8,10)];
    for &(i,j) in &skeleton {
        if i < kpts.len() && j < kpts.len() {
            let (x1,y1,c1) = kpts[i]; let (x2,y2,c2) = kpts[j];
            if c1 > 0.4 && c2 > 0.4 {
                imageproc::drawing::draw_line_segment_mut(img, ((x1-dx)/scale, (y1-dy)/scale), ((x2-dx)/scale, (y2-dy)/scale), Rgb([0,100,255]));
            }
        }
    }
    for (idx, (kx, ky, kc)) in kpts.iter().enumerate() {
        if *kc > 0.4 && idx > 4 {
            imageproc::drawing::draw_filled_circle_mut(img, (((kx - dx) / scale) as i32, ((ky - dy) / scale) as i32), 3, Rgb([255, 255, 255]));
        }
    }
}

fn run_image(path: &str, face_model: &mut YoloModel, pose_model: &mut YoloModel, recog_model: &mut FaceRecognizer, known: &[KnownPerson]) -> Result<()> {
    let img = image::open(path)?;
    let mut rgb_image = img.to_rgb8();
    let (face_dets, pose_dets, recognitions) = analyze_frame_fast(&rgb_image, face_model, pose_model, recog_model, known)?;
    let _ = draw_overlay(&mut rgb_image, &face_dets, &pose_dets, &recognitions)?;
    rgb_image.save("output.jpg")?;
    println!("Saved result to output.jpg");
    Ok(())
}
