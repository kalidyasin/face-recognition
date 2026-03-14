use anyhow::Result;
use clap::{Parser, Subcommand};
use image::{DynamicImage, Rgb, RgbImage};
use minifb::{Key, Window, WindowOptions};
use ndarray::Array1;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

mod models;
mod utils;

use models::{Detection, FaceRecognizer, YoloModel};

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
const POSE_MODEL_URL: &str =
    "https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov8n-pose.onnx";
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

    println!("Loading known faces...");
    let mut known_people = load_known_people(&mut recog_model)?;
    println!("Total People in Database: {}", known_people.len());

    match cli.command {
        Commands::Realtime { camera_index } => {
            run_realtime(
                camera_index,
                &mut face_model,
                &mut pose_model,
                &mut recog_model,
                &mut known_people,
            )?;
        }
        Commands::Image { path } => {
            run_image(
                &path,
                &mut face_model,
                &mut pose_model,
                &mut recog_model,
                &known_people,
            )?;
        }
    }

    Ok(())
}

fn check_download(url: &str, path: &str) -> Result<()> {
    if !Path::new(path).exists() {
        println!("Downloading {} (this may take a while)...", path);
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;
        let mut response = client.get(url).send()?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Download failed: {}", response.status()));
        }
        let mut file = std::fs::File::create(path)?;
        std::io::copy(&mut response, &mut file)?;
        println!(
            "  - Finished. Size: {} MB",
            std::fs::metadata(path)?.len() / 1024 / 1024
        );
    }
    Ok(())
}

fn load_known_people(recog_model: &mut FaceRecognizer) -> Result<Vec<KnownPerson>> {
    let base_dir = Path::new("known_faces");
    if !base_dir.exists() {
        std::fs::create_dir_all(base_dir)?;
        return Ok(vec![]);
    }

    let mut people_map: HashMap<String, Vec<Array1<f32>>> = HashMap::new();

    for entry in std::fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            for img_entry in std::fs::read_dir(&path)? {
                let img_path = img_entry?.path();
                if is_image(&img_path) {
                    if let Ok(img) = image::open(&img_path) {
                        if let Ok(emb) = recog_model.get_embedding(&img.to_rgb8()) {
                            people_map.entry(name.clone()).or_default().push(emb);
                        }
                    }
                }
            }
        } else if is_image(&path) {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            if let Ok(img) = image::open(&path) {
                if let Ok(emb) = recog_model.get_embedding(&img.to_rgb8()) {
                    people_map.entry(name).or_default().push(emb);
                }
            }
        }
    }

    Ok(people_map
        .into_iter()
        .map(|(name, embeddings)| KnownPerson { name, embeddings })
        .collect())
}

fn is_image(path: &Path) -> bool {
    path.is_file()
        && (path
            .extension()
            .map_or(false, |e| e == "jpg" || e == "png" || e == "jpeg"))
}

fn run_realtime(
    index: u32,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
    recog_model: &mut FaceRecognizer,
    known: &mut Vec<KnownPerson>,
) -> Result<()> {
    let index = CameraIndex::Index(index);
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;

    let width = camera.resolution().width_x;
    let height = camera.resolution().height_y;

    let mut window = Window::new(
        "AI Tracker - [R] Reload",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )?;
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut unknown_trackers: HashMap<usize, (Instant, RgbImage, Array1<f32>)> = HashMap::new();
    let mut last_enrollment_time = Instant::now();

    // Performance Cache: Only run recognition every N frames
    let mut frame_count = 0;
    let mut recog_cache: Vec<String> = Vec::new();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame = camera.frame()?;
        let mut rgb_image = frame.decode_image::<RgbFormat>()?;

        if window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            if let Ok(new_known) = load_known_people(recog_model) {
                *known = new_known;
                println!("Database reloaded. People: {}", known.len());
            }
        }

        // Run Recognition every 3 frames to reduce lag
        let run_recog = frame_count % 3 == 0;
        let (face_dets, pose_dets, recognitions) = analyze_frame(
            &rgb_image,
            face_model,
            pose_model,
            recog_model,
            known,
            run_recog,
            &recog_cache,
        )?;

        if run_recog {
            recog_cache = recognitions.iter().map(|(name, _)| name.clone()).collect();
        }

        // Auto-Enrollment Logic
        let mut active_indices = Vec::new();
        for (idx, (name, emb)) in recognitions.iter().enumerate() {
            if name == "Unknown" {
                active_indices.push(idx);
                let entry = unknown_trackers
                    .entry(idx)
                    .or_insert_with(|| (Instant::now(), RgbImage::new(1, 1), emb.clone()));
                if let Some(det) = face_dets.get(idx) {
                    let (scale, dx, dy) =
                        utils::get_scale_params(rgb_image.width(), rgb_image.height(), 640, 640);
                    let x =
                        (((det.bbox.0 - dx) / scale).max(0.0) as u32).min(rgb_image.width() - 1);
                    let y =
                        (((det.bbox.1 - dy) / scale).max(0.0) as u32).min(rgb_image.height() - 1);
                    let w = ((det.bbox.2 / scale) as u32).min(rgb_image.width() - x);
                    let h = ((det.bbox.3 / scale) as u32).min(rgb_image.height() - y);
                    if w > 20 && h > 20 {
                        entry.1 = image::imageops::crop_imm(&rgb_image, x, y, w, h).to_image();
                    }
                }

                if entry.0.elapsed() > Duration::from_millis(3000)
                    && last_enrollment_time.elapsed() > Duration::from_millis(5000)
                {
                    let mut really_unknown = true;
                    for p in known.iter() {
                        for p_emb in &p.embeddings {
                            if models::cosine_similarity(emb, p_emb) > 0.35 {
                                really_unknown = false;
                                break;
                            }
                        }
                    }
                    if really_unknown {
                        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                        let new_name = format!("Person_{}", ts);
                        let folder = PathBuf::from("known_faces").join(&new_name);
                        std::fs::create_dir_all(&folder)?;
                        entry.1.save(folder.join("auto_enrolled.jpg"))?;
                        known.push(KnownPerson {
                            name: new_name.clone(),
                            embeddings: vec![emb.clone()],
                        });
                        println!("Auto-Enrolled: {}", new_name);
                        last_enrollment_time = Instant::now();
                        unknown_trackers.clear();
                        break;
                    }
                }
            }
        }
        unknown_trackers.retain(|k, _| active_indices.contains(k));

        let (display_buffer, nf, np) =
            draw_overlay(&mut rgb_image, &face_dets, &pose_dets, &recognitions)?;
        window.set_title(&format!(
            "Faces: {} | Bodies: {} | Database: {}",
            nf,
            np,
            known.len()
        ));
        window.update_with_buffer(&display_buffer, width as usize, height as usize)?;
        frame_count += 1;
    }
    Ok(())
}

fn run_image(
    path: &str,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
    recog_model: &mut FaceRecognizer,
    known: &[KnownPerson],
) -> Result<()> {
    let img = image::open(path)?;
    let mut rgb_image = img.to_rgb8();
    let (face_dets, pose_dets, recognitions) = analyze_frame(
        &rgb_image,
        face_model,
        pose_model,
        recog_model,
        known,
        true,
        &vec![],
    )?;
    let _ = draw_overlay(&mut rgb_image, &face_dets, &pose_dets, &recognitions)?;
    rgb_image.save("output.jpg")?;
    println!("Saved result to output.jpg");
    Ok(())
}

fn analyze_frame(
    img: &image::RgbImage,
    face_model: &mut YoloModel,
    pose_model: &mut YoloModel,
    recog_model: &mut FaceRecognizer,
    known: &[KnownPerson],
    run_recog: bool,
    cached_names: &Vec<String>,
) -> Result<(Vec<Detection>, Vec<Detection>, Vec<(String, Array1<f32>)>)> {
    let (input, scale, dx, dy) =
        utils::preprocess_yolo(&DynamicImage::ImageRgb8(img.clone()), (640, 640));

    let face_raw_dets = face_model.detect(input.clone(), 0.4)?;
    let pose_raw_dets = pose_model.detect(input, 0.4)?;

    let mut face_boxes = Vec::new();
    for (i, det) in face_raw_dets.iter().enumerate() {
        face_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_faces = utils::nms(&face_boxes, 0.45);
    let mut face_dets = Vec::new();
    let mut recognitions = Vec::new();

    for (idx_in_keep, &idx) in keep_faces.iter().enumerate() {
        let det = &face_raw_dets[idx];
        face_dets.push(det.clone());

        let mut name = "Unknown".to_string();
        let mut current_emb = Array1::zeros(512);

        if run_recog {
            let x_raw = ((det.bbox.0 - dx) / scale).max(0.0);
            let y_raw = ((det.bbox.1 - dy) / scale).max(0.0);
            let w_box = det.bbox.2 / scale;
            let h_box = det.bbox.3 / scale;
            let x = (x_raw as u32).min(img.width() - 1);
            let y = (y_raw as u32).min(img.height() - 1);
            let w = (w_box as u32).min(img.width() - x);
            let h = (h_box as u32).min(img.height() - y);

            if w > 10 && h > 10 {
                let face_crop = image::imageops::crop_imm(img, x, y, w, h).to_image();
                if let Ok(emb) = recog_model.get_embedding(&face_crop) {
                    current_emb = emb.clone();
                    let mut best_score = 0.0;
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
                    if best_score > 0.35 {
                        name = format!("{} ({:.2})", best_name, best_score);
                    }
                }
            }
        } else if let Some(cached) = cached_names.get(idx_in_keep) {
            name = cached.clone();
        }
        recognitions.push((name, current_emb));
    }

    let mut pose_boxes = Vec::new();
    for (i, det) in pose_raw_dets.iter().enumerate() {
        pose_boxes.push((det.bbox.0, det.bbox.1, det.bbox.2, det.bbox.3, det.score, i));
    }
    let keep_pose = utils::nms(&pose_boxes, 0.45);
    let mut pose_dets = Vec::new();
    for &idx in &keep_pose {
        pose_dets.push(pose_raw_dets[idx].clone());
    }

    Ok((face_dets, pose_dets, recognitions))
}

fn draw_overlay(
    img: &mut image::RgbImage,
    face_dets: &Vec<Detection>,
    pose_dets: &Vec<Detection>,
    recognitions: &Vec<(String, Array1<f32>)>,
) -> Result<(Vec<u32>, usize, usize)> {
    let (scale, dx, dy) = utils::get_scale_params(img.width(), img.height(), 640, 640);

    for (idx, det) in face_dets.iter().enumerate() {
        let x = ((det.bbox.0 - dx) / scale).max(0.0);
        let y = ((det.bbox.1 - dy) / scale).max(0.0);
        let w = det.bbox.2 / scale;
        let h = det.bbox.3 / scale;
        let name = &recognitions[idx].0;

        utils::draw_bbox(img, (x, y, w, h), Rgb([0, 255, 0]), Some(name));

        for (i, (kx, ky, _kc)) in det.keypoints.iter().enumerate() {
            let color = match i {
                0 | 1 => Rgb([0, 255, 255]),
                2 => Rgb([255, 255, 0]),
                3 | 4 => Rgb([255, 0, 255]),
                _ => Rgb([255, 0, 0]),
            };
            imageproc::drawing::draw_filled_circle_mut(
                img,
                (((kx - dx) / scale) as i32, ((ky - dy) / scale) as i32),
                3,
                color,
            );
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

    for det in pose_dets {
        draw_skeleton(img, &det.keypoints, dx, dy, scale);
    }

    let buffer: Vec<u32> = img
        .as_raw()
        .par_chunks_exact(3)
        .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32))
        .collect();
    Ok((buffer, face_dets.len(), pose_dets.len()))
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
fn draw_skeleton(img: &mut RgbImage, kpts: &Vec<(f32, f32, f32)>, dx: f32, dy: f32, scale: f32) {
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
    // RESTORED: Draw dots (joints) for the body tracking
    for (idx, (kx, ky, kc)) in kpts.iter().enumerate() {
        if *kc > 0.4 && idx > 4 {
            // Skip face points from pose model to avoid clutter
            imageproc::drawing::draw_filled_circle_mut(
                img,
                (((kx - dx) / scale) as i32, ((ky - dy) / scale) as i32),
                3,
                Rgb([255, 255, 255]),
            );
        }
    }
}
