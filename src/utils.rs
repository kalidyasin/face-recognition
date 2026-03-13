use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_line_segment_mut, draw_hollow_rect_mut};
use ndarray::{Array, Array4};

// Pre-process image for YOLO (Letterbox resize - Centered)
pub fn preprocess_yolo(img: &DynamicImage, target_size: (u32, u32)) -> (Array4<f32>, f32, f32, f32) {
    let (w, h) = img.dimensions();
    let (tw, th) = target_size;

    let scale = (tw as f32 / w as f32).min(th as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let resized_rgb = resized.to_rgb8();
    
    let mut canvas = ImageBuffer::from_pixel(tw, th, Rgb([114, 114, 114]));
    
    let dx = (tw - new_w) / 2;
    let dy = (th - new_h) / 2;
    
    image::imageops::overlay(&mut canvas, &resized_rgb, dx as i64, dy as i64);

    let mut input = Array::zeros((1, 3, th as usize, tw as usize));
    for (x, y, pixel) in canvas.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        input[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
        input[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
        input[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
    }

    (input, scale, dx as f32, dy as f32)
}

// Draw bounding box
pub fn draw_bbox(img: &mut RgbImage, rect: (f32, f32, f32, f32), color: Rgb<u8>, _label: Option<&str>) {
    let (x, y, w, h) = rect;
    let rw = w.round() as u32;
    let rh = h.round() as u32;
    if rw == 0 || rh == 0 {
        return;
    }
    // Ensure coordinates are within bounds
    let rect_struct = imageproc::rect::Rect::at(x as i32, y as i32).of_size(rw, rh);
    draw_hollow_rect_mut(img, rect_struct, color);
}

// Simple IoU
pub fn iou(box1: &(f32, f32, f32, f32), box2: &(f32, f32, f32, f32)) -> f32 {
    let (x1, y1, w1, h1) = box1;
    let (x2, y2, w2, h2) = box2;

    let xi1 = x1.max(*x2);
    let yi1 = y1.max(*y2);
    let xi2 = (x1 + w1).min(x2 + w2);
    let yi2 = (y1 + h1).min(y2 + h2);

    let inter_w = (xi2 - xi1).max(0.0);
    let inter_h = (yi2 - yi1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union_area = area1 + area2 - inter_area;

    if union_area == 0.0 { 0.0 } else { inter_area / union_area }
}

// NMS
pub fn nms(boxes: &Vec<(f32, f32, f32, f32, f32, usize)>, iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].4.partial_cmp(&boxes[a].4).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = Vec::new();
    while !indices.is_empty() {
        let current = indices[0];
        keep.push(current);
        // Only keep boxes that have IoU <= threshold
        let mut next_indices = Vec::new();
        for &idx in indices.iter().skip(1) {
             let box1 = (boxes[current].0, boxes[current].1, boxes[current].2, boxes[current].3);
             let box2 = (boxes[idx].0, boxes[idx].1, boxes[idx].2, boxes[idx].3);
             if iou(&box1, &box2) <= iou_threshold {
                 next_indices.push(idx);
             }
        }
        indices = next_indices;
    }
    keep
}
