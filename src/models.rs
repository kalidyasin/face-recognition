use anyhow::Result;
use ndarray::{Array, Array1, Array4};
use ort::session::Session;
use ort::value::Tensor;

pub struct YoloModel {
    session: Session,
}

impl YoloModel {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("  - Building session for {}...", model_path);
        let session = Session::builder()?.commit_from_file(model_path)?;
        println!("  - Session built.");
        Ok(Self { session })
    }

    pub fn detect(
        &mut self,
        input_tensor: Array<f32, ndarray::Dim<[usize; 4]>>,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let tensor = Tensor::from_array(input_tensor)?;
        // Use positional inputs for robustness if names are uncertain, 
        // but YOLOv8 exports usually name it "images"
        let inputs = ort::inputs!["images" => tensor];
        let outputs = self.session.run(inputs)?;

        let mut detections = Vec::new();

        if outputs.len() > 1 {
            for (_name, value) in outputs.iter() {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                if dims.len() == 4 && dims[1] == 80 {
                    let h = dims[2];
                    let w = dims[3];
                    let stride = 640 / h;
                    Self::decode_head(data, w, h, stride, conf_threshold, &mut detections);
                } else if dims.len() == 3 && dims[1] == 56 {
                    Self::decode_concatenated(
                        data,
                        dims[1],
                        dims[2],
                        conf_threshold,
                        &mut detections,
                    );
                }
            }
        } else {
            let value = outputs.values().next().unwrap();
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            Self::decode_concatenated(data, dims[1], dims[2], conf_threshold, &mut detections);
        }

        Ok(detections)
    }

    fn decode_head(
        data: &[f32],
        w: usize,
        h: usize,
        stride: usize,
        conf_threshold: f32,
        detections: &mut Vec<Detection>,
    ) {
        for y in 0..h {
            for x in 0..w {
                let offset = y * w + x;
                let score_raw = data[64 * h * w + offset];
                let score = sigmoid(score_raw);

                if score > conf_threshold {
                    let mut dists = [0.0f32; 4];
                    for i in 0..4 {
                        let mut softmax_sum = 0.0;
                        let mut weighted_sum = 0.0;
                        let mut max_val = -f32::INFINITY;
                        for j in 0..16 {
                            let val = data[(i * 16 + j) * h * w + offset];
                            if val > max_val { max_val = val; }
                        }
                        for j in 0..16 {
                            let exp_val = (data[(i * 16 + j) * h * w + offset] - max_val).exp();
                            softmax_sum += exp_val;
                            weighted_sum += exp_val * (j as f32);
                        }
                        dists[i] = weighted_sum / softmax_sum;
                    }

                    let cx = x as f32 + 0.5;
                    let cy = y as f32 + 0.5;
                    let x1 = (cx - dists[0]) * (stride as f32);
                    let y1 = (cy - dists[1]) * (stride as f32);
                    let x2 = (cx + dists[2]) * (stride as f32);
                    let y2 = (cy + dists[3]) * (stride as f32);

                    let mut keypoints = Vec::new();
                    for i in 0..5 {
                        let px_raw = data[(65 + i * 3) * h * w + offset];
                        let py_raw = data[(66 + i * 3) * h * w + offset];
                        let ps_raw = data[(67 + i * 3) * h * w + offset];

                        let lx = (px_raw * 2.0 + x as f32) * (stride as f32);
                        let ly = (py_raw * 2.0 + y as f32) * (stride as f32);
                        let ls = sigmoid(ps_raw);
                        keypoints.push((lx, ly, ls));
                    }

                    detections.push(Detection {
                        bbox: (x1, y1, x2 - x1, y2 - y1),
                        score,
                        keypoints,
                    });
                }
            }
        }
    }

    fn decode_concatenated(
        data: &[f32],
        channels: usize,
        anchors: usize,
        conf_threshold: f32,
        detections: &mut Vec<Detection>,
    ) {
        for i in 0..anchors {
            let score = data[4 * anchors + i];
            if score > conf_threshold {
                let cx = data[0 * anchors + i];
                let cy = data[1 * anchors + i];
                let w = data[2 * anchors + i];
                let h = data[3 * anchors + i];
                let x1 = cx - w / 2.0;
                let y1 = cy - h / 2.0;

                let mut keypoints = Vec::new();
                if channels > 5 {
                    let kpt_start = 5;
                    let step = if channels == 56 { 3 } else { 2 };
                    for k in 0..((channels - kpt_start) / step) {
                        let c_idx = kpt_start + k * step;
                        let kx = data[(c_idx) * anchors + i];
                        let ky = data[(c_idx + 1) * anchors + i];
                        let kc = if step == 3 {
                            data[(c_idx + 2) * anchors + i]
                        } else {
                            1.0
                        };
                        keypoints.push((kx, ky, kc));
                    }
                }

                detections.push(Detection {
                    bbox: (x1, y1, w, h),
                    score,
                    keypoints,
                });
            }
        }
    }
}

pub struct FaceRecognizer {
    session: Session,
}

impl FaceRecognizer {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("  - Building session for {}...", model_path);
        let session = Session::builder()?.commit_from_file(model_path)?;
        println!("  - Session built.");
        Ok(Self { session })
    }

    pub fn get_embedding(&mut self, face_img: &image::RgbImage) -> Result<Array1<f32>> {
        let resized = image::imageops::resize(face_img, 112, 112, image::imageops::FilterType::Triangle);
        let mut input = Array4::zeros((1, 3, 112, 112));
        for (x, y, pixel) in resized.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            // Preprocessing: BGR and (pixel - 127.5) / 128.0
            input[[0, 0, y as usize, x as usize]] = (b as f32 - 127.5) / 128.0;
            input[[0, 1, y as usize, x as usize]] = (g as f32 - 127.5) / 128.0;
            input[[0, 2, y as usize, x as usize]] = (r as f32 - 127.5) / 128.0;
        }
        let tensor = Tensor::from_array(input)?;
        
        // Auto-detect input name for the recognition model
        let input_name = self.session.inputs()[0].name().to_string();
        let inputs: Vec<(String, ort::session::SessionInputValue)> = vec![(input_name, tensor.into())];
        
        let outputs = self.session.run(inputs)?;
        let output_value = outputs.values().next().unwrap();
        let (_, data) = output_value.try_extract_tensor::<f32>()?;
        
        let mut embedding = Array1::from_vec(data.to_vec());
        let norm = embedding.dot(&embedding).sqrt();
        embedding /= norm.max(1e-6);
        Ok(embedding)
    }
}

pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.dot(b)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: (f32, f32, f32, f32),
    pub score: f32,
    pub keypoints: Vec<(f32, f32, f32)>,
}
