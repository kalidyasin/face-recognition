use anyhow::Result;
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

pub struct YoloModel {
    session: Session,
    input_shape: (u32, u32),
}

impl YoloModel {
    pub fn new(model_path: &str) -> Result<Self> {
        println!("  - Building session for {}...", model_path);
        let session = Session::builder()?.commit_from_file(model_path)?;
        println!("  - Session built.");
        Ok(Self {
            session,
            input_shape: (640, 640),
        })
    }

    pub fn detect(
        &mut self,
        input_tensor: Array<f32, ndarray::Dim<[usize; 4]>>,
        conf_threshold: f32,
    ) -> Result<Vec<Detection>> {
        let tensor = Tensor::from_array(input_tensor)?;
        let inputs = ort::inputs!["images" => tensor];
        let outputs = self.session.run(inputs)?;

        let mut detections = Vec::new();

        // Check if we have multiple outputs (raw heads) or a single concatenated one
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
                            if val > max_val {
                                max_val = val;
                            }
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
                        let px = data[(65 + i * 3) * h * w + offset];
                        let py = data[(66 + i * 3) * h * w + offset];
                        let ps = data[(67 + i * 3) * h * w + offset];
                        let lx = (px * 2.0 + x as f32) * (stride as f32);
                        let ly = (py * 2.0 + y as f32) * (stride as f32);
                        let ls = sigmoid(ps);
                        keypoints.push((lx, ly, ls));
                    }

                    detections.push(Detection {
                        bbox: (x1, y1, x2 - x1, y2 - y1),
                        score,
                        keypoints,
                        class_id: 0,
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
                    class_id: 0,
                });
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: (f32, f32, f32, f32), // x, y, w, h
    pub score: f32,
    pub keypoints: Vec<(f32, f32, f32)>, // x, y, conf
    pub class_id: usize,
}
