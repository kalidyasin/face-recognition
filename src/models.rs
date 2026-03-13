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
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        println!("  - Session built.");
        Ok(Self {
            session,
            input_shape: (640, 640),
        })
    }

    pub fn detect(&mut self, input_tensor: Array<f32, ndarray::Dim<[usize; 4]>>, conf_threshold: f32) -> Result<Vec<Detection>> {
        let tensor = Tensor::from_array(input_tensor)?;
        let inputs = ort::inputs!["images" => tensor];
        let outputs = self.session.run(inputs)?;
        
        let output_value = &outputs["output0"];
        let (shape, data) = output_value.try_extract_tensor::<f32>()?;
        
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let channels = dims[1];
        let anchors = dims[2];
        
        let mut detections = Vec::new();

        for i in 0..anchors {
            let score_idx = 4 * anchors + i;
            let score = data[score_idx];
            
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
                     
                     let step = if channels == 56 { 3 } else { 
                         if channels == 15 { 2 } else { 3 } 
                     };

                     for k in 0..((channels - kpt_start) / step) {
                         let c_idx = kpt_start + k * step;
                         let kx = data[(c_idx) * anchors + i];
                         let ky = data[(c_idx + 1) * anchors + i];
                         let kc = if step == 3 { data[(c_idx + 2) * anchors + i] } else { 1.0 }; 
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
        
        Ok(detections)
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: (f32, f32, f32, f32), // x, y, w, h
    pub score: f32,
    pub keypoints: Vec<(f32, f32, f32)>, // x, y, conf
    pub class_id: usize,
}
