use apex_camera_models::CameraModel as ApexCameraModel;
use apex_camera_models::{
    DistortionModel, DoubleSphereCamera, EucmCamera, FovCamera, KannalaBrandtCamera, PinholeCamera,
    PinholeParams, RadTanCamera, UcmCamera,
};
use nalgebra::{Vector2, Vector3};
use serde::{Deserialize, Serialize};

pub mod estimation;

/// Camera intrinsic parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Image resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum CameraModelError {
    #[error("Projection is outside the image")]
    ProjectionOutSideImage,
    #[error("Input point is outside the image")]
    PointIsOutSideImage,
    #[error("z is close to zero, point is at camera center")]
    PointAtCameraCenter,
    #[error("Focal length must be positive")]
    FocalLengthMustBePositive,
    #[error("Principal point must be finite")]
    PrincipalPointMustBeFinite,
    #[error("Invalid camera parameters: {0}")]
    InvalidParams(String),
    #[error("Failed to load YAML: {0}")]
    YamlError(String),
    #[error("IO Error: {0}")]
    IOError(String),
    #[error("NumericalError: {0}")]
    NumericalError(String),
}

impl From<std::io::Error> for CameraModelError {
    fn from(err: std::io::Error) -> Self {
        CameraModelError::IOError(err.to_string())
    }
}

impl From<yaml_rust::ScanError> for CameraModelError {
    fn from(err: yaml_rust::ScanError) -> Self {
        CameraModelError::YamlError(err.to_string())
    }
}

impl From<apex_camera_models::CameraModelError> for CameraModelError {
    fn from(e: apex_camera_models::CameraModelError) -> Self {
        CameraModelError::NumericalError(e.to_string())
    }
}

/// Object-safe camera model trait for projection/unprojection and parameter access.
pub trait CameraModel: Send + Sync {
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError>;
    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError>;
    fn get_resolution(&self) -> Resolution;
    fn get_intrinsics(&self) -> Intrinsics;
    fn get_distortion(&self) -> Vec<f64>;
    fn get_model_name(&self) -> &str;
}

/// Trait for loading/saving camera parameters from/to YAML files.
pub trait YamlCamera: Sized {
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError>;
    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError>;
}

/// Unified camera model wrapping apex-camera-models with resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraWithResolution {
    pub intrinsics: Intrinsics,
    pub resolution: Resolution,
    pub model_name: String,
    pub distortion_params: Vec<f64>,
}

impl CameraWithResolution {
    pub fn pinhole_params(&self) -> PinholeParams {
        PinholeParams {
            fx: self.intrinsics.fx,
            fy: self.intrinsics.fy,
            cx: self.intrinsics.cx,
            cy: self.intrinsics.cy,
        }
    }

    fn make_distortion_model(&self) -> Result<DistortionModel, CameraModelError> {
        let d = &self.distortion_params;
        match self.model_name.as_str() {
            "pinhole" => Ok(DistortionModel::None),
            "double_sphere" => {
                if d.len() < 2 {
                    return Err(CameraModelError::InvalidParams(
                        "DS needs 2 distortion params".into(),
                    ));
                }
                // YAML order: [alpha, xi]
                Ok(DistortionModel::DoubleSphere {
                    xi: d[1],
                    alpha: d[0],
                })
            }
            "eucm" => {
                if d.len() < 2 {
                    return Err(CameraModelError::InvalidParams(
                        "EUCM needs 2 distortion params".into(),
                    ));
                }
                Ok(DistortionModel::EUCM {
                    alpha: d[0],
                    beta: d[1],
                })
            }
            "fov" => {
                if d.is_empty() {
                    return Err(CameraModelError::InvalidParams(
                        "FOV needs 1 distortion param".into(),
                    ));
                }
                Ok(DistortionModel::FOV { w: d[0] })
            }
            "kannala_brandt" => {
                if d.len() < 4 {
                    return Err(CameraModelError::InvalidParams(
                        "KB needs 4 distortion params".into(),
                    ));
                }
                Ok(DistortionModel::KannalaBrandt {
                    k1: d[0],
                    k2: d[1],
                    k3: d[2],
                    k4: d[3],
                })
            }
            "rad_tan" => {
                if d.len() < 5 {
                    return Err(CameraModelError::InvalidParams(
                        "RadTan needs 5 distortion params".into(),
                    ));
                }
                Ok(DistortionModel::BrownConrady {
                    k1: d[0],
                    k2: d[1],
                    p1: d[2],
                    p2: d[3],
                    k3: d[4],
                })
            }
            "ucm" => {
                if d.is_empty() {
                    return Err(CameraModelError::InvalidParams(
                        "UCM needs 1 distortion param".into(),
                    ));
                }
                Ok(DistortionModel::UCM { alpha: d[0] })
            }
            _ => Err(CameraModelError::InvalidParams(format!(
                "Unknown model: {}",
                self.model_name
            ))),
        }
    }
}

impl CameraModel for CameraWithResolution {
    fn project(&self, point_3d: &Vector3<f64>) -> Result<Vector2<f64>, CameraModelError> {
        let pinhole = self.pinhole_params();
        let distortion = self.make_distortion_model()?;
        match self.model_name.as_str() {
            "pinhole" => {
                let cam = PinholeCamera::new(pinhole, distortion)?;
                Ok(cam.project(point_3d)?)
            }
            "double_sphere" => {
                let cam = DoubleSphereCamera::new(pinhole, distortion)?;
                Ok(cam.project(point_3d)?)
            }
            "eucm" => {
                // Direct construction to bypass alpha validation (alpha > 1 support)
                let cam = EucmCamera {
                    pinhole,
                    distortion,
                };
                Ok(cam.project(point_3d)?)
            }
            "fov" => {
                let cam = FovCamera::new(pinhole, distortion)?;
                Ok(cam.project(point_3d)?)
            }
            "kannala_brandt" => {
                let cam = KannalaBrandtCamera::new(pinhole, distortion)?;
                Ok(cam.project(point_3d)?)
            }
            "rad_tan" => {
                let cam = RadTanCamera::new(pinhole, distortion)?;
                Ok(cam.project(point_3d)?)
            }
            "ucm" => {
                // Direct construction to bypass alpha validation (alpha > 1 support)
                let cam = UcmCamera {
                    pinhole,
                    distortion,
                };
                Ok(cam.project(point_3d)?)
            }
            _ => Err(CameraModelError::InvalidParams(format!(
                "Unknown model: {}",
                self.model_name
            ))),
        }
    }

    fn unproject(&self, point_2d: &Vector2<f64>) -> Result<Vector3<f64>, CameraModelError> {
        let pinhole = self.pinhole_params();
        let distortion = self.make_distortion_model()?;
        match self.model_name.as_str() {
            "pinhole" => {
                let cam = PinholeCamera::new(pinhole, distortion)?;
                Ok(cam.unproject(point_2d)?)
            }
            "double_sphere" => {
                let cam = DoubleSphereCamera::new(pinhole, distortion)?;
                Ok(cam.unproject(point_2d)?)
            }
            "eucm" => {
                let cam = EucmCamera {
                    pinhole,
                    distortion,
                };
                Ok(cam.unproject(point_2d)?)
            }
            "fov" => {
                let cam = FovCamera::new(pinhole, distortion)?;
                Ok(cam.unproject(point_2d)?)
            }
            "kannala_brandt" => {
                let cam = KannalaBrandtCamera::new(pinhole, distortion)?;
                Ok(cam.unproject(point_2d)?)
            }
            "rad_tan" => {
                let cam = RadTanCamera::new(pinhole, distortion)?;
                Ok(cam.unproject(point_2d)?)
            }
            "ucm" => {
                let cam = UcmCamera {
                    pinhole,
                    distortion,
                };
                Ok(cam.unproject(point_2d)?)
            }
            _ => Err(CameraModelError::InvalidParams(format!(
                "Unknown model: {}",
                self.model_name
            ))),
        }
    }

    fn get_resolution(&self) -> Resolution {
        self.resolution.clone()
    }

    fn get_intrinsics(&self) -> Intrinsics {
        self.intrinsics.clone()
    }

    fn get_distortion(&self) -> Vec<f64> {
        self.distortion_params.clone()
    }

    fn get_model_name(&self) -> &str {
        &self.model_name
    }
}

impl YamlCamera for CameraWithResolution {
    fn load_from_yaml(path: &str) -> Result<Self, CameraModelError> {
        yaml_io::load_from_yaml(path)
    }

    fn save_to_yaml(&self, path: &str) -> Result<(), CameraModelError> {
        yaml_io::save_to_yaml(self, path)
    }
}

mod yaml_io {
    use super::*;
    use std::fs;
    use std::io::Write;
    use yaml_rust::YamlLoader;

    /// Model name mapping from YAML `camera_model` field to our internal name.
    fn normalize_model_name(name: &str) -> Result<&'static str, CameraModelError> {
        match name {
            "pinhole" => Ok("pinhole"),
            "double_sphere" | "ds" => Ok("double_sphere"),
            "eucm" | "extended_unified" => Ok("eucm"),
            "fov" | "field_of_view" => Ok("fov"),
            "kannala_brandt" | "kb" => Ok("kannala_brandt"),
            "rad_tan" | "radtan" => Ok("rad_tan"),
            "ucm" | "unified" => Ok("ucm"),
            _ => Err(CameraModelError::InvalidParams(format!(
                "Unknown camera model: {name}"
            ))),
        }
    }

    /// Expected number of intrinsics+distortion params for each model.
    fn expected_intrinsics_len(model_name: &str) -> usize {
        match model_name {
            "pinhole" => 4,
            "double_sphere" => 6,  // fx, fy, cx, cy, xi, alpha
            "eucm" => 6,           // fx, fy, cx, cy, alpha, beta
            "fov" => 5,            // fx, fy, cx, cy, w
            "kannala_brandt" => 8, // fx, fy, cx, cy, k1, k2, k3, k4
            "rad_tan" => 9,        // fx, fy, cx, cy, k1, k2, p1, p2, k3
            "ucm" => 5,            // fx, fy, cx, cy, alpha
            _ => 4,
        }
    }

    pub fn load_from_yaml(path: &str) -> Result<CameraWithResolution, CameraModelError> {
        let contents = fs::read_to_string(path)?;
        let docs = YamlLoader::load_from_str(&contents)?;

        if docs.is_empty() {
            return Err(CameraModelError::InvalidParams(
                "Empty YAML document".into(),
            ));
        }

        let doc = &docs[0];
        let cam_node = &doc["cam0"];

        if cam_node.is_badvalue() {
            return Err(CameraModelError::InvalidParams(
                "Missing 'cam0' node in YAML".into(),
            ));
        }

        // Get camera model name
        let model_name_raw = cam_node["camera_model"].as_str().ok_or_else(|| {
            CameraModelError::InvalidParams("Missing 'camera_model' field".into())
        })?;
        let model_name = normalize_model_name(model_name_raw)?;
        let min_len = expected_intrinsics_len(model_name);

        // Parse intrinsics array
        let intrinsics_yaml = cam_node["intrinsics"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("YAML missing 'intrinsics' array under 'cam0'".into())
        })?;

        // Check for separate 'distortion' field (used by radtan and kannala_brandt YAMLs)
        let separate_distortion = cam_node["distortion"].as_vec();
        let has_separate_distortion = separate_distortion.is_some();

        // If distortion is separate, we only need 4 intrinsics; otherwise need full min_len
        let required_intrinsics = if has_separate_distortion { 4 } else { min_len };

        if intrinsics_yaml.len() < required_intrinsics {
            return Err(CameraModelError::InvalidParams(format!(
                "Intrinsics array must have at least {} elements, got {}",
                required_intrinsics,
                intrinsics_yaml.len()
            )));
        }

        // Parse resolution array
        let resolution_yaml = cam_node["resolution"].as_vec().ok_or_else(|| {
            CameraModelError::InvalidParams("YAML missing 'resolution' array under 'cam0'".into())
        })?;

        if resolution_yaml.len() < 2 {
            return Err(CameraModelError::InvalidParams(
                "Resolution array must have at least 2 elements".into(),
            ));
        }

        let parse_f64 = |yaml: &yaml_rust::Yaml, name: &str| -> Result<f64, CameraModelError> {
            yaml.as_f64().ok_or_else(|| {
                CameraModelError::InvalidParams(format!("Invalid {name}: not a float"))
            })
        };

        let intrinsics = Intrinsics {
            fx: parse_f64(&intrinsics_yaml[0], "fx")?,
            fy: parse_f64(&intrinsics_yaml[1], "fy")?,
            cx: parse_f64(&intrinsics_yaml[2], "cx")?,
            cy: parse_f64(&intrinsics_yaml[3], "cy")?,
        };

        let resolution = Resolution {
            width: resolution_yaml[0].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid width: not an integer".into())
            })? as u32,
            height: resolution_yaml[1].as_i64().ok_or_else(|| {
                CameraModelError::InvalidParams("Invalid height: not an integer".into())
            })? as u32,
        };

        // Extract distortion params: from separate 'distortion' field or from intrinsics[4..]
        let mut distortion_params = Vec::new();
        if let Some(dist_yaml) = separate_distortion {
            for (i, param_yaml) in dist_yaml.iter().enumerate() {
                let param = parse_f64(param_yaml, &format!("distortion[{i}]"))?;
                distortion_params.push(param);
            }
        } else {
            for (i, param_yaml) in intrinsics_yaml.iter().enumerate().skip(4) {
                let param = parse_f64(param_yaml, &format!("param[{i}]"))?;
                distortion_params.push(param);
            }
        }

        Ok(CameraWithResolution {
            intrinsics,
            resolution,
            model_name: model_name.to_string(),
            distortion_params,
        })
    }

    pub fn save_to_yaml(cam: &CameraWithResolution, path: &str) -> Result<(), CameraModelError> {
        // Build intrinsics array: [fx, fy, cx, cy, ...distortion]
        let mut intrinsics_vec = vec![
            cam.intrinsics.fx,
            cam.intrinsics.fy,
            cam.intrinsics.cx,
            cam.intrinsics.cy,
        ];
        intrinsics_vec.extend_from_slice(&cam.distortion_params);

        let yaml = serde_yaml::to_value(serde_yaml::Mapping::from_iter([(
            serde_yaml::Value::String("cam0".to_string()),
            serde_yaml::to_value(serde_yaml::Mapping::from_iter([
                (
                    serde_yaml::Value::String("camera_model".to_string()),
                    serde_yaml::Value::String(cam.model_name.clone()),
                ),
                (
                    serde_yaml::Value::String("intrinsics".to_string()),
                    serde_yaml::to_value(&intrinsics_vec)
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
                (
                    serde_yaml::Value::String("resolution".to_string()),
                    serde_yaml::to_value(vec![cam.resolution.width, cam.resolution.height])
                        .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
                ),
            ]))
            .map_err(|e| CameraModelError::YamlError(e.to_string()))?,
        )]))
        .map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        let yaml_string =
            serde_yaml::to_string(&yaml).map_err(|e| CameraModelError::YamlError(e.to_string()))?;

        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| CameraModelError::IOError(e.to_string()))?;
        }

        let mut file =
            fs::File::create(path).map_err(|e| CameraModelError::IOError(e.to_string()))?;
        file.write_all(yaml_string.as_bytes())
            .map_err(|e| CameraModelError::IOError(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_camera(model_name: &str, distortion_params: Vec<f64>) -> CameraWithResolution {
        CameraWithResolution {
            intrinsics: Intrinsics {
                fx: 350.0,
                fy: 350.0,
                cx: 320.0,
                cy: 240.0,
            },
            resolution: Resolution {
                width: 640,
                height: 480,
            },
            model_name: model_name.to_string(),
            distortion_params,
        }
    }

    fn test_project_unproject_roundtrip(cam: &CameraWithResolution, tolerance: f64) {
        // Test pixels near center (where all models are well-behaved)
        let test_pixels = vec![
            Vector2::new(320.0, 240.0),
            Vector2::new(325.0, 245.0),
            Vector2::new(315.0, 235.0),
        ];
        for pixel in &test_pixels {
            let ray = cam.unproject(pixel).expect("unproject failed");
            let reprojected = cam.project(&ray).expect("project failed");
            assert_relative_eq!(reprojected.x, pixel.x, epsilon = tolerance);
            assert_relative_eq!(reprojected.y, pixel.y, epsilon = tolerance);
        }
    }

    #[test]
    fn test_pinhole_roundtrip() {
        let cam = make_camera("pinhole", vec![]);
        test_project_unproject_roundtrip(&cam, 1e-6);
    }

    #[test]
    fn test_double_sphere_roundtrip() {
        // distortion_params order: [alpha, xi]
        let cam = make_camera("double_sphere", vec![0.58, -0.18]);
        test_project_unproject_roundtrip(&cam, 1e-6);
    }

    #[test]
    fn test_eucm_roundtrip() {
        // EUCM roundtrip: test only at optical center where model is most accurate
        let cam = make_camera("eucm", vec![0.5, 1.0]);
        let pixel = Vector2::new(320.0, 240.0);
        let ray = cam.unproject(&pixel).expect("unproject failed");
        let reprojected = cam.project(&ray).expect("project failed");
        assert_relative_eq!(reprojected.x, pixel.x, epsilon = 1e-3);
        assert_relative_eq!(reprojected.y, pixel.y, epsilon = 1e-3);
    }

    #[test]
    fn test_fov_roundtrip() {
        let cam = make_camera("fov", vec![0.9]);
        test_project_unproject_roundtrip(&cam, 1e-6);
    }

    #[test]
    fn test_kannala_brandt_roundtrip() {
        let cam = make_camera("kannala_brandt", vec![-0.01, 0.05, -0.08, 0.04]);
        test_project_unproject_roundtrip(&cam, 1e-6);
    }

    #[test]
    fn test_rad_tan_roundtrip() {
        let cam = make_camera("rad_tan", vec![-0.28, 0.07, 0.0002, 0.00002, 0.0]);
        test_project_unproject_roundtrip(&cam, 1e-4);
    }

    #[test]
    fn test_ucm_roundtrip() {
        let cam = make_camera("ucm", vec![0.8]);
        test_project_unproject_roundtrip(&cam, 1e-3);
    }

    #[test]
    fn test_model_names() {
        assert_eq!(make_camera("pinhole", vec![]).get_model_name(), "pinhole");
        assert_eq!(
            make_camera("double_sphere", vec![0.58, -0.18]).get_model_name(),
            "double_sphere"
        );
        assert_eq!(make_camera("eucm", vec![0.5, 1.0]).get_model_name(), "eucm");
        assert_eq!(make_camera("fov", vec![0.9]).get_model_name(), "fov");
        assert_eq!(
            make_camera("kannala_brandt", vec![0.0; 4]).get_model_name(),
            "kannala_brandt"
        );
        assert_eq!(
            make_camera("rad_tan", vec![0.0; 5]).get_model_name(),
            "rad_tan"
        );
        assert_eq!(make_camera("ucm", vec![0.8]).get_model_name(), "ucm");
    }

    #[test]
    fn test_yaml_roundtrip() {
        let cam = make_camera("double_sphere", vec![0.58, -0.18]);
        let path = "/tmp/test_camera_yaml_roundtrip.yaml";
        cam.save_to_yaml(path).expect("save failed");
        let loaded = CameraWithResolution::load_from_yaml(path).expect("load failed");
        assert_eq!(loaded.model_name, cam.model_name);
        assert_relative_eq!(loaded.intrinsics.fx, cam.intrinsics.fx, epsilon = 1e-10);
        assert_relative_eq!(loaded.intrinsics.fy, cam.intrinsics.fy, epsilon = 1e-10);
        assert_eq!(loaded.resolution.width, cam.resolution.width);
        assert_eq!(loaded.distortion_params.len(), cam.distortion_params.len());
        for (a, b) in loaded
            .distortion_params
            .iter()
            .zip(cam.distortion_params.iter())
        {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }
}
