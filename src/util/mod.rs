use std::fs;
use std::path::Path;

use crate::camera::CameraModelError;

// Module declarations
mod error_metrics;
mod image_quality;
mod point_sampling;
mod reporting;
mod undistort;
mod validation;
// Re-export all public items from sub-modules
pub use error_metrics::{ProjectionError, compute_reprojection_error};
pub use image_quality::{
    ImageQualityMetrics, calculate_psnr, calculate_ssim, compute_image_quality_metrics,
    create_combined_projection_image, create_combined_projection_image_on_reference,
    create_projection_image, load_image, model_projection_visualization,
    save_model_projection_image,
};
pub use point_sampling::{export_point_correspondences, sample_points};
pub use reporting::{
    ConversionMetrics, display_detailed_results, display_input_model_parameters,
    display_results_summary, export_conversion_results,
};
pub use undistort::{InterpolationMethod, undistort_image};
pub use validation::{RegionValidation, ValidationResults, validate_conversion_accuracy};
/// Ensure the output directory exists
pub fn ensure_output_dir() -> Result<(), UtilError> {
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).map_err(|e| {
            UtilError::InvalidParams(format!("Failed to create output directory: {e}"))
        })?;
    }
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub enum UtilError {
    #[error("Camera model does not exist")]
    CameraModelDoesNotExist,
    #[error("Numerical error in computation: {0}")]
    NumericalError(String),
    #[error("Matrix singularity detected")]
    SingularMatrix,
    #[error("Zero projection points")]
    ZeroProjectionPoints,
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
}

impl From<std::io::Error> for UtilError {
    fn from(err: std::io::Error) -> Self {
        UtilError::NumericalError(err.to_string())
    }
}

impl From<CameraModelError> for UtilError {
    fn from(err: CameraModelError) -> Self {
        UtilError::NumericalError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraWithResolution, YamlCamera};

    #[test]
    fn test_sample_points() {
        let input_path = "samples/double_sphere.yaml";
        let camera_model = CameraWithResolution::load_from_yaml(input_path).unwrap();
        let n = 100_usize;
        let (points_2d, points_3d) = sample_points(Some(&camera_model), n).unwrap();

        // Check that we have some valid points
        assert!(!points_2d.is_empty(), "No valid 2D-points were generated");

        assert!(
            !points_3d.is_empty(),
            "No valid 3D-points were generated with camera model"
        );

        assert!(
            points_2d.ncols() == points_3d.ncols(),
            "Number of 2D and 3D points should be equal"
        );

        // Check that all 3D points have z > 0
        for col_idx in 0..points_3d.ncols() {
            let point_3d = points_3d.column(col_idx);
            assert!(point_3d[2] > 0.0, "3D point has z <= 0");
        }
    }
}
