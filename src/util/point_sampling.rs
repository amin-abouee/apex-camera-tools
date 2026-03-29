//! Point sampling and correspondence export utilities.
//!
//! This module provides functionality for generating sample points across an image
//! and exporting 3D-2D point correspondences for camera calibration and conversion.

use crate::camera::{CameraModel, CameraModelError};
use nalgebra::{Matrix2xX, Matrix3xX, Vector2};
use std::fs::File;
use std::io::Write;

use super::{UtilError, ensure_output_dir};

/// Generate a grid of sample points that are evenly distributed across the image,
/// optionally unprojecting them to 3D using a provided camera model.
///
/// This function creates a grid of 2D points across the image and unprojects them
/// to 3D rays using the provided camera model. Only points that successfully unproject
/// with z > 0 (in front of the camera) are included in the result.
///
/// # Arguments
///
/// * `camera_model` - Camera model to use for unprojection. If None, 3D points will be
///   on a plane at z=1.0
/// * `n` - The approximate number of points to generate
///
/// # Returns
///
/// * A tuple containing:
///   * Matrix2xX where each column represents a 2D point with pixel coordinates
///   * Matrix3xX where each column represents the corresponding 3D point
///
/// # Errors
///
/// Returns a `CameraModelError` if the camera model operations fail.
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::camera::KannalaBrandtModel;
/// use apex_camera_models::util::sample_points;
///
/// let model = KannalaBrandtModel::load_from_yaml("camera.yaml")?;
/// let (points_2d, points_3d) = sample_points(Some(&model), 500)?;
/// println!("Generated {} point correspondences", points_2d.ncols());
/// ```
pub fn sample_points<T>(
    camera_model: Option<&T>,
    n: usize,
) -> Result<(Matrix2xX<f64>, Matrix3xX<f64>), CameraModelError>
where
    T: ?Sized + CameraModel,
{
    let width = camera_model.unwrap().get_resolution().width as f64;
    let height = camera_model.unwrap().get_resolution().height as f64;
    // Calculate the number of cells in each dimension
    let num_cells_x = (n as f64 * (width / height)).sqrt().round() as i32;
    let num_cells_y = (n as f64 * (height / width)).sqrt().round() as i32;

    // Calculate the dimensions of each cell
    let cell_width = width / num_cells_x as f64;
    let cell_height = height / num_cells_y as f64;

    // Calculate total number of points
    let total_points = (num_cells_x * num_cells_y) as usize;

    // Create a matrix with the appropriate size
    let mut points_2d_matrix = Matrix2xX::zeros(total_points);

    // Generate a point at the center of each cell
    let mut idx = 0;
    for i in 0..num_cells_y {
        for j in 0..num_cells_x {
            let x = (j as f64 + 0.5) * cell_width;
            let y = (i as f64 + 0.5) * cell_height;
            points_2d_matrix.set_column(idx, &Vector2::new(x, y));
            idx += 1;
        }
    }

    // Unwrap the camera model (safe because we checked it's Some)
    let camera_model = camera_model.unwrap();

    // Prepare vectors to store valid points
    let mut valid_2d_points = Vec::new();
    let mut valid_3d_points = Vec::new();

    // Unproject each 2D point and filter for z > 0
    for col_idx in 0..points_2d_matrix.ncols() {
        let point_2d = points_2d_matrix.column(col_idx);
        let p2d = Vector2::new(point_2d[0], point_2d[1]);

        // Try to unproject the point
        if let Ok(p3d) = camera_model.unproject(&p2d) {
            // Only keep points with z > 0
            if p3d.z > 0.0 {
                // Store the valid 2D point
                valid_2d_points.push(p2d);

                // Store the corresponding 3D point
                valid_3d_points.push(p3d);
            }
        }
    }

    // Convert vectors to matrices
    let n_valid = valid_2d_points.len();
    let mut points_2d_result = Matrix2xX::zeros(n_valid);
    let mut points_3d_result = Matrix3xX::zeros(n_valid);

    for (idx, (p2d, p3d)) in valid_2d_points
        .iter()
        .zip(valid_3d_points.iter())
        .enumerate()
    {
        points_2d_result.set_column(idx, p2d);
        points_3d_result.set_column(idx, p3d);
    }

    Ok((points_2d_result, points_3d_result))
}

/// Export point correspondences to CSV and Rust format files.
///
/// This function saves 3D-2D point correspondences to two file formats:
/// - CSV format for general use and analysis
/// - Rust code format for easy import into Rust programs
///
/// # Arguments
///
/// * `points_3d` - Matrix of 3D points (3×N)
/// * `points_2d` - Matrix of corresponding 2D points (2×N)
/// * `filename_prefix` - Prefix for output files (files will be created in `output/` directory)
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
///
/// # Errors
///
/// * `UtilError::InvalidParams` - If 3D and 2D point counts don't match
/// * `UtilError::NumericalError` - If file creation or writing fails
///
/// # Examples
///
/// ```rust,ignore
/// use apex_camera_models::util::export_point_correspondences;
/// use nalgebra::{Matrix2xX, Matrix3xX};
///
/// // Assuming you have points_3d and points_2d matrices
/// export_point_correspondences(&points_3d, &points_2d, "my_calibration")?;
/// // Creates: output/my_calibration.csv and output/my_calibration_rust.txt
/// ```
pub fn export_point_correspondences(
    points_3d: &Matrix3xX<f64>,
    points_2d: &Matrix2xX<f64>,
    filename_prefix: &str,
) -> Result<(), UtilError> {
    println!("\n💾 Exporting Point Correspondences:");

    if points_3d.ncols() != points_2d.ncols() {
        return Err(UtilError::InvalidParams(
            "3D and 2D point counts must match".to_string(),
        ));
    }

    // Ensure output directory exists
    ensure_output_dir()?;

    // Export to CSV format
    let csv_filename = format!("output/{filename_prefix}.csv");
    let mut csv_file = File::create(&csv_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create CSV file: {e}")))?;

    writeln!(
        csv_file,
        "# 3D-2D Point Correspondences from Rust Implementation"
    )?;
    writeln!(csv_file, "# Format: x3d,y3d,z3d,x2d,y2d")?;
    writeln!(csv_file, "# Total points: {}", points_3d.ncols())?;

    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i);
        let p2d = points_2d.column(i);
        writeln!(
            csv_file,
            "{:.15},{:.15},{:.15},{:.15},{:.15}",
            p3d[0], p3d[1], p3d[2], p2d[0], p2d[1]
        )?;
    }

    // Export to Rust format
    let rust_filename = format!("output/{filename_prefix}_rust.txt");
    let mut rust_file = File::create(&rust_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create Rust file: {e}")))?;

    writeln!(rust_file, "// 3D-2D Point Correspondences for Rust Import")?;
    writeln!(rust_file, "// Generated from Rust fisheye-tools")?;
    writeln!(rust_file, "let points_3d = Matrix3xX::from_columns(&[")?;

    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i);
        write!(
            rust_file,
            "    Vector3::new({:.15}, {:.15}, {:.15})",
            p3d[0], p3d[1], p3d[2]
        )?;
        if i < points_3d.ncols() - 1 {
            writeln!(rust_file, ",")?;
        } else {
            writeln!(rust_file)?;
        }
    }
    writeln!(rust_file, "]);")?;
    writeln!(rust_file)?;

    writeln!(rust_file, "let points_2d = Matrix2xX::from_columns(&[")?;
    for i in 0..points_2d.ncols() {
        let p2d = points_2d.column(i);
        write!(
            rust_file,
            "    Vector2::new({:.15}, {:.15})",
            p2d[0], p2d[1]
        )?;
        if i < points_2d.ncols() - 1 {
            writeln!(rust_file, ",")?;
        } else {
            writeln!(rust_file)?;
        }
    }
    writeln!(rust_file, "]);")?;

    println!("Exported {} point correspondences to:", points_3d.ncols());
    println!("  - {csv_filename} (CSV format)");
    println!("  - {rust_filename} (Rust code format)");

    Ok(())
}
