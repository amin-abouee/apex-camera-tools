//! Image quality assessment and projection visualization utilities.
//!
//! This module provides functionality for:
//! - Computing image quality metrics (PSNR, SSIM)
//! - Creating projection visualizations
//! - Loading and saving images
//! - Comparing camera model projections visually

use crate::camera::CameraModel;
use image::{GrayImage, Rgb, RgbImage};
use nalgebra::{Matrix3xX, Vector2};
use serde::{Deserialize, Serialize};

use super::{UtilError, ensure_output_dir};

/// Image quality metrics for assessing camera model accuracy.
///
/// Contains PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
/// values for comparing projections from different camera models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualityMetrics {
    /// Peak Signal-to-Noise Ratio in dB (higher is better)
    pub psnr: f64,
    /// Structural Similarity Index from 0 to 1 (1 is perfect similarity)
    pub ssim: f64,
}

/// Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
///
/// PSNR is a widely used metric for image quality assessment. Higher values
/// indicate better quality/similarity. Black pixels are skipped in the calculation.
///
/// # Arguments
///
/// * `img1` - First image
/// * `img2` - Second image
///
/// # Returns
///
/// * `Result<f64, UtilError>` - PSNR value in dB (higher is better)
///
/// # Errors
///
/// * `UtilError::InvalidParams` - If images have different dimensions
pub fn calculate_psnr(img1: &RgbImage, img2: &RgbImage) -> Result<f64, UtilError> {
    if img1.dimensions() != img2.dimensions() {
        return Err(UtilError::InvalidParams(
            "Images must have the same dimensions".to_string(),
        ));
    }

    let (width, height) = img1.dimensions();
    let mut mse = 0.0;
    let mut valid_pixels = 0;

    for y in 0..height {
        for x in 0..width {
            let pixel1 = img1.get_pixel(x, y);
            let pixel2 = img2.get_pixel(x, y);

            // Skip black pixels (consider them as invalid regions)
            if pixel1[0] != 0
                || pixel1[1] != 0
                || pixel1[2] != 0
                || pixel2[0] != 0
                || pixel2[1] != 0
                || pixel2[2] != 0
            {
                for c in 0..3 {
                    let diff = pixel1[c] as f64 - pixel2[c] as f64;
                    mse += diff * diff;
                }
                valid_pixels += 3; // 3 channels
            }
        }
    }

    if valid_pixels == 0 {
        return Ok(f64::INFINITY); // Perfect match for empty images
    }

    mse /= valid_pixels as f64;

    if mse <= 1e-10 {
        Ok(f64::INFINITY) // Perfect match
    } else {
        Ok(10.0 * (255.0 * 255.0 / mse).log10())
    }
}

/// Calculate Structural Similarity Index (SSIM) between two images.
///
/// SSIM is a perception-based metric that considers luminance, contrast, and structure.
/// Values range from 0 to 1, where 1 indicates perfect similarity.
///
/// # Arguments
///
/// * `img1` - First image
/// * `img2` - Second image
///
/// # Returns
///
/// * `Result<f64, UtilError>` - SSIM value between 0 and 1
///
/// # Errors
///
/// * `UtilError::InvalidParams` - If images have different dimensions
pub fn calculate_ssim(img1: &RgbImage, img2: &RgbImage) -> Result<f64, UtilError> {
    if img1.dimensions() != img2.dimensions() {
        return Err(UtilError::InvalidParams(
            "Images must have the same dimensions".to_string(),
        ));
    }

    // Convert to grayscale for SSIM calculation
    let gray1 = rgb_to_grayscale(img1);
    let gray2 = rgb_to_grayscale(img2);

    // SSIM constants
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);

    let (width, height) = gray1.dimensions();
    let mut sum1 = 0.0;
    let mut count = 0;

    // Simple sliding window SSIM calculation
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut local_sum1 = 0.0;
            let mut local_sum2 = 0.0;
            let mut local_count = 0;

            // 3x3 window
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;

                    let val1 = gray1.get_pixel(nx, ny)[0] as f64;
                    let val2 = gray2.get_pixel(nx, ny)[0] as f64;

                    local_sum1 += val1;
                    local_sum2 += val2;
                    local_count += 1;
                }
            }

            if local_count > 0 {
                let mu1 = local_sum1 / local_count as f64;
                let mu2 = local_sum2 / local_count as f64;

                let mut sigma1_sq = 0.0;
                let mut sigma2_sq = 0.0;
                let mut sigma12 = 0.0;

                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = (x as i32 + dx) as u32;
                        let ny = (y as i32 + dy) as u32;

                        let val1 = gray1.get_pixel(nx, ny)[0] as f64;
                        let val2 = gray2.get_pixel(nx, ny)[0] as f64;

                        sigma1_sq += (val1 - mu1).powi(2);
                        sigma2_sq += (val2 - mu2).powi(2);
                        sigma12 += (val1 - mu1) * (val2 - mu2);
                    }
                }

                sigma1_sq /= (local_count - 1) as f64;
                sigma2_sq /= (local_count - 1) as f64;
                sigma12 /= (local_count - 1) as f64;

                let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
                let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);

                if denominator > 0.0 {
                    sum1 += numerator / denominator;
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        Ok(sum1 / count as f64)
    } else {
        Ok(1.0) // Perfect similarity for empty images
    }
}

/// Convert RGB image to grayscale using standard luminance formula.
///
/// This is a private helper function used internally by SSIM calculation.
fn rgb_to_grayscale(img: &RgbImage) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut gray_img = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let gray_value =
                (0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64) as u8;
            gray_img.put_pixel(x, y, image::Luma([gray_value]));
        }
    }

    gray_img
}

/// Load an image from file path.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Loaded RGB image
///
/// # Errors
///
/// * `UtilError::InvalidParams` - If image cannot be loaded
pub fn load_image(image_path: &str) -> Result<RgbImage, UtilError> {
    let img = image::open(image_path)
        .map_err(|e| UtilError::InvalidParams(format!("Failed to load image: {e}")))?;

    Ok(img.to_rgb8())
}

/// Compute image quality metrics by comparing projections from two camera models.
///
/// This function projects 3D points using both camera models, creates visualization
/// images, and computes PSNR and SSIM metrics to assess the similarity.
///
/// # Arguments
///
/// * `input_model` - Source camera model
/// * `output_model` - Target camera model
/// * `optimization_points_3d` - The 3D points used in optimization
/// * `output_model_name` - Name of the output model for file naming
/// * `reference_image` - Optional reference image for coloring pixels
///
/// # Returns
///
/// * `Result<ImageQualityMetrics, UtilError>` - PSNR and SSIM values
///
/// # Errors
///
/// * `UtilError::ZeroProjectionPoints` - If no valid projections could be computed
pub fn compute_image_quality_metrics(
    input_model: &dyn CameraModel,
    output_model: &dyn CameraModel,
    optimization_points_3d: &Matrix3xX<f64>,
    output_model_name: &str,
    reference_image: Option<&RgbImage>,
) -> Result<ImageQualityMetrics, UtilError> {
    // Determine image dimensions - prefer reference image size if available
    let (width, height) = if let Some(ref_img) = reference_image {
        (ref_img.width(), ref_img.height())
    } else {
        let resolution = input_model.get_resolution();
        (resolution.width, resolution.height)
    };

    // Project the optimization 3D points using both models
    let mut input_projections = Vec::new();
    let mut output_projections = Vec::new();
    let mut valid_colors = Vec::new();

    for col_idx in 0..optimization_points_3d.ncols() {
        let point_3d = optimization_points_3d.column(col_idx);

        // Project using both models
        if let (Ok(input_proj), Ok(output_proj)) = (
            input_model.project(&point_3d.into()),
            output_model.project(&point_3d.into()),
        ) {
            // Check if output projection is within image bounds
            if output_proj.x >= 0.0
                && output_proj.x < width as f64
                && output_proj.y >= 0.0
                && output_proj.y < height as f64
            {
                input_projections.push(input_proj);
                output_projections.push(output_proj);

                // Colors will be determined when drawing projections
                valid_colors.push(Rgb([255, 255, 255]));
            }
        }
    }

    if output_projections.is_empty() {
        return Err(UtilError::ZeroProjectionPoints);
    }

    // Create projection image with both input and output projections for comparison
    let combined_display_image = if let Some(ref_img) = reference_image {
        create_combined_projection_image_on_reference(
            &input_projections,
            &output_projections,
            ref_img,
        )?
    } else {
        create_combined_projection_image(&input_projections, &output_projections, width, height)?
    };

    // Save the combined projection image
    save_model_projection_image(&combined_display_image, output_model_name)?;

    // Create black background images for PSNR/SSIM computation (not saved)
    let input_comp_image =
        create_projection_image(&input_projections, &valid_colors, width, height)?;
    let output_comp_image =
        create_projection_image(&output_projections, &valid_colors, width, height)?;

    // Calculate quality metrics using black background comparison images
    let psnr = calculate_psnr(&input_comp_image, &output_comp_image)?;
    let ssim = calculate_ssim(&input_comp_image, &output_comp_image)?;

    Ok(ImageQualityMetrics { psnr, ssim })
}

/// Create an image with projected points drawn as colored circles.
///
/// # Arguments
///
/// * `projections` - Vector of 2D projection points
/// * `colors` - Vector of colors for each point
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image
pub fn create_projection_image(
    projections: &[Vector2<f64>],
    colors: &[Rgb<u8>],
    width: u32,
    height: u32,
) -> Result<RgbImage, UtilError> {
    let mut img = RgbImage::new(width, height);

    // Fill with black background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }

    // Draw each projection as a small colored circle
    for (projection, color) in projections.iter().zip(colors.iter()) {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, *color);
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Create an image with both input and output projections on a reference image.
///
/// Input projections are drawn in green, output projections in magenta.
///
/// # Arguments
///
/// * `input_projections` - Vector of 2D projection points from input model
/// * `output_projections` - Vector of 2D projection points from output model
/// * `reference_image` - Reference image to draw points on
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image with both projections
pub fn create_combined_projection_image_on_reference(
    input_projections: &[Vector2<f64>],
    output_projections: &[Vector2<f64>],
    reference_image: &RgbImage,
) -> Result<RgbImage, UtilError> {
    let mut img = reference_image.clone();

    // Draw input projections as green circles
    for projection in input_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0])); // Green for input
                    }
                }
            }
        }
    }

    // Draw output projections as magenta circles
    for projection in output_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([255, 0, 255])); // Magenta for output
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Create an image with both input and output projections on black background.
///
/// Input projections are drawn in green, output projections in magenta.
///
/// # Arguments
///
/// * `input_projections` - Vector of 2D projection points from input model
/// * `output_projections` - Vector of 2D projection points from output model
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// * `Result<RgbImage, UtilError>` - Generated image with both projections
pub fn create_combined_projection_image(
    input_projections: &[Vector2<f64>],
    output_projections: &[Vector2<f64>],
    width: u32,
    height: u32,
) -> Result<RgbImage, UtilError> {
    let mut img = RgbImage::new(width, height);

    // Fill with black background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([0, 0, 0]);
    }

    // Draw input projections as green circles
    for projection in input_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0])); // Green for input
                    }
                }
            }
        }
    }

    // Draw output projections as magenta circles
    for projection in output_projections.iter() {
        let center_x = projection.x.round() as i32;
        let center_y = projection.y.round() as i32;

        // Draw a small circle (radius 2)
        let radius = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = center_x + dx;
                    let y = center_y + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, Rgb([255, 0, 255])); // Magenta for output
                    }
                }
            }
        }
    }

    Ok(img)
}

/// Save model projection image to the output directory with model-specific naming.
///
/// # Arguments
///
/// * `image` - Image to save
/// * `model_name` - Name of the model for filename
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
///
/// # Errors
///
/// * `UtilError::InvalidParams` - If image saving fails
pub fn save_model_projection_image(image: &RgbImage, model_name: &str) -> Result<(), UtilError> {
    ensure_output_dir()?;

    // Convert model name to lowercase and replace spaces with underscores
    let filename_safe_name = model_name.to_lowercase().replace(' ', "_");
    let filename = format!("output/{filename_safe_name}_projection.png");

    image
        .save(&filename)
        .map_err(|e| UtilError::InvalidParams(format!("Failed to save projection image: {e}")))?;

    println!("📸 Saved projection image: {filename}");
    Ok(())
}

/// Create and save input model projection visualization from 2D points.
///
/// This function creates a visualization of 2D projected points, either on a reference image
/// (if provided) or on a black background. Points are drawn as green circles.
///
/// # Arguments
///
/// * `points_2d` - Matrix of 2D projection points
/// * `model_name` - Name of the input model for filename generation
/// * `reference_image` - Optional reference image to project points onto
/// * `camera_resolution` - Camera resolution for image dimensions when no reference image
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
pub fn model_projection_visualization(
    points_2d: &nalgebra::Matrix2xX<f64>,
    model_name: &str,
    reference_image: Option<&RgbImage>,
    camera_resolution: (u32, u32), // (width, height)
) -> Result<(), UtilError> {
    // Convert Matrix2xX to Vec<Vector2<f64>>
    let mut projections = Vec::new();
    for col_idx in 0..points_2d.ncols() {
        let point = points_2d.column(col_idx);
        projections.push(Vector2::new(point[0], point[1]));
    }

    // Create the projection image
    let projection_image = if let Some(ref_img) = reference_image {
        // Project points onto the reference image
        let mut img = ref_img.clone();

        // Draw each projection as a green circle
        for projection in projections.iter() {
            let center_x = projection.x.round() as i32;
            let center_y = projection.y.round() as i32;

            // Draw a small circle (radius 2)
            let radius = 2;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx * dx + dy * dy <= radius * radius {
                        let x = center_x + dx;
                        let y = center_y + dy;

                        if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                            img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0]));
                            // Green
                        }
                    }
                }
            }
        }
        img
    } else {
        // Create projection on black background
        let green_colors = vec![Rgb([0, 255, 0]); projections.len()];
        create_projection_image(
            &projections,
            &green_colors,
            camera_resolution.0,
            camera_resolution.1,
        )?
    };

    // Save the image
    ensure_output_dir()?;
    let filename = format!("output/{}_projection.png", model_name.to_lowercase());
    projection_image
        .save(&filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to save projection image: {e}")))?;

    println!("Saved input model projection visualization: {filename}");
    Ok(())
}
