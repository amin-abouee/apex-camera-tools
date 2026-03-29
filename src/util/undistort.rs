//! Minimal image undistortion utilities

use super::UtilError;
use crate::camera::{CameraModel, Intrinsics};
use image::{Rgb, RgbImage};
use nalgebra::Vector3;

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    Nearest,
    Bilinear,
}

pub fn undistort_image(
    input_image: &RgbImage,
    camera_model: &dyn CameraModel,
    target_intrinsics: Option<Intrinsics>,
    interpolation: InterpolationMethod,
) -> Result<RgbImage, UtilError> {
    let resolution = camera_model.get_resolution();
    let (width, height) = input_image.dimensions();

    if width != resolution.width || height != resolution.height {
        return Err(UtilError::InvalidParams(format!(
            "Image {}x{} doesn't match model {}x{}",
            width, height, resolution.width, resolution.height
        )));
    }

    let target = target_intrinsics.unwrap_or_else(|| camera_model.get_intrinsics());
    let mut output = RgbImage::new(width, height);

    for v_out in 0..height {
        for u_out in 0..width {
            let x_norm = (u_out as f64 - target.cx) / target.fx;
            let y_norm = (v_out as f64 - target.cy) / target.fy;
            let ray_3d = Vector3::new(x_norm, y_norm, 1.0);

            if let Ok(source_pixel) = camera_model.project(&ray_3d)
                && let Some(color) =
                    interpolate_pixel(input_image, source_pixel.x, source_pixel.y, interpolation)
            {
                output.put_pixel(u_out, v_out, color);
            }
        }
    }
    Ok(output)
}

fn interpolate_pixel(
    image: &RgbImage,
    x: f64,
    y: f64,
    method: InterpolationMethod,
) -> Option<Rgb<u8>> {
    let (width, height) = image.dimensions();

    match method {
        InterpolationMethod::Nearest => {
            let u = x.round() as i32;
            let v = y.round() as i32;
            if u >= 0 && u < width as i32 && v >= 0 && v < height as i32 {
                Some(*image.get_pixel(u as u32, v as u32))
            } else {
                None
            }
        }
        InterpolationMethod::Bilinear => {
            let x0 = x.floor();
            let y0 = y.floor();
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;

            if x0 < 0.0 || x1 >= width as f64 || y0 < 0.0 || y1 >= height as f64 {
                return None;
            }

            let x0_u = x0 as u32;
            let y0_u = y0 as u32;
            let x1_u = x1 as u32;
            let y1_u = y1 as u32;

            let p00 = image.get_pixel(x0_u, y0_u);
            let p10 = image.get_pixel(x1_u, y0_u);
            let p01 = image.get_pixel(x0_u, y1_u);
            let p11 = image.get_pixel(x1_u, y1_u);

            let wx = x - x0;
            let wy = y - y0;
            let wx_inv = 1.0 - wx;
            let wy_inv = 1.0 - wy;

            let mut result = Rgb([0u8; 3]);
            for c in 0..3 {
                let val = p00[c] as f64 * wx_inv * wy_inv
                    + p10[c] as f64 * wx * wy_inv
                    + p01[c] as f64 * wx_inv * wy
                    + p11[c] as f64 * wx * wy;
                result[c] = val.round().clamp(0.0, 255.0) as u8;
            }
            Some(result)
        }
    }
}
