//! Linear estimation functions for initializing camera model parameters.
//!
//! These functions provide initial parameter estimates from 2D-3D point correspondences,
//! used as starting points for nonlinear optimization in the convert command.

use super::{CameraModelError, CameraWithResolution};
use log::info;

/// Linear estimation for Double Sphere model.
/// Estimates alpha (xi is set to 0).
pub fn linear_estimation_double_sphere(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_2d.ncols() != points_3d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }

    let num_points = points_2d.ncols();
    let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
    let mut b = nalgebra::DVector::zeros(num_points * 2);

    for i in 0..num_points {
        let x = points_3d[(0, i)];
        let y = points_3d[(1, i)];
        let z = points_3d[(2, i)];
        let u = points_2d[(0, i)];
        let v = points_2d[(1, i)];

        let d = (x * x + y * y + z * z).sqrt();
        let u_cx = u - cam.intrinsics.cx;
        let v_cy = v - cam.intrinsics.cy;

        a[(i * 2, 0)] = u_cx * (d - z);
        a[(i * 2 + 1, 0)] = v_cy * (d - z);

        b[i * 2] = (cam.intrinsics.fx * x) - (u_cx * z);
        b[i * 2 + 1] = (cam.intrinsics.fy * y) - (v_cy * z);
    }

    let svd = a.svd(true, true);
    let alpha = match svd.solve(&b, 1e-10) {
        Ok(sol) => sol[0],
        Err(err_msg) => {
            return Err(CameraModelError::NumericalError(err_msg.to_string()));
        }
    };

    let alpha = alpha.clamp(0.01, 1.0);
    info!("DS linear estimation: alpha = {}, xi = 0.0", alpha);

    cam.distortion_params = vec![alpha, 0.0]; // [alpha, xi]
    Ok(())
}

/// Linear estimation for EUCM model.
/// Estimates alpha (beta is fixed to 1.0).
pub fn linear_estimation_eucm(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_2d.ncols() != points_3d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }

    let num_points = points_2d.ncols();
    let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
    let mut b = nalgebra::DVector::zeros(num_points * 2);

    for i in 0..num_points {
        let x = points_3d[(0, i)];
        let y = points_3d[(1, i)];
        let z = points_3d[(2, i)];
        let u = points_2d[(0, i)];
        let v = points_2d[(1, i)];

        let d = (x * x + y * y + z * z).sqrt();
        let u_cx = u - cam.intrinsics.cx;
        let v_cy = v - cam.intrinsics.cy;

        a[(i * 2, 0)] = u_cx * (d - z);
        a[(i * 2 + 1, 0)] = v_cy * (d - z);

        b[i * 2] = cam.intrinsics.fx * x - u_cx * z;
        b[i * 2 + 1] = cam.intrinsics.fy * y - v_cy * z;
    }

    let svd = a.svd(true, true);
    let solution = match svd.solve(&b, 1e-10) {
        Ok(sol) => sol,
        Err(err_msg) => {
            return Err(CameraModelError::NumericalError(err_msg.to_string()));
        }
    };

    let alpha = solution[0].clamp(0.01, 2.0);
    let beta = 1.0;
    info!("EUCM linear estimation: alpha = {}, beta = {}", alpha, beta);

    cam.distortion_params = vec![alpha, beta];
    Ok(())
}

/// Linear estimation for FOV model.
/// Grid search over w to find best fit.
pub fn linear_estimation_fov(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_2d.ncols() != points_3d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }

    let num_points = points_2d.ncols();
    if num_points < 2 {
        return Err(CameraModelError::InvalidParams(
            "Need at least 2 points for FOV linear estimation".into(),
        ));
    }

    let mut best_w = 1.0;
    let mut best_error = f64::INFINITY;

    for w_test in (10..300).map(|i| i as f64 / 100.0) {
        let mut error_sum = 0.0;
        let mut valid_count = 0;

        for i in 0..num_points {
            let x = points_3d[(0, i)];
            let y = points_3d[(1, i)];
            let z = points_3d[(2, i)];
            let u_observed = points_2d[(0, i)];
            let v_observed = points_2d[(1, i)];

            let r2 = x * x + y * y;
            let r = r2.sqrt();

            let tan_w_half = (w_test / 2.0).tan();
            let atan_wrd = (2.0 * tan_w_half * r).atan2(z);

            let eps_sqrt = f64::EPSILON.sqrt();
            let rd = if r2 < eps_sqrt {
                2.0 * tan_w_half / w_test
            } else {
                atan_wrd / (r * w_test)
            };

            let mx = x * rd;
            let my = y * rd;

            let u_predicted = cam.intrinsics.fx * mx + cam.intrinsics.cx;
            let v_predicted = cam.intrinsics.fy * my + cam.intrinsics.cy;

            let error =
                ((u_predicted - u_observed).powi(2) + (v_predicted - v_observed).powi(2)).sqrt();

            if error.is_finite() {
                error_sum += error;
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            let avg_error = error_sum / valid_count as f64;
            if avg_error < best_error {
                best_error = avg_error;
                best_w = w_test;
            }
        }
    }

    let w = best_w.clamp(0.01, 3.0);
    info!(
        "FOV linear estimation: w = {}, avg_error = {}",
        w, best_error
    );

    cam.distortion_params = vec![w];
    Ok(())
}

/// Linear estimation for Kannala-Brandt model.
/// Estimates k1, k2, k3, k4 via SVD.
pub fn linear_estimation_kannala_brandt(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_3d.ncols() != points_2d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }
    if points_3d.ncols() < 4 {
        return Err(CameraModelError::InvalidParams(
            "Not enough points for linear estimation (need at least 4)".into(),
        ));
    }

    let num_points = points_3d.ncols();
    let mut a_mat = nalgebra::DMatrix::zeros(num_points * 2, 4);
    let mut b_vec = nalgebra::DVector::zeros(num_points * 2);

    for i in 0..num_points {
        let p3d = points_3d.column(i);
        let p2d = points_2d.column(i);

        let x_world = p3d.x;
        let y_world = p3d.y;
        let z_world = p3d.z;
        let u_img = p2d.x;
        let v_img = p2d.y;

        if z_world <= f64::EPSILON {
            continue;
        }

        let r_world = (x_world * x_world + y_world * y_world).sqrt();
        let theta = r_world.atan2(z_world);

        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        a_mat[(i * 2, 0)] = theta3;
        a_mat[(i * 2, 1)] = theta5;
        a_mat[(i * 2, 2)] = theta7;
        a_mat[(i * 2, 3)] = theta9;

        a_mat[(i * 2 + 1, 0)] = theta3;
        a_mat[(i * 2 + 1, 1)] = theta5;
        a_mat[(i * 2 + 1, 2)] = theta7;
        a_mat[(i * 2 + 1, 3)] = theta9;

        let x_r = if r_world < f64::EPSILON {
            0.0
        } else {
            x_world / r_world
        };
        let y_r = if r_world < f64::EPSILON {
            0.0
        } else {
            y_world / r_world
        };

        if x_r.abs() > f64::EPSILON {
            b_vec[i * 2] = (u_img - cam.intrinsics.cx) / (cam.intrinsics.fx * x_r) - theta;
        } else {
            b_vec[i * 2] = if (u_img - cam.intrinsics.cx).abs() < f64::EPSILON {
                -theta
            } else {
                0.0
            };
        }

        if y_r.abs() > f64::EPSILON {
            b_vec[i * 2 + 1] = (v_img - cam.intrinsics.cy) / (cam.intrinsics.fy * y_r) - theta;
        } else {
            b_vec[i * 2 + 1] = if (v_img - cam.intrinsics.cy).abs() < f64::EPSILON {
                -theta
            } else {
                0.0
            };
        }
    }

    let svd = a_mat.svd(true, true);
    let x_coeffs = svd
        .solve(&b_vec, f64::EPSILON)
        .map_err(|e_str| CameraModelError::NumericalError(format!("SVD solve failed: {e_str}")))?;

    cam.distortion_params = vec![x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]];
    info!(
        "KB linear estimation: k1={}, k2={}, k3={}, k4={}",
        x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]
    );
    Ok(())
}

/// Linear estimation for Radial-Tangential model.
/// Estimates k1, k2, k3 (tangential p1, p2 set to 0).
pub fn linear_estimation_rad_tan(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_2d.ncols() != points_3d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }

    let num_points = points_2d.ncols();
    if num_points < 3 {
        return Err(CameraModelError::InvalidParams(
            "Need at least 3 points for RadTan linear estimation".into(),
        ));
    }

    let mut a = nalgebra::DMatrix::zeros(num_points * 2, 3);
    let mut b = nalgebra::DVector::zeros(num_points * 2);

    let fx = cam.intrinsics.fx;
    let fy = cam.intrinsics.fy;
    let cx = cam.intrinsics.cx;
    let cy = cam.intrinsics.cy;

    for i in 0..num_points {
        let x = points_3d[(0, i)];
        let y = points_3d[(1, i)];
        let z = points_3d[(2, i)];
        let u = points_2d[(0, i)];
        let v = points_2d[(1, i)];

        let x_norm = x / z;
        let y_norm = y / z;
        let r2 = x_norm * x_norm + y_norm * y_norm;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let u_undist = fx * x_norm + cx;
        let v_undist = fy * y_norm + cy;

        a[(i * 2, 0)] = fx * x_norm * r2;
        a[(i * 2, 1)] = fx * x_norm * r4;
        a[(i * 2, 2)] = fx * x_norm * r6;

        a[(i * 2 + 1, 0)] = fy * y_norm * r2;
        a[(i * 2 + 1, 1)] = fy * y_norm * r4;
        a[(i * 2 + 1, 2)] = fy * y_norm * r6;

        b[i * 2] = u - u_undist;
        b[i * 2 + 1] = v - v_undist;
    }

    let svd = a.svd(true, true);
    let distortion_coeffs = match svd.solve(&b, 1e-10) {
        Ok(sol) => sol,
        Err(err_msg) => {
            return Err(CameraModelError::NumericalError(err_msg.to_string()));
        }
    };

    cam.distortion_params = vec![
        distortion_coeffs[0], // k1
        distortion_coeffs[1], // k2
        0.0,                  // p1
        0.0,                  // p2
        distortion_coeffs[2], // k3
    ];

    info!(
        "RadTan linear estimation: k1={}, k2={}, k3={}",
        distortion_coeffs[0], distortion_coeffs[1], distortion_coeffs[2]
    );
    Ok(())
}

/// Linear estimation for UCM model.
/// Estimates alpha.
pub fn linear_estimation_ucm(
    cam: &mut CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
) -> Result<(), CameraModelError> {
    if points_2d.ncols() != points_3d.ncols() {
        return Err(CameraModelError::InvalidParams(
            "Number of 2D and 3D points must match".into(),
        ));
    }

    let num_points = points_2d.ncols();
    let mut a = nalgebra::DMatrix::zeros(num_points * 2, 1);
    let mut b = nalgebra::DVector::zeros(num_points * 2);

    for i in 0..num_points {
        let x = points_3d[(0, i)];
        let y = points_3d[(1, i)];
        let z = points_3d[(2, i)];
        let u = points_2d[(0, i)];
        let v = points_2d[(1, i)];

        let d = (x * x + y * y + z * z).sqrt();
        let u_cx = u - cam.intrinsics.cx;
        let v_cy = v - cam.intrinsics.cy;

        a[(i * 2, 0)] = u_cx * (d - z);
        a[(i * 2 + 1, 0)] = v_cy * (d - z);

        b[i * 2] = (cam.intrinsics.fx * x) - (u_cx * z);
        b[i * 2 + 1] = (cam.intrinsics.fy * y) - (v_cy * z);
    }

    let svd = a.svd(true, true);
    let alpha = match svd.solve(&b, 1e-10) {
        Ok(sol) => sol[0],
        Err(err_msg) => {
            return Err(CameraModelError::NumericalError(err_msg.to_string()));
        }
    };

    let alpha = if alpha <= 0.0 { 0.01 } else { alpha };
    info!("UCM linear estimation: alpha = {}", alpha);

    cam.distortion_params = vec![alpha];
    Ok(())
}
