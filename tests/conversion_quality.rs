//! Conversion quality tests — sanity check against fisheye-calib-adapter (C++).
//!
//! Reference parameters taken from fisheye-calib-adapter/dataset/kalibr/*.yml,
//! which were calibrated from the same 752×480 fisheye camera.
//!
//! The test strategy mirrors fisheye-calib-adapter's validation:
//!   1. Sample 500 2D points from the input model
//!   2. Unproject to 3D rays
//!   3. Linear estimation → Levenberg-Marquardt optimization → target model
//!   4. Check reprojection RMSE against threshold
//!   5. Compare estimated parameters against the C++ reference values

use camera_tools::camera::{CameraModel, CameraWithResolution, YamlCamera};
use camera_tools::convert;
use camera_tools::util;
use nalgebra::{Matrix2xX, Matrix3xX, Vector2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_ds_camera() -> CameraWithResolution {
    CameraWithResolution::load_from_yaml("samples/double_sphere.yaml")
        .expect("Failed to load samples/double_sphere.yaml")
}

fn sample_500(camera: &CameraWithResolution) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
    util::sample_points(Some(camera), 500).expect("Failed to sample 500 points")
}

/// Project 3D points through a camera model to produce 2D-3D correspondences.
/// Only keeps points that successfully project within the image bounds.
fn reproject_points(
    model: &CameraWithResolution,
    points_3d: &Matrix3xX<f64>,
) -> (Matrix2xX<f64>, Matrix3xX<f64>) {
    let mut valid_2d = Vec::new();
    let mut valid_3d = Vec::new();

    let w = model.get_resolution().width as f64;
    let h = model.get_resolution().height as f64;

    for i in 0..points_3d.ncols() {
        let p3d = points_3d.column(i).into_owned();
        if let Ok(p2d) = model.project(&p3d)
            && p2d.x >= 0.0
            && p2d.x < w
            && p2d.y >= 0.0
            && p2d.y < h
        {
            valid_2d.push(p2d);
            valid_3d.push(p3d);
        }
    }

    let n = valid_2d.len();
    let mut pts2d = Matrix2xX::zeros(n);
    let mut pts3d = Matrix3xX::zeros(n);
    for (i, (p2, p3)) in valid_2d.iter().zip(valid_3d.iter()).enumerate() {
        pts2d.set_column(i, &Vector2::new(p2.x, p2.y));
        pts3d.set_column(i, p3);
    }
    (pts2d, pts3d)
}

// ---------------------------------------------------------------------------
// C++ fisheye-calib-adapter reference parameters (dataset/kalibr/*.yml)
// All calibrated from the same 752×480 camera as our double_sphere.yaml.
// ---------------------------------------------------------------------------

mod cpp_ref {
    pub const KB_FX: f64 = 461.58688085556616;
    pub const KB_FY: f64 = 460.2811732644195;
    pub const KB_CX: f64 = 366.28603126815506;
    pub const KB_CY: f64 = 249.08026891791644;

    pub const EUCM_FX: f64 = 460.92588889854693;
    pub const EUCM_FY: f64 = 459.59327980206865;
    pub const EUCM_CX: f64 = 365.81762690405503;
    pub const EUCM_CY: f64 = 248.93674967920276;
    pub const EUCM_ALPHA: f64 = 0.5864679700626143;
    pub const EUCM_BETA: f64 = 1.139248617863534;

    pub const RADTAN_FX: f64 = 459.2595646276051;
    pub const RADTAN_FY: f64 = 457.7579979599497;
    pub const RADTAN_CX: f64 = 366.021259576013;
    pub const RADTAN_CY: f64 = 248.29906707021513;
    pub const RADTAN_K1: f64 = -0.2895683327836746;
    pub const RADTAN_K2: f64 = 0.07964702146041833;
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tolerance,
        "{label}: actual={actual:.6}, expected={expected:.6}, diff={diff:.6}, tol={tolerance}"
    );
}

// ===========================================================================
// Forward conversion quality tests: DS → each model
// ===========================================================================

#[test]
fn test_ds_to_kannala_brandt_quality() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);
    let metrics = convert::convert_to_kannala_brandt(&ds, &pts3d, &pts2d, None)
        .expect("DS→KB conversion failed");

    let rmse = metrics.final_reprojection_error.rmse;
    assert!(
        rmse < 5.0,
        "DS→KB RMSE {rmse:.4} px exceeds threshold 5.0 px"
    );

    // KB distortion params can find equivalent but numerically different local minima,
    // so only compare intrinsics against the C++ reference.
    let m = &metrics.model;
    assert_close("KB fx", m.intrinsics.fx, cpp_ref::KB_FX, 15.0);
    assert_close("KB fy", m.intrinsics.fy, cpp_ref::KB_FY, 15.0);
    assert_close("KB cx", m.intrinsics.cx, cpp_ref::KB_CX, 5.0);
    assert_close("KB cy", m.intrinsics.cy, cpp_ref::KB_CY, 5.0);
}

#[test]
fn test_ds_to_eucm_quality() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);
    let metrics =
        convert::convert_to_eucm(&ds, &pts3d, &pts2d, None).expect("DS→EUCM conversion failed");

    let rmse = metrics.final_reprojection_error.rmse;
    assert!(
        rmse < 2.0,
        "DS→EUCM RMSE {rmse:.4} px exceeds threshold 2.0 px"
    );

    let m = &metrics.model;
    assert_close("EUCM fx", m.intrinsics.fx, cpp_ref::EUCM_FX, 15.0);
    assert_close("EUCM fy", m.intrinsics.fy, cpp_ref::EUCM_FY, 15.0);
    assert_close("EUCM cx", m.intrinsics.cx, cpp_ref::EUCM_CX, 5.0);
    assert_close("EUCM cy", m.intrinsics.cy, cpp_ref::EUCM_CY, 5.0);
    assert_close(
        "EUCM alpha",
        m.distortion_params[0],
        cpp_ref::EUCM_ALPHA,
        0.15,
    );
    assert_close(
        "EUCM beta",
        m.distortion_params[1],
        cpp_ref::EUCM_BETA,
        0.25,
    );
}

#[test]
fn test_ds_to_ucm_quality() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);
    let metrics =
        convert::convert_to_ucm(&ds, &pts3d, &pts2d, None).expect("DS→UCM conversion failed");

    let rmse = metrics.final_reprojection_error.rmse;
    assert!(
        rmse < 5.0,
        "DS→UCM RMSE {rmse:.4} px exceeds threshold 5.0 px"
    );
    // No C++ reference for same resolution — UCM ref is 1920×1080 (different camera)
}

#[test]
fn test_ds_to_fov_quality() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);
    let metrics =
        convert::convert_to_fov(&ds, &pts3d, &pts2d, None).expect("DS→FOV conversion failed");

    let rmse = metrics.final_reprojection_error.rmse;
    assert!(
        rmse < 10.0,
        "DS→FOV RMSE {rmse:.4} px exceeds threshold 10.0 px"
    );
}

#[test]
fn test_ds_to_radtan_quality() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);
    let metrics = convert::convert_to_rad_tan(&ds, &pts3d, &pts2d, None)
        .expect("DS→RadTan conversion failed");

    let rmse = metrics.final_reprojection_error.rmse;
    assert!(
        rmse.is_finite(),
        "DS→RadTan RMSE should be finite, got {rmse}"
    );
    assert!(
        rmse < 50.0,
        "DS→RadTan RMSE {rmse:.4} px exceeds existence threshold 50.0 px"
    );

    let m = &metrics.model;
    assert_close("RadTan fx", m.intrinsics.fx, cpp_ref::RADTAN_FX, 30.0);
    assert_close("RadTan fy", m.intrinsics.fy, cpp_ref::RADTAN_FY, 30.0);
    assert_close("RadTan cx", m.intrinsics.cx, cpp_ref::RADTAN_CX, 10.0);
    assert_close("RadTan cy", m.intrinsics.cy, cpp_ref::RADTAN_CY, 10.0);
    assert_close(
        "RadTan k1",
        m.distortion_params[0],
        cpp_ref::RADTAN_K1,
        0.15,
    );
    assert_close(
        "RadTan k2",
        m.distortion_params[1],
        cpp_ref::RADTAN_K2,
        0.10,
    );
}

// ===========================================================================
// Round-trip tests: DS → X → DS
//
// The second leg reprojects the original 3D points through the intermediate
// model to get new 2D correspondences, avoiding the issue where some models
// (EUCM, UCM) have narrower valid unprojection regions than DS.
// ===========================================================================

#[test]
fn test_round_trip_ds_eucm_ds() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);

    // Leg 1: DS → EUCM
    let fwd = convert::convert_to_eucm(&ds, &pts3d, &pts2d, None).expect("DS→EUCM failed");
    assert!(
        fwd.final_reprojection_error.rmse < 2.0,
        "Leg 1 (DS→EUCM) RMSE {:.4} px",
        fwd.final_reprojection_error.rmse
    );

    // Leg 2: EUCM → DS (reproject original 3D points through EUCM)
    let (pts2d_e, pts3d_e) = reproject_points(&fwd.model, &pts3d);
    assert!(
        pts2d_e.ncols() > 0,
        "No valid reprojections through EUCM model"
    );
    let back = convert::convert_to_double_sphere(&fwd.model, &pts3d_e, &pts2d_e, None)
        .expect("EUCM→DS failed");
    assert!(
        back.final_reprojection_error.rmse < 5.0,
        "Leg 2 (EUCM→DS) RMSE {:.4} px",
        back.final_reprojection_error.rmse
    );
}

#[test]
fn test_round_trip_ds_ucm_ds() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);

    let fwd = convert::convert_to_ucm(&ds, &pts3d, &pts2d, None).expect("DS→UCM failed");
    assert!(
        fwd.final_reprojection_error.rmse < 5.0,
        "Leg 1 (DS→UCM) RMSE {:.4} px",
        fwd.final_reprojection_error.rmse
    );

    let (pts2d_u, pts3d_u) = reproject_points(&fwd.model, &pts3d);
    assert!(
        pts2d_u.ncols() > 0,
        "No valid reprojections through UCM model"
    );
    let back = convert::convert_to_double_sphere(&fwd.model, &pts3d_u, &pts2d_u, None)
        .expect("UCM→DS failed");
    assert!(
        back.final_reprojection_error.rmse < 5.0,
        "Leg 2 (UCM→DS) RMSE {:.4} px",
        back.final_reprojection_error.rmse
    );
}

#[test]
fn test_round_trip_ds_kannala_brandt_ds() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);

    let fwd = convert::convert_to_kannala_brandt(&ds, &pts3d, &pts2d, None).expect("DS→KB failed");
    assert!(
        fwd.final_reprojection_error.rmse < 5.0,
        "Leg 1 (DS→KB) RMSE {:.4} px",
        fwd.final_reprojection_error.rmse
    );

    let (pts2d_k, pts3d_k) = reproject_points(&fwd.model, &pts3d);
    assert!(
        pts2d_k.ncols() > 0,
        "No valid reprojections through KB model"
    );
    let back = convert::convert_to_double_sphere(&fwd.model, &pts3d_k, &pts2d_k, None)
        .expect("KB→DS failed");
    assert!(
        back.final_reprojection_error.rmse < 5.0,
        "Leg 2 (KB→DS) RMSE {:.4} px",
        back.final_reprojection_error.rmse
    );
}

#[test]
fn test_round_trip_ds_fov_ds() {
    let ds = load_ds_camera();
    let (pts2d, pts3d) = sample_500(&ds);

    let fwd = convert::convert_to_fov(&ds, &pts3d, &pts2d, None).expect("DS→FOV failed");
    assert!(
        fwd.final_reprojection_error.rmse < 10.0,
        "Leg 1 (DS→FOV) RMSE {:.4} px",
        fwd.final_reprojection_error.rmse
    );

    let (pts2d_f, pts3d_f) = reproject_points(&fwd.model, &pts3d);
    assert!(
        pts2d_f.ncols() > 0,
        "No valid reprojections through FOV model"
    );
    let back = convert::convert_to_double_sphere(&fwd.model, &pts3d_f, &pts2d_f, None)
        .expect("FOV→DS failed");
    assert!(
        back.final_reprojection_error.rmse < 10.0,
        "Leg 2 (FOV→DS) RMSE {:.4} px",
        back.final_reprojection_error.rmse
    );
}
