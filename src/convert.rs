use crate::camera::{CameraModel, CameraWithResolution, estimation};
use crate::util::{self, ConversionMetrics, ValidationResults};
use apex_camera_models::DistortionModel;
use apex_solver::core::problem::{Problem, VariableEnum};
use apex_solver::factors::{OnlyIntrinsics, ProjectionFactor};
use apex_solver::manifold::se3::SE3;
use apex_solver::optimizer::levenberg_marquardt::{LevenbergMarquardt, LevenbergMarquardtConfig};
use apex_solver::{
    DoubleSphereCamera, EucmCamera, FovCamera, KannalaBrandtCamera, ManifoldType, RadTanCamera,
    UcmCamera,
};
use log::info;
use nalgebra::DVector;
use std::collections::HashMap;
use std::time::Instant;

pub fn lm_config() -> LevenbergMarquardtConfig {
    LevenbergMarquardtConfig::new()
        .with_max_iterations(100)
        .with_cost_tolerance(1e-6)
        .with_parameter_tolerance(1e-8)
        .with_gradient_tolerance(1e-6)
}

fn extract_optimized(
    result: &apex_solver::optimizer::SolverResult<HashMap<String, VariableEnum>>,
    key: &str,
) -> Result<DVector<f64>, Box<dyn std::error::Error>> {
    let var = result
        .parameters
        .get(key)
        .ok_or("Variable not found in result")?;
    Ok(var.to_vector())
}

fn default_validation() -> ValidationResults {
    ValidationResults {
        center_error: f64::NAN,
        near_center_error: f64::NAN,
        mid_region_error: f64::NAN,
        edge_region_error: f64::NAN,
        far_edge_error: f64::NAN,
        average_error: f64::NAN,
        max_error: f64::NAN,
        status: "NEEDS IMPROVEMENT".to_string(),
        region_data: vec![],
    }
}

pub fn convert_to_double_sphere(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "double_sphere".to_string(),
        distortion_params: vec![0.5, 0.1],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_double_sphere(&mut model, points_3d, points_2d)?;

    let ds_alpha = model.distortion_params[0];
    let ds_xi = model.distortion_params[1];

    let camera = DoubleSphereCamera::new(
        model.pinhole_params(),
        DistortionModel::DoubleSphere {
            xi: ds_xi,
            alpha: ds_alpha,
        },
    )?;
    let factor: ProjectionFactor<DoubleSphereCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        ds_xi,
        ds_alpha,
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, -5.0, 5.0);
    problem.set_variable_bounds("params", 5, 1e-6, 1.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params[0] = p[5]; // alpha
            model.distortion_params[1] = p[4]; // xi
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "double_sphere_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Double Sphere".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}

pub fn convert_to_kannala_brandt(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "kannala_brandt".to_string(),
        distortion_params: vec![0.0; 4],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_kannala_brandt(&mut model, points_3d, points_2d)?;

    let camera = KannalaBrandtCamera::new(
        model.pinhole_params(),
        DistortionModel::KannalaBrandt {
            k1: model.distortion_params[0],
            k2: model.distortion_params[1],
            k3: model.distortion_params[2],
            k4: model.distortion_params[3],
        },
    )?;
    let factor: ProjectionFactor<KannalaBrandtCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortion_params[0],
        model.distortion_params[1],
        model.distortion_params[2],
        model.distortion_params[3],
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, -5.0, 5.0);
    problem.set_variable_bounds("params", 5, -5.0, 5.0);
    problem.set_variable_bounds("params", 6, -5.0, 5.0);
    problem.set_variable_bounds("params", 7, -5.0, 5.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params = vec![p[4], p[5], p[6], p[7]];
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "kannala_brandt_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Kannala-Brandt".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}

pub fn convert_to_rad_tan(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "rad_tan".to_string(),
        distortion_params: vec![0.0; 5],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_rad_tan(&mut model, points_3d, points_2d)?;

    let camera = RadTanCamera::new(
        model.pinhole_params(),
        DistortionModel::BrownConrady {
            k1: model.distortion_params[0],
            k2: model.distortion_params[1],
            p1: model.distortion_params[2],
            p2: model.distortion_params[3],
            k3: model.distortion_params[4],
        },
    )?;
    let factor: ProjectionFactor<RadTanCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortion_params[0],
        model.distortion_params[1],
        model.distortion_params[2],
        model.distortion_params[3],
        model.distortion_params[4],
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, -5.0, 5.0);
    problem.set_variable_bounds("params", 5, -5.0, 5.0);
    problem.set_variable_bounds("params", 6, -1.0, 1.0);
    problem.set_variable_bounds("params", 7, -1.0, 1.0);
    problem.set_variable_bounds("params", 8, -5.0, 5.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params = vec![p[4], p[5], p[6], p[7], p[8]];
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "radial_tangential_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Radial-Tangential".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}

pub fn convert_to_ucm(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "ucm".to_string(),
        distortion_params: vec![0.5],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_ucm(&mut model, points_3d, points_2d)?;

    let camera = UcmCamera::new(
        model.pinhole_params(),
        DistortionModel::UCM {
            alpha: model.distortion_params[0],
        },
    )?;
    let factor: ProjectionFactor<UcmCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortion_params[0],
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, 1e-6, 10.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params = vec![p[4]];
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}

pub fn convert_to_eucm(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "eucm".to_string(),
        distortion_params: vec![0.5, 1.0],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_eucm(&mut model, points_3d, points_2d)?;

    let camera = EucmCamera::new(
        model.pinhole_params(),
        DistortionModel::EUCM {
            alpha: model.distortion_params[0],
            beta: model.distortion_params[1],
        },
    )?;
    let factor: ProjectionFactor<EucmCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortion_params[0],
        model.distortion_params[1],
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, 1e-6, 1.0);
    problem.set_variable_bounds("params", 5, 1e-6, 5.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params = vec![p[4], p[5]];
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "extended_unified_camera_model_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Extended Unified Camera Model".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}

pub fn convert_to_fov(
    input_model: &CameraWithResolution,
    points_3d: &nalgebra::Matrix3xX<f64>,
    points_2d: &nalgebra::Matrix2xX<f64>,
    reference_image: Option<&image::RgbImage>,
) -> Result<ConversionMetrics, Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let mut model = CameraWithResolution {
        intrinsics: input_model.get_intrinsics(),
        resolution: input_model.get_resolution(),
        model_name: "fov".to_string(),
        distortion_params: vec![1.0],
    };

    let initial_error = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    estimation::linear_estimation_fov(&mut model, points_3d, points_2d)?;

    let camera = FovCamera::new(
        model.pinhole_params(),
        DistortionModel::FOV {
            w: model.distortion_params[0],
        },
    )?;
    let factor: ProjectionFactor<FovCamera, OnlyIntrinsics> =
        ProjectionFactor::new(points_2d.clone(), camera)
            .with_fixed_pose(SE3::identity())
            .with_fixed_landmarks(points_3d.clone());

    let mut problem = Problem::new();
    problem.add_residual_block(&["params"], Box::new(factor), None);

    let initial_params = DVector::from_vec(vec![
        model.intrinsics.fx,
        model.intrinsics.fy,
        model.intrinsics.cx,
        model.intrinsics.cy,
        model.distortion_params[0],
    ]);

    problem.set_variable_bounds("params", 0, 1.0, 2000.0);
    problem.set_variable_bounds("params", 1, 1.0, 2000.0);
    problem.set_variable_bounds("params", 2, 0.0, 2000.0);
    problem.set_variable_bounds("params", 3, 0.0, 2000.0);
    problem.set_variable_bounds("params", 4, 1e-6, 3.0);

    let mut initial_values = HashMap::new();
    initial_values.insert("params".to_string(), (ManifoldType::RN, initial_params));

    let mut solver = LevenbergMarquardt::with_config(lm_config());
    let optimization_result = solver.optimize(&problem, &initial_values);
    let optimization_time = start_time.elapsed().as_millis() as f64;

    let convergence_status = match &optimization_result {
        Ok(result) => {
            let p = extract_optimized(result, "params")?;
            model.intrinsics.fx = p[0];
            model.intrinsics.fy = p[1];
            model.intrinsics.cx = p[2];
            model.intrinsics.cy = p[3];
            model.distortion_params = vec![p[4]];
            "Converged".to_string()
        }
        Err(_) => "Linear Only".to_string(),
    };

    let reprojection_result = util::compute_reprojection_error(Some(&model), points_3d, points_2d)?;
    let validation_results = util::validate_conversion_accuracy(&model, input_model)
        .unwrap_or_else(|_| default_validation());

    let image_quality = util::compute_image_quality_metrics(
        input_model,
        &model,
        points_3d,
        "fov_apex",
        reference_image,
    )
    .map_err(|e| info!("Failed to compute image quality metrics: {e:?}"))
    .ok();

    let metrics = ConversionMetrics {
        model,
        model_name: "Field-of-View".to_string(),
        final_reprojection_error: reprojection_result,
        initial_reprojection_error: initial_error,
        optimization_time_ms: optimization_time,
        convergence_status,
        validation_results,
        image_quality,
    };
    util::display_detailed_results(&metrics);
    Ok(metrics)
}
