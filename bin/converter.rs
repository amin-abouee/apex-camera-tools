use camera_tools::camera::{CameraModel, CameraWithResolution, YamlCamera};
use camera_tools::{convert, util};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "converter", about = "Convert camera models between formats")]
struct Args {
    /// Type of the input camera model (kb, ds, radtan, ucm, eucm, fov, pinhole)
    #[arg(short = 'i', long)]
    input_model: String,

    /// Path to the input model YAML file
    #[arg(short = 'p', long)]
    input_path: PathBuf,

    /// Number of sample points to generate for optimization (default: 500)
    #[arg(short = 'n', long, default_value = "500")]
    num_points: usize,

    /// Path to the input image for processing and quality assessment
    #[arg(short = 'm', long)]
    image_path: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("CAMERA MODEL CONVERTER - APEX-SOLVER");
    println!("=====================================");
    println!("Optimizer: apex-solver v1.2.1");
    println!("Method: Analytical Jacobians (hand-derived)");
    println!(
        "Input model: {} -> Converting to all supported target models",
        args.input_model.to_lowercase()
    );
    println!("Input file: {:?}", args.input_path);
    if let Some(ref image_path) = args.image_path {
        println!("Input image: {image_path:?}");
    } else {
        println!("Input image: None (synthetic image will be generated)");
    }
    println!("Sample points: {}", args.num_points);
    println!();

    let input_path_str = args
        .input_path
        .to_str()
        .ok_or("Invalid input path string")?;
    let input_model = CameraWithResolution::load_from_yaml(input_path_str)?;

    util::display_input_model_parameters(&args.input_model, &input_model);

    let reference_image = if let Some(ref image_path) = args.image_path {
        let image_path_str = image_path.to_str().ok_or("Invalid image path string")?;
        match util::load_image(image_path_str) {
            Ok(img) => {
                println!("Successfully loaded input image: {image_path:?}");
                println!("Image dimensions: {}x{}", img.width(), img.height());
                Some(img)
            }
            Err(e) => {
                println!("Failed to load image: {e}");
                println!("Continuing with synthetic image generation...");
                None
            }
        }
    } else {
        None
    };

    let (points_2d, points_3d) = util::sample_points(Some(&input_model), args.num_points)?;

    println!("\nGenerating Sample Points:");
    println!("Requested points: {}", args.num_points);
    println!("Generated {} sample points", points_3d.ncols());
    println!("\nCreating 3D-2D Point Correspondences:");
    println!(
        "Valid 3D-2D correspondences: {} / {}",
        points_2d.ncols(),
        args.num_points
    );

    util::export_point_correspondences(&points_3d, &points_2d, "point_correspondences_apex")?;

    let resolution = input_model.get_resolution();
    util::model_projection_visualization(
        &points_2d,
        &args.input_model,
        reference_image.as_ref(),
        (resolution.width, resolution.height),
    )?;

    println!("\nStep 3: Converting to All Target Models");
    println!("========================================");
    let mut all_metrics = Vec::new();
    let model_lower = args.input_model.to_lowercase();

    if !matches!(model_lower.as_str(), "ds" | "double_sphere") {
        println!("\nConverting {model_lower} -> Double Sphere");
        if let Ok(m) = convert::convert_to_double_sphere(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    if !matches!(model_lower.as_str(), "kb" | "kannala_brandt") {
        println!("\nConverting {model_lower} -> Kannala-Brandt");
        if let Ok(m) = convert::convert_to_kannala_brandt(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    if !matches!(model_lower.as_str(), "radtan" | "rad_tan") {
        println!("\nConverting {model_lower} -> Radial-Tangential");
        if let Ok(m) = convert::convert_to_rad_tan(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    if !matches!(model_lower.as_str(), "ucm" | "unified") {
        println!("\nConverting {model_lower} -> Unified Camera Model (UCM)");
        if let Ok(m) = convert::convert_to_ucm(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    if !matches!(model_lower.as_str(), "eucm" | "extended_unified") {
        println!("\nConverting {model_lower} -> Extended Unified Camera Model (EUCM)");
        if let Ok(m) = convert::convert_to_eucm(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    if !matches!(model_lower.as_str(), "fov" | "field_of_view") {
        println!("\nConverting {model_lower} -> Field-of-View (FOV)");
        if let Ok(m) = convert::convert_to_fov(
            &input_model,
            &points_3d,
            &points_2d,
            reference_image.as_ref(),
        ) {
            all_metrics.push(m);
        }
    }

    util::display_results_summary(&all_metrics, &args.input_model);

    Ok(())
}
