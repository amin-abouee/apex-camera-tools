use camera_tools::camera::{CameraModel, CameraWithResolution, YamlCamera};
use camera_tools::util::{InterpolationMethod, undistort_image};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "undistort",
    about = "Undistort images using camera calibration"
)]
struct Args {
    /// Input image path
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Camera calibration YAML file
    #[arg(short = 'c', long)]
    calib: PathBuf,

    /// Output image path
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Target focal length X (optional, defaults to source fx)
    #[arg(long)]
    target_fx: Option<f64>,

    /// Target focal length Y (optional, defaults to source fy)
    #[arg(long)]
    target_fy: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("Image Undistortion Tool");
    println!("=======================");
    println!("Input: {:?}", args.input);
    println!("Calibration: {:?}", args.calib);
    println!("Output: {:?}", args.output);
    println!();

    let calib_path = args.calib.to_str().ok_or("Invalid calibration path")?;
    let model = CameraWithResolution::load_from_yaml(calib_path)?;

    println!("Loaded {} camera model", model.get_model_name());
    let intrinsics = model.get_intrinsics();
    println!(
        "  fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    );
    let resolution = model.get_resolution();
    println!("  Resolution: {}x{}", resolution.width, resolution.height);
    println!();

    let img = image::open(&args.input)?.to_rgb8();
    println!("Loaded input image: {}x{}", img.width(), img.height());

    let target_intrinsics = if args.target_fx.is_some() || args.target_fy.is_some() {
        let mut target = intrinsics.clone();
        if let Some(fx) = args.target_fx {
            target.fx = fx;
        }
        if let Some(fy) = args.target_fy {
            target.fy = fy;
        }
        println!(
            "Using custom target focal lengths: fx={:.2}, fy={:.2}",
            target.fx, target.fy
        );
        Some(target)
    } else {
        None
    };

    println!("Undistorting image...");
    let undistorted = undistort_image(
        &img,
        &model,
        target_intrinsics,
        InterpolationMethod::Bilinear,
    )?;

    undistorted.save(&args.output)?;
    println!("Saved undistorted image to: {:?}", args.output);
    println!();
    println!("Done!");

    Ok(())
}
