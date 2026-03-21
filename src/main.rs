mod camera;
mod commands;
mod util;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "camera-tools", about = "Camera model conversion and image undistortion")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a camera model to all supported target models
    Convert(commands::convert::ConvertArgs),
    /// Undistort an image using a camera calibration file
    Undistort(commands::undistort::UndistortArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Undistort(args) => commands::undistort::run(args),
    }
}
