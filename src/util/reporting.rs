//! Reporting and display utilities for camera model conversions.
//!
//! This module provides functionality for displaying and exporting conversion results,
//! including console output formatting and file export in various formats.

use crate::camera::{CameraModel, CameraWithResolution};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

use super::error_metrics::ProjectionError;
use super::image_quality::ImageQualityMetrics;
use super::validation::ValidationResults;
use super::{ensure_output_dir, UtilError};

/// Comprehensive metrics for camera model conversion evaluation.
///
/// Contains all information about a conversion from one camera model to another,
/// including error statistics, optimization performance, and validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMetrics {
    /// The converted camera model
    pub model: CameraWithResolution,
    /// Human-readable name of the model
    pub model_name: String,
    /// Final reprojection error after optimization
    pub final_reprojection_error: ProjectionError,
    /// Initial reprojection error before optimization
    pub initial_reprojection_error: ProjectionError,
    /// Time taken for optimization in milliseconds
    pub optimization_time_ms: f64,
    /// Status message about convergence
    pub convergence_status: String,
    /// Validation results across different image regions
    pub validation_results: ValidationResults,
    /// Optional image quality metrics (PSNR, SSIM)
    pub image_quality: Option<ImageQualityMetrics>,
}

/// Display input camera model parameters in a formatted way.
///
/// Shows the intrinsic parameters and distortion coefficients specific
/// to each camera model type.
///
/// # Arguments
///
/// * `model_type` - Type identifier of the camera model (kb, ds, radtan, etc.)
/// * `camera_model` - The camera model instance
pub fn display_input_model_parameters(model_type: &str, camera_model: &dyn CameraModel) {
    println!("📷 Input Model Parameters:");
    let intrinsics = camera_model.get_intrinsics();
    let distortion = camera_model.get_distortion();

    match model_type.to_lowercase().as_str() {
        "ds" | "double_sphere" => {
            println!(
                "DS parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, xi={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "kb" | "kannala_brandt" => {
            println!(
                "KB parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, k3={:.6}, k4={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3]
            );
        }
        "radtan" | "rad_tan" => {
            println!(
                "RadTan parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}, k3={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]
            );
        }
        "ucm" | "unified" => {
            println!(
                "UCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, distortion[0]
            );
        }
        "eucm" | "extended_unified" => {
            println!(
                "EUCM parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, alpha={:.6}, beta={:.6}",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                distortion[0],
                distortion[1]
            );
        }
        "pinhole" => {
            println!(
                "Pinhole parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
        _ => {
            println!(
                "Model parameters: fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            );
        }
    }
}

/// Display detailed results for a single camera model conversion.
///
/// Shows comprehensive information including model parameters, reprojection errors,
/// validation results across different image regions, and image quality metrics.
///
/// # Arguments
///
/// * `metrics` - Conversion metrics containing all the results
pub fn display_detailed_results(metrics: &ConversionMetrics) {
    println!("\n📊 Final Output Model Parameters:");
    println!("{:?}", metrics.model);
    println!("computing time(ms): {:.0}", metrics.optimization_time_ms);

    println!("\n🧪 EVALUATION AND VALIDATION:");
    println!("=============================");

    // Print detailed reprojection error information
    let final_error = &metrics.final_reprojection_error;
    println!("Reprojection Error Statistics:");
    println!("  Mean: {:.8} px", final_error.mean);
    println!("  RMSE: {:.8} px", final_error.rmse);
    println!("  Min: {:.8} px", final_error.min);
    println!("  Max: {:.8} px", final_error.max);
    println!("  Std Dev: {:.8} px", final_error.stddev);
    println!("  Median: {:.8} px", final_error.median);

    // Print validation results with actual calculated values
    let validation = &metrics.validation_results;
    println!("\n🎯 Conversion Accuracy Validation:");

    let region_names = [
        "Center",
        "Near Center",
        "Mid Region",
        "Edge Region",
        "Far Edge",
    ];
    let region_errors = [
        validation.center_error,
        validation.near_center_error,
        validation.mid_region_error,
        validation.edge_region_error,
        validation.far_edge_error,
    ];

    for (i, (name, error)) in region_names.iter().zip(region_errors.iter()).enumerate() {
        if !error.is_nan() {
            // Use actual projected coordinates if available from region_data
            if i < validation.region_data.len() {
                let region = &validation.region_data[i];
                if let (Some(input_coords), Some(output_coords)) =
                    (&region.input_projection, &region.output_projection)
                {
                    println!(
                        "  {}: Input({:.2}, {:.2}) → Output({:.2}, {:.2}) | Error: {:.4} px",
                        name,
                        input_coords.0,
                        input_coords.1,
                        output_coords.0,
                        output_coords.1,
                        error
                    );
                } else {
                    println!("  {name}: Projection failed");
                }
            } else {
                // Fallback for when region_data is not populated
                println!("  {name}: Error: {error:.4} px");
            }
        } else {
            println!("  {name}: Projection failed");
        }
    }

    println!(
        "  📈 Average Error: {:.4} px, Max Error: {:.4} px",
        validation.average_error, validation.max_error
    );

    if validation.status == "EXCELLENT" {
        println!("  ✅ Conversion Accuracy: {}", validation.status);
    } else {
        println!("  ⚠️  Conversion Accuracy: {}", validation.status);
    }

    // Print image quality metrics if available
    if let Some(ref image_quality) = metrics.image_quality {
        println!("\n📸 Image Quality Assessment:");
        println!("  PSNR: {:.2} dB", image_quality.psnr);
        println!("  SSIM: {:.4}", image_quality.ssim);
    }
}

/// Export conversion results to a detailed text file.
///
/// Creates a comprehensive report including:
/// - Conversion results table with errors and timing
/// - Performance analysis (best accuracy, fastest conversion)
/// - Detailed metrics for each model
/// - Validation and image quality results
///
/// # Arguments
///
/// * `metrics` - Vector of conversion metrics for all models
/// * `input_model_type` - Type of input model (for filename generation)
///
/// # Returns
///
/// * `Result<(), UtilError>` - Success or error
///
/// # Errors
///
/// * `UtilError::NumericalError` - If file creation or writing fails
pub fn export_conversion_results(
    metrics: &[ConversionMetrics],
    input_model_type: &str,
) -> Result<(), UtilError> {
    ensure_output_dir()?;

    let report_filename = format!(
        "output/camera_conversion_results_{}.txt",
        input_model_type.to_lowercase()
    );
    let mut report_file = File::create(&report_filename)
        .map_err(|e| UtilError::NumericalError(format!("Failed to create report file: {e}")))?;

    writeln!(
        report_file,
        "FISHEYE CAMERA MODEL CONVERSION ANALYSIS REPORT - RUST IMPLEMENTATION"
    )?;
    writeln!(
        report_file,
        "====================================================================="
    )?;
    writeln!(report_file)?;
    writeln!(
        report_file,
        "INPUT MODEL TYPE: {}",
        input_model_type.to_uppercase()
    )?;
    writeln!(report_file, "OPTIMIZATION FRAMEWORK: tiny-solver")?;
    writeln!(report_file, "ALGORITHM: Levenberg-Marquardt")?;
    writeln!(report_file)?;

    if metrics.is_empty() {
        writeln!(report_file, "❌ No conversions performed (input model type not supported for conversion or no target models available)")?;
        return Ok(());
    }

    // Export detailed results table
    writeln!(report_file, "CONVERSION RESULTS TABLE")?;
    writeln!(report_file, "========================")?;
    writeln!(
        report_file,
        "{:<32} | {:>15} | {:>15} | {:>13} | {:>15}",
        "Target Model", "Final Error", "Improvement", "Time (ms)", "Convergence"
    )?;
    writeln!(
        report_file,
        "{:<32} | {:>15} | {:>15} | {:>13} | {:>15}",
        "", "(pixels)", "(pixels)", "", "Status"
    )?;
    writeln!(
        report_file,
        "{:-<32}-+-{:-<15}-+-{:-<15}-+-{:-<13}-+-{:-<15}",
        "", "", "", "", ""
    )?;

    for metric in metrics {
        let improvement =
            metric.initial_reprojection_error.mean - metric.final_reprojection_error.mean;
        writeln!(
            report_file,
            "{:<32} | {:>13.6}   | {:>13.6}   | {:>11.2}   | {:<15}",
            metric.model_name,
            metric.final_reprojection_error.mean,
            improvement,
            metric.optimization_time_ms,
            metric.convergence_status
        )?;
    }
    writeln!(report_file)?;

    // Performance analysis
    writeln!(report_file, "PERFORMANCE ANALYSIS")?;
    writeln!(report_file, "====================")?;

    let best_accuracy = metrics.iter().min_by(|a, b| {
        a.final_reprojection_error
            .mean
            .partial_cmp(&b.final_reprojection_error.mean)
            .unwrap()
    });
    let fastest_conversion = metrics.iter().min_by(|a, b| {
        a.optimization_time_ms
            .partial_cmp(&b.optimization_time_ms)
            .unwrap()
    });

    if let Some(best) = best_accuracy {
        writeln!(
            report_file,
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error.mean
        )?;
    }
    if let Some(fastest) = fastest_conversion {
        writeln!(
            report_file,
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        )?;
    }

    let avg_error = metrics
        .iter()
        .map(|m| m.final_reprojection_error.mean)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_time =
        metrics.iter().map(|m| m.optimization_time_ms).sum::<f64>() / metrics.len() as f64;

    writeln!(
        report_file,
        "📊 Average Reprojection Error: {avg_error:.6} pixels"
    )?;
    writeln!(
        report_file,
        "📊 Average Optimization Time: {avg_time:.2} ms"
    )?;
    writeln!(report_file)?;

    // Detailed metrics for each model
    writeln!(report_file, "DETAILED MODEL RESULTS")?;
    writeln!(report_file, "======================")?;

    for metric in metrics {
        writeln!(report_file, "\n{} MODEL:", metric.model_name.to_uppercase())?;
        writeln!(report_file, "{}", "-".repeat(metric.model_name.len() + 7))?;
        writeln!(report_file, "Final Parameters: {:?}", metric.model)?;
        writeln!(
            report_file,
            "Optimization Time: {:.2} ms",
            metric.optimization_time_ms
        )?;
        writeln!(
            report_file,
            "Convergence Status: {}",
            metric.convergence_status
        )?;

        writeln!(report_file, "\nReprojection Error Statistics:")?;
        writeln!(
            report_file,
            "  Mean: {:.8} px",
            metric.final_reprojection_error.mean
        )?;
        writeln!(
            report_file,
            "  RMSE: {:.8} px",
            metric.final_reprojection_error.rmse
        )?;
        writeln!(
            report_file,
            "  Min: {:.8} px",
            metric.final_reprojection_error.min
        )?;
        writeln!(
            report_file,
            "  Max: {:.8} px",
            metric.final_reprojection_error.max
        )?;
        writeln!(
            report_file,
            "  Std Dev: {:.8} px",
            metric.final_reprojection_error.stddev
        )?;
        writeln!(
            report_file,
            "  Median: {:.8} px",
            metric.final_reprojection_error.median
        )?;

        writeln!(report_file, "\nConversion Accuracy:")?;
        let validation = &metric.validation_results;
        writeln!(
            report_file,
            "  Average Error: {:.4} px",
            validation.average_error
        )?;
        writeln!(report_file, "  Max Error: {:.4} px", validation.max_error)?;
        writeln!(report_file, "  Status: {}", validation.status)?;

        if let Some(ref image_quality) = metric.image_quality {
            writeln!(report_file, "\nImage Quality Assessment:")?;
            writeln!(report_file, "  PSNR: {:.2} dB", image_quality.psnr)?;
            writeln!(report_file, "  SSIM: {:.4}", image_quality.ssim)?;
        }
    }

    Ok(())
}

/// Display results summary table and performance analysis.
///
/// Prints a formatted table of conversion results to the console,
/// along with performance analysis and automatically exports results to a file.
///
/// # Arguments
///
/// * `metrics` - Vector of conversion metrics for all models
/// * `input_model_type` - Type of input model
pub fn display_results_summary(metrics: &[ConversionMetrics], input_model_type: &str) {
    // Step 4: Generate comprehensive benchmark report
    println!("\n📊 Step 4: Conversion Results Summary");
    println!("====================================");

    if metrics.is_empty() {
        println!("❌ No conversions performed (input model type not supported for conversion or no target models available)");
        return;
    }

    // Print detailed results table
    println!("\n📋 CONVERSION RESULTS TABLE");
    println!("┌────────────────────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("│ Target Model                   │ Final Error     │ Improvement     │ Time (ms)       │ Convergence     │");
    println!("│                                │ (pixels)        │ (pixels)        │                 │ Status          │");
    println!("├────────────────────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┤");

    for metric in metrics {
        let improvement =
            metric.initial_reprojection_error.mean - metric.final_reprojection_error.mean;
        println!(
            "│ {:<30} │ {:>13.6}   │ {:>13.6}   │ {:>13.2}   │ {:<15} │",
            metric.model_name,
            metric.final_reprojection_error.mean,
            improvement,
            metric.optimization_time_ms,
            metric.convergence_status
        );
    }
    println!("└────────────────────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘");

    // Step 5: Performance analysis
    println!("\n📈 Step 5: Performance Analysis");
    println!("===============================");

    let best_accuracy = metrics.iter().min_by(|a, b| {
        a.final_reprojection_error
            .mean
            .partial_cmp(&b.final_reprojection_error.mean)
            .unwrap()
    });
    let fastest_conversion = metrics.iter().min_by(|a, b| {
        a.optimization_time_ms
            .partial_cmp(&b.optimization_time_ms)
            .unwrap()
    });

    if let Some(best) = best_accuracy {
        println!(
            "🏆 Best Accuracy: {} ({:.6} pixels)",
            best.model_name, best.final_reprojection_error.mean
        );
    }
    if let Some(fastest) = fastest_conversion {
        println!(
            "⚡ Fastest Conversion: {} ({:.2} ms)",
            fastest.model_name, fastest.optimization_time_ms
        );
    }

    // Calculate average metrics
    let avg_error = metrics
        .iter()
        .map(|m| m.final_reprojection_error.mean)
        .sum::<f64>()
        / metrics.len() as f64;
    let avg_time =
        metrics.iter().map(|m| m.optimization_time_ms).sum::<f64>() / metrics.len() as f64;

    println!("📊 Average Reprojection Error: {avg_error:.6} pixels");
    println!("📊 Average Optimization Time: {avg_time:.2} ms");

    // Step 6: Export results using util module
    println!("\n💾 Step 6: Exporting Results");
    println!("============================");

    // Export detailed analysis report
    if let Err(e) = export_conversion_results(metrics, input_model_type) {
        println!("⚠️  Failed to export results: {e}");
    } else {
        let report_filename = format!(
            "output/camera_conversion_results_{}.txt",
            input_model_type.to_lowercase()
        );
        println!("📄 Results exported to: {report_filename}");
    }
}
