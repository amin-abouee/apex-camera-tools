# Camera Tools

A Rust library and CLI toolkit for fisheye and wide-angle camera model conversion and image undistortion. Built on [`apex-camera-models`](https://crates.io/crates/apex-camera-models) and [`apex-solver`](https://crates.io/crates/apex-solver).

## Supported Camera Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| **Pinhole** | `fx, fy, cx, cy` | Standard cameras, no distortion |
| **Double Sphere (DS)** | `fx, fy, cx, cy, xi, alpha` | Wide-angle and fisheye cameras |
| **Kannala-Brandt (KB)** | `fx, fy, cx, cy, k1..k4` | Fisheye cameras (equidistant-like) |
| **Radial-Tangential (RadTan)** | `fx, fy, cx, cy, k1, k2, p1, p2, k3` | Standard lenses (OpenCV-compatible) |
| **Unified Camera Model (UCM)** | `fx, fy, cx, cy, alpha` | Wide-angle with single sphere |
| **Extended UCM (EUCM)** | `fx, fy, cx, cy, alpha, beta` | Flexible wide-angle model |
| **Field-of-View (FOV)** | `fx, fy, cx, cy, w` | Simple fisheye approximation |

## Binaries

The crate provides two CLI tools:

### `converter`

Converts a camera model to all other supported target models using Levenberg-Marquardt optimization with analytical Jacobians.

```bash
# Convert a Double Sphere model to all targets
cargo run --bin converter -- -i ds -p samples/double_sphere.yaml

# With custom point count
cargo run --bin converter -- -i kb -p samples/kannala_brandt.yaml -n 1000

# With reference image for PSNR/SSIM quality assessment
cargo run --bin converter -- -i ds -p samples/double_sphere.yaml -m image.png
```

**Options:**

| Flag | Description |
|------|-------------|
| `-i, --input-model` | Input model type: `ds`, `kb`, `radtan`, `ucm`, `eucm`, `fov`, `pinhole` |
| `-p, --input-path` | Path to YAML calibration file |
| `-n, --num-points` | Sample points for optimization (default: 500) |
| `-m, --image-path` | Optional reference image for quality metrics |

### `undistort`

Removes lens distortion from an image using a camera calibration file.

```bash
cargo run --bin undistort -- -i input.png -c samples/double_sphere.yaml -o output.png

# With custom target focal length
cargo run --bin undistort -- -i input.png -c calib.yaml -o output.png --target-fx 400 --target-fy 400
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
camera-tools = "0.4"
```

### Load and convert camera models

```rust
use camera_tools::camera::{CameraWithResolution, CameraModel, YamlCamera};
use camera_tools::commands::convert;
use camera_tools::util;

// Load a camera model from YAML
let model = CameraWithResolution::load_from_yaml("samples/double_sphere.yaml")?;

// Sample 500 2D-3D point correspondences
let (points_2d, points_3d) = util::sample_points(Some(&model), 500)?;

// Convert to Kannala-Brandt
let metrics = convert::convert_to_kannala_brandt(&model, &points_3d, &points_2d, None)?;
println!("RMSE: {:.4} px", metrics.final_reprojection_error.rmse);
println!("Converted model: {:?}", metrics.model);
```

### Project and unproject points

```rust
use camera_tools::camera::{CameraWithResolution, CameraModel, YamlCamera};
use nalgebra::{Vector2, Vector3};

let model = CameraWithResolution::load_from_yaml("samples/double_sphere.yaml")?;

// Project a 3D point to 2D
let point_3d = Vector3::new(0.1, 0.2, 1.0);
let point_2d = model.project(&point_3d)?;

// Unproject a 2D pixel to a 3D ray
let pixel = Vector2::new(400.0, 250.0);
let ray = model.unproject(&pixel)?;
```

## Calibration YAML Format

```yaml
cam0:
  camera_model: double_sphere
  intrinsics: [fx, fy, cx, cy, alpha, xi]
  resolution: [width, height]
```

Sample calibrations are in `samples/`.

## Conversion Algorithm

The conversion pipeline follows the approach from [Fisheye-Calib-Adapter](https://arxiv.org/abs/2407.12405):

1. Sample N points uniformly across the source image
2. Unproject to 3D rays using the source camera model
3. Closed-form linear estimation of target model parameters
4. Nonlinear refinement via Levenberg-Marquardt (analytical Jacobians)
5. Compute reprojection error and validation metrics

## Testing

```bash
cargo test              # all tests (unit + integration)
cargo test --lib        # 10 unit tests (project/unproject roundtrips, YAML I/O)
cargo test --test conversion_quality  # 9 integration tests (conversion quality + C++ reference comparison)
```

The integration tests validate conversions from Double Sphere to all target models and round-trips (DS -> X -> DS), with reprojection RMSE thresholds and parameter comparison against the C++ [fisheye-calib-adapter](https://github.com/AIT-Assistive-Autonomous-Systems/fisheye_calib_adapter) reference implementation.

## License

MIT
