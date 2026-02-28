//! Image PCN Inference Binary
//!
//! Loads a binary checkpoint and reconstructs a random CIFAR-10 test image,
//! saving input and output as PPM files (simple, no extra deps).

use pcn::core::{TanhActivation, PCN};
use pcn::data::image::{
    self, load_cifar10_test, reconstruction_mse, reconstruction_psnr, CIFAR_PIXELS,
};
use pcn::Config;

use ndarray::Array1;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Load PCN from binary checkpoint (matches save_checkpoint in train_image).
fn load_binary_checkpoint(path: &Path) -> Result<PCN, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut off = 0;

    let num_layers = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
    off += 4;

    let mut dims = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let d = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        dims.push(d);
    }

    let mut pcn = PCN::with_activation(dims.clone(), Box::new(TanhActivation))?;

    // Load weight matrices (skip index 0)
    for l in 1..num_layers {
        let rows = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let cols = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let n = rows * cols;
        let floats: Vec<f32> = (0..n)
            .map(|i| {
                let s = off + i * 4;
                f32::from_le_bytes(data[s..s + 4].try_into().unwrap())
            })
            .collect();
        off += n * 4;

        let w = ndarray::Array2::from_shape_vec((rows, cols), floats)
            .map_err(|e| format!("Weight matrix shape error: {e}"))?;
        pcn.w[l] = w;
    }

    // Load biases
    let mut bias_idx = 0;
    while off + 4 <= data.len() {
        let blen = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let floats: Vec<f32> = (0..blen)
            .map(|i| {
                let s = off + i * 4;
                f32::from_le_bytes(data[s..s + 4].try_into().unwrap())
            })
            .collect();
        off += blen * 4;

        if bias_idx < pcn.b.len() {
            pcn.b[bias_idx] = Array1::from_vec(floats);
        }
        bias_idx += 1;
    }

    eprintln!("Loaded checkpoint: dims={dims:?}, {bias_idx} bias vectors");
    Ok(pcn)
}

/// Save a 32x32 RGB image as PPM (trivial format, no deps).
fn save_ppm(path: &Path, pixels: &[(u8, u8, u8)], width: usize, height: usize, scale: usize) {
    let sw = width * scale;
    let sh = height * scale;
    let mut f = fs::File::create(path).expect("create PPM");
    write!(f, "P6\n{sw} {sh}\n255\n").unwrap();
    for y in 0..sh {
        for x in 0..sw {
            let (r, g, b) = pixels[(y / scale) * width + (x / scale)];
            f.write_all(&[r, g, b]).unwrap();
        }
    }
}

/// Convert HWC interleaved [0,1] flat vector to RGB pixels.
/// Data layout: [R,G,B, R,G,B, ...] for each of 1024 pixels.
fn flat_to_rgb(flat: &[f32]) -> Vec<(u8, u8, u8)> {
    let mut pixels = Vec::with_capacity(1024);
    for i in 0..1024 {
        let r = (flat[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
        let g = (flat[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
        let b = (flat[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
        pixels.push((r, g, b));
    }
    pixels
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let checkpoint_path = args
        .iter()
        .position(|a| a == "--checkpoint")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| {
            PathBuf::from("data/checkpoints/image-wide-v2-tuned/image-pcn-epoch-1000.bin")
        });

    let data_dir = args
        .iter()
        .position(|a| a == "--data-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| {
            PathBuf::from("/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin")
        });

    let out_dir = args
        .iter()
        .position(|a| a == "--out-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("data/output"));

    let image_idx: Option<usize> = args
        .iter()
        .position(|a| a == "--index")
        .and_then(|i| args[i + 1].parse().ok());

    let relax_steps: usize = args
        .iter()
        .position(|a| a == "--relax-steps")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(80);

    let alpha: f32 = args
        .iter()
        .position(|a| a == "--alpha")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(0.025);

    fs::create_dir_all(&out_dir).ok();

    // Load checkpoint
    eprintln!("Loading checkpoint: {}", checkpoint_path.display());
    let pcn = load_binary_checkpoint(&checkpoint_path).expect("Failed to load checkpoint");

    // Load test data
    eprintln!("Loading CIFAR-10 test data from: {}", data_dir.display());
    let test_data = load_cifar10_test(&data_dir).expect("Failed to load test data");
    eprintln!("  {} test images", test_data.num_images);

    // Compute channel stats for normalization
    let (means, stds) = image::compute_channel_stats(&test_data.images, 3);
    eprintln!(
        "  Means: [{:.4}, {:.4}, {:.4}]",
        means[0], means[1], means[2]
    );
    eprintln!(
        "  Stds:  [{:.4}, {:.4}, {:.4}]",
        stds[0], stds[1], stds[2]
    );

    // Pick image
    let idx = image_idx.unwrap_or_else(|| {
        use rand::Rng;
        rand::thread_rng().gen_range(0..test_data.num_images)
    });
    let label = test_data.labels[idx];
    let class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ];
    eprintln!(
        "\nImage #{idx}: {} (label={label})",
        class_names.get(label as usize).unwrap_or(&"?")
    );

    // Save original (unnormalized)
    let original = test_data.images.row(idx).to_owned();
    let input_rgb = flat_to_rgb(original.as_slice().unwrap());
    let input_path = out_dir.join("input.ppm");
    save_ppm(&input_path, &input_rgb, 32, 32, 8);
    eprintln!("  Saved input: {}", input_path.display());

    // Normalize (HWC layout: pixel i has channels at i*3, i*3+1, i*3+2)
    let mut normalized = original.clone();
    for i in 0..1024 {
        for c in 0..3 {
            normalized[i * 3 + c] = (normalized[i * 3 + c] - means[c]) / stds[c];
        }
    }

    // Run PCN inference
    let config = Config {
        relax_steps,
        alpha,
        clamp_output: true,
        use_layer_norm: true,
        ..Default::default()
    };

    eprintln!("Running inference ({relax_steps} relax steps, alpha={alpha})...");
    let l_max = pcn.dims().len() - 1;
    let mut state = pcn.init_state_from_input(&normalized);
    state.x[0].assign(&normalized);

    // Only clamp input — let output settle freely for real reconstruction
    for _step in 0..relax_steps {
        let _ = pcn.compute_errors(&mut state);
        let _ = pcn.relax_step(&mut state, alpha);
        state.x[0].assign(&normalized);
    }
    let _ = pcn.compute_errors(&mut state);

    // Reconstruction = mu[0] (top-down prediction of input layer)
    let recon = &state.mu[0];

    // Denormalize reconstruction (HWC layout)
    let mut recon_denorm = recon.clone();
    for i in 0..1024 {
        for c in 0..3 {
            recon_denorm[i * 3 + c] = recon_denorm[i * 3 + c] * stds[c] + means[c];
        }
    }

    let mse = reconstruction_mse(&normalized, recon);
    let psnr = reconstruction_psnr(mse);
    eprintln!("  MSE: {mse:.6}, PSNR: {psnr:.1} dB");

    // Save reconstruction
    let output_rgb = flat_to_rgb(recon_denorm.as_slice().unwrap());
    let output_path = out_dir.join("output.ppm");
    save_ppm(&output_path, &output_rgb, 32, 32, 8);
    eprintln!("  Saved output: {}", output_path.display());

    // Save side-by-side comparison
    let mut combined = Vec::with_capacity(32 * 68);
    for y in 0..32 {
        for x in 0..68 {
            if x < 32 {
                combined.push(input_rgb[y * 32 + x]);
            } else if x < 36 {
                combined.push((128, 128, 128));
            } else {
                combined.push(output_rgb[y * 32 + (x - 36)]);
            }
        }
    }
    let comp_path = out_dir.join("comparison.ppm");
    save_ppm(&comp_path, &combined, 68, 32, 8);
    eprintln!("  Saved comparison: {}", comp_path.display());

    eprintln!("\nDone.");
}
