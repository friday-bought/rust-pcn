//! Custom image PCN inference — loads a 32x32 RGB raw file and runs reconstruction.

use pcn::core::{TanhActivation, PCN};
use pcn::data::image::{self, load_cifar10_test, reconstruction_mse, reconstruction_psnr};
use pcn::Config;

use ndarray::Array1;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

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
    for l in 1..num_layers {
        let rows = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let cols = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let n = rows * cols;
        let floats: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(data[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
            .collect();
        off += n * 4;
        pcn.w[l] = ndarray::Array2::from_shape_vec((rows, cols), floats)
            .map_err(|e| format!("{e}"))?;
    }
    let mut bias_idx = 0;
    while off + 4 <= data.len() {
        let blen = u32::from_le_bytes(data[off..off + 4].try_into()?) as usize;
        off += 4;
        let floats: Vec<f32> = (0..blen)
            .map(|i| f32::from_le_bytes(data[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
            .collect();
        off += blen * 4;
        if bias_idx < pcn.b.len() {
            pcn.b[bias_idx] = Array1::from_vec(floats);
        }
        bias_idx += 1;
    }
    eprintln!("Loaded: dims={dims:?}");
    Ok(pcn)
}

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

fn flat_to_rgb(flat: &[f32]) -> Vec<(u8, u8, u8)> {
    (0..1024)
        .map(|i| {
            let r = (flat[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (flat[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (flat[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
            (r, g, b)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let checkpoint_path = args.iter().position(|a| a == "--checkpoint")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("data/checkpoints/image-wide-v2-tuned/image-pcn-epoch-1000.bin"));

    let image_path = args.iter().position(|a| a == "--image")
        .map(|i| PathBuf::from(&args[i + 1]))
        .expect("--image <path to 32x32 RGB raw file> required");

    let cifar_dir = args.iter().position(|a| a == "--data-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin"));

    let out_dir = args.iter().position(|a| a == "--out-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("data/output"));

    let relax_steps: usize = args.iter().position(|a| a == "--relax-steps")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(80);

    let alpha: f32 = args.iter().position(|a| a == "--alpha")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(0.025);

    fs::create_dir_all(&out_dir).ok();

    let pcn = load_binary_checkpoint(&checkpoint_path).expect("Failed to load checkpoint");

    // Load raw 32x32 RGB image (3072 bytes, HWC interleaved)
    let raw_bytes = fs::read(&image_path).expect("Failed to read image");
    assert_eq!(raw_bytes.len(), 3072, "Expected 32x32x3 = 3072 bytes");

    let original: Vec<f32> = raw_bytes.iter().map(|&b| f32::from(b) / 255.0).collect();
    let original_arr = ndarray::Array1::from_vec(original.clone());

    // Compute normalization stats from CIFAR test set (to match training)
    eprintln!("Computing normalization stats from CIFAR...");
    let test_data = load_cifar10_test(&cifar_dir).expect("load test");
    let (means, stds) = image::compute_channel_stats(&test_data.images, 3);
    eprintln!("  Means: [{:.4}, {:.4}, {:.4}]", means[0], means[1], means[2]);

    // Normalize custom image with same stats
    let mut normalized = original_arr.clone();
    for i in 0..1024 {
        for c in 0..3 {
            normalized[i * 3 + c] = (normalized[i * 3 + c] - means[c]) / stds[c];
        }
    }

    // Save input
    let input_rgb = flat_to_rgb(&original);
    save_ppm(&out_dir.join("custom_input.ppm"), &input_rgb, 32, 32, 8);

    // Run inference
    eprintln!("Running inference ({relax_steps} steps)...");
    let l_max = pcn.dims().len() - 1;
    let mut state = pcn.init_state_from_input(&normalized);
    state.x[0].assign(&normalized);

    // Only clamp input — let output settle freely
    for _ in 0..relax_steps {
        let _ = pcn.compute_errors(&mut state);
        let _ = pcn.relax_step(&mut state, alpha);
        state.x[0].assign(&normalized);
    }
    let _ = pcn.compute_errors(&mut state);

    let recon = &state.mu[0];

    // Denormalize
    let mut recon_denorm = recon.clone();
    for i in 0..1024 {
        for c in 0..3 {
            recon_denorm[i * 3 + c] = recon_denorm[i * 3 + c] * stds[c] + means[c];
        }
    }

    let mse = reconstruction_mse(&normalized, recon);
    let psnr = reconstruction_psnr(mse);
    eprintln!("  MSE: {mse:.6}, PSNR: {psnr:.1} dB");

    let output_rgb = flat_to_rgb(recon_denorm.as_slice().unwrap());
    save_ppm(&out_dir.join("custom_output.ppm"), &output_rgb, 32, 32, 8);

    // Comparison
    let mut combined = Vec::with_capacity(32 * 68);
    for y in 0..32 {
        for x in 0..68 {
            if x < 32 { combined.push(input_rgb[y * 32 + x]); }
            else if x < 36 { combined.push((128, 128, 128)); }
            else { combined.push(output_rgb[y * 32 + (x - 36)]); }
        }
    }
    save_ppm(&out_dir.join("custom_comparison.ppm"), &combined, 68, 32, 8);

    eprintln!("Saved to {}", out_dir.display());
}
