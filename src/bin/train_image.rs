//! Image PCN Training Binary
//!
//! Trains a Predictive Coding Network on CIFAR-10 images using:
//! - Patch-based encoding (8×8 patches, ViT-style)
//! - Reconstruction objective (autoencoder): predicts own input
//! - Optional classification mode with one-hot targets
//! - Same PCN core: energy minimization, Hebbian learning, convergence relaxation
//!
//! ## Architecture
//!
//! Default (reconstruction): `[3072, 512, 384, 256, 3072]`
//! Classification mode:      `[3072, 512, 384, 256, 10]`
//!
//! The hidden layers [512, 384, 256] mirror the text PCN for eventual
//! multimodal bridging at the 256-dim latent space.
//!
//! ## Usage
//!
//! ```bash
//! pcn-train-image \
//!   --data-dir /bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin \
//!   --mode reconstruction \
//!   --epochs 100 \
//!   --batch-size 256 \
//!   --checkpoint-dir /workspace/data/checkpoints/image-v1
//! ```

use pcn::core::{TanhActivation, PCN};
use pcn::data::image::{
    self, load_cifar10_test, load_cifar10_train, one_hot_encode, reconstruction_mse,
    reconstruction_psnr, PatchConfig, CIFAR_NUM_CLASSES,
};
use pcn::pool::BufferPool;
use pcn::training::{train_batch, train_batch_parallel};
use pcn::Config;

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Training mode for the image PCN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingMode {
    /// Autoencoder: input = target = image pixels (reconstruction)
    Reconstruction,
    /// Classification: target = one-hot class label
    Classification,
}

/// Per-epoch metrics for JSON logging.
#[derive(Debug, Serialize, Deserialize)]
struct ImageEpochMetrics {
    epoch: usize,
    mode: String,
    avg_loss: f32,
    train_accuracy: Option<f32>,
    test_accuracy: Option<f32>,
    reconstruction_mse: Option<f32>,
    reconstruction_psnr_db: Option<f32>,
    test_reconstruction_mse: Option<f32>,
    test_reconstruction_psnr_db: Option<f32>,
    elapsed_secs: f32,
    samples_per_sec: f32,
    per_layer_energy: Vec<f32>,
}

/// CLI argument parsing (minimal, no external deps).
struct Args {
    data_dir: PathBuf,
    checkpoint_dir: PathBuf,
    metrics_file: PathBuf,
    mode: TrainingMode,
    epochs: usize,
    batch_size: usize,
    relax_steps: usize,
    alpha: f32,
    eta: f32,
    eta_per_layer: Option<Vec<f32>>,
    hidden_sizes: Vec<usize>,
    eval_every: usize,
    max_train_samples: Option<usize>,
    patch_size: usize,
    buffer_capacity: usize,
    checkpoint_every: usize,
    normalize_input: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from(
                "/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin",
            ),
            checkpoint_dir: PathBuf::from("/workspace/data/checkpoints/image-v1"),
            metrics_file: PathBuf::from("/workspace/data/output/metrics-image.jsonl"),
            mode: TrainingMode::Reconstruction,
            epochs: 100,
            batch_size: 256,
            relax_steps: 80,
            alpha: 0.025,
            eta: 0.015,
            eta_per_layer: None,
            hidden_sizes: vec![512, 384, 256],
            eval_every: 5,
            max_train_samples: None,
            patch_size: 8,
            buffer_capacity: 512,
            checkpoint_every: 25,
            normalize_input: true,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = env::args().collect();
    let mut i = 1;

    while i < argv.len() {
        match argv[i].as_str() {
            "--data-dir" => {
                i += 1;
                args.data_dir = PathBuf::from(&argv[i]);
            }
            "--checkpoint-dir" => {
                i += 1;
                args.checkpoint_dir = PathBuf::from(&argv[i]);
            }
            "--metrics-file" => {
                i += 1;
                args.metrics_file = PathBuf::from(&argv[i]);
            }
            "--mode" => {
                i += 1;
                args.mode = match argv[i].as_str() {
                    "reconstruction" | "recon" => TrainingMode::Reconstruction,
                    "classification" | "classify" => TrainingMode::Classification,
                    other => {
                        eprintln!("Unknown mode: {other}. Using 'reconstruction'.");
                        TrainingMode::Reconstruction
                    }
                };
            }
            "--epochs" => {
                i += 1;
                args.epochs = argv[i].parse().unwrap_or(args.epochs);
            }
            "--batch-size" => {
                i += 1;
                args.batch_size = argv[i].parse().unwrap_or(args.batch_size);
            }
            "--relax-steps" => {
                i += 1;
                args.relax_steps = argv[i].parse().unwrap_or(args.relax_steps);
            }
            "--alpha" => {
                i += 1;
                args.alpha = argv[i].parse().unwrap_or(args.alpha);
            }
            "--eta" => {
                i += 1;
                args.eta = argv[i].parse().unwrap_or(args.eta);
            }
            "--eta-per-layer" => {
                i += 1;
                args.eta_per_layer = Some(
                    argv[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect(),
                );
            }
            "--hidden-sizes" => {
                i += 1;
                args.hidden_sizes = argv[i]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
            }
            "--eval-every" => {
                i += 1;
                args.eval_every = argv[i].parse().unwrap_or(args.eval_every);
            }
            "--max-train-samples" => {
                i += 1;
                args.max_train_samples = argv[i].parse().ok();
            }
            "--patch-size" => {
                i += 1;
                args.patch_size = argv[i].parse().unwrap_or(args.patch_size);
            }
            "--buffer-capacity" => {
                i += 1;
                args.buffer_capacity = argv[i].parse().unwrap_or(args.buffer_capacity);
            }
            "--checkpoint-every" => {
                i += 1;
                args.checkpoint_every = argv[i].parse().unwrap_or(args.checkpoint_every);
            }
            "--no-normalize" => {
                args.normalize_input = false;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
            }
        }
        i += 1;
    }

    args
}

fn print_usage() {
    eprintln!(
        r#"pcn-train-image: Train an image PCN on CIFAR-10

USAGE:
    pcn-train-image [OPTIONS]

OPTIONS:
    --data-dir <PATH>           CIFAR-10 binary data directory
    --checkpoint-dir <PATH>     Directory for saving checkpoints
    --metrics-file <PATH>       Path for JSONL metrics output
    --mode <MODE>               Training mode: reconstruction | classification
    --epochs <N>                Number of training epochs (default: 100)
    --batch-size <N>            Mini-batch size (default: 256)
    --relax-steps <N>           Relaxation steps per sample (default: 80)
    --alpha <F>                 State update rate (default: 0.025)
    --eta <F>                   Weight learning rate (default: 0.015)
    --eta-per-layer <F,F,...>   Per-layer learning rates (comma-separated)
    --hidden-sizes <N,N,...>    Hidden layer sizes (default: 512,384,256)
    --eval-every <N>            Evaluate on test set every N epochs (default: 5)
    --max-train-samples <N>     Limit training samples (for debugging)
    --patch-size <N>            Patch size in pixels (default: 8, options: 4, 8)
    --buffer-capacity <N>       Buffer pool capacity (default: 512)
    --checkpoint-every <N>      Save checkpoint every N epochs (default: 25)
    --no-normalize              Skip per-channel normalization
    --help, -h                  Show this help"#
    );
}

/// Shuffle indices in-place using Fisher-Yates algorithm.
fn shuffle_indices(indices: &mut [usize]) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
}

/// Compute top-1 classification accuracy from PCN output states.
///
/// For each sample, checks if the argmax of the output layer
/// matches the true label.
#[allow(clippy::cast_precision_loss)]
fn classification_accuracy(
    pcn: &PCN,
    inputs: &Array2<f32>,
    labels: &[u8],
    config: &Config,
) -> f32 {
    let l_max = pcn.dims().len() - 1;
    let num_samples = inputs.nrows();
    let mut correct = 0usize;

    for i in 0..num_samples {
        let input = inputs.row(i).to_owned();
        let mut state = pcn.init_state_from_input(&input);
        state.x[0].assign(&input);

        // Relax (inference mode: don't clamp output)
        for _ in 0..config.relax_steps {
            let _ = pcn.compute_errors(&mut state);
            let _ = pcn.relax_step(&mut state, config.alpha);
            state.x[0].assign(&input);
        }
        let _ = pcn.compute_errors(&mut state);

        // Get predicted class from output layer
        let output = &state.x[l_max];
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if predicted == labels[i] as usize {
            correct += 1;
        }
    }

    correct as f32 / num_samples as f32
}

/// Compute average reconstruction MSE and PSNR on a test set.
#[allow(clippy::cast_precision_loss)]
fn reconstruction_eval(
    pcn: &PCN,
    inputs: &Array2<f32>,
    config: &Config,
    max_samples: usize,
) -> (f32, f32) {
    let num_samples = inputs.nrows().min(max_samples);
    let mut total_mse = 0.0f32;

    for i in 0..num_samples {
        let input = inputs.row(i).to_owned();
        let mut state = pcn.init_state_from_input(&input);
        state.x[0].assign(&input);

        // Relax (inference: don't clamp output)
        for _ in 0..config.relax_steps {
            let _ = pcn.compute_errors(&mut state);
            let _ = pcn.relax_step(&mut state, config.alpha);
            state.x[0].assign(&input);
        }
        let _ = pcn.compute_errors(&mut state);

        // The output layer predicts the input layer via the top-down pathway.
        // Compute reconstruction from the top-down prediction of layer 0.
        // mu[0] is the network's reconstruction of the input.
        total_mse += reconstruction_mse(&input, &state.mu[0]);
    }

    let avg_mse = total_mse / num_samples as f32;
    let psnr = reconstruction_psnr(avg_mse);
    (avg_mse, psnr)
}

/// Save network weights to a checkpoint file.
fn save_checkpoint(pcn: &PCN, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Simple binary format: dims, then weight matrices, then biases
    let mut data = Vec::new();

    // Write number of layers
    let num_layers = pcn.dims().len();
    data.extend_from_slice(&(num_layers as u32).to_le_bytes());

    // Write dimensions
    for &dim in pcn.dims() {
        data.extend_from_slice(&(dim as u32).to_le_bytes());
    }

    // Write weight matrices (skip index 0 placeholder)
    for l in 1..pcn.w.len() {
        let shape = pcn.w[l].shape();
        data.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        data.extend_from_slice(&(shape[1] as u32).to_le_bytes());
        for &val in pcn.w[l].iter() {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Write bias vectors
    for bias in &pcn.b {
        data.extend_from_slice(&(bias.len() as u32).to_le_bytes());
        for &val in bias.iter() {
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    fs::write(path, data)?;
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    let args = parse_args();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║          Image PCN Training — CIFAR-10              ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    let mode_str = match args.mode {
        TrainingMode::Reconstruction => "reconstruction",
        TrainingMode::Classification => "classification",
    };
    println!("Mode:            {mode_str}");
    println!("Data dir:        {}", args.data_dir.display());
    println!("Hidden layers:   {:?}", args.hidden_sizes);
    println!("Patch size:      {}×{}", args.patch_size, args.patch_size);
    println!("Epochs:          {}", args.epochs);
    println!("Batch size:      {}", args.batch_size);
    println!("Relax steps:     {}", args.relax_steps);
    println!("Alpha:           {}", args.alpha);
    println!("Eta:             {}", args.eta);
    if let Some(ref per_layer) = args.eta_per_layer {
        println!("Eta per layer:   {per_layer:?}");
    }
    println!("Normalize:       {}", args.normalize_input);
    println!("Checkpoint dir:  {}", args.checkpoint_dir.display());
    println!("Metrics file:    {}", args.metrics_file.display());
    println!();

    // --- Load Data ---
    println!("Loading CIFAR-10 training data...");
    let start = Instant::now();
    let train_data = match load_cifar10_train(&args.data_dir) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("ERROR: Failed to load training data: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "  Loaded {} training images in {:.1}s",
        train_data.num_images,
        start.elapsed().as_secs_f32()
    );

    println!("Loading CIFAR-10 test data...");
    let test_data = match load_cifar10_test(&args.data_dir) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("ERROR: Failed to load test data: {e}");
            std::process::exit(1);
        }
    };
    println!("  Loaded {} test images", test_data.num_images);

    // --- Optionally limit training samples ---
    let (mut train_images, train_labels) = if let Some(max) = args.max_train_samples {
        let n = max.min(train_data.num_images);
        println!("  Limiting to {n} training samples");
        (
            train_data.images.slice(ndarray::s![..n, ..]).to_owned(),
            train_data.labels[..n].to_vec(),
        )
    } else {
        (train_data.images, train_data.labels)
    };
    let mut test_images = test_data.images;
    let test_labels = test_data.labels;

    // --- Normalize ---
    if args.normalize_input {
        println!("Computing per-channel statistics...");
        let (means, stds) = image::compute_channel_stats(&train_images, 3);
        println!(
            "  Means: [{:.4}, {:.4}, {:.4}]",
            means[0], means[1], means[2]
        );
        println!(
            "  Stds:  [{:.4}, {:.4}, {:.4}]",
            stds[0], stds[1], stds[2]
        );

        image::normalize_channels(&mut train_images, &means, &stds, 3);
        image::normalize_channels(&mut test_images, &means, &stds, 3);
    }

    // --- Patch Encoding ---
    let patch_config = match args.patch_size {
        4 => PatchConfig::cifar10_4x4(),
        8 => PatchConfig::cifar10_8x8(),
        _ => {
            eprintln!(
                "Unsupported patch size: {}. Using 8×8.",
                args.patch_size
            );
            PatchConfig::cifar10_8x8()
        }
    };

    println!(
        "Patch config: {}×{} patches of {}×{} = {} dim each, total input = {}",
        patch_config.patches_x(),
        patch_config.patches_y(),
        patch_config.patch_height,
        patch_config.patch_width,
        patch_config.patch_dim(),
        patch_config.total_input_dim()
    );

    // Note: For both 8×8 and 4×4 patches on CIFAR-10, total_input_dim = 3072
    // which equals CIFAR_PIXELS. The patch extraction just reorders pixels.
    // This is by design: patches are a spatial reordering, not a dimensionality change.

    let input_dim = patch_config.total_input_dim();

    // --- Build Network Architecture ---
    let output_dim = match args.mode {
        TrainingMode::Reconstruction => input_dim, // Autoencoder: predict own input
        TrainingMode::Classification => CIFAR_NUM_CLASSES, // 10 classes
    };

    let mut dims = Vec::new();
    dims.push(input_dim);
    dims.extend_from_slice(&args.hidden_sizes);
    dims.push(output_dim);

    println!("Network architecture: {dims:?}");
    println!(
        "  Total parameters: ~{}",
        estimate_params(&dims)
    );
    println!();

    // --- Create PCN ---
    let mut pcn = match PCN::with_activation(dims.clone(), Box::new(TanhActivation)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Failed to create PCN: {e}");
            std::process::exit(1);
        }
    };

    let config = Config {
        relax_steps: args.relax_steps,
        alpha: args.alpha,
        eta: args.eta,
        clamp_output: true,
    };

    // --- Prepare Targets ---
    let train_targets = match args.mode {
        TrainingMode::Reconstruction => train_images.clone(), // Target = input
        TrainingMode::Classification => {
            one_hot_encode(&train_labels, CIFAR_NUM_CLASSES)
        }
    };

    // Buffer pool for parallel training
    let pool = BufferPool::new(&dims, args.buffer_capacity);

    // --- Create output directories ---
    if let Err(e) = fs::create_dir_all(&args.checkpoint_dir) {
        eprintln!("WARNING: Could not create checkpoint dir: {e}");
    }
    if let Some(parent) = args.metrics_file.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("WARNING: Could not create metrics dir: {e}");
        }
    }

    // --- Open metrics file ---
    let mut metrics_file = match fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.metrics_file)
    {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("WARNING: Could not open metrics file: {e}");
            None
        }
    };

    // --- Training Loop ---
    let num_train = train_images.nrows();
    let num_batches = num_train.div_ceil(args.batch_size);

    println!("═══ Starting Training ═══");
    println!(
        "  {} samples, {} batches/epoch, {} epochs",
        num_train, num_batches, args.epochs
    );
    println!();

    let training_start = Instant::now();

    for epoch in 1..=args.epochs {
        let epoch_start = Instant::now();

        // Shuffle training data
        let mut indices: Vec<usize> = (0..num_train).collect();
        shuffle_indices(&mut indices);

        let mut epoch_loss = 0.0f32;
        let mut epoch_samples = 0usize;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * args.batch_size;
            let end_idx = (start_idx + args.batch_size).min(num_train);
            let current_batch_size = end_idx - start_idx;

            // Extract batch
            let mut batch_inputs = Array2::zeros((current_batch_size, train_images.ncols()));
            let mut batch_targets = Array2::zeros((current_batch_size, train_targets.ncols()));

            for (local, &global) in indices[start_idx..end_idx].iter().enumerate() {
                batch_inputs
                    .row_mut(local)
                    .assign(&train_images.row(global));
                batch_targets
                    .row_mut(local)
                    .assign(&train_targets.row(global));
            }

            // Train batch (use parallel if batch is large enough)
            let batch_metrics = if current_batch_size >= 16 {
                match train_batch_parallel(
                    &mut pcn,
                    &batch_inputs,
                    &batch_targets,
                    &config,
                    &pool,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("ERROR in batch {batch_idx}: {e}");
                        continue;
                    }
                }
            } else {
                match train_batch(&mut pcn, &batch_inputs, &batch_targets, &config) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("ERROR in batch {batch_idx}: {e}");
                        continue;
                    }
                }
            };

            epoch_loss += batch_metrics.avg_loss * current_batch_size as f32;
            epoch_samples += current_batch_size;
        }

        let epoch_elapsed = epoch_start.elapsed().as_secs_f32();
        let avg_loss = if epoch_samples > 0 {
            epoch_loss / epoch_samples as f32
        } else {
            0.0
        };
        let samples_per_sec = epoch_samples as f32 / epoch_elapsed;

        // --- Evaluation ---
        let mut metrics = ImageEpochMetrics {
            epoch,
            mode: mode_str.to_string(),
            avg_loss,
            train_accuracy: None,
            test_accuracy: None,
            reconstruction_mse: None,
            reconstruction_psnr_db: None,
            test_reconstruction_mse: None,
            test_reconstruction_psnr_db: None,
            elapsed_secs: epoch_elapsed,
            samples_per_sec,
            per_layer_energy: Vec::new(),
        };

        let eval_epoch = epoch % args.eval_every == 0 || epoch == 1 || epoch == args.epochs;

        if eval_epoch {
            match args.mode {
                TrainingMode::Reconstruction => {
                    // Evaluate reconstruction quality on train subset
                    let (train_mse, train_psnr) =
                        reconstruction_eval(&pcn, &train_images, &config, 500);
                    metrics.reconstruction_mse = Some(train_mse);
                    metrics.reconstruction_psnr_db = Some(train_psnr);

                    // Evaluate on test set
                    let (test_mse, test_psnr) =
                        reconstruction_eval(&pcn, &test_images, &config, 500);
                    metrics.test_reconstruction_mse = Some(test_mse);
                    metrics.test_reconstruction_psnr_db = Some(test_psnr);

                    println!(
                        "Epoch {epoch:>4}/{} | loss={avg_loss:.4} | train_MSE={train_mse:.6} PSNR={train_psnr:.1}dB | test_MSE={test_mse:.6} PSNR={test_psnr:.1}dB | {samples_per_sec:.0} samples/s | {epoch_elapsed:.1}s",
                        args.epochs
                    );
                }
                TrainingMode::Classification => {
                    // Evaluate on subset for speed
                    let test_subset_n = 1000.min(test_images.nrows());
                    let test_subset = test_images.slice(ndarray::s![..test_subset_n, ..]).to_owned();
                    let test_labels_subset = &test_labels[..test_subset_n];

                    let test_acc =
                        classification_accuracy(&pcn, &test_subset, test_labels_subset, &config);
                    metrics.test_accuracy = Some(test_acc);

                    println!(
                        "Epoch {epoch:>4}/{} | loss={avg_loss:.4} | test_acc={:.1}% | {samples_per_sec:.0} samples/s | {epoch_elapsed:.1}s",
                        args.epochs,
                        test_acc * 100.0
                    );
                }
            }
        } else {
            println!(
                "Epoch {epoch:>4}/{} | loss={avg_loss:.4} | {samples_per_sec:.0} samples/s | {epoch_elapsed:.1}s",
                args.epochs
            );
        }

        // --- Write Metrics ---
        if let Some(ref mut f) = metrics_file {
            if let Ok(json) = serde_json::to_string(&metrics) {
                let _ = writeln!(f, "{json}");
            }
        }

        // --- Checkpoint ---
        if epoch % args.checkpoint_every == 0 || epoch == args.epochs {
            let ckpt_path = args
                .checkpoint_dir
                .join(format!("image-pcn-epoch-{epoch:04}.bin"));
            match save_checkpoint(&pcn, &ckpt_path) {
                Ok(()) => println!("  Checkpoint saved: {}", ckpt_path.display()),
                Err(e) => eprintln!("  WARNING: Failed to save checkpoint: {e}"),
            }
        }
    }

    let total_elapsed = training_start.elapsed().as_secs_f32();
    println!();
    println!("═══ Training Complete ═══");
    println!("  Total time: {total_elapsed:.1}s");
    println!("  Metrics: {}", args.metrics_file.display());
    println!("  Checkpoints: {}", args.checkpoint_dir.display());

    // Print pool stats
    let pool_stats = pool.stats();
    println!(
        "  Buffer pool: {} allocated, {:.0}% hit rate",
        pool_stats.total_allocated,
        pool_stats.hit_rate * 100.0
    );
}

/// Estimate total parameter count for a given architecture.
#[allow(clippy::cast_precision_loss)]
fn estimate_params(dims: &[usize]) -> String {
    let mut total = 0usize;
    for i in 1..dims.len() {
        total += dims[i - 1] * dims[i]; // weights
        total += dims[i - 1]; // biases
    }
    if total > 1_000_000 {
        format!("{:.1}M", total as f64 / 1_000_000.0)
    } else if total > 1_000 {
        format!("{:.1}K", total as f64 / 1_000.0)
    } else {
        format!("{total}")
    }
}
