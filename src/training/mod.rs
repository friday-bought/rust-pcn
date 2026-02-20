//! Training loops, convergence checks, and metrics.
//!
//! Phase 3 Implementation: Mini-batch training with epoch support.
//! - Single-sample training (baseline)
//! - Mini-batch training with gradient accumulation
//! - Epoch-level training with shuffling
//! - Metrics tracking and aggregation

use crate::core::PCN;
use crate::Config;
use ndarray::{Array1, Array2, Axis};

/// Metrics computed during training.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Total prediction error energy
    pub energy: f32,
    /// Layer-wise error magnitudes
    pub layer_errors: Vec<f32>,
    /// Classification accuracy (if applicable)
    pub accuracy: Option<f32>,
}

/// Mini-batch training epoch metrics.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Average loss across epoch
    pub avg_loss: f32,
    /// Total samples processed
    pub num_samples: usize,
    /// Number of batches processed
    pub num_batches: usize,
    /// Per-batch losses
    pub batch_losses: Vec<f32>,
}

/// Train the network on a single sample.
///
/// # Algorithm
/// 1. Initialize state from input
/// 2. Clamp input and target (supervised training)
/// 3. Relax for `config.relax_steps` iterations
/// 4. Compute final errors
/// 5. Update weights using Hebbian rule
/// 6. Return energy and layer-wise errors
pub fn train_sample(
    pcn: &mut PCN,
    input: &Array1<f32>,
    target: &Array1<f32>,
    config: &Config,
) -> Result<Metrics, String> {
    // Validate dimensions
    if input.len() != pcn.dims()[0] {
        return Err(format!(
            "Input dim mismatch: expected {}, got {}",
            pcn.dims()[0],
            input.len()
        ));
    }
    let l_max = pcn.dims().len() - 1;
    if target.len() != pcn.dims()[l_max] {
        return Err(format!(
            "Target dim mismatch: expected {}, got {}",
            pcn.dims()[l_max],
            target.len()
        ));
    }

    // Initialize state from input
    let mut state = pcn
        .init_state_from_input(input)
        .map_err(|e| e.to_string())?;

    // Clamp input and output
    state.x[0] = input.clone();
    state.x[l_max] = target.clone();

    // Relax to equilibrium
    pcn.relax(&mut state, config.relax_steps, config.alpha)
        .map_err(|e| e.to_string())?;

    // Update weights using Hebbian rule
    pcn.update_weights(&state, config.eta)
        .map_err(|e| e.to_string())?;

    // Compute metrics
    let energy = pcn.compute_energy(&state);
    let layer_errors: Vec<f32> = state.eps.iter().map(|e| e.norm_l2()).collect();

    Ok(Metrics {
        energy,
        layer_errors,
        accuracy: None,
    })
}

/// Train on a mini-batch of samples with weight gradient accumulation.
///
/// # Algorithm
/// 1. For each sample in batch:
///    - Initialize state
///    - Clamp input and target
///    - Relax to equilibrium
///    - Accumulate weight gradients (no update yet)
/// 2. Perform single batch weight update with accumulated gradients
/// 3. Return aggregated metrics
///
/// **Advantages over per-sample training:**
/// - Reduced weight update overhead (one update vs. batch_size updates)
/// - Better gradient estimates through noise averaging
/// - Faster convergence on coherent patterns
pub fn train_batch(
    pcn: &mut PCN,
    batch_inputs: &Array2<f32>,  // Shape: (batch_size, input_dim)
    batch_targets: &Array2<f32>, // Shape: (batch_size, output_dim)
    config: &Config,
) -> Result<EpochMetrics, String> {
    let batch_size = batch_inputs.nrows();

    // Validate dimensions
    if batch_inputs.ncols() != pcn.dims()[0] {
        return Err(format!(
            "Batch input dim: expected {}, got {}",
            pcn.dims()[0],
            batch_inputs.ncols()
        ));
    }
    let l_max = pcn.dims().len() - 1;
    if batch_targets.ncols() != pcn.dims()[l_max] {
        return Err(format!(
            "Batch target dim: expected {}, got {}",
            pcn.dims()[l_max],
            batch_targets.ncols()
        ));
    }

    // Accumulate gradients over batch
    let mut accumulated_weight_grads: Vec<Array2<f32>> =
        pcn.w.iter().map(|w| Array2::zeros(w.dim())).collect();
    let mut accumulated_bias_grads: Vec<Array1<f32>> =
        pcn.b.iter().map(|b| Array1::zeros(b.len())).collect();

    let mut total_energy = 0.0f32;
    let mut batch_losses = Vec::new();

    // Process each sample in the batch
    for i in 0..batch_size {
        let input = batch_inputs.row(i).to_owned();
        let target = batch_targets.row(i).to_owned();

        // Initialize state from input
        let mut state = pcn
            .init_state_from_input(&input)
            .map_err(|e| e.to_string())?;

        // Clamp input and output
        state.x[0] = input;
        state.x[l_max] = target;

        // Relax to equilibrium
        pcn.relax(&mut state, config.relax_steps, config.alpha)
            .map_err(|e| e.to_string())?;

        // Accumulate energy
        let sample_energy = pcn.compute_energy(&state);
        total_energy += sample_energy;
        batch_losses.push(sample_energy);

        // Accumulate weight gradients: ε^ℓ-1 ⊗ f(x^ℓ)
        for l in 1..=l_max {
            let f_x_l = pcn.activation.apply(&state.x[l]);

            // Outer product via broadcasting
            let eps_col = state.eps[l - 1].view().insert_axis(Axis(1));
            let fx_row = f_x_l.view().insert_axis(Axis(0));
            let delta_w = &eps_col * &fx_row;

            accumulated_weight_grads[l] = &accumulated_weight_grads[l] + &delta_w;
            accumulated_bias_grads[l - 1] = &accumulated_bias_grads[l - 1] + &state.eps[l - 1];
        }
    }

    // Apply batch weight update (average by batch size)
    let batch_scale = config.eta / batch_size as f32;
    for l in 1..=l_max {
        pcn.w[l] = &pcn.w[l] + &(batch_scale * &accumulated_weight_grads[l]);
        pcn.b[l - 1] = &pcn.b[l - 1] + batch_scale * &accumulated_bias_grads[l - 1];
    }

    let avg_loss = total_energy / batch_size as f32;

    Ok(EpochMetrics {
        avg_loss,
        num_samples: batch_size,
        num_batches: 1,
        batch_losses,
    })
}

/// Train for one full epoch over a dataset.
///
/// # Algorithm
/// 1. Optionally shuffle dataset indices
/// 2. Divide into batches of specified size
/// 3. For each batch, call `train_batch()`
/// 4. Aggregate and return epoch-level metrics
///
/// # Arguments
/// - `pcn`: Network to train
/// - `inputs`: Full dataset, shape (num_samples, input_dim)
/// - `targets`: Full targets, shape (num_samples, output_dim)
/// - `batch_size`: Samples per batch (32-128 typical)
/// - `config`: Training hyperparameters
/// - `shuffle`: Whether to randomize sample order
pub fn train_epoch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    batch_size: usize,
    config: &Config,
    shuffle: bool,
) -> Result<EpochMetrics, String> {
    let num_samples = inputs.nrows();

    if batch_size == 0 {
        return Err("Batch size must be > 0".to_string());
    }
    if num_samples != targets.nrows() {
        return Err(format!(
            "Sample count mismatch: inputs={}, targets={}",
            num_samples, targets.nrows()
        ));
    }

    // Create and optionally shuffle indices
    let mut indices: Vec<usize> = (0..num_samples).collect();
    if shuffle {
        shuffle_indices(&mut indices);
    }

    // Process batches
    let mut all_batch_losses = Vec::new();
    let mut total_energy = 0.0f32;
    let num_batches = (num_samples + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let current_batch_size = end - start;

        // Extract batch rows via indices
        let batch_indices = &indices[start..end];

        let mut batch_inputs = Array2::zeros((current_batch_size, inputs.ncols()));
        let mut batch_targets = Array2::zeros((current_batch_size, targets.ncols()));

        for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
            batch_inputs.row_mut(local_idx).assign(&inputs.row(global_idx));
            batch_targets.row_mut(local_idx).assign(&targets.row(global_idx));
        }

        // Train this batch
        let batch_metrics = train_batch(pcn, &batch_inputs, &batch_targets, config)?;
        all_batch_losses.extend(&batch_metrics.batch_losses);
        total_energy += batch_metrics.avg_loss * current_batch_size as f32;
    }

    let avg_loss = total_energy / num_samples as f32;

    Ok(EpochMetrics {
        avg_loss,
        num_samples,
        num_batches,
        batch_losses: all_batch_losses,
    })
}

/// Compute energy of a state using the Energy principle.
///
/// Energy = (1/2) * Σ_ℓ ||ε^ℓ||²
///
/// Where ε^ℓ is the prediction error at layer ℓ.
/// This is already implemented in PCN::compute_energy(),
/// but provided here for compatibility.
pub fn compute_energy(state: &crate::core::State) -> f32 {
    let mut energy = 0.0f32;
    for eps in &state.eps {
        let sq_norm = eps.dot(eps);
        energy += sq_norm;
    }
    0.5 * energy
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PCN;

    #[test]
    fn test_metrics() {
        let metrics = Metrics {
            energy: 0.5,
            layer_errors: vec![0.1, 0.2],
            accuracy: Some(0.95),
        };
        assert!(metrics.energy > 0.0);
    }

    #[test]
    fn test_shuffle_indices() {
        let mut indices = vec![0, 1, 2, 3, 4];
        let original = indices.clone();
        shuffle_indices(&mut indices);

        // Check all indices still present
        assert_eq!(indices.len(), original.len());
        for i in &original {
            assert!(indices.contains(i));
        }
    }

    #[test]
    fn test_train_batch_shape() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![2, 3, 2]).unwrap();

        let batch_inputs = Array2::from_elem((4, 2), 0.1);
        let batch_targets = Array2::from_elem((4, 2), 0.5);

        let result = train_batch(&mut pcn, &batch_inputs, &batch_targets, &config);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.num_samples, 4);
        assert!(metrics.avg_loss >= 0.0);
        assert_eq!(metrics.batch_losses.len(), 4);
    }

    #[test]
    fn test_train_epoch_basic() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![2, 3, 2]).unwrap();

        let inputs = Array2::from_elem((8, 2), 0.1);
        let targets = Array2::from_elem((8, 2), 0.5);

        let result = train_epoch(&mut pcn, &inputs, &targets, 2, &config, false);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_eq!(metrics.num_samples, 8);
        assert_eq!(metrics.num_batches, 4);
        assert!(metrics.avg_loss >= 0.0);
    }

    #[test]
    fn test_train_epoch_uneven_batches() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![2, 3, 2]).unwrap();

        // 13 = 4*3 + 1, so last batch is size 1
        let inputs = Array2::from_elem((13, 2), 0.1);
        let targets = Array2::from_elem((13, 2), 0.5);

        let metrics = train_epoch(&mut pcn, &inputs, &targets, 4, &config, false)
            .expect("epoch training failed");

        assert_eq!(metrics.num_samples, 13);
        assert_eq!(metrics.num_batches, 4); // ceil(13/4) = 4
        assert_eq!(metrics.batch_losses.len(), 13); // All 13 samples recorded
    }

    #[test]
    fn test_train_batch_dimension_mismatch() {
        let config = Config::default();
        let mut pcn = PCN::new(vec![4, 6, 3]).unwrap();

        // Wrong input dimension
        let batch_inputs = Array2::from_elem((4, 5), 0.1);
        let batch_targets = Array2::from_elem((4, 3), 0.5);

        let result = train_batch(&mut pcn, &batch_inputs, &batch_targets, &config);
        assert!(result.is_err());
    }
}
