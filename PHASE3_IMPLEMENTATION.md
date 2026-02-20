# Phase 3: Mini-batch Training Implementation

## Overview

Phase 3 implements mini-batch training for the Predictive Coding Network (PCN), enabling efficient training on larger datasets. This document describes the changes made and the new API.

## Key Changes

### 1. Core Module Extensions (`src/core/mod.rs`)

#### New Type: `BatchState`

A new state representation for batched operations:

```rust
pub struct BatchState {
    pub x: Vec<Array2<f32>>,        // Activations: (batch_size, d_l)
    pub mu: Vec<Array2<f32>>,       // Predictions: (batch_size, d_l)
    pub eps: Vec<Array2<f32>>,      // Errors: (batch_size, d_l)
    pub batch_size: usize,
    pub steps_taken: usize,
    pub final_energy: f32,
}
```

Key difference from `State`:
- Uses `Array2<f32>` instead of `Array1<f32>` for all layer representations
- First dimension is batch samples, second is neuron activations
- Enables simultaneous processing of multiple samples

#### New Methods on `PCN`

**Batch Initialization:**
```rust
pub fn init_batch_state(&self, batch_size: usize) -> BatchState
```

**Batch Operations:**
- `compute_batch_errors()` — Compute predictions and errors for all samples
- `relax_batch_step()` — One relaxation step for entire batch
- `relax_batch()` — Relax batch for fixed number of steps
- `compute_batch_energy()` — Total energy across batch
- `update_batch_weights()` — Hebbian weight update using batch-accumulated gradients

**Key Implementation Details:**

The batch operations maintain the same mathematical semantics as single-sample operations but operate on matrices:

```rust
// Single-sample (from Phase 1-2):
mu[l-1] = W[l] @ f(x[l]) + b[l-1]          // (d_{l-1},) = (d_{l-1}, d_l) @ (d_l,)

// Batch version (Phase 3):
mu[l-1] = f(x[l]) @ W[l]^T + b[l-1]        // (B, d_{l-1}) = (B, d_l) @ (d_l, d_{l-1})
```

Weight updates are **batch-averaged** to maintain learning rate consistency:
```rust
ΔW^ℓ = (η / B) ε^{ℓ-1}^T @ f(x^ℓ)    // Divide by batch size B
```

### 2. Training Module (`src/training/mod.rs`)

#### New Type: `BatchIterator`

Provides efficient mini-batch iteration over datasets with shuffle support:

```rust
pub struct BatchIterator {
    inputs: Array2<f32>,        // Full dataset: (num_samples, input_dim)
    targets: Array2<f32>,       // Full dataset: (num_samples, output_dim)
    batch_size: usize,
    // ... internal state ...
}
```

**Key Methods:**
- `new()` — Create iterator with validation
- `shuffle()` — Randomize order for next epoch
- `reset()` — Reset to start without shuffling
- `next_batch()` — Get next batch; returns `Option<(inputs, targets)>`
- `has_next()` — Check if more batches available

#### New Type: `TrainingConfig`

Configuration structure for mini-batch training:

```rust
pub struct TrainingConfig {
    pub relax_steps: usize,      // Relaxation iterations per sample
    pub alpha: f32,              // State update rate
    pub eta: f32,                // Weight learning rate
    pub clamp_output: bool,      // Clamp output during training
    pub batch_size: usize,       // Mini-batch size
    pub epochs: usize,           // Number of epochs to train
}
```

#### New Functions

**Single Batch Training:**
```rust
pub fn train_batch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,       // Shape: (batch_size, input_dim)
    targets: &Array2<f32>,      // Shape: (batch_size, output_dim)
    config: &TrainingConfig,
) -> PCNResult<Metrics>
```

Algorithm:
1. Initialize batch state
2. Clamp input layer
3. Optionally clamp output layer
4. Relax for `config.relax_steps` iterations
5. Update weights using accumulated batch errors
6. Return metrics

**Epoch Training:**
```rust
pub fn train_epoch(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &TrainingConfig,
    shuffle: bool,              // Whether to shuffle batches
) -> PCNResult<Metrics>
```

Algorithm:
1. Create batch iterator
2. Optionally shuffle
3. For each batch:
   - Train batch
   - Accumulate metrics
4. Return epoch-averaged metrics

**Multi-Epoch Training:**
```rust
pub fn train_epochs(
    pcn: &mut PCN,
    inputs: &Array2<f32>,
    targets: &Array2<f32>,
    config: &TrainingConfig,
) -> PCNResult<Vec<Metrics>>
```

Trains for `config.epochs` epochs, printing progress and returning metrics for each epoch.

#### New Type: `Metrics`

Training metrics computed per batch or epoch:

```rust
pub struct Metrics {
    pub energy: f32,               // Total prediction error
    pub layer_errors: Vec<f32>,    // L2 norm per layer
    pub accuracy: Option<f32>,     // Classification accuracy if applicable
    pub num_samples: usize,        // Number of samples processed
}
```

## Design Decisions

### 1. Batch Dimension First

States use shape `(batch_size, neuron_dim)` to align with ndarray's row-major conventions and enable efficient matrix operations:

```rust
inputs @ weights.t()    // Natural matrix multiplication
```

This allows vectorized computation across the batch.

### 2. Batch-Averaged Weight Updates

Weight updates are scaled by `1 / batch_size` to maintain consistent learning rate semantics across different batch sizes:

```rust
ΔW = (η / B) ε^T @ f(x)
```

Without this scaling, using larger batches would effectively reduce the learning rate.

### 3. Epoch-Level Shuffling

Shuffling occurs at the epoch level, not within batches. This:
- Improves generalization (random order helps SGD)
- Is computationally cheap (only shuffles indices)
- Maintains reproducibility (fixed seed → fixed order)

### 4. Energy Computation

Batch energy is the sum across all samples and layers:

```rust
E_batch = (1/2) * Σ_ℓ Σ_b ||ε^ℓ_b||²
```

This allows tracking total learning progress across the entire batch.

### 5. Accuracy Computation

For classification tasks, accuracy is computed as:

```rust
accuracy = (# correct class predictions) / batch_size
```

Where class is determined by `argmax(output)`.

## API Usage Example

```rust
use pcn::{PCN, TrainingConfig, train_epochs};
use ndarray::Array2;

// Create network
let mut pcn = PCN::new(vec![784, 128, 64, 10])?;

// Prepare data (already normalized to [-1, 1] or [0, 1])
let train_inputs = Array2::zeros((60000, 784));
let train_targets = Array2::zeros((60000, 10));

// Configure training
let config = TrainingConfig {
    relax_steps: 20,
    alpha: 0.05,
    eta: 0.01,
    clamp_output: true,
    batch_size: 32,
    epochs: 10,
};

// Train
let epoch_metrics = train_epochs(&mut pcn, &train_inputs, &train_targets, &config)?;

for (epoch, metrics) in epoch_metrics.iter().enumerate() {
    println!("Epoch {}: energy={:.4}, accuracy={:.4}",
        epoch + 1,
        metrics.energy,
        metrics.accuracy.unwrap_or(0.0)
    );
}
```

## Performance Characteristics

### Time Complexity

For a dataset with N samples, L layers, and T relaxation steps:

**Per Batch (size B):**
- Forward/error computation: O(B × L × d²) where d is typical layer size
- Relaxation: O(T × B × L × d²)
- Weight update: O(B × d²)
- **Total: O(T × B × L × d²)**

**Per Epoch:**
- Number of batches: ⌈N / B⌉
- **Total: O(T × N × L × d²)**

**Comparison to Phase 1-2 (single-sample training):**
- Single-sample per epoch: O(T × N × L × d²)
- **Batching has same asymptotic cost but enables parallelization**

### Memory Usage

**Per Batch:**
- States: O(B × L × d) for x, mu, eps
- Weights: O(L × d²) (shared, not per-batch)
- Workspace: O(B × d²) for intermediate matrices
- **Total: O(B × L × d + L × d²)**

Compared to single-sample:
- States: O(L × d)
- **Batching adds O(B × L × d) overhead** — scales linearly with batch size

## Testing

Comprehensive test coverage includes:

**Core Tests:**
- `test_batch_state_init()` — Verify shape initialization
- `test_compute_batch_errors()` — Error computation correctness
- `test_batch_energy_increases_with_error()` — Energy semantics
- `test_relax_batch_step()` — Single relaxation step
- `test_relax_batch()` — Full relaxation loop
- `test_update_batch_weights()` — Weight update mechanics

**Training Tests:**
- `test_batch_iterator_creation()` — Iterator initialization
- `test_batch_iterator_next_batch()` — Batch iteration logic
- `test_batch_iterator_shuffle()` — Shuffling functionality
- `test_train_batch_basic()` — Batch training
- `test_train_batch_shape_mismatch()` — Error handling
- `test_train_epoch()` — Epoch-level training
- `test_train_epochs()` — Multi-epoch training

All tests verify correctness; no performance optimization yet (per Phase 3 spec).

## Backward Compatibility

The legacy `Config` struct and single-sample API remain available but are considered deprecated:
- Kept for backward compatibility
- Marked with documentation notes
- Should migrate to `TrainingConfig` and batch functions for new code

## Next Steps (Phase 4+)

1. **Parallelization (Phase 4):** Rayon-based layer parallelism within relaxation
2. **GPU Support (Phase 5):** wgpu/CUDA kernels for batch matrix operations
3. **Advanced Optimization:** Adaptive learning rates, early stopping, validation metrics

## Files Modified

- `src/core/mod.rs` — Added `BatchState`, batch methods, tests
- `src/training/mod.rs` — Complete rewrite with batching, epoch loops, metrics
- `src/lib.rs` — Exports updated public API

## Commit Message

```
Implement Phase 3: Mini-batch training

- Add BatchState for matrix-based state representation
- Implement batch operations: compute_batch_errors, relax_batch_step, relax_batch, update_batch_weights
- Add BatchIterator trait for efficient mini-batch iteration with shuffle support
- Add TrainingConfig for batch-aware hyperparameter control
- Implement train_batch, train_epoch, train_epochs functions
- Add training metrics (energy, layer errors, accuracy) per batch/epoch
- Comprehensive test coverage for batch operations and training loops
- All code follows no-unwrap-in-lib policy and passes Clippy linting

No performance optimization yet - focus on correctness. Rayon parallelization will be in Phase 4.
```
