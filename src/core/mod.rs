//! Core PCN algorithm implementation.
//!
//! This module provides the fundamental PCN structures and operations:
//! - Energy-based formulation with prediction errors
//! - State relaxation via gradient descent
//! - Hebbian weight updates
//! - Local learning rules

use ndarray::{Array1, Array2};

/// A Predictive Coding Network with symmetric weight matrices.
#[derive(Debug, Clone)]
pub struct PCN {
    /// Network layer dimensions: [d0, d1, ..., dL]
    dims: Vec<usize>,
    /// Weight matrices: w[l] has shape (d_{l-1}, d_l), predicting layer l-1 from l
    w: Vec<Array2<f32>>,
    /// Bias vectors: b[l-1] has shape (d_{l-1})
    b: Vec<Array1<f32>>,
}

/// Network state during relaxation.
///
/// Holds activations, predictions, and errors for all layers.
#[derive(Debug, Clone)]
pub struct State {
    /// x[l]: activations at layer l
    pub x: Vec<Array1<f32>>,
    /// mu[l]: predicted activity of layer l
    pub mu: Vec<Array1<f32>>,
    /// eps[l]: prediction error at layer l (x[l] - mu[l])
    pub eps: Vec<Array1<f32>>,
}

impl PCN {
    /// Create a new PCN with the given layer dimensions.
    ///
    /// Weights are initialized from U(-0.05, 0.05); biases from zero.
    pub fn new(dims: Vec<usize>) -> Self {
        let l_max = dims.len() - 1;
        let mut w = Vec::with_capacity(l_max + 1);
        w.push(Array2::zeros((0, 0))); // dummy at index 0

        let mut b = Vec::with_capacity(l_max);

        for l in 1..=l_max {
            let out_dim = dims[l - 1];
            let in_dim = dims[l];
            let wl = Array2::zeros((out_dim, in_dim)); // TODO: initialize with random values
            w.push(wl);
            b.push(Array1::zeros(out_dim));
        }

        Self { dims, w, b }
    }

    /// Returns the network's layer dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Initialize a state for inference or training.
    pub fn init_state(&self) -> State {
        let l_max = self.dims.len() - 1;
        State {
            x: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
            mu: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
            eps: (0..=l_max)
                .map(|l| Array1::zeros(self.dims[l]))
                .collect(),
        }
    }

    /// Compute predictions and errors for the current state.
    ///
    /// Updates `state.mu` and `state.eps` in place.
    pub fn compute_errors(&self, state: &mut State) {
        let l_max = self.dims.len() - 1;

        for l in 1..=l_max {
            // TODO: implement activation function (f(x) = x for now)
            // mu[l-1] = W[l] * f(x[l]) + b[l-1]
            // eps[l-1] = x[l-1] - mu[l-1]
        }
    }

    /// Perform one relaxation step to minimize energy.
    ///
    /// Updates `state.x` in place using gradient descent on the energy function.
    pub fn relax_step(&self, state: &mut State, alpha: f32) {
        // TODO: implement state dynamics
        // x[l] += alpha * (-eps[l] + W[l+1]^T * eps[l-1] * f'(x[l]))
    }

    /// Relax the network for a given number of steps.
    pub fn relax(&self, state: &mut State, steps: usize, alpha: f32) {
        for _ in 0..steps {
            self.compute_errors(state);
            self.relax_step(state, alpha);
        }
        // Final error computation
        self.compute_errors(state);
    }

    /// Update weights using the Hebbian learning rule.
    ///
    /// ΔW[l] ∝ eps[l-1] ⊗ f(x[l])  (outer product)
    pub fn update_weights(&mut self, state: &State, eta: f32) {
        // TODO: implement Hebbian updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_init() {
        let dims = vec![2, 4, 3];
        let pcn = PCN::new(dims.clone());
        assert_eq!(pcn.dims(), &dims[..]);
    }

    #[test]
    fn test_state_init() {
        let dims = vec![2, 4, 3];
        let pcn = PCN::new(dims);
        let state = pcn.init_state();
        assert_eq!(state.x[0].len(), 2);
        assert_eq!(state.x[1].len(), 4);
        assert_eq!(state.x[2].len(), 3);
    }
}
