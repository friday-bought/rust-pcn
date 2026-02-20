# /architecture — PCN Project Architecture Skill

Use this skill to:
- **Understand** the Predictive Coding Network design (algorithm, why it works)
- **Design** new features or layers
- **Debug** architectural issues
- **Refactor** while maintaining invariants
- **Document** design decisions

## Quick Links

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — Full PCN derivation, math, and design choices
- **[src/](./src/)** — Source tree (core/, data/, training/, utils/)
- **[GitHub Issues](https://github.com/your-account/pcn-rust/issues)** — Task tracking

## Core Concepts at a Glance

### Energy Minimization
```
E = (1/2) * Σ ||ε^ℓ-1||²

Lower energy = better predictions up and down the network.
```

### Two Populations Per Layer
- **State neurons** `x^ℓ`: the actual layer activity
- **Error neurons** `ε^ℓ`: difference between actual and predicted

### State Dynamics (Relaxation)
Each neuron adjusts itself to minimize energy:
```
x^ℓ += α * (-ε^ℓ + (W^ℓ+1)^T ε^ℓ-1 ⊙ f'(x^ℓ))
```

**Two competing forces:**
1. Align with top-down prediction (−ε^ℓ)
2. Better predict the layer below ((W^ℓ+1)^T ε^ℓ-1)

### Weight Updates (Local Learning)
```
ΔW^ℓ ∝ ε^ℓ-1 ⊗ f(x^ℓ)     (Hebbian rule)
```

Same-sign activity at pre- and post-synaptic neurons strengthens the weight.

## Design Phases

| Phase | Goal | Activation | Key Milestones |
|-------|------|-----------|-----------------|
| **1** | Linear kernel works | Identity `f(x)=x` | Energy decreases, can learn XOR |
| **2** | Nonlinear networks | Tanh, ReLU | Validation on MNIST, convergence-based stopping |
| **3** | Production ready | Batching, optimization | Criterion benchmarks, >80% test coverage |
| **4** | Advanced | Separate weights, noise | Sparsity, precision scalars |
| **5** | GPU-scale | CUDA/wgpu | Kubernetes training, real datasets |

## Making Decisions

### Adding a Feature

Before coding:

1. **Check ARCHITECTURE.md** — Is it mentioned as a design choice?
2. **File an issue** — Describe what, why, trade-offs
3. **Design in the issue** — Comments from agents should converge on approach
4. **Reference the issue in your PR**

### Activation Functions

- **Linear `f(x)=x`**: For Phase 1 only. Makes energy quadratic; easy analysis.
- **Tanh `f(x)=tanh(x)`**: Smooth, bounded, good for Phase 2+.
- **Leaky ReLU `f(x)=αx (x<0), x (x≥0)`**: Fast, biologically plausible, requires tuning.

Add activation via a trait (not a match on each layer).

### Weight Initialization

Current: `U(-0.05, 0.05)` (uniform small random).

**May need adjustment based on depth and activation:**
- Deeper networks → smaller initialization
- ReLU → Xavier or He initialization

Document any change in a doc comment.

### Relaxation Steps

Current: Fixed `T` (e.g., 20 steps).

**When to move to energy-based:**
- When accuracy plateaus with fixed T
- When you want adaptive compute
- Measure wall-clock time before and after

Use an issue to track this decision.

## Testing Strategy

### Unit Tests

Test each module in isolation:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_decreases() {
        // Create small network, verify E goes down during relaxation
    }

    #[test]
    fn test_hebbian_update() {
        // Verify weight update rule math
    }
}
```

### Integration Tests

Test full training loop:
```rust
#[test]
fn test_xor_learning() {
    // Train PCN on XOR, verify >95% accuracy after convergence
}
```

### Benchmarks

Track performance across phases:
```rust
// benches/relaxation.rs
// Measure time per relaxation step as network size scales
```

## Code Style

### Guard Clauses

```rust
// ✓ Good: early return
pub fn validate_dims(dims: &[usize]) -> Result<(), String> {
    if dims.is_empty() {
        return Err("dims must not be empty".to_string());
    }
    if dims.iter().any(|&d| d == 0) {
        return Err("all dims must be > 0".to_string());
    }
    Ok(())
}

// ✗ Avoid: nested ifs
pub fn validate_dims(dims: &[usize]) -> Result<(), String> {
    if !dims.is_empty() {
        if dims.iter().all(|&d| d > 0) {
            Ok(())
        } else {
            Err("...")
        }
    } else {
        Err("...")
    }
}
```

### Error Handling

```rust
// ✓ Good: explicit Result
pub fn train_sample(&mut self, input: &Array1<f32>, target: &Array1<f32>) -> Result<f32, TrainError> {
    // ...
}

// ✗ Avoid: unwrap() or panic!() in library code
let x = some_risky_op().unwrap();  // only in main() or tests
```

### Immutability by Default

```rust
// ✓ Good: take by reference, mutate in place when needed
pub fn relax_step(&self, state: &mut State, alpha: f32) { ... }

// ✗ Avoid: returning a new copy if mutation is the goal
pub fn relax_step(&self, state: &State, alpha: f32) -> State { ... }
```

## Debugging Checklist

**Network diverges (energy increases):**
- Check step sizes `alpha` and `eta` — are they too large?
- Verify clamping logic — are input/output being fixed?
- Plot layer-wise errors — which layer is misbehaving?

**Accuracy doesn't improve:**
- Check data normalization (should be roughly [-1, 1] or [0, 1])
- Increase relaxation steps T
- Try different initializations

**Out of memory:**
- Reduce batch size
- Reduce layer dimensions for debugging
- Profile with `cargo flamegraph`

## Refactoring Safely

1. **Run tests** before touching anything: `cargo test --all`
2. **Make one change** (rename, extract function, etc.)
3. **Test again**: `cargo test && cargo clippy && cargo fmt`
4. **Commit with reference to the refactoring issue**

**Never delete working code.** Stage it in a feature flag or `deprecated` module until confident.

## Communication

When making architectural decisions:
- **Link to ARCHITECTURE.md** sections for context
- **Propose alternatives** and trade-offs
- **Reference published papers** (Millidge et al., Rao & Ballard) for learning rule variants
- **Include measurements** (energy decrease, accuracy, wall-clock time)

## Resources

- **Energy minimization:** See ARCHITECTURE.md § Energy-Based Formulation
- **Locality:** ARCHITECTURE.md § Locality & Parallelism
- **Training loop:** src/training/lib.rs
- **Tests:** tests/ directory

