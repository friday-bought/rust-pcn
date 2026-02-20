# PCN (Predictive Coding Networks) in Rust

A production-grade implementation of Predictive Coding Networks from first principles, trained on local datasets without pre-trained models.

## Overview

This project implements Predictive Coding Networks (PCNs), a biologically-plausible alternative to backpropagation that:
- **Operates locally** — each neuron responds only to adjacent-layer signals
- **Learns continuously** — no separate forward/backward phases; computation and learning happen in parallel
- **Scales horizontally** — massively parallelizable architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full derivation and design rationale.

## Project Structure

```
src/
├── core/              # PCN algorithm kernel (networks, layers, state)
├── data/              # Dataset loading and preprocessing
├── training/          # Training loops, convergence, metrics
├── utils/             # Math utilities, activations, statistics
└── bin/               # CLI tools and experiments

tests/                 # Unit and integration tests
examples/              # Example scripts
docs/                  # Technical documentation
```

## Build & Development

### Prerequisites
- Rust 1.70+ (ideally latest stable)
- `cargo` and `rustfmt` installed
- Git with clean working directory

### Quick Start

```bash
# Build with strict linting
cargo build --release

# Run tests (all must pass)
cargo test --all

# Format and lint
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings

# Run a simple example
cargo run --example linear_training
```

## Development Principles

- **Guard clauses and early returns** — prevent nested conditionals
- **Immutable by default** — mutate only where it's semantically necessary
- **Explicit error handling** — no panics in library code; `Result<T>` for fallible operations
- **No unsafe code** unless absolutely justified (and documented)
- **Continuous integration** — all changes tested before commit
- **Iterative refinement** — build working phases, then optimize; never delete working code

## Implementation Phases

### Phase 1: Linear PCN (Current)
- [ ] Core PCN struct with symmetric weights
- [ ] Energy computation
- [ ] State relaxation (gradient descent on energy)
- [ ] Hebbian weight updates
- [ ] Training loop with fixed relaxation steps
- [ ] Unit tests and basic MNIST validation

### Phase 2: Nonlinear PCN
- [ ] Add tanh activation (smooth derivatives)
- [ ] Implement leaky ReLU variant
- [ ] Convergence-based stopping (instead of fixed T)
- [ ] Integration tests on toy problems (XOR, spirals)

### Phase 3: Batching & Performance
- [ ] Mini-batch training
- [ ] Typed array buffers and reuse
- [ ] Rayon-based parallelization
- [ ] Benchmarks (criterion)

### Phase 4: Advanced Features
- [ ] Separate feedback weights (biologically-closer variant)
- [ ] Precision scalars per layer
- [ ] Sparsity penalties
- [ ] Noise injection

### Phase 5: GPU Training (Kubernetes)
- [ ] GPU kernel support (via wgpu or CUDA bindings)
- [ ] Kubernetes deployment with resource tracking
- [ ] Large-scale dataset training

## Data

All training uses local datasets from `/bulk-storage/localdocs`. No pre-trained models.

## Code Quality

- **Linting:** `cargo clippy` — treat warnings as errors
- **Formatting:** `cargo fmt` (non-negotiable)
- **Testing:** All public APIs have tests; >80% coverage expected
- **Commits:** One feature per commit; include issue reference

## Contributing

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes; keep commits atomic
3. Test: `cargo test && cargo clippy && cargo fmt`
4. Push and reference issue in PR

## References

- [Predictive Coding: A Biologically-Plausible Learning Algorithm](https://arxiv.org/abs/2301.07755)
- ["Why Does the Brain Do Backprop?" — Neuron Perspectives](https://www.youtube.com/watch?v=wNPBSesj4kY)
- Original transcript analysis: `docs/pcn-derivation.md`

## Status

**Current Phase:** Architecture finalization and sub-agent research planning.

See [GitHub Issues](https://github.com/your-account/pcn-rust/issues) for task tracking.
