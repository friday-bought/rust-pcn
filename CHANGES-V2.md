# PCN V2 Changes

**Date:** 2026-02-22
**Author:** Opus (subagent)

## Summary

V2 implements 7 recommendations from the prior audit plus depth analysis fixes. Key changes:

### Depth Analysis Fixes
- **Relaxation steps**: 32 → 100 (3x more for 5-layer equilibrium)
- **Alpha**: 0.05 → 0.025 (halved for stability in deeper networks)
- **Per-layer eta scaling**: Middle layers get 2-3x boost to compensate for gradient attenuation
- **Architecture**: Widened bottleneck `[3104, 512, 384, 256, 97]` (was `[3104, 512, 256, 256, 97]`)

### Feature 1: Adaptive Encoding Strategy (PRIMARY NOVEL CONTRIBUTION)
Hybrid PCN that dynamically switches between sparse (SEAL-like, amplified learning) and dense (standard, dampened learning) encoding based on real-time prediction error variance per layer.

**How it works:**
- Tracks running variance of prediction errors per layer via EMA
- High variance → sparse mode: amplify learning rate (1.0-2.0x) to respond to surprises
- Low variance → dense mode: dampen learning rate (0.5-1.0x) to trust steady-state learning
- Threshold is configurable (`--adaptive-encoding-threshold`)
- Works independently of SEAL but complements it

**New types:** `AdaptiveEncodingState` in `training/mod.rs`
**New CLI flags:** `--adaptive-encoding`, `--adaptive-encoding-threshold`

### Feature 2: Section Boundary Warmup
Reduces learning rate for the first N epochs when starting a new data section to prevent the crash-then-recover pattern observed in SEAL training.

**How it works:**
- First `boundary_warmup_epochs` epochs use `eta * boundary_warmup_lr_factor`
- Both base eta and per-layer eta are scaled
- Restores original eta after warmup
- Applies to both book sections and Ollama text batches

**New CLI flags:** `--boundary-warmup-epochs`, `--boundary-warmup-lr-factor`

### Feature 3: Architecture Scaling Adjustments
- Widened bottleneck from `[512, 256, 256]` to `[512, 384, 256]`
- Each layer now compresses by ~1.5:1 ratio instead of 2:1 cliff + flat 1:1
- Tapered design provides natural hierarchy of increasingly abstract representations

### Feature 4: Text-Type Awareness
Automatic text classification and hyperparameter adjustment for the Ollama curriculum phase.

**How it works:**
- Analyzes text for punctuation ratio, digit ratio, uppercase ratio, word length
- Classifies as Fiction/Technical/Mixed/Unknown
- Adjusts eta factor, relax steps factor per text type
- Technical text → lower LR (0.8x), more relax steps (1.2x)
- Fiction → baseline (1.0x)
- Logged in metrics for research

**New types:** `TextType`, `TextTypeTracker` in `training/mod.rs`

### Feature 5: Convergence Tracking
The code already tracks both top-1 and top-5 accuracy. V2 ensures:
- Top-5 is always computed (not just on eval epochs)
- Convergence is tracked in the energy-accuracy monitor

### Feature 6: Top-5 Always On
Verified that top-5 accuracy is computed in all eval paths (book training and Ollama). Both GPU and CPU paths compute top-5.

### Feature 7: Energy-Accuracy Coupling Monitor
Detects when energy drops but accuracy doesn't follow — a sign of compression without learning.

**How it works:**
- Maintains sliding window of recent energy and accuracy values
- Compares first-half vs second-half averages for trend detection
- If energy decreasing by >5% but accuracy improving by <1%: decoupling detected
- After 2 consecutive detections: reduces eta by correction factor
- Automatically restores eta when coupling resumes
- Correction factor has a floor (0.1x) to prevent total stall

**New types:** `EnergyAccuracyMonitor` in `training/mod.rs`
**New CLI flags:** `--energy-accuracy-monitor`, `--energy-accuracy-window`

### Per-Layer Eta Scaling
- New CLI flag `--eta-per-layer "0.02,0.04,0.06,0.02"`
- Middle layers (512→384, 384→256) get 2-3x boost
- Input and output layers keep base rate
- Supported in both CPU and GPU training paths

### Additional Config Changes
- `Config` struct extended with per-layer eta, adaptive encoding, boundary warmup, energy-accuracy fields
- All new fields have sensible defaults (backwards compatible)
- All new types implement `Serialize/Deserialize` for checkpointing

## Pod Spec Changes

Old: `pcn-train-unified`
```
--hidden-sizes 512,256,256
--relax-steps 32
--eta 0.02
(no --alpha flag, default 0.05)
```

New: `pcn-train-v2`
```
--hidden-sizes 512,384,256
--relax-steps 100
--alpha 0.025
--eta 0.02
--eta-per-layer "0.02,0.04,0.06,0.02"
--seal-ema-decay 0.3
--seal-sensitivity 2.0
--adaptive-encoding
--adaptive-encoding-threshold 0.5
--boundary-warmup-epochs 3
--boundary-warmup-lr-factor 0.3
--energy-accuracy-monitor
--energy-accuracy-window 5
```

## Files Changed

### Library (`src/lib.rs`)
- Extended `Config` struct with 8 new fields
- Re-exported new types from training module

### Training (`src/training/mod.rs`)
- Added `AdaptiveEncodingState` struct and logic
- Added `EnergyAccuracyMonitor` struct and logic
- Added `TextType` enum and `TextTypeTracker` struct
- Modified `apply_accumulated_gradients` for per-layer eta
- Modified `apply_accumulated_gradients_seal` for per-layer eta

### Binary (`src/bin/train.rs`)
- Added CLI flags for all new features
- Integrated boundary warmup in section training loop
- Integrated adaptive encoding state
- Integrated energy-accuracy monitor
- Integrated text type tracker in Ollama phase
- Enhanced metrics JSON with new feature data

### GPU (`src/gpu/mod.rs`, `src/gpu/tensors.rs`)
- Updated weight update functions for per-layer eta support

### Kubernetes
- New pod spec: `k8s/pcn-train-v2.yaml`
- Build pod spec: `k8s/build-pod.yaml`
