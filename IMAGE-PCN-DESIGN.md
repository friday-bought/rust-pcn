# Image PCN Design Document

**Date:** 2026-02-22
**Goal:** Extend the text PCN to visual processing with CIFAR-10, preparing for multimodal fusion.

---

## 1. Architecture Overview

### Text PCN (existing, running)
```
[3104, 512, 384, 256, 97]
  ↑     ↑    ↑    ↑    ↑
  │     │    │  LATENT  │
  │     │    │  SPACE   │
  32×97 │    │   256d   └─ 97 chars (output)
  input │    384
        512
```

### Image PCN (new)
```
[3072, 512, 384, 256, 3072]    (reconstruction mode)
[3072, 512, 384, 256, 10]      (classification mode)
  ↑     ↑    ↑    ↑    ↑
  │     │    │  LATENT  │
  │     │    │  SPACE   │
  32×32 │    │   256d   └─ reconstruction: predict own pixels
  ×3    │    384             classification: predict class
  =3072 512
```

### Design Rationale

- **Hidden layers `[512, 384, 256]`** are identical to the text PCN by design
- **Layer 3 (256 dims)** is the shared latent space for future multimodal bridging
- **5 layers** in both networks ensures compatible depth for cross-modal alignment
- **Reconstruction mode** (primary): natural for PCN's top-down generative nature — the network learns to predict its own input from internal representations, developing rich visual features without labels
- **Classification mode** (secondary): for benchmarking and supervised evaluation

---

## 2. Input Encoding: Patch-Based Processing

### Why Patches?

Raw pixels lack spatial structure when flattened. Patch-based encoding (ViT-style):
1. Preserves local spatial relationships within each patch
2. Creates natural "tokens" analogous to text characters
3. Enables future autoregressive processing (next-patch prediction)
4. Keeps total dimension unchanged (spatial reordering, not compression)

### Patch Configuration

**Default: 8×8 patches on 32×32 CIFAR images**

```
32×32 image → 4×4 grid of 8×8 patches
Each patch: 8×8×3 = 192 dimensions (RGB)
16 patches × 192 dims = 3072 total (= 32×32×3, lossless)
```

```
┌────────┬────────┬────────┬────────┐
│ P(0,0) │ P(1,0) │ P(2,0) │ P(3,0) │  ← Row 0 of patches
│  192d  │  192d  │  192d  │  192d  │
├────────┼────────┼────────┼────────┤
│ P(0,1) │ P(1,1) │ P(2,1) │ P(3,1) │  ← Row 1
│  192d  │  192d  │  192d  │  192d  │
├────────┼────────┼────────┼────────┤
│ P(0,2) │ P(1,2) │ P(2,2) │ P(3,2) │  ← Row 2
│  192d  │  192d  │  192d  │  192d  │
├────────┼────────┼────────┼────────┤
│ P(0,3) │ P(1,3) │ P(2,3) │ P(3,3) │  ← Row 3
│  192d  │  192d  │  192d  │  192d  │
└────────┴────────┴────────┴────────┘
       16 patches × 192d = 3072d input
```

**Alternative: 4×4 patches (finer granularity)**
```
32×32 image → 8×8 grid of 4×4 patches
64 patches × 48 dims = 3072 total
```

The 4×4 variant may be useful for future autoregressive experiments where each "step" processes one small patch.

### Pixel Format

- CIFAR binary: channel-planar `[R:1024][G:1024][B:1024]`
- Internal: interleaved HWC `[R,G,B, R,G,B, ...]`
- Normalized: per-channel z-score `(pixel - μ) / σ` (optional, recommended)
- Within patches: HWC format maintained for spatial locality

---

## 3. Training Objective: Reconstruction vs Classification

### Reconstruction (Primary, Recommended)

```
Input:  image pixels (3072d, patch-encoded)
Target: same image pixels (3072d)
Loss:   PCN energy = Σ ||ε^ℓ||²
```

**Why reconstruction for multimodal fusion:**

1. **Self-supervised:** No labels needed, can scale to any image dataset
2. **Rich representations:** The network must learn the full structure of images, not just class-discriminative features
3. **Natural PCN fit:** PCN's energy function already measures reconstruction error — the top-down pathway IS a generative model
4. **Modality-agnostic latent:** The 256-dim latent space learns to encode "what matters" about images without being biased toward classification
5. **Compatible with text:** The text PCN also learns to reconstruct/predict its input — both networks develop generative latent spaces, making them natural partners for bridging

**Metrics:**
- MSE (mean squared error) between input and μ[0] (top-down prediction of layer 0)
- PSNR (peak signal-to-noise ratio) in dB — human-interpretable quality measure
- Energy (PCN prediction error energy) — should decrease during training

### Classification (Secondary, for Benchmarking)

```
Input:  image pixels (3072d, patch-encoded)
Target: one-hot class label (10d)
Loss:   PCN energy focused on label prediction
```

**Useful for:**
- Validating that the network learns meaningful features
- Comparing against known CIFAR-10 benchmarks
- Debugging (easier to see if 10-class accuracy improves than MSE)

**NOT recommended for multimodal fusion** because:
- Label-specific features are too task-dependent
- 10-dim output is too compressed to capture rich visual semantics
- Latent space becomes classification-oriented, not general-purpose

---

## 4. Hyperparameter Selection

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Activation | Tanh | Bounded output for stable reconstruction |
| Relax steps | 80 | Slightly less than text (100) since images are lower-dimensional output |
| Alpha (state LR) | 0.025 | Same as text V2, proven stable for 5-layer networks |
| Eta (weight LR) | 0.015 | Slightly lower than text (0.02) — images have higher input dim |
| Per-layer eta | [0.015, 0.03, 0.045, 0.015] | Middle layers boosted 2-3× (same pattern as text V2) |
| Batch size | 256 | Balance between gradient quality and memory |
| Patch size | 8×8 | Good balance of spatial locality and patch count |
| Normalization | Per-channel z-score | Standard for image networks; helps tanh stay in linear regime |

### Per-Layer Eta Rationale (for reconstruction mode)

```
Layer 1: W[1] shape (3072, 512)  — η=0.015 (base, large fan-in)
Layer 2: W[2] shape (512, 384)   — η=0.030 (2× boost, mid-level features)  
Layer 3: W[3] shape (384, 256)   — η=0.045 (3× boost, abstraction bottleneck)
Layer 4: W[4] shape (256, 3072)  — η=0.015 (base, reconstruction head)
```

Middle layers get boosted to compensate for gradient attenuation in deep networks (same insight as text PCN V2).

---

## 5. Multimodal Bridging Plan

### The Shared Latent Space

Both networks have a 256-dimensional representation at layer 3:

```
TEXT PCN:                    IMAGE PCN:
[3104] ← input              [3072] ← input (patches)
  ↓                            ↓
[512]                        [512]
  ↓                            ↓
[384]                        [384]
  ↓                            ↓
[256] ═══ BRIDGE LAYER ═══ [256]
  ↓                            ↓
[97]  ← text output          [3072] ← image reconstruction
```

### Phase 1: Independent Pre-training (Current)
Train each modality independently until good within-modality performance:
- Text: character prediction accuracy
- Image: reconstruction MSE/PSNR

### Phase 2: Contrastive Alignment
After pre-training, align the latent spaces using paired data:

```rust
// Paired sample: (image, text_description)
let image_latent = image_pcn.get_layer(3);  // 256d
let text_latent = text_pcn.get_layer(3);    // 256d

// Contrastive loss: pull paired latents together, push unpaired apart
let alignment_loss = contrastive_loss(image_latent, text_latent);
```

Options:
- **CLIP-style contrastive:** Match image-text pairs, separate non-pairs
- **CKA (Centered Kernel Alignment):** Soft alignment that preserves within-modality structure
- **Simple MSE bridging:** For paired data, minimize ||z_img - z_text||²

### Phase 3: Cross-Modal Generation
Once aligned, one modality can drive the other:

```
Image → [512, 384, 256] → inject into text PCN layer 3 → generate text
Text → [512, 384, 256] → inject into image PCN layer 3 → generate image
```

### Phase 4: Unified Multimodal PCN
Eventually, merge into a single network:

```
              [256] ← SHARED LATENT
             ╱     ╲
       [384]         [384]
      ╱                   ╲
   [512]                 [512]
   ╱                         ╲
[3104]                     [3072]
text input               image input (patches)
```

Both modalities feed into the same latent space, with modality-specific encoder/decoder arms.

---

## 6. Multimodal Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                    MULTIMODAL PCN STACK (Future)                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  TEXT ARM                SHARED             IMAGE ARM               ║
║  ────────               ──────             ─────────               ║
║                                                                    ║
║  [97] output                              [3072] reconstructed     ║
║    ↑                                         ↑                     ║
║  [3104] input     ╔═══════════╗          [3072] input (patches)    ║
║    ↓              ║           ║             ↓                      ║
║  [512] ──────────►║           ║◄────────  [512]                    ║
║    ↓              ║  BRIDGE   ║             ↓                      ║
║  [384] ──────────►║   LAYER   ║◄────────  [384]                    ║
║    ↓              ║           ║             ↓                      ║
║  [256] ══════════►║  [256+256]║◄════════  [256]                    ║
║                   ║  or [256] ║                                    ║
║                   ║ (aligned) ║                                    ║
║                   ╚═══════════╝                                    ║
║                                                                    ║
║  Bridge options:                                                   ║
║  1. Linear projection: z_shared = W_bridge @ [z_text; z_image]    ║
║  2. Contrastive alignment: minimize dist(z_text, z_image)          ║
║  3. Shared weights: same W[3] matrix for both modalities           ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 7. Dataset: CIFAR-10

### Specifications

| Property | Value |
|----------|-------|
| Name | CIFAR-10 |
| Source | https://www.cs.toronto.edu/~kriz/cifar.html |
| License | MIT-like (free for research and commercial use) |
| Images | 60,000 (50K train + 10K test) |
| Dimensions | 32×32×3 (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Format | Binary: 1 byte label + 3072 bytes pixels per record |
| Download size | 162 MB compressed |
| Location | `/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin/` |

### Why CIFAR-10?

1. **Small and fast:** 162 MB download, fits in RAM, fast epoch times
2. **Well-understood:** Thousands of published benchmarks for comparison
3. **Consistent dimensions:** All images are exactly 32×32×3
4. **Good for validation:** Before scaling to larger datasets, verify the image PCN architecture works
5. **No auth required:** Direct HTTP download, curl-friendly

### Future Datasets

After validating on CIFAR-10, candidates for scaling:
- **CIFAR-100:** Same format, 100 classes (more nuanced features)
- **STL-10:** 96×96 images (test higher resolution)
- **Tiny ImageNet:** 200 classes, 64×64 (bridge to full ImageNet)
- **CC0 image collections from Hugging Face** (for true open licensing)

---

## 8. Implementation Files

### New Files

| File | Purpose |
|------|---------|
| `src/data/image.rs` | CIFAR-10 loader, patch encoding, normalization, metrics |
| `src/bin/train_image.rs` | Image training binary with CLI, eval loop, checkpointing |
| `k8s/pcn-image-train.yaml` | Kubernetes pod spec with init container for dataset download |
| `IMAGE-PCN-DESIGN.md` | This document |

### Modified Files

| File | Change |
|------|--------|
| `src/data/mod.rs` | Added `pub mod image;` |
| `src/lib.rs` | Re-exported `data::image as image_data` |
| `Cargo.toml` | Added `[[bin]]` entry for `pcn-train-image` |

### Dataset Scripts

| File | Purpose |
|------|---------|
| `/bulk-storage/datasets/images/download-cifar10.sh` | Download + verify + extract CIFAR-10 binary |

---

## 9. Memory Budget (RTX 3050 6GB)

### Reconstruction mode: [3072, 512, 384, 256, 3072]

**Parameters:**
- W[1]: 3072×512 = 1,572,864 floats = 6.0 MB
- W[2]: 512×384 = 196,608 floats = 0.75 MB
- W[3]: 384×256 = 98,304 floats = 0.38 MB
- W[4]: 256×3072 = 786,432 floats = 3.0 MB
- Biases: ~7,220 floats = 0.03 MB
- **Total weights: ~10.2 MB**

**Per-sample state (x, mu, eps for all layers):**
- 3 × (3072 + 512 + 384 + 256 + 3072) = 3 × 7,296 = 21,888 floats = 85 KB

**Batch of 256:**
- 256 × 85 KB ≈ 21.3 MB (state only, during parallel relaxation)

**Training data in memory:**
- 50,000 × 3,072 × 4 bytes = 614 MB (train images)
- 10,000 × 3,072 × 4 bytes = 123 MB (test images)

**Total estimate:** ~770 MB — comfortably fits in CPU RAM (8 GB pod limit).
Currently CPU-only; GPU would reduce to ~50 MB on device.

---

## 10. Evaluation Strategy

### Reconstruction Mode Metrics

1. **MSE** (Mean Squared Error): Average pixel-level reconstruction error
   - Good: < 0.01 (normalized pixels)
   - Target: < 0.005

2. **PSNR** (Peak Signal-to-Noise Ratio): Quality in dB
   - Acceptable: > 20 dB
   - Good: > 25 dB
   - Excellent: > 30 dB

3. **Per-layer energy**: Track energy contribution per layer to detect bottlenecks

### Classification Mode Metrics

1. **Top-1 accuracy**: Primary benchmark
   - Random chance: 10%
   - Decent PCN: > 40%
   - Good: > 55%
   - (Note: SotA CNNs get 96%+, but PCN is a different paradigm)

2. **Per-class accuracy**: Identify easy/hard classes

### Latent Space Quality (for bridging readiness)

1. **Cluster separation**: Do same-class images cluster in 256d latent space?
2. **Linear probing:** Train a linear classifier on frozen layer-3 activations
3. **Reconstruction from latent:** Clamp layer 3, decode to pixels — are images recognizable?

---

## 11. Build & Deploy

### Building

The image training binary must be compiled on the host or in a build pod:

```bash
# In the build pod or on theshire:
cd /workspace
cargo build --release --bin pcn-train-image
```

The build pod spec (`k8s/build-pod.yaml`) can be extended to also build `pcn-train-image`.

### Running

```bash
# Apply the pod (after building):
kubectl apply -f k8s/pcn-image-train.yaml

# Monitor:
kubectl logs -n pcn-train pcn-image-train -f

# Note: Don't run simultaneously with pcn-train-v2 if GPU-limited
```

### Monitoring

```bash
# Watch metrics:
kubectl exec -n pcn-train pcn-image-train -- \
  tail -f /workspace/data/output/metrics-image.jsonl | jq '.'

# Check reconstruction quality:
kubectl exec -n pcn-train pcn-image-train -- \
  tail -1 /workspace/data/output/metrics-image.jsonl | jq '.reconstruction_mse, .reconstruction_psnr_db'
```
