# PCN Depth Analysis: Why 5 Layers Underperforms 3 Layers

**Date:** 2026-02-22  
**Analyst:** Opus (read-only codebase audit + PCN theory analysis)  
**Network under analysis:** `[3104, 512, 256, 256, 97]` (5 layers, 4 weight matrices)  
**Comparison baseline:** Previous 3-layer `[1552, 256, 97]` which approached ~20% accuracy  
**Current accuracy:** 4-5% top-1, ~19% top-5 on angular~20 (section 88, round 424)

---

## Executive Summary

The 5-layer network underperforms the 3-layer network primarily due to **insufficient relaxation depth for the added layers**, **error signal attenuation through the 256→256 bottleneck**, and **hyperparameters tuned for a shallower architecture**. The current run IS still learning (accuracy trending up section-over-section) but at a fraction of the rate it should be. The most impactful single change would be **increasing relaxation steps from 32 to 64-100** for the 5-layer network. However, the optimal path is to restart with corrected hyperparameters — the current run's learning rate has already shaped the weight landscape suboptimally, and continuing will yield diminishing returns.

---

## 1. Diagnosis: Why 5 Layers Underperforms 3 Layers

### 1.1 The Relaxation Depth Problem (PRIMARY CAUSE)

**PCN Theory:** In a PCN, relaxation is the analog of both forward and backward passes in backpropagation. During relaxation, error signals must propagate bidirectionally through the network to find the energy minimum. The number of relaxation steps required scales at minimum linearly with network depth — each step propagates information approximately one layer in each direction.

**The 3-layer network** had 2 weight matrices (input→hidden, hidden→output). With the previous config of 8 relaxation steps (from the earlier audit), this gave a ratio of **4 steps per weight matrix** — enough for error signals to make ~2 full round trips through the network.

**The 5-layer network** has 4 weight matrices (input→512, 512→256, 256→256, 256→output). With 32 relaxation steps, this gives **8 steps per weight matrix** — but this is misleading. The key metric isn't steps per matrix, it's **steps for the deepest error signal to make round trips**. In a 5-layer network, an error at the output must traverse 4 layers to reach the input, and the input's response must traverse 4 layers back. That's 8 layers of propagation for one round trip.

With 32 steps, the network gets approximately **4 full round trips** — but the attenuation through tanh derivatives means that by the 3rd-4th round trip, the gradient signal through 4 layers of `(1 - tanh²(x))` multiplication is vanishingly small. In the 3-layer network, even 8 steps gave sufficient round trips because the signal only had to traverse 2 layers.

**Empirical evidence:** The energy IS decreasing (from 253 to ~16), which means relaxation is doing *something*. But the states aren't reaching true equilibrium — they're settling into a local configuration that minimizes the easily-reachable energy but leaves the deeper inter-layer relationships underlearned.

**Key citation:** Millidge et al. (2022) show that PCN convergence to the backprop solution requires relaxation to approximate equilibrium. With insufficient steps, the Hebbian updates are computed from a non-equilibrium state, introducing noise into the weight updates that accumulates over training.

### 1.2 Error Signal Attenuation Through the 256→256 Bottleneck

The state update rule for internal layers is:

```
x^ℓ += α * (-ε^ℓ + W[l]^T ε[l-1] ⊙ f'(x^ℓ))
```

The `f'(x^ℓ) = 1 - tanh²(x^ℓ)` factor means that for any neuron where `|x^ℓ|` is moderate (say, tanh(x) ≈ ±0.76 for x = ±1), the derivative is only 0.42. For each layer the error signal passes through, it gets multiplied by this derivative AND by the weight matrix transpose.

In the 3-layer network, the error from layer 0 (input) to reach the weight update of W[2] passes through **1 layer** of derivatives. In the 5-layer network, the error from layer 0 to influence the weight update of W[4] passes through **3 layers** of derivatives. The multiplicative attenuation is roughly:

- 3-layer: `f'(x¹)` ≈ 0.42 (one intermediate layer)
- 5-layer: `f'(x¹) × f'(x²) × f'(x³)` ≈ 0.42³ ≈ 0.074

This means the gradient signal reaching the deepest weights in the 5-layer network is approximately **5.7x weaker** than in the 3-layer network, even before considering weight matrix conditioning.

The two 256-dimensional hidden layers (layers 2 and 3 in the 5-layer architecture `[3104, 512, 256, 256, 97]`) create a particular problem: they're an **information bottleneck followed by no expansion**. Layer 1 (512 dims) compresses 3104→512 (6:1 ratio). Layer 2 compresses 512→256 (2:1). Layer 3 maintains 256→256 (1:1). The 256→256 transition is neither compressing nor expanding — it's just adding depth without adding representational capacity, while incurring the full cost of another layer of gradient attenuation.

### 1.3 Learning Rate Not Scaled for Depth

The current config uses uniform learning rates: `alpha=0.05` for relaxation and `eta=0.02` for weight updates across all layers.

**The problem:** In deeper PCNs, the inner layers receive weaker error signals (as shown in §1.2). With a uniform learning rate, the inner layers' weights update too slowly relative to the outer layers. The outer layers (W[1] near input, W[4] near output) see strong error signals and update aggressively, while the inner layers (W[2], W[3]) barely budge.

This creates a **representation learning imbalance**: the network learns strong input encoding (W[1]) and output decoding (W[4]) but fails to learn the intermediate transformations (W[2], W[3]) that should extract hierarchical features. The 256→256 layers become approximately random projections — they transform the signal but not in a useful direction.

In the 3-layer network, there was only one weight matrix between input and output encoding — no "inner layer" problem existed.

### 1.4 Alpha (Relaxation Rate) × Depth Interaction

Alpha = 0.05 controls how aggressively states update during relaxation. In a 3-layer network, this produces stable dynamics because the state update at any layer only receives feedback from 1-2 nearby layers.

In a 5-layer network, the dynamics are more complex. Each relaxation step creates a cascade: updating x[1] changes ε[0] and ε[1], which affects x[2]'s update, which changes ε[1] and ε[2], and so on. With 4 internal layers all updating simultaneously with alpha=0.05, the system can oscillate rather than converge. The errors from each step haven't fully propagated before the next step begins.

This is analogous to a learning rate that's too high for the problem's curvature — the system bounces around the energy minimum rather than settling into it.

### 1.5 The Window Size Change (3-layer was 16, 5-layer is 32)

The 3-layer network used a 16-character window (input dim = 16 × 97 = 1552). The 5-layer network uses a 32-character window (input dim = 32 × 97 = 3104). This means:

1. **Doubled input dimensionality** — the first weight matrix W[1] is 3104×512 instead of 1552×256. This is a much harder compression problem (6:1 vs 6:1 ratio, but in a much higher-dimensional space).
2. **Quadrupled the number of distinct input patterns** — longer windows create more unique contexts, making the prediction problem harder.
3. **The larger window captures more context** — which should help, but only if the network can actually learn to use it. With insufficient relaxation and attenuated gradients, the extra context is noise rather than signal.

This change alone could account for a 2-3x accuracy reduction even without the depth increase.

### 1.6 SEAL Interaction with Depth

The SEAL system modulates learning rates per-layer based on surprise ratios. The current config uses:
- `decay=0.1` (slow EMA tracking)
- `sensitivity=3`
- `mod_range=[0.5, 1.8]`
- `boundary_reset=true`
- `blend=0.5`

From the prior audit (the 3-layer version), we know that SEAL's modulation factors hover very close to 1.0 (range ~0.93-1.07) because batch-averaged errors have low variance. Adding more layers doesn't change this fundamental dynamic — batch averaging still smooths out per-sample surprises.

However, SEAL's interaction with depth has a subtle issue: **the output layer's epsilon is always zero** (because output is clamped during training). This means the SEAL surprise state for the output layer always computes a degenerate surprise ratio. While the prior audit showed this modulation value isn't applied to any weight matrix (the loop uses `modulation[l-1]`), the EMA tracking for this layer is wasted computation that could interact with boundary resets in unexpected ways.

More importantly, SEAL doesn't address the fundamental depth problems. It can't compensate for insufficient relaxation or gradient attenuation — it only adjusts the *magnitude* of the learning signal, not its *path* through the network.

### 1.7 The 36.5% Peak Accuracy Mystery

The task notes that peak accuracy of 36.5% was observed "earlier, different book sections." This likely occurred on:
- A particularly predictable text section (e.g., repeated structures, simple vocabulary)
- Early in training when the network was overfitting to easy patterns
- Before the network had been exposed to enough diverse data to "average down" its specialized representations

The current 4-5% on angular~20 (Angular documentation) is a very different domain from literary fiction. Technical documentation has different character distributions, longer tokens, more punctuation variety, and less natural-language predictability. The gap between 36.5% and 4-5% likely reflects **domain difficulty** more than network degradation.

For comparison, the 3-layer network on the same technical documentation was also much lower than its fiction peaks (from the audit notes: the smaller model was getting ~2-3% on Angular docs early on, climbing to 5-8% after extended training). So the 5-layer network's 4-5% on Angular is actually **in the same ballpark** as the 3-layer network was at a similar training stage on technical docs.

**The real comparison should be on the same text type.** If the 3-layer network was approaching 20% on fiction/Ansible and the 5-layer is at 4-5% on Angular, these aren't directly comparable.

---

## 2. Training Trajectory Analysis: Continue or Restart?

### 2.1 Is It Still Learning?

Looking at the section-by-section trajectory on angular~20:

| Section | Accuracy Range | Top-5 Range | Trend |
|---------|---------------|-------------|-------|
| ~83 | 0.9-1.6% | 9-13% | ↑ |
| ~84 | 1.3-1.8% | 10-11.7% | ↑ slight |
| ~85 | 2-2.4% | 11-13.9% | ↑ |
| ~86 | 1.7-2.8% | 12.6-15% | ↑ |
| ~87 | 2.8-4.2% | 14-18.9% | ↑ strong |
| ~88 | 3.9-4.9% | 14.6-19.5% | ↑ |

**Yes, it's still learning.** Accuracy is roughly doubling every 3-4 sections (from ~1% at section 83 to ~5% at section 88). Top-5 accuracy is approaching 20%, which is the same level the 3-layer network achieved on top-1. This suggests the 5-layer network is learning useful representations — it just distributes probability mass less efficiently (correct answer is in top-5 but not top-1).

### 2.2 Recommendation: Restart with New Hyperparameters

Despite the positive trajectory, **the current run is suboptimal and should be restarted.** Here's why:

1. **The learning rate (eta=0.02) has been training all 4 weight matrices for 424 rounds.** The inner layers (W[2], W[3]) have been receiving attenuated gradients and updating slowly, while the outer layers (W[1], W[4]) have been updating aggressively. This has likely created a **weight magnitude imbalance** — the outer weights are well-trained but the inner weights are still near-random. Continuing with corrected hyperparameters would need to overcome this imbalance.

2. **The energy has dropped from 253 to 16.2.** This massive drop with relatively low accuracy means the network has found a low-energy configuration that doesn't correspond to good predictions. Continuing from this state, even with better hyperparameters, means climbing out of this energy valley — which is harder than starting fresh.

3. **7 hours of training is a sunk cost.** With corrected hyperparameters, the network should reach the current accuracy within 1-2 hours and then keep climbing. The time saved by restarting exceeds the time spent recovering from suboptimal weights.

**Exception:** If Jeremy values the scientific observation of seeing whether the current run can self-correct, continuing has research value. But for maximizing final accuracy, restart.

---

## 3. Specific Recommendations (Ranked by Expected Impact)

### Rank 1: Increase Relaxation Steps (32 → 80-100)

**Expected impact:** 2-3x accuracy improvement  
**Cost:** ~2.5-3x training time per sample (relaxation dominates compute)

**Reasoning:** This is the single most impactful change. The 5-layer network has 4 weight matrices. Each relaxation step propagates information approximately one layer. For the error signal to make a full round trip (output→input→output), it needs ~8 steps. For convergence to approximate equilibrium, research suggests 3-5 full round trips minimum, requiring 24-40 steps at the bare minimum.

32 steps is at the lower bound. 80-100 steps gives 10-12 full round trips, which should be sufficient for a 5-layer network to reach near-equilibrium before weight updates. This means the Hebbian updates will be computed from a much more accurate equilibrium state, leading to better weight learning.

**Recommended value:** Start with 80. Monitor energy convergence during relaxation (add logging of per-step energy if possible). If energy is still decreasing at step 80, increase to 100-128.

**GPU memory consideration:** Relaxation doesn't increase memory footprint (same buffers reused), only compute time. The RTX 3050 6GB should handle this fine.

### Rank 2: Per-Layer Learning Rate Scaling for Eta

**Expected impact:** 1.5-2x accuracy improvement  
**Cost:** Requires code change in weight update loop

**Reasoning:** The inner layers (W[2], W[3] — the 512→256 and 256→256 connections) receive attenuated error signals due to tanh derivative multiplication. To compensate, they should have higher learning rates.

**Recommended scaling:** Use layer-dependent eta:
```
eta[1] = 0.02      # Input→512 (strong signal, normal rate)
eta[2] = 0.04      # 512→256 (one layer of attenuation, 2x boost)
eta[3] = 0.06      # 256→256 (two layers of attenuation, 3x boost)  
eta[4] = 0.02      # 256→Output (strong error signal from clamped output)
```

The output layer (W[4]) gets strong gradients because the output is clamped — the error ε at the layer below is large and direct. The input layer (W[1]) gets strong gradients because ε[0] = x[0] - μ[0] is the raw input mismatch. The middle layers need the boost.

**Alternative approach:** Use a single eta=0.02 but add **gradient scaling** — multiply each layer's gradient by `depth_factor = (l_max - l + 1)` before the weight update. This achieves the same effect without explicit per-layer rates.

### Rank 3: Widen the Bottleneck (256→256 to 384→384 or 512→256)

**Expected impact:** 1.3-1.5x accuracy improvement  
**Cost:** ~2x GPU memory for weight matrices, ~1.5x compute

**Reasoning:** The current architecture `[3104, 512, 256, 256, 97]` has a sharp compression at layer 2 (512→256, 2:1 ratio) and then a flat bottleneck at layer 3 (256→256, 1:1). The 256-dimensional bottleneck may be too narrow to represent the full diversity of character-level patterns needed for 97-class prediction from a 32-char window.

**Recommended architectures (pick one):**

1. **Wider bottleneck:** `[3104, 512, 384, 384, 97]` — more capacity in the middle, gradual compression
2. **Tapered:** `[3104, 512, 384, 256, 97]` — each layer compresses by ~1.5:1, more principled
3. **Wider throughout:** `[3104, 768, 384, 256, 97]` — more capacity everywhere

Option 2 (tapered) is theoretically best for PCNs because it creates a natural hierarchy of increasingly abstract representations. Each layer compresses by a similar ratio, avoiding the sharp 2:1 cliff at layer 2.

**GPU memory check:** For `[3104, 512, 384, 256, 97]`:
- W[1]: 3104×512 = 1.59M params
- W[2]: 512×384 = 197K params  
- W[3]: 384×256 = 98K params
- W[4]: 256×97 = 25K params
- Total: ~1.91M params (+ biases ≈ 1.25K)
- At float32: ~7.6 MB for weights alone

Batch state for 512 samples, 5 layers: 512 × (3104+512+384+256+97) × 3 (x, mu, eps) × 4 bytes ≈ 27 MB. Well within 6GB GPU memory.

### Rank 4: Reduce Alpha (0.05 → 0.02-0.03) for Deeper Network

**Expected impact:** 1.2-1.5x accuracy improvement  
**Cost:** Requires more relaxation steps (compensated by Rank 1)

**Reasoning:** With 4 internal layers all updating simultaneously, alpha=0.05 may cause oscillation during relaxation. A smaller alpha means each step makes a more conservative adjustment, allowing the cascade of inter-layer updates to settle more smoothly.

**Recommended value:** alpha=0.025 (half of current). Combined with Rank 1 (80-100 steps), this gives 80 steps × 0.025 = 2.0 "total relaxation budget" versus current 32 × 0.05 = 1.6. The network gets more steps and smaller steps, leading to smoother convergence.

**The alpha × steps product** should be roughly proportional to network depth. For 5 layers: aim for alpha × steps ≈ 2.0-2.5. For 3 layers: alpha × steps ≈ 1.0-1.5 was sufficient.

### Rank 5: Tune SEAL for Depth

**Expected impact:** 1.1-1.3x accuracy improvement  
**Cost:** Config change only

**Current SEAL config issues with depth:**

1. **`decay=0.1` is too slow** — the EMA tracks so closely that surprise ratios are always near 1.0. For a deeper network processing many batches per section, the EMA becomes stale. Increase to 0.3-0.5 for more responsive surprise tracking.

2. **`sensitivity=3` may be reasonable** — but with near-unity surprise ratios, even sensitivity=3 maps through the steep sigmoid center, producing modulation near 1.0. Consider lowering to 1.5-2.0 to spread the response.

3. **`mod_range=[0.5, 1.8]` is fine** — centered at 1.0 when surprise is neutral. No change needed.

4. **Consider disabling SEAL for initial depth experiments** — SEAL adds complexity without clear benefit at the current accuracy levels. Once the baseline 5-layer network is performing well, re-enable SEAL and tune it.

### Rank 6: Reduce Window Size (32 → 24) or Use Skip Connections

**Expected impact:** 1.1-1.2x accuracy improvement  
**Cost:** Reduces context but simplifies learning

**Reasoning:** The jump from 16-char to 32-char windows doubled the input dimensionality and quadrupled the input space. If the goal is to test depth effects, reducing the window to 24 (input dim = 2328) would be a middle ground that still captures more context than the 3-layer network but doesn't overwhelm the deeper architecture.

**Alternative:** Keep 32-char windows but add skip connections (residual connections) that bypass the 256→256 layer. This would let the network learn the identity mapping for that layer while still having the option to learn useful transformations. This requires a code change but is a well-understood technique for training deeper networks.

---

## 4. Suggested Hyperparameter Sets

### Option A: Conservative (Minimal Changes)

```
Network: [3104, 512, 256, 256, 97]  (unchanged)
Relax steps: 80                     (was 32)
Alpha: 0.03                         (was 0.05)
Eta: 0.02                           (unchanged)
Batch size: 512                     (unchanged)
Epochs/section: 40                  (unchanged)
SEAL: disabled for initial testing
```

**Rationale:** Only changes relaxation parameters. If this works, the diagnosis is confirmed — insufficient relaxation was the primary issue.

### Option B: Recommended (Best Expected Performance)

```
Network: [3104, 512, 384, 256, 97]  (wider bottleneck)
Relax steps: 100                    (was 32)
Alpha: 0.025                        (was 0.05)
Eta: [0.02, 0.04, 0.05, 0.02]      (per-layer, boost middle)
Batch size: 512                     (unchanged)
Epochs/section: 40                  (unchanged)
SEAL: disabled initially, re-enable after baseline established
```

**Rationale:** Addresses all major issues. Per-layer eta compensates for gradient attenuation. Wider bottleneck provides more capacity. More relaxation steps ensure equilibrium.

### Option C: Aggressive (Maximum Depth Exploitation)

```
Network: [3104, 768, 512, 256, 97]  (much wider)
Relax steps: 128                    (maximum budget)
Alpha: 0.02                         (very conservative)
Eta: [0.015, 0.03, 0.04, 0.015]    (per-layer, heavy middle boost)
Batch size: 256                     (smaller for GPU memory)
Epochs/section: 60                  (more training per section)
SEAL: enabled with decay=0.3, sensitivity=2.0
```

**Rationale:** Maximizes network capacity and relaxation quality at the cost of training speed. Batch size reduced to 256 to fit the wider network in 6GB GPU memory. More epochs per section to exploit the larger network's slower convergence.

---

## 5. Long-Term Architecture Suggestions for Scaling PCN Depth

### 5.1 Residual/Skip Connections for PCN

The most impactful architectural change for deeper PCNs would be **residual connections**. In the PCN framework, this means adding an identity pathway alongside each weight matrix:

```
μ^{ℓ-1} = W^ℓ f(x^ℓ) + b^{ℓ-1} + γ × x^ℓ    (when dims match)
```

Where γ is a learnable scalar controlling how much of the "skip" signal to include. This would:
- Allow gradients to flow directly through the skip path (no derivative attenuation)
- Let the network learn the identity mapping for layers that don't need transformation
- Make depth nearly free in terms of gradient flow (as in ResNets)

For dimension mismatches (e.g., 512→256), use a simple linear projection in the skip path.

### 5.2 Layer Normalization for PCN

Adding layer normalization to the state updates would help with depth:

```
x^ℓ_normalized = (x^ℓ - mean(x^ℓ)) / (std(x^ℓ) + ε)
```

This would prevent the tanh saturation problem — if state values drift toward ±∞ in deeper layers, tanh saturates and derivatives go to zero. Layer norm keeps states centered, maintaining healthy derivatives.

### 5.3 Adaptive Relaxation Steps

Rather than fixed relaxation steps, use **convergence-based stopping** (already implemented in the codebase as `relax_with_convergence`). Set a generous max_steps (e.g., 200) and let each sample converge to its own equilibrium. Easy samples will converge in 20-30 steps; hard samples might need 100+. This amortizes compute across sample difficulty.

The current codebase already has this capability but it's not being used in the training binary (which uses fixed steps).

### 5.4 Separate Feedback Weights

For deeper networks (6+ layers), consider using separate forward and feedback weight matrices as described in ARCHITECTURE.md (Option B). This breaks the weight symmetry requirement, which becomes harder to maintain in deeper networks. The feedback weights can be smaller (lower rank) than the forward weights, reducing compute for error backpropagation while maintaining the full forward prediction capacity.

### 5.5 Precision-Weighted Errors

Instead of uniform error weighting, assign per-layer **precision scalars** that modulate how much each layer's error contributes to the total energy:

```
E = (1/2) * Σ_ℓ π_ℓ ||ε^ℓ||²
```

Where π_ℓ is learned or scheduled. Inner layers can have higher precision (more weight on their errors), forcing the network to learn accurate intermediate representations. This is inspired by the "precision weighting" concept from predictive coding theory (Rao & Ballard, 1999).

### 5.6 Gradual Depth Increase (Training Strategy)

Rather than jumping from 3 to 5 layers, use **progressive depth training**:

1. Train a 3-layer `[3104, 512, 97]` network until it plateaus
2. Insert a new layer: `[3104, 512, 256, 97]` — initialize W_new as identity-like, copy old weights
3. Train until plateau
4. Insert another: `[3104, 512, 256, 256, 97]` — same strategy
5. Continue as needed

This ensures each new layer starts from a useful baseline rather than random initialization. The existing layers' weights provide a warm start that the new layer can build upon.

---

## 6. Comparison Table: 3-Layer vs 5-Layer Analysis

| Factor | 3-Layer `[1552, 256, 97]` | 5-Layer `[3104, 512, 256, 256, 97]` | Impact |
|--------|--------------------------|--------------------------------------|--------|
| Weight matrices | 2 | 4 | 2x more params to learn |
| Relaxation round trips needed | ~2-3 | ~5-6 | 2.5x more steps needed |
| Steps per round trip | 4 (8 steps) | 8 | Marginal per step |
| Tanh derivative attenuation | 1 layer (0.42x) | 3 layers (0.074x) | 5.7x signal loss |
| Input dimensionality | 1552 | 3104 | 2x harder compression |
| Window context | 16 chars | 32 chars | 4x input space |
| Min useful relax steps | ~8-12 | ~40-64 | Need 3-5x more |
| Actual relax steps | 8 (prev config) | 32 | Insufficient for depth |
| Bottleneck width | 256 (1 layer) | 256 (2 layers) | Double bottleneck |
| Information ratio (in→out) | 16:1 | 32:1 | 2x harder task |

---

## 7. What Would Success Look Like?

With corrected hyperparameters (Option B above), a well-tuned 5-layer PCN on this task should achieve:

- **On fiction (Dracula, Austen, Twain):** 15-25% top-1 accuracy (the deeper network should eventually surpass the 3-layer's ~20% by capturing longer-range dependencies via the 32-char window)
- **On technical docs (Angular, Ansible):** 8-15% top-1 accuracy (lower than fiction due to domain difficulty, but competitive)
- **Top-5 accuracy:** 35-50% on fiction, 25-35% on technical docs
- **Energy:** Should stabilize around 5-8 (vs current 16)
- **Training time to reach 10% accuracy:** ~100-200 rounds (vs current 424 rounds at only 5%)

The 32-character window gives the 5-layer network a theoretical advantage the 3-layer never had — it can learn dependencies across a much wider context. But realizing this advantage requires the deeper layers to actually learn useful transformations, which requires the hyperparameter corrections outlined above.

---

## 8. Final Verdict

### Should the current run continue?

**No, restart with Option A or B.** The current run has been training for 7 hours with suboptimal relaxation, producing weights that are imbalanced (outer layers overtrained, inner layers undertrained). While accuracy is still trending up, the rate of improvement (~0.5% per section) means reaching the 3-layer network's 20% would take hundreds more sections at the current rate.

A restart with 80-100 relaxation steps and adjusted alpha would likely reach 5% within 1-2 hours and continue climbing much faster.

### What's the single most important thing to change?

**Relaxation steps: 32 → 80+.** This is the most PCN-theory-grounded fix. Deeper networks need more relaxation to reach equilibrium. Everything else is secondary.

### Is 5 layers the right depth for this task?

**4 layers** (`[3104, 512, 256, 97]`) might be the sweet spot — one more hidden layer than the 3-layer network for additional representational capacity, but without the 256→256 flat bottleneck. If 5 layers is desired, make the architecture tapered: `[3104, 512, 384, 256, 97]`.

### Is this a PCN-specific problem or a universal depth problem?

**PCN-specific.** Backprop networks handle depth through automatic differentiation — gradients flow through the entire computation graph regardless of depth (though they can also vanish). PCNs rely on iterative relaxation to approximate this, and relaxation quality degrades with insufficient steps. This is a known theoretical limitation of PCNs (Millidge et al., 2022; Whittington & Bogacz, 2017) that becomes practically significant at 4+ layers.

---

## Appendix: PCN Relaxation Theory

### Why Relaxation Steps Scale with Depth

Consider a 5-layer PCN with clamped input (layer 0) and clamped output (layer 4). The internal layers (1, 2, 3) must find states that minimize total energy. In the first relaxation step:

1. Errors ε[0], ε[1], ε[2], ε[3] are computed
2. x[1] updates based on ε[1] and ε[0]
3. x[2] updates based on ε[2] and ε[1]  
4. x[3] updates based on ε[3] and ε[2]

But these updates happen simultaneously — x[2]'s update uses the *old* ε[1], not the updated one. It takes another full step for the changes in x[1] to propagate through ε[1] to affect x[2]. This is a **communication delay** inherent to parallel PCN relaxation.

For the clamped output's information to reach x[1]:
- Step 1: ε[3] influences x[3]
- Step 2: x[3]'s change propagates through ε[2] to x[2]
- Step 3: x[2]'s change propagates through ε[1] to x[1]

That's 3 steps minimum for one-way propagation in a 5-layer network. For x[1] to adjust AND for that adjustment to propagate back to x[3], we need 3+3 = 6 steps for one round trip.

For convergence, research suggests 3-5 round trips, giving a minimum of 18-30 steps. The current 32 steps is right at the bare minimum, leaving no margin for the attenuation effects described in §1.2.

### The Equilibrium Quality Problem

The Hebbian weight update `ΔW^ℓ = η ε^{ℓ-1} ⊗ f(x^ℓ)` assumes the state is at or near equilibrium. At equilibrium, the errors ε represent the *true* prediction mismatches that the weights should correct. Away from equilibrium, the errors are contaminated by transient dynamics — they reflect the relaxation process, not the actual input-output relationship.

Weight updates from non-equilibrium states introduce systematic bias. Over many training steps, this bias compounds, leading the weights in directions that reduce relaxation transients rather than prediction errors. This is why the energy drops (the network gets better at relaxing quickly) but accuracy doesn't improve proportionally (the weights aren't learning the right input-output mapping).

---

*This analysis is read-only. No code or configuration was modified.*
