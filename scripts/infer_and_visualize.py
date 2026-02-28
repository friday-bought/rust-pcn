#!/usr/bin/env python3
"""Load a PCN checkpoint and run inference on a random CIFAR-10 test image.

Produces input.png and output.png side by side.
"""
import struct
import numpy as np
from pathlib import Path
import sys

def load_checkpoint(path):
    """Load PCN checkpoint: dims, weights, biases."""
    data = Path(path).read_bytes()
    off = 0
    
    num_layers = struct.unpack_from('<I', data, off)[0]; off += 4
    dims = []
    for _ in range(num_layers):
        d = struct.unpack_from('<I', data, off)[0]; off += 4
        dims.append(d)
    
    weights = [None]  # index 0 is placeholder
    for l in range(1, num_layers):
        rows = struct.unpack_from('<I', data, off)[0]; off += 4
        cols = struct.unpack_from('<I', data, off)[0]; off += 4
        n = rows * cols
        w = np.array(struct.unpack_from(f'<{n}f', data, off)); off += n * 4
        weights.append(w.reshape(rows, cols))
    
    biases = []
    for _ in range(num_layers):
        blen = struct.unpack_from('<I', data, off)[0]; off += 4
        b = np.array(struct.unpack_from(f'<{blen}f', data, off)); off += blen * 4
        biases.append(b)
    
    return dims, weights, biases

def load_cifar10_test(data_dir):
    """Load CIFAR-10 test batch."""
    path = Path(data_dir) / 'test_batch.bin'
    raw = path.read_bytes()
    n = len(raw) // 3073
    images = np.zeros((n, 3072), dtype=np.float32)
    labels = []
    for i in range(n):
        off = i * 3073
        labels.append(raw[off])
        pixels = np.frombuffer(raw, dtype=np.uint8, count=3072, offset=off+1)
        images[i] = pixels.astype(np.float32) / 255.0
    return images, labels

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def tanh(x):
    return np.tanh(x)

def pcn_reconstruct(dims, weights, biases, inp, alpha=0.025, relax_steps=80):
    """Run PCN inference (reconstruction mode): clamp input, let output settle."""
    L = len(dims)
    
    # Init states
    x = [np.zeros(d, dtype=np.float32) for d in dims]
    x[0] = inp.copy()
    
    # Forward init: propagate through weights to set initial states
    for l in range(1, L):
        x[l] = tanh(weights[l] @ x[l-1] + biases[l])
    
    # Relaxation
    for step in range(relax_steps):
        # Compute predictions (mu) - top-down
        mu = [np.zeros(d, dtype=np.float32) for d in dims]
        for l in range(1, L):
            mu[l-1] = tanh(weights[l].T @ x[l])  # Wait, need to check direction
        
        # Actually, looking at the Rust code:
        # mu[l] = W[l+1]^T @ x[l+1] for prediction of layer l from layer l+1
        # But the weight matrix W[l] maps layer l-1 -> layer l
        # So prediction of layer l from above = W[l+1]^T @ x[l+1]
        # And error[l] = x[l] - mu[l]
        
        # Recompute mu correctly
        mu = [np.zeros(d, dtype=np.float32) for d in dims]
        for l in range(0, L-1):
            # Prediction of layer l from layer l+1
            # W[l+1] has shape (dims[l+1], dims[l]) — maps l -> l+1
            # So W[l+1]^T maps l+1 -> l
            mu[l] = weights[l+1].T @ x[l+1]
        
        # Compute errors
        errors = [x[l] - mu[l] for l in range(L)]
        
        # Update hidden states (not input, not output in reconstruction)
        for l in range(1, L):
            # State update: reduce prediction error
            # dx = -error[l] + W[l]^T @ error[l-1]  ... hmm
            # Actually from the Rust code, the update is:
            # dx[l] = -alpha * (error[l] - tanh'(x[l]) * (W[l+1]^T @ error[l+1]))
            # But let me just use simple gradient descent on energy
            grad = errors[l].copy()
            if l < L - 1:
                # Contribution from layer above
                grad -= weights[l+1].T @ errors[l+1] * (1 - x[l]**2)  # tanh derivative
            x[l] = x[l] - alpha * grad
        
        # Re-clamp input
        x[0] = inp.copy()
    
    # The reconstruction is mu[0] — the top-down prediction of the input
    mu_final = [np.zeros(d, dtype=np.float32) for d in dims]
    for l in range(0, L-1):
        mu_final[l] = weights[l+1].T @ x[l+1]
    
    return mu_final[0]

def cifar_to_rgb(flat, channel_stats=None):
    """Convert CIFAR flat array (CHW, normalized) to HWC uint8 RGB image."""
    if channel_stats:
        means, stds = channel_stats
        # Denormalize
        for c in range(3):
            flat[c*1024:(c+1)*1024] = flat[c*1024:(c+1)*1024] * stds[c] + means[c]
    
    # Clip to [0, 1]
    flat = np.clip(flat, 0, 1)
    
    # CIFAR format: [R_plane, G_plane, B_plane] each 32x32
    r = flat[0:1024].reshape(32, 32)
    g = flat[1024:2048].reshape(32, 32)
    b = flat[2048:3072].reshape(32, 32)
    
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)

def save_png(img, path):
    """Save RGB image as PNG using pure Python (no PIL dependency)."""
    import zlib
    h, w, _ = img.shape
    
    def write_chunk(f, chunk_type, data):
        f.write(struct.pack('>I', len(data)))
        f.write(chunk_type)
        f.write(data)
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        f.write(struct.pack('>I', crc))
    
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        
        ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        write_chunk(f, b'IHDR', ihdr)
        
        raw = b''
        for y in range(h):
            raw += b'\x00'  # filter byte
            raw += img[y].tobytes()
        
        compressed = zlib.compress(raw)
        write_chunk(f, b'IDAT', compressed)
        write_chunk(f, b'IEND', b'')

def save_png_upscaled(img, path, scale=8):
    """Save with nearest-neighbor upscale for visibility."""
    h, w, c = img.shape
    big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    save_png(big, path)

def main():
    ckpt = "/home/kadajett/dev/rust-pcn/data/checkpoints/image-wide-v2-tuned/image-pcn-epoch-1000.bin"
    data_dir = "/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin"
    out_dir = Path("/home/node/.openclaw/workspace/pcn-rust/data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading checkpoint...")
    dims, weights, biases = load_checkpoint(ckpt)
    print(f"  Architecture: {dims}")
    print(f"  Weight shapes: {[w.shape if w is not None else None for w in weights]}")
    
    print("Loading CIFAR-10 test data...")
    images, labels = load_cifar10_test(data_dir)
    print(f"  Loaded {len(labels)} test images")
    
    # Compute channel stats for normalization (same as training)
    means = [images[:, c*1024:(c+1)*1024].mean() for c in range(3)]
    stds = [images[:, c*1024:(c+1)*1024].std() for c in range(3)]
    print(f"  Means: {means}")
    print(f"  Stds: {stds}")
    
    # Normalize
    images_norm = images.copy()
    for c in range(3):
        images_norm[:, c*1024:(c+1)*1024] = (images_norm[:, c*1024:(c+1)*1024] - means[c]) / stds[c]
    
    # Pick a random image
    np.random.seed(42)
    idx = np.random.randint(0, len(labels))
    label = labels[idx]
    print(f"\nSelected image #{idx}: {CIFAR_CLASSES[label]} (label={label})")
    
    inp = images_norm[idx]
    
    print("Running PCN inference (80 relax steps)...")
    reconstruction = pcn_reconstruct(dims, weights, biases, inp, alpha=0.025, relax_steps=80)
    
    # Compute MSE
    mse = np.mean((inp - reconstruction) ** 2)
    print(f"  Reconstruction MSE: {mse:.6f}")
    
    # Save images
    input_rgb = cifar_to_rgb(images[idx].copy())  # Use unnormalized original
    output_rgb = cifar_to_rgb(reconstruction.copy(), channel_stats=(means, stds))
    
    save_png_upscaled(input_rgb, out_dir / "input.png", scale=8)
    save_png_upscaled(output_rgb, out_dir / "output.png", scale=8)
    
    # Also save a side-by-side
    h, w = 32, 32
    combined = np.zeros((h, w*2 + 4, 3), dtype=np.uint8)
    combined[:, :w] = input_rgb
    combined[:, w+4:] = output_rgb
    combined[:, w:w+4] = 128  # gray separator
    save_png_upscaled(combined, out_dir / "comparison.png", scale=8)
    
    print(f"\nSaved:")
    print(f"  {out_dir}/input.png")
    print(f"  {out_dir}/output.png")
    print(f"  {out_dir}/comparison.png")

if __name__ == '__main__':
    main()
