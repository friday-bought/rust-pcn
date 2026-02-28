#!/usr/bin/env python3
"""Pure-Python PCN inference — no numpy required."""
import struct, zlib, math, random, os
from pathlib import Path

def load_checkpoint(path):
    data = Path(path).read_bytes()
    off = 0
    num_layers = struct.unpack_from('<I', data, off)[0]; off += 4
    dims = []
    for _ in range(num_layers):
        d = struct.unpack_from('<I', data, off)[0]; off += 4
        dims.append(d)
    
    weights = [None]
    for l in range(1, num_layers):
        rows = struct.unpack_from('<I', data, off)[0]; off += 4
        cols = struct.unpack_from('<I', data, off)[0]; off += 4
        n = rows * cols
        flat = list(struct.unpack_from(f'<{n}f', data, off)); off += n * 4
        # Store as list of rows
        w = [flat[r*cols:(r+1)*cols] for r in range(rows)]
        weights.append((rows, cols, w))
    
    biases = []
    while off < len(data):
        blen = struct.unpack_from('<I', data, off)[0]; off += 4
        b = list(struct.unpack_from(f'<{blen}f', data, off)); off += blen * 4
        biases.append(b)
    # Pad with zeros for any missing layers
    while len(biases) < num_layers:
        biases.append([0.0] * dims[len(biases)])
    
    return dims, weights, biases

def mat_vec(w_tuple, v):
    """Matrix-vector multiply. w_tuple = (rows, cols, list_of_rows)"""
    rows, cols, w = w_tuple
    return [sum(w[r][c] * v[c] for c in range(cols)) for r in range(rows)]

def mat_T_vec(w_tuple, v):
    """Transpose matrix-vector multiply."""
    rows, cols, w = w_tuple
    result = [0.0] * cols
    for r in range(rows):
        for c in range(cols):
            result[c] += w[r][c] * v[r]
    return result

def tanh_vec(v):
    return [math.tanh(x) for x in v]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def vec_scale(v, s):
    return [x * s for x in v]

def load_cifar10_test(data_dir):
    path = os.path.join(data_dir, 'test_batch.bin')
    raw = open(path, 'rb').read()
    n = len(raw) // 3073
    images = []
    labels = []
    for i in range(n):
        off = i * 3073
        labels.append(raw[off])
        pixels = [raw[off+1+j] / 255.0 for j in range(3072)]
        images.append(pixels)
    return images, labels

def pcn_reconstruct(dims, weights, biases, inp, alpha=0.025, relax_steps=80):
    L = len(dims)
    x = [[0.0]*d for d in dims]
    x[0] = inp[:]
    
    # W[l] stored as (dims[l-1], dims[l]) — so forward = W^T @ x, backward = W @ x
    # Forward: x[l] = tanh(W[l]^T @ x[l-1] + b[l])  (W^T maps l-1 -> l)
    # Prediction of l from l+1: mu[l] = W[l+1] @ x[l+1]  (W maps l+1 -> l)
    
    # Forward init
    for l in range(1, L):
        raw = vec_add(mat_T_vec(weights[l], x[l-1]), biases[l])
        x[l] = tanh_vec(raw)
    
    # Relaxation
    import time
    for step in range(relax_steps):
        if step % 5 == 0:
            print(f"  step {step}/{relax_steps}...", flush=True)
        # Predictions from above: mu[l] = W[l+1] @ x[l+1]
        mu = [[0.0]*d for d in dims]
        for l in range(0, L-1):
            mu[l] = mat_vec(weights[l+1], x[l+1])
        
        # Errors
        errors = [vec_sub(x[l], mu[l]) for l in range(L)]
        
        # Update hidden states
        for l in range(1, L):
            grad = errors[l][:]
            if l < L - 1:
                # W[l+1] maps l -> l+1, so W[l+1]^T maps l+1 -> l... wait
                # W[l+1] stored as (dims[l], dims[l+1])
                # To propagate error from l+1 to l: W[l+1] @ errors[l+1]
                above_contrib = mat_vec(weights[l+1], errors[l+1])
                dtanh = [1 - x[l][j]**2 for j in range(dims[l])]
                grad = [grad[j] - above_contrib[j] * dtanh[j] for j in range(dims[l])]
            x[l] = [max(-10, min(10, x[l][j] - alpha * grad[j])) for j in range(dims[l])]
        
        x[0] = inp[:]
    
    # Final reconstruction: mu[0] = W[1] @ x[1]
    mu_final = mat_vec(weights[1], x[1])
    return mu_final

def save_png(pixels_rgb, w, h, path, scale=8):
    """Save RGB image as upscaled PNG."""
    sw, sh = w * scale, h * scale
    
    def write_chunk(f, ct, d):
        f.write(struct.pack('>I', len(d)))
        f.write(ct)
        f.write(d)
        f.write(struct.pack('>I', zlib.crc32(ct + d) & 0xffffffff))
    
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        write_chunk(f, b'IHDR', struct.pack('>IIBBBBB', sw, sh, 8, 2, 0, 0, 0))
        
        raw = bytearray()
        for y in range(sh):
            raw.append(0)  # filter
            sy = y // scale
            for x in range(sw):
                sx = x // scale
                idx = sy * w + sx
                raw.append(pixels_rgb[idx][0])
                raw.append(pixels_rgb[idx][1])
                raw.append(pixels_rgb[idx][2])
        
        write_chunk(f, b'IDAT', zlib.compress(bytes(raw)))
        write_chunk(f, b'IEND', b'')

def flat_to_rgb(flat):
    """CIFAR CHW [0,1] -> list of (R,G,B) tuples, HWC order."""
    pixels = []
    for y in range(32):
        for x in range(32):
            r = max(0, min(255, int(flat[0*1024 + y*32 + x] * 255)))
            g = max(0, min(255, int(flat[1*1024 + y*32 + x] * 255)))
            b = max(0, min(255, int(flat[2*1024 + y*32 + x] * 255)))
            pixels.append((r, g, b))
    return pixels

def denorm_flat(flat, means, stds):
    """Denormalize a CHW flat vector."""
    out = flat[:]
    for c in range(3):
        for i in range(1024):
            out[c*1024 + i] = out[c*1024 + i] * stds[c] + means[c]
    return out

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    ckpt = "/home/kadajett/dev/rust-pcn/data/checkpoints/image-wide-v2-tuned/image-pcn-epoch-1000.bin"
    data_dir = "/bulk-storage/datasets/images/cifar-10/cifar-10-batches-bin"
    out_dir = "/home/node/.openclaw/workspace/pcn-rust/data/output"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading checkpoint...")
    dims, weights, biases = load_checkpoint(ckpt)
    print(f"  Architecture: {dims}")
    
    print("Loading CIFAR-10 test data...")
    images, labels = load_cifar10_test(data_dir)
    print(f"  {len(labels)} test images")
    
    # Channel stats (use subset for speed — pure python is slow)
    subset = images[:500]
    means = [0.0]*3
    stds = [0.0]*3
    ns = len(subset)
    for c in range(3):
        s = 0.0
        for img in subset:
            for i in range(1024):
                s += img[c*1024+i]
        means[c] = s / (ns * 1024)
    for c in range(3):
        s = 0.0
        for img in subset:
            for i in range(1024):
                s += (img[c*1024+i] - means[c])**2
        stds[c] = (s / (ns * 1024)) ** 0.5
    print(f"  Means: {means}")
    print(f"  Stds: {stds}")
    
    # Pick random image
    random.seed(42)
    idx = random.randint(0, len(labels)-1)
    label = labels[idx]
    print(f"\nImage #{idx}: {CLASSES[label]}")
    
    # Normalize
    inp = images[idx][:]
    for c in range(3):
        for i in range(1024):
            inp[c*1024+i] = (inp[c*1024+i] - means[c]) / stds[c]
    
    print("Running PCN inference (80 steps)...")
    recon = pcn_reconstruct(dims, weights, biases, inp, alpha=0.025, relax_steps=20)
    
    mse = sum((inp[i] - recon[i])**2 for i in range(3072)) / 3072
    print(f"  MSE: {mse:.6f}")
    
    # Save input
    input_rgb = flat_to_rgb(images[idx])
    save_png(input_rgb, 32, 32, os.path.join(out_dir, "input.png"), scale=8)
    
    # Save output (denormalize reconstruction)
    recon_denorm = denorm_flat(recon, means, stds)
    output_rgb = flat_to_rgb(recon_denorm)
    save_png(output_rgb, 32, 32, os.path.join(out_dir, "output.png"), scale=8)
    
    # Side by side
    combined = []
    for y in range(32):
        for x in range(68):  # 32 + 4 gap + 32
            if x < 32:
                combined.append(input_rgb[y*32+x])
            elif x < 36:
                combined.append((128,128,128))
            else:
                combined.append(output_rgb[y*32+(x-36)])
    save_png(combined, 68, 32, os.path.join(out_dir, "comparison.png"), scale=8)
    
    print(f"\nSaved to {out_dir}/")
    print("  input.png | output.png | comparison.png")

if __name__ == '__main__':
    main()
