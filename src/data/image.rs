//! Image dataset loading and preprocessing for PCN training.
//!
//! Supports CIFAR-10 binary format and provides patch-based encoding
//! for ViT-style image processing in predictive coding networks.
//!
//! ## CIFAR-10 Binary Format
//!
//! Each file contains 10,000 records, each structured as:
//! ```text
//! [label: u8] [red: 1024×u8] [green: 1024×u8] [blue: 1024×u8]
//! ```
//! Total: 3073 bytes per record. Pixels are row-major within each channel plane.
//!
//! ## Patch Encoding
//!
//! Images are split into non-overlapping patches (like Vision Transformers):
//! - 32×32 image with 8×8 patches → 16 patches, each 192 dims (8×8×3)
//! - Patches are flattened and concatenated into a single input vector
//! - This preserves spatial locality while enabling sequence-like processing

use ndarray::{Array1, Array2};
use std::fs;
use std::io;
use std::path::Path;

/// CIFAR-10 image dimensions.
pub const CIFAR_HEIGHT: usize = 32;
/// CIFAR-10 image width.
pub const CIFAR_WIDTH: usize = 32;
/// CIFAR-10 number of color channels.
pub const CIFAR_CHANNELS: usize = 3;
/// Total pixels per image (32 × 32 × 3 = 3072).
pub const CIFAR_PIXELS: usize = CIFAR_HEIGHT * CIFAR_WIDTH * CIFAR_CHANNELS;
/// Bytes per CIFAR-10 record (1 label + 3072 pixels).
pub const CIFAR_RECORD_SIZE: usize = 1 + CIFAR_PIXELS;
/// Number of classes in CIFAR-10.
pub const CIFAR_NUM_CLASSES: usize = 10;

/// CIFAR-10 class label names.
pub const CIFAR_CLASS_NAMES: [&str; CIFAR_NUM_CLASSES] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// Configuration for patch-based image encoding.
#[derive(Debug, Clone)]
pub struct PatchConfig {
    /// Image height in pixels.
    pub image_height: usize,
    /// Image width in pixels.
    pub image_width: usize,
    /// Number of color channels.
    pub channels: usize,
    /// Patch height in pixels.
    pub patch_height: usize,
    /// Patch width in pixels.
    pub patch_width: usize,
}

impl PatchConfig {
    /// Create a default CIFAR-10 patch config with 8×8 patches.
    #[must_use]
    pub fn cifar10_8x8() -> Self {
        Self {
            image_height: CIFAR_HEIGHT,
            image_width: CIFAR_WIDTH,
            channels: CIFAR_CHANNELS,
            patch_height: 8,
            patch_width: 8,
        }
    }

    /// Create a CIFAR-10 patch config with 4×4 patches (finer granularity).
    #[must_use]
    pub fn cifar10_4x4() -> Self {
        Self {
            image_height: CIFAR_HEIGHT,
            image_width: CIFAR_WIDTH,
            channels: CIFAR_CHANNELS,
            patch_height: 4,
            patch_width: 4,
        }
    }

    /// Number of patches along the height axis.
    #[must_use]
    pub fn patches_y(&self) -> usize {
        self.image_height / self.patch_height
    }

    /// Number of patches along the width axis.
    #[must_use]
    pub fn patches_x(&self) -> usize {
        self.image_width / self.patch_width
    }

    /// Total number of patches per image.
    #[must_use]
    pub fn num_patches(&self) -> usize {
        self.patches_y() * self.patches_x()
    }

    /// Dimensions of a single flattened patch (patch_h × patch_w × channels).
    #[must_use]
    pub fn patch_dim(&self) -> usize {
        self.patch_height * self.patch_width * self.channels
    }

    /// Total input dimension: all patches concatenated.
    #[must_use]
    pub fn total_input_dim(&self) -> usize {
        self.num_patches() * self.patch_dim()
    }
}

/// A loaded image dataset with normalized pixel values and labels.
#[derive(Debug, Clone)]
pub struct ImageDataset {
    /// Pixel data normalized to \[0, 1\], shape: (num_images, pixels_per_image).
    /// Channel layout: interleaved RGB per pixel (H×W×C format).
    pub images: Array2<f32>,
    /// Class labels (0-9 for CIFAR-10).
    pub labels: Vec<u8>,
    /// Number of images.
    pub num_images: usize,
    /// Pixels per image.
    pub image_dim: usize,
}

/// Load a single CIFAR-10 binary batch file.
///
/// Returns normalized pixel data in HWC (interleaved) format and labels.
///
/// # Errors
///
/// Returns `io::Error` if the file cannot be read or has unexpected size.
pub fn load_cifar10_batch(path: &Path) -> io::Result<(Vec<Vec<f32>>, Vec<u8>)> {
    let data = fs::read(path)?;
    let num_records = data.len() / CIFAR_RECORD_SIZE;

    if data.len() % CIFAR_RECORD_SIZE != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "File size {} is not a multiple of record size {}",
                data.len(),
                CIFAR_RECORD_SIZE
            ),
        ));
    }

    let mut images = Vec::with_capacity(num_records);
    let mut labels = Vec::with_capacity(num_records);

    for i in 0..num_records {
        let offset = i * CIFAR_RECORD_SIZE;
        let label = data[offset];
        labels.push(label);

        // CIFAR binary: [R:1024][G:1024][B:1024] (channel-planar)
        // We convert to HWC (interleaved) format: [R,G,B, R,G,B, ...]
        let r_start = offset + 1;
        let g_start = r_start + 1024;
        let b_start = g_start + 1024;

        let mut pixels = Vec::with_capacity(CIFAR_PIXELS);
        for pixel_idx in 0..1024 {
            // Normalize to [0, 1]
            pixels.push(f32::from(data[r_start + pixel_idx]) / 255.0);
            pixels.push(f32::from(data[g_start + pixel_idx]) / 255.0);
            pixels.push(f32::from(data[b_start + pixel_idx]) / 255.0);
        }

        images.push(pixels);
    }

    Ok((images, labels))
}

/// Load the full CIFAR-10 training set (5 batches, 50,000 images).
///
/// # Errors
///
/// Returns `io::Error` if any batch file cannot be read.
pub fn load_cifar10_train(data_dir: &Path) -> io::Result<ImageDataset> {
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    for batch_num in 1..=5 {
        let batch_path = data_dir.join(format!("data_batch_{batch_num}.bin"));
        let (images, labels) = load_cifar10_batch(&batch_path)?;
        all_images.extend(images);
        all_labels.extend(labels);
    }

    let num_images = all_images.len();
    let image_dim = CIFAR_PIXELS;

    // Convert to Array2
    let mut data = Array2::zeros((num_images, image_dim));
    for (i, img) in all_images.iter().enumerate() {
        for (j, &val) in img.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    Ok(ImageDataset {
        images: data,
        labels: all_labels,
        num_images,
        image_dim,
    })
}

/// Load the CIFAR-10 test set (1 batch, 10,000 images).
///
/// # Errors
///
/// Returns `io::Error` if the test batch file cannot be read.
pub fn load_cifar10_test(data_dir: &Path) -> io::Result<ImageDataset> {
    let batch_path = data_dir.join("test_batch.bin");
    let (images, labels) = load_cifar10_batch(&batch_path)?;

    let num_images = images.len();
    let image_dim = CIFAR_PIXELS;

    let mut data = Array2::zeros((num_images, image_dim));
    for (i, img) in images.iter().enumerate() {
        for (j, &val) in img.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    Ok(ImageDataset {
        images: data,
        labels,
        num_images,
        image_dim,
    })
}

/// Extract patches from an image and concatenate them into a flat vector.
///
/// Given an image in HWC format (H×W×C interleaved), extracts non-overlapping
/// patches in raster order (left-to-right, top-to-bottom) and concatenates
/// them into a single vector suitable for PCN input.
///
/// # Arguments
///
/// - `image`: flat pixel array in HWC format, length = H × W × C
/// - `config`: patch configuration specifying image and patch dimensions
///
/// # Returns
///
/// Flat vector of length `num_patches × patch_dim`, where patches are
/// ordered raster-scan and pixels within each patch are in HWC format.
#[must_use]
pub fn extract_patches(image: &[f32], config: &PatchConfig) -> Array1<f32> {
    let total_dim = config.total_input_dim();
    let mut patches = Array1::zeros(total_dim);

    let mut patch_offset = 0;
    for py in 0..config.patches_y() {
        for px in 0..config.patches_x() {
            // Extract one patch
            for row in 0..config.patch_height {
                for col in 0..config.patch_width {
                    let img_row = py * config.patch_height + row;
                    let img_col = px * config.patch_width + col;
                    let img_pixel_idx = (img_row * config.image_width + img_col) * config.channels;

                    for c in 0..config.channels {
                        patches[patch_offset] = image[img_pixel_idx + c];
                        patch_offset += 1;
                    }
                }
            }
        }
    }

    patches
}

/// Reconstruct an image from concatenated patches.
///
/// Inverse of `extract_patches`: takes a flat vector of concatenated patches
/// and reconstructs the original image in HWC format.
///
/// # Arguments
///
/// - `patches`: flat vector of length `num_patches × patch_dim`
/// - `config`: patch configuration specifying image and patch dimensions
///
/// # Returns
///
/// Flat pixel array in HWC format, length = H × W × C.
#[must_use]
pub fn reconstruct_from_patches(patches: &Array1<f32>, config: &PatchConfig) -> Vec<f32> {
    let total_pixels = config.image_height * config.image_width * config.channels;
    let mut image = vec![0.0f32; total_pixels];

    let mut patch_offset = 0;
    for py in 0..config.patches_y() {
        for px in 0..config.patches_x() {
            for row in 0..config.patch_height {
                for col in 0..config.patch_width {
                    let img_row = py * config.patch_height + row;
                    let img_col = px * config.patch_width + col;
                    let img_pixel_idx = (img_row * config.image_width + img_col) * config.channels;

                    for c in 0..config.channels {
                        image[img_pixel_idx + c] = patches[patch_offset];
                        patch_offset += 1;
                    }
                }
            }
        }
    }

    image
}

/// Extract patches from a batch of images.
///
/// # Arguments
///
/// - `images`: image data, shape (num_images, H×W×C)
/// - `config`: patch configuration
///
/// # Returns
///
/// Batch of patch vectors, shape (num_images, num_patches × patch_dim).
#[must_use]
pub fn extract_patches_batch(images: &Array2<f32>, config: &PatchConfig) -> Array2<f32> {
    let num_images = images.nrows();
    let total_dim = config.total_input_dim();
    let mut result = Array2::zeros((num_images, total_dim));

    for i in 0..num_images {
        let image_data = images.row(i);
        // Collect row into a Vec to avoid unwrap on as_slice
        let image_vec: Vec<f32> = image_data.iter().copied().collect();
        let patches = extract_patches(&image_vec, config);
        result.row_mut(i).assign(&patches);
    }

    result
}

/// Create one-hot encoded labels for classification.
///
/// # Arguments
///
/// - `labels`: vector of class labels (0-indexed)
/// - `num_classes`: total number of classes
///
/// # Returns
///
/// One-hot encoded matrix, shape (num_labels, num_classes).
#[must_use]
pub fn one_hot_encode(labels: &[u8], num_classes: usize) -> Array2<f32> {
    let num_labels = labels.len();
    let mut encoded = Array2::zeros((num_labels, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        encoded[[i, label as usize]] = 1.0;
    }

    encoded
}

/// Compute per-channel mean and standard deviation for normalization.
///
/// # Arguments
///
/// - `images`: image data in HWC format, shape (num_images, H×W×C)
/// - `channels`: number of color channels
///
/// # Returns
///
/// Tuple of (mean, std) vectors, each of length `channels`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_channel_stats(images: &Array2<f32>, channels: usize) -> (Vec<f32>, Vec<f32>) {
    let num_images = images.nrows();
    let pixels_per_channel = images.ncols() / channels;
    let total_pixels = num_images * pixels_per_channel;

    let mut means = vec![0.0f32; channels];
    let mut vars = vec![0.0f32; channels];

    // Compute means
    for i in 0..num_images {
        let row = images.row(i);
        for pixel_idx in 0..pixels_per_channel {
            for c in 0..channels {
                means[c] += row[pixel_idx * channels + c];
            }
        }
    }
    for mean in &mut means {
        *mean /= total_pixels as f32;
    }

    // Compute variances
    for i in 0..num_images {
        let row = images.row(i);
        for pixel_idx in 0..pixels_per_channel {
            for c in 0..channels {
                let diff = row[pixel_idx * channels + c] - means[c];
                vars[c] += diff * diff;
            }
        }
    }
    let stds: Vec<f32> = vars
        .iter()
        .map(|&v| (v / total_pixels as f32).sqrt().max(1e-8))
        .collect();

    (means, stds)
}

/// Normalize images using per-channel mean and standard deviation.
///
/// Applies z-score normalization: `(pixel - mean) / std` per channel.
///
/// # Arguments
///
/// - `images`: mutable image data in HWC format, shape (num_images, H×W×C)
/// - `means`: per-channel means
/// - `stds`: per-channel standard deviations
/// - `channels`: number of color channels
pub fn normalize_channels(
    images: &mut Array2<f32>,
    means: &[f32],
    stds: &[f32],
    channels: usize,
) {
    let num_images = images.nrows();
    let pixels_per_channel = images.ncols() / channels;

    for i in 0..num_images {
        let mut row = images.row_mut(i);
        for pixel_idx in 0..pixels_per_channel {
            for c in 0..channels {
                let idx = pixel_idx * channels + c;
                row[idx] = (row[idx] - means[c]) / stds[c];
            }
        }
    }
}

/// Compute mean squared error between two image vectors.
///
/// Used to measure reconstruction quality.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn reconstruction_mse(original: &Array1<f32>, reconstructed: &Array1<f32>) -> f32 {
    let diff = original - reconstructed;
    diff.dot(&diff) / diff.len() as f32
}

/// Compute peak signal-to-noise ratio (PSNR) for reconstruction quality.
///
/// Assumes pixel values in \[0, 1\] range.
/// Higher is better; typical good values are 20-40 dB.
#[must_use]
pub fn reconstruction_psnr(mse: f32) -> f32 {
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    // For [0,1] range, max pixel value = 1.0
    10.0 * (1.0 / mse).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_config_cifar10_8x8() {
        let config = PatchConfig::cifar10_8x8();
        assert_eq!(config.patches_y(), 4);
        assert_eq!(config.patches_x(), 4);
        assert_eq!(config.num_patches(), 16);
        assert_eq!(config.patch_dim(), 192); // 8×8×3
        assert_eq!(config.total_input_dim(), 3072); // 16 × 192 = 3072 = 32×32×3
    }

    #[test]
    fn test_patch_config_cifar10_4x4() {
        let config = PatchConfig::cifar10_4x4();
        assert_eq!(config.patches_y(), 8);
        assert_eq!(config.patches_x(), 8);
        assert_eq!(config.num_patches(), 64);
        assert_eq!(config.patch_dim(), 48); // 4×4×3
        assert_eq!(config.total_input_dim(), 3072); // 64 × 48 = 3072
    }

    #[test]
    fn test_extract_and_reconstruct_patches() {
        let config = PatchConfig::cifar10_8x8();

        // Create a test image with known pattern
        let mut image = vec![0.0f32; CIFAR_PIXELS];
        for i in 0..CIFAR_PIXELS {
            image[i] = (i as f32) / (CIFAR_PIXELS as f32);
        }

        let patches = extract_patches(&image, &config);
        assert_eq!(patches.len(), config.total_input_dim());

        let reconstructed = reconstruct_from_patches(&patches, &config);
        assert_eq!(reconstructed.len(), CIFAR_PIXELS);

        // Verify roundtrip is exact
        for (i, (&orig, &recon)) in image.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - recon).abs() < 1e-6,
                "Mismatch at pixel {i}: orig={orig}, recon={recon}"
            );
        }
    }

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 3, 7, 9, 1];
        let encoded = one_hot_encode(&labels, 10);

        assert_eq!(encoded.shape(), &[5, 10]);
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[0, 1]], 0.0);
        assert_eq!(encoded[[1, 3]], 1.0);
        assert_eq!(encoded[[4, 1]], 1.0);
    }

    #[test]
    fn test_reconstruction_mse() {
        let a = Array1::from_vec(vec![0.0, 1.0, 0.5]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.5]);
        assert_eq!(reconstruction_mse(&a, &b), 0.0);

        let c = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let mse = reconstruction_mse(&a, &c);
        assert!(mse > 0.0);
    }

    #[test]
    fn test_reconstruction_psnr() {
        let psnr = reconstruction_psnr(0.01);
        assert!(psnr > 0.0);
        assert!(psnr < 100.0);

        let psnr_perfect = reconstruction_psnr(0.0);
        assert!(psnr_perfect.is_infinite());
    }

    #[test]
    fn test_channel_stats() {
        // Create uniform image data
        let mut images = Array2::zeros((2, 6)); // 2 images, 2 pixels × 3 channels
        // Image 1: all pixels = [0.2, 0.4, 0.6]
        images[[0, 0]] = 0.2;
        images[[0, 1]] = 0.4;
        images[[0, 2]] = 0.6;
        images[[0, 3]] = 0.2;
        images[[0, 4]] = 0.4;
        images[[0, 5]] = 0.6;
        // Image 2: same
        images[[1, 0]] = 0.2;
        images[[1, 1]] = 0.4;
        images[[1, 2]] = 0.6;
        images[[1, 3]] = 0.2;
        images[[1, 4]] = 0.4;
        images[[1, 5]] = 0.6;

        let (means, stds) = compute_channel_stats(&images, 3);
        assert!((means[0] - 0.2).abs() < 1e-5);
        assert!((means[1] - 0.4).abs() < 1e-5);
        assert!((means[2] - 0.6).abs() < 1e-5);
        // Std should be ~0 (clamped to 1e-8)
        for std in &stds {
            assert!(*std < 1e-5);
        }
    }
}
