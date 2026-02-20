//! Dataset loading and preprocessing.
//!
//! Handles loading from local files in /bulk-storage/localdocs,
//! normalization, shuffling, and batching.

/// Load a dataset from a local path.
pub fn load_dataset(_path: &str) -> Result<Vec<(Vec<f32>, Vec<f32>)>, String> {
    // TODO: implement dataset loading
    // Support: JSON, CSV, binary formats
    Err("not yet implemented".to_string())
}

/// Normalize data to a target range.
pub fn normalize(data: &mut [f32], min: f32, max: f32) {
    let data_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let data_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = data_max - data_min;

    if range == 0.0 {
        return;
    }

    for v in data {
        *v = min + ((*v - data_min) / range) * (max - min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        normalize(&mut data, -1.0, 1.0);

        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
    }
}
