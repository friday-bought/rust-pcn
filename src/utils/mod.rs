//! Math utilities, activation functions, and helpers.

/// Activation function: identity (for linear networks).
#[inline]
pub fn identity(x: f32) -> f32 {
    x
}

/// Derivative of identity activation.
#[inline]
pub fn d_identity(_x: f32) -> f32 {
    1.0
}

/// Activation function: tanh.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Derivative of tanh activation.
#[inline]
pub fn d_tanh(x: f32) -> f32 {
    let t = x.tanh();
    1.0 - t * t
}

/// Activation function: leaky ReLU.
#[inline]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Derivative of leaky ReLU.
#[inline]
pub fn d_leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        assert_eq!(identity(3.0), 3.0);
        assert_eq!(d_identity(0.0), 1.0);
    }

    #[test]
    fn test_tanh() {
        let x = 0.5;
        let y = tanh(x);
        assert!(y > 0.0 && y < x);
        assert!(d_tanh(x) > 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(2.0, 0.01), 2.0);
        assert_eq!(leaky_relu(-1.0, 0.01), -0.01);
        assert_eq!(d_leaky_relu(2.0, 0.01), 1.0);
        assert_eq!(d_leaky_relu(-1.0, 0.01), 0.01);
    }
}
