//! Rust bindings for HPTT (High-Performance Tensor Transposition)
//!
//! This crate provides safe Rust bindings to the HPTT C++ library for
//! high-performance out-of-place tensor transpositions.
//!
//! # Example
//!
//! ```no_run
//! use hptt::transpose_f64;
//!
//! // Transpose a 3D tensor [2, 3, 4] -> [3, 2, 4]
//! let input = vec![1.0; 2 * 3 * 4];
//! let mut output = vec![0.0; 3 * 2 * 4];
//!
//! transpose_f64(
//!     &[1, 0, 2],  // permutation: swap first two dims
//!     1.0,         // alpha
//!     &input,
//!     &[2, 3, 4],  // shape
//!     0.0,         // beta (overwrite)
//!     &mut output,
//!     1,           // single thread
//! ).expect("transpose failed");
//! ```

mod ffi;

pub use num_complex::{Complex32, Complex64};
use std::os::raw::c_int;

/// Error type for HPTT operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Dimension mismatch between permutation and shape
    DimensionMismatch { perm_len: usize, shape_len: usize },
    /// Invalid permutation (not a valid permutation of 0..n-1)
    InvalidPermutation,
    /// Buffer size mismatch
    BufferSizeMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch {
                perm_len,
                shape_len,
            } => {
                write!(
                    f,
                    "Permutation length ({}) doesn't match shape length ({})",
                    perm_len, shape_len
                )
            }
            Error::InvalidPermutation => write!(f, "Invalid permutation"),
            Error::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Validate permutation and shape compatibility
fn validate_permutation(perm: &[usize], shape: &[usize]) -> Result<()> {
    if perm.len() != shape.len() {
        return Err(Error::DimensionMismatch {
            perm_len: perm.len(),
            shape_len: shape.len(),
        });
    }

    // Check that perm is a valid permutation of 0..n-1
    let mut seen = vec![false; perm.len()];
    for &p in perm {
        if p >= perm.len() {
            return Err(Error::InvalidPermutation);
        }
        if seen[p] {
            return Err(Error::InvalidPermutation);
        }
        seen[p] = true;
    }

    Ok(())
}

/// Calculate total number of elements from shape
fn total_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Transpose a double-precision (f64) tensor
///
/// Performs the operation: `B[perm] = alpha * A + beta * B[perm]`
///
/// # Arguments
///
/// * `perm` - Permutation of dimensions (e.g., `[1, 0, 2]` swaps first two dimensions)
/// * `alpha` - Scaling factor for input tensor
/// * `input` - Input tensor data in row-major or column-major order
/// * `shape` - Shape of input tensor
/// * `beta` - Scaling factor for output (0.0 to overwrite, 1.0 to accumulate)
/// * `output` - Output tensor buffer (must be pre-allocated)
/// * `num_threads` - Number of OpenMP threads (0 for default)
///
/// # Errors
///
/// Returns an error if:
/// - Permutation length doesn't match shape length
/// - Permutation is invalid
/// - Buffer sizes don't match tensor size
pub fn transpose_f64(
    perm: &[usize],
    alpha: f64,
    input: &[f64],
    shape: &[usize],
    beta: f64,
    output: &mut [f64],
    num_threads: usize,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape);
    if input.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: output.len(),
        });
    }

    // Convert to i32 for C API
    let perm_i32: Vec<c_int> = perm.iter().map(|&x| x as c_int).collect();
    let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
    let dim = shape.len() as c_int;
    let num_threads = num_threads as c_int;

    unsafe {
        ffi::dTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(), // outerSizeA
            beta,
            output.as_mut_ptr(),
            std::ptr::null(), // outerSizeB
            num_threads,
            1, // row-major (C order)
        );
    }

    Ok(())
}

/// Transpose a single-precision (f32) tensor
///
/// Same as `transpose_f64` but for single-precision floats.
pub fn transpose_f32(
    perm: &[usize],
    alpha: f32,
    input: &[f32],
    shape: &[usize],
    beta: f32,
    output: &mut [f32],
    num_threads: usize,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape);
    if input.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: output.len(),
        });
    }

    let perm_i32: Vec<c_int> = perm.iter().map(|&x| x as c_int).collect();
    let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
    let dim = shape.len() as c_int;
    let num_threads = num_threads as c_int;

    unsafe {
        ffi::sTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(),
            beta,
            output.as_mut_ptr(),
            std::ptr::null(),
            num_threads,
            1, // row-major (C order)
        );
    }

    Ok(())
}

/// Transpose a single-precision complex (Complex32) tensor
///
/// Same as `transpose_f32` but for single-precision complex numbers.
pub fn transpose_c32(
    perm: &[usize],
    alpha: Complex32,
    input: &[Complex32],
    shape: &[usize],
    beta: Complex32,
    output: &mut [Complex32],
    num_threads: usize,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape);
    if input.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: output.len(),
        });
    }

    let perm_i32: Vec<c_int> = perm.iter().map(|&x| x as c_int).collect();
    let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
    let dim = shape.len() as c_int;
    let num_threads = num_threads as c_int;

    unsafe {
        ffi::cTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            false, // conjA: no conjugation
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(),
            beta,
            output.as_mut_ptr(),
            std::ptr::null(),
            num_threads,
            1, // row-major (C order)
        );
    }

    Ok(())
}

/// Transpose a double-precision complex (Complex64) tensor
///
/// Same as `transpose_f64` but for double-precision complex numbers.
pub fn transpose_c64(
    perm: &[usize],
    alpha: Complex64,
    input: &[Complex64],
    shape: &[usize],
    beta: Complex64,
    output: &mut [Complex64],
    num_threads: usize,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape);
    if input.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: input.len(),
        });
    }
    if output.len() != total {
        return Err(Error::BufferSizeMismatch {
            expected: total,
            actual: output.len(),
        });
    }

    let perm_i32: Vec<c_int> = perm.iter().map(|&x| x as c_int).collect();
    let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
    let dim = shape.len() as c_int;
    let num_threads = num_threads as c_int;

    unsafe {
        ffi::zTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            false, // conjA: no conjugation
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(),
            beta,
            output.as_mut_ptr(),
            std::ptr::null(),
            num_threads,
            1, // row-major (C order)
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_permutation() {
        assert!(validate_permutation(&[0, 1, 2], &[2, 3, 4]).is_ok());
        assert!(validate_permutation(&[1, 0, 2], &[2, 3, 4]).is_ok());
        assert!(validate_permutation(&[2, 1, 0], &[2, 3, 4]).is_ok());

        // Dimension mismatch
        assert!(validate_permutation(&[0, 1], &[2, 3, 4]).is_err());

        // Invalid permutation (duplicate)
        assert!(validate_permutation(&[0, 0, 2], &[2, 3, 4]).is_err());

        // Invalid permutation (out of range)
        assert!(validate_permutation(&[0, 1, 3], &[2, 3, 4]).is_err());
    }

    #[test]
    fn test_total_elements() {
        assert_eq!(total_elements(&[2, 3, 4]), 24);
        assert_eq!(total_elements(&[10]), 10);
        assert_eq!(total_elements(&[]), 1);
    }
}
