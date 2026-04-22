//! Rust bindings for HPTT (High-Performance Tensor Transposition)
//!
//! This crate provides safe Rust bindings to the HPTT C++ library for
//! high-performance out-of-place tensor transpositions.
//!
//! # Example
//!
//! ```no_run
//! use hptt::{transpose_f64, MemoryOrder};
//!
//! // Transpose a 3D tensor [2, 3, 4] -> [3, 2, 4]
//! let input = vec![1.0; 2 * 3 * 4];
//! let mut output = vec![0.0; 3 * 2 * 4];
//!
//! transpose_f64(
//!     &[1, 0, 2],           // permutation: swap first two dims
//!     1.0,                  // alpha
//!     &input,
//!     &[2, 3, 4],           // shape
//!     0.0,                  // beta (overwrite)
//!     &mut output,
//!     1,                    // single thread
//!     MemoryOrder::RowMajor,
//! ).expect("transpose failed");
//! ```

mod ffi;

pub use num_complex::{Complex32, Complex64};
use std::os::raw::c_int;

/// Memory layout order for tensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// Row-major (C order): rightmost index is stride-1
    RowMajor,
    /// Column-major (Fortran order): leftmost index is stride-1
    ColumnMajor,
}

impl MemoryOrder {
    fn to_hptt_flag(self) -> c_int {
        match self {
            MemoryOrder::RowMajor => 1,
            MemoryOrder::ColumnMajor => 0,
        }
    }
}

/// Error type for HPTT operations
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Dimension mismatch between permutation and shape
    DimensionMismatch { perm_len: usize, shape_len: usize },
    /// Invalid permutation (not a valid permutation of 0..n-1)
    InvalidPermutation,
    /// Buffer size mismatch
    BufferSizeMismatch { expected: usize, actual: usize },
    /// Value does not fit in C API integer range (c_int / i32)
    ValueOutOfRange { field: &'static str, value: usize },
    /// Tensor element count overflowed usize while multiplying shape dimensions
    ElementCountOverflow,
    /// Outer size length doesn't match shape length
    OuterSizeLengthMismatch { expected: usize, actual: usize },
    /// Outer size is smaller than shape size for a dimension
    OuterSizeTooSmall {
        dim: usize,
        outer_size: usize,
        shape_size: usize,
    },
    /// num_threads must be >= 1; 0 is not accepted
    NumThreadsZero,
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
            Error::ValueOutOfRange { field, value } => {
                write!(f, "Value out of range for {}: {}", field, value)
            }
            Error::ElementCountOverflow => write!(f, "Tensor element count overflow"),
            Error::OuterSizeLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Outer size length ({}) doesn't match shape length ({})",
                    actual, expected
                )
            }
            Error::OuterSizeTooSmall {
                dim,
                outer_size,
                shape_size,
            } => {
                write!(
                    f,
                    "Outer size ({}) is smaller than shape size ({}) for dimension {}",
                    outer_size, shape_size, dim
                )
            }
            Error::NumThreadsZero => write!(f, "num_threads must be >= 1"),
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
fn total_elements(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(Error::ElementCountOverflow)
}

/// Validate outer_size against shape: each outer_size[i] >= required_size[i]
fn validate_outer_size(
    outer_size: &[usize],
    required_size: &[usize],
    ndim: usize,
) -> Result<()> {
    if outer_size.len() != ndim {
        return Err(Error::OuterSizeLengthMismatch {
            expected: ndim,
            actual: outer_size.len(),
        });
    }
    for i in 0..ndim {
        if outer_size[i] < required_size[i] {
            return Err(Error::OuterSizeTooSmall {
                dim: i,
                outer_size: outer_size[i],
                shape_size: required_size[i],
            });
        }
    }
    Ok(())
}

fn usize_to_c_int(value: usize, field: &'static str) -> Result<c_int> {
    i32::try_from(value)
        .map(|v| v as c_int)
        .map_err(|_| Error::ValueOutOfRange { field, value })
}

fn usize_slice_to_c_int_vec(values: &[usize], field: &'static str) -> Result<Vec<c_int>> {
    values
        .iter()
        .copied()
        .map(|v| usize_to_c_int(v, field))
        .collect()
}

// HPTT's C++ core does not accept numThreads == 0: it triggers an
// "Internal error: primefactorization for 0" and calls exit(-1) inside
// createPlan. Reject 0 at the Rust boundary with a recoverable error.
fn validate_num_threads(num_threads: usize) -> Result<c_int> {
    if num_threads == 0 {
        return Err(Error::NumThreadsZero);
    }
    usize_to_c_int(num_threads, "num_threads")
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
/// * `num_threads` - Number of OpenMP threads (must be >= 1; 0 is rejected)
/// * `order` - Memory layout order (row-major or column-major)
///
/// # Errors
///
/// Returns an error if:
/// - Permutation length doesn't match shape length
/// - Permutation is invalid
/// - Buffer sizes don't match tensor size
/// - `num_threads` is 0
pub fn transpose_f64(
    perm: &[usize],
    alpha: f64,
    input: &[f64],
    shape: &[usize],
    beta: f64,
    output: &mut [f64],
    num_threads: usize,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape)?;
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
    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let dim = usize_to_c_int(shape.len(), "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

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
            order.to_hptt_flag(),
        );
    }

    Ok(())
}

/// Transpose a sub-tensor of a double-precision (f64) tensor
///
/// Same as `transpose_f64` but operates on a sub-tensor within a larger allocation.
/// `outer_size_a` and `outer_size_b` specify the allocated (padded) dimensions of the
/// input and output buffers respectively. Each `outer_size[i]` must be >= the
/// corresponding shape dimension.
pub fn transpose_f64_sub(
    perm: &[usize],
    alpha: f64,
    input: &[f64],
    shape: &[usize],
    outer_size_a: &[usize],
    beta: f64,
    output: &mut [f64],
    outer_size_b: &[usize],
    num_threads: usize,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;
    let ndim = shape.len();
    validate_outer_size(outer_size_a, shape, ndim)?;
    let permuted_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    validate_outer_size(outer_size_b, &permuted_shape, ndim)?;

    let total_a = total_elements(outer_size_a)?;
    if input.len() < total_a {
        return Err(Error::BufferSizeMismatch {
            expected: total_a,
            actual: input.len(),
        });
    }
    let total_b = total_elements(outer_size_b)?;
    if output.len() < total_b {
        return Err(Error::BufferSizeMismatch {
            expected: total_b,
            actual: output.len(),
        });
    }

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let outer_a_i32 = usize_slice_to_c_int_vec(outer_size_a, "outer_size_a element")?;
    let outer_b_i32 = usize_slice_to_c_int_vec(outer_size_b, "outer_size_b element")?;
    let dim = usize_to_c_int(ndim, "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::dTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            input.as_ptr(),
            shape_i32.as_ptr(),
            outer_a_i32.as_ptr(),
            beta,
            output.as_mut_ptr(),
            outer_b_i32.as_ptr(),
            num_threads,
            order.to_hptt_flag(),
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
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape)?;
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

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let dim = usize_to_c_int(shape.len(), "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

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
            order.to_hptt_flag(),
        );
    }

    Ok(())
}

/// Transpose a sub-tensor of a single-precision (f32) tensor
///
/// Same as `transpose_f32` but operates on a sub-tensor within a larger allocation.
pub fn transpose_f32_sub(
    perm: &[usize],
    alpha: f32,
    input: &[f32],
    shape: &[usize],
    outer_size_a: &[usize],
    beta: f32,
    output: &mut [f32],
    outer_size_b: &[usize],
    num_threads: usize,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;
    let ndim = shape.len();
    validate_outer_size(outer_size_a, shape, ndim)?;
    let permuted_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    validate_outer_size(outer_size_b, &permuted_shape, ndim)?;

    let total_a = total_elements(outer_size_a)?;
    if input.len() < total_a {
        return Err(Error::BufferSizeMismatch {
            expected: total_a,
            actual: input.len(),
        });
    }
    let total_b = total_elements(outer_size_b)?;
    if output.len() < total_b {
        return Err(Error::BufferSizeMismatch {
            expected: total_b,
            actual: output.len(),
        });
    }

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let outer_a_i32 = usize_slice_to_c_int_vec(outer_size_a, "outer_size_a element")?;
    let outer_b_i32 = usize_slice_to_c_int_vec(outer_size_b, "outer_size_b element")?;
    let dim = usize_to_c_int(ndim, "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::sTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            input.as_ptr(),
            shape_i32.as_ptr(),
            outer_a_i32.as_ptr(),
            beta,
            output.as_mut_ptr(),
            outer_b_i32.as_ptr(),
            num_threads,
            order.to_hptt_flag(),
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
    conj: bool,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape)?;
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

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let dim = usize_to_c_int(shape.len(), "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::cTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            conj,
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(),
            beta,
            output.as_mut_ptr(),
            std::ptr::null(),
            num_threads,
            order.to_hptt_flag(),
        );
    }

    Ok(())
}

/// Transpose a sub-tensor of a single-precision complex (Complex32) tensor
///
/// Same as `transpose_c32` but operates on a sub-tensor within a larger allocation.
pub fn transpose_c32_sub(
    perm: &[usize],
    alpha: Complex32,
    input: &[Complex32],
    shape: &[usize],
    outer_size_a: &[usize],
    beta: Complex32,
    output: &mut [Complex32],
    outer_size_b: &[usize],
    num_threads: usize,
    conj: bool,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;
    let ndim = shape.len();
    validate_outer_size(outer_size_a, shape, ndim)?;
    let permuted_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    validate_outer_size(outer_size_b, &permuted_shape, ndim)?;

    let total_a = total_elements(outer_size_a)?;
    if input.len() < total_a {
        return Err(Error::BufferSizeMismatch {
            expected: total_a,
            actual: input.len(),
        });
    }
    let total_b = total_elements(outer_size_b)?;
    if output.len() < total_b {
        return Err(Error::BufferSizeMismatch {
            expected: total_b,
            actual: output.len(),
        });
    }

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let outer_a_i32 = usize_slice_to_c_int_vec(outer_size_a, "outer_size_a element")?;
    let outer_b_i32 = usize_slice_to_c_int_vec(outer_size_b, "outer_size_b element")?;
    let dim = usize_to_c_int(ndim, "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::cTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            conj,
            input.as_ptr(),
            shape_i32.as_ptr(),
            outer_a_i32.as_ptr(),
            beta,
            output.as_mut_ptr(),
            outer_b_i32.as_ptr(),
            num_threads,
            order.to_hptt_flag(),
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
    conj: bool,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;

    let total = total_elements(shape)?;
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

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let dim = usize_to_c_int(shape.len(), "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::zTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            conj,
            input.as_ptr(),
            shape_i32.as_ptr(),
            std::ptr::null(),
            beta,
            output.as_mut_ptr(),
            std::ptr::null(),
            num_threads,
            order.to_hptt_flag(),
        );
    }

    Ok(())
}

/// Transpose a sub-tensor of a double-precision complex (Complex64) tensor
///
/// Same as `transpose_c64` but operates on a sub-tensor within a larger allocation.
pub fn transpose_c64_sub(
    perm: &[usize],
    alpha: Complex64,
    input: &[Complex64],
    shape: &[usize],
    outer_size_a: &[usize],
    beta: Complex64,
    output: &mut [Complex64],
    outer_size_b: &[usize],
    num_threads: usize,
    conj: bool,
    order: MemoryOrder,
) -> Result<()> {
    validate_permutation(perm, shape)?;
    let ndim = shape.len();
    validate_outer_size(outer_size_a, shape, ndim)?;
    let permuted_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    validate_outer_size(outer_size_b, &permuted_shape, ndim)?;

    let total_a = total_elements(outer_size_a)?;
    if input.len() < total_a {
        return Err(Error::BufferSizeMismatch {
            expected: total_a,
            actual: input.len(),
        });
    }
    let total_b = total_elements(outer_size_b)?;
    if output.len() < total_b {
        return Err(Error::BufferSizeMismatch {
            expected: total_b,
            actual: output.len(),
        });
    }

    let perm_i32 = usize_slice_to_c_int_vec(perm, "perm element")?;
    let shape_i32 = usize_slice_to_c_int_vec(shape, "shape element")?;
    let outer_a_i32 = usize_slice_to_c_int_vec(outer_size_a, "outer_size_a element")?;
    let outer_b_i32 = usize_slice_to_c_int_vec(outer_size_b, "outer_size_b element")?;
    let dim = usize_to_c_int(ndim, "shape length")?;
    let num_threads = validate_num_threads(num_threads)?;

    unsafe {
        ffi::zTensorTranspose(
            perm_i32.as_ptr(),
            dim,
            alpha,
            conj,
            input.as_ptr(),
            shape_i32.as_ptr(),
            outer_a_i32.as_ptr(),
            beta,
            output.as_mut_ptr(),
            outer_b_i32.as_ptr(),
            num_threads,
            order.to_hptt_flag(),
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
        assert_eq!(total_elements(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(total_elements(&[10]).unwrap(), 10);
        assert_eq!(total_elements(&[]).unwrap(), 1);
    }

    #[test]
    fn test_total_elements_overflow() {
        let shape = [usize::MAX, 2];
        assert!(matches!(
            total_elements(&shape),
            Err(Error::ElementCountOverflow)
        ));
    }

    #[test]
    fn test_usize_to_c_int_range_check() {
        let too_large = i32::MAX as usize + 1;
        assert!(matches!(
            usize_to_c_int(too_large, "num_threads"),
            Err(Error::ValueOutOfRange {
                field: "num_threads",
                value
            }) if value == too_large
        ));
    }

    #[test]
    fn test_num_threads_zero_rejected() {
        let input = vec![0.0; 6];
        let mut output = vec![0.0; 6];
        let result = transpose_f64(
            &[1, 0],
            1.0,
            &input,
            &[2, 3],
            0.0,
            &mut output,
            0,
            MemoryOrder::RowMajor,
        );
        assert!(matches!(result, Err(Error::NumThreadsZero)));
    }

    // Contract: README's install snippet must advertise the same major.minor
    // as the crate version. Guards against silent drift between Cargo.toml
    // and README when we cut a release.
    #[test]
    fn readme_install_version_matches_crate_version() {
        let readme = include_str!("../README.md");
        let version = env!("CARGO_PKG_VERSION");
        let major_minor = version
            .rsplit_once('.')
            .expect("CARGO_PKG_VERSION missing '.'")
            .0;
        let expected = format!("hptt = \"{}\"", major_minor);
        assert!(
            readme.contains(&expected),
            "README install snippet should contain `{}` (crate version is {}), \
             but it does not. Update README.md to match Cargo.toml.",
            expected,
            version
        );
    }

    #[test]
    fn test_transpose_f64_2d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; input.len()];

        transpose_f64(&[1, 0], 1.0, &input, &[2, 3], 0.0, &mut output, 1, MemoryOrder::RowMajor)
            .unwrap();

        assert_eq!(output, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_c32_2d() {
        let input = vec![
            Complex32::new(1.0, 10.0),
            Complex32::new(2.0, 20.0),
            Complex32::new(3.0, 30.0),
            Complex32::new(4.0, 40.0),
        ];
        let mut output = vec![Complex32::new(0.0, 0.0); input.len()];

        transpose_c32(
            &[1, 0],
            Complex32::new(1.0, 0.0),
            &input,
            &[2, 2],
            Complex32::new(0.0, 0.0),
            &mut output,
            1,
            false,
            MemoryOrder::RowMajor,
        )
        .unwrap();

        assert_eq!(
            output,
            vec![
                Complex32::new(1.0, 10.0),
                Complex32::new(3.0, 30.0),
                Complex32::new(2.0, 20.0),
                Complex32::new(4.0, 40.0),
            ]
        );
    }

    #[test]
    fn test_transpose_c32_2d_conj() {
        let input = vec![
            Complex32::new(1.0, 10.0),
            Complex32::new(2.0, 20.0),
            Complex32::new(3.0, 30.0),
            Complex32::new(4.0, 40.0),
        ];
        let mut output = vec![Complex32::new(0.0, 0.0); input.len()];

        transpose_c32(
            &[1, 0],
            Complex32::new(1.0, 0.0),
            &input,
            &[2, 2],
            Complex32::new(0.0, 0.0),
            &mut output,
            1,
            true,
            MemoryOrder::RowMajor,
        )
        .unwrap();

        // Conjugate transpose: transpose + negate imaginary parts
        assert_eq!(
            output,
            vec![
                Complex32::new(1.0, -10.0),
                Complex32::new(3.0, -30.0),
                Complex32::new(2.0, -20.0),
                Complex32::new(4.0, -40.0),
            ]
        );
    }

    #[test]
    fn test_transpose_f64_sub_2d() {
        // 2x3 sub-tensor within a 4x5 allocation (row-major)
        // Logical data:
        //   [[1, 2, 3],
        //    [4, 5, 6]]
        // Stored in 4x5 buffer with padding
        let mut input = vec![0.0; 4 * 5];
        // Row 0: indices 0..3
        input[0] = 1.0;
        input[1] = 2.0;
        input[2] = 3.0;
        // Row 1: indices 5..8 (stride 5)
        input[5] = 4.0;
        input[6] = 5.0;
        input[7] = 6.0;

        // Output: 3x2 sub-tensor within a 5x4 allocation
        let mut output = vec![0.0; 5 * 4];

        transpose_f64_sub(
            &[1, 0],
            1.0,
            &input,
            &[2, 3],
            &[4, 5],
            0.0,
            &mut output,
            &[5, 4],
            1,
            MemoryOrder::RowMajor,
        )
        .unwrap();

        // Expected transposed 3x2 in 5x4 buffer (stride 4):
        //   [[1, 4],
        //    [2, 5],
        //    [3, 6]]
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 4.0);
        assert_eq!(output[4], 2.0);
        assert_eq!(output[5], 5.0);
        assert_eq!(output[8], 3.0);
        assert_eq!(output[9], 6.0);
    }

    #[test]
    fn test_transpose_sub_outer_size_too_small() {
        let input = vec![0.0; 20];
        let mut output = vec![0.0; 20];

        let result = transpose_f64_sub(
            &[1, 0],
            1.0,
            &input,
            &[2, 3],
            &[1, 5], // outer_size_a[0]=1 < shape[0]=2
            0.0,
            &mut output,
            &[5, 4],
            1,
            MemoryOrder::RowMajor,
        );

        assert!(matches!(
            result,
            Err(Error::OuterSizeTooSmall {
                dim: 0,
                outer_size: 1,
                shape_size: 2,
            })
        ));
    }

    #[test]
    fn test_transpose_sub_outer_size_length_mismatch() {
        let input = vec![0.0; 20];
        let mut output = vec![0.0; 20];

        let result = transpose_f64_sub(
            &[1, 0],
            1.0,
            &input,
            &[2, 3],
            &[4, 5, 6], // length 3 != ndim 2
            0.0,
            &mut output,
            &[5, 4],
            1,
            MemoryOrder::RowMajor,
        );

        assert!(matches!(
            result,
            Err(Error::OuterSizeLengthMismatch {
                expected: 2,
                actual: 3,
            })
        ));
    }
}
