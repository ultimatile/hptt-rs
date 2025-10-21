//! Low-level FFI bindings to HPTT C API

use std::os::raw::{c_double, c_float, c_int};

#[link(name = "hptt", kind = "static")]
unsafe extern "C" {
    /// Single-precision tensor transpose
    ///
    /// # Arguments
    /// * `perm` - Permutation array
    /// * `dim` - Number of dimensions
    /// * `alpha` - Scaling factor for input
    /// * `A` - Input tensor data
    /// * `sizeA` - Size of each dimension of A
    /// * `outerSizeA` - Outer size (for sub-tensors, NULL for full tensor)
    /// * `beta` - Scaling factor for output (0.0 for overwrite, 1.0 for accumulate)
    /// * `B` - Output tensor data
    /// * `outerSizeB` - Outer size of B (NULL for full tensor)
    /// * `numThreads` - Number of OpenMP threads
    /// * `useRowMajor` - 0 for column-major, 1 for row-major
    pub fn sTensorTranspose(
        perm: *const c_int,
        dim: c_int,
        alpha: c_float,
        A: *const c_float,
        sizeA: *const c_int,
        outerSizeA: *const c_int,
        beta: c_float,
        B: *mut c_float,
        outerSizeB: *const c_int,
        numThreads: c_int,
        useRowMajor: c_int,
    );

    /// Double-precision tensor transpose
    ///
    /// Same arguments as sTensorTranspose but with double precision
    pub fn dTensorTranspose(
        perm: *const c_int,
        dim: c_int,
        alpha: c_double,
        A: *const c_double,
        sizeA: *const c_int,
        outerSizeA: *const c_int,
        beta: c_double,
        B: *mut c_double,
        outerSizeB: *const c_int,
        numThreads: c_int,
        useRowMajor: c_int,
    );

    // Complex versions are available but not exposed initially
    // pub fn cTensorTranspose(...);
    // pub fn zTensorTranspose(...);
}
