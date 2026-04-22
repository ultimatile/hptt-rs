# hptt-rs

Rust bindings for HPTT (High-Performance Tensor Transposition).

## About

This crate provides Rust bindings to the HPTT C++ library for out-of-place tensor transpositions:

```
B_π(i₁i₂...iₙ) ← α × A_i₁i₂...iₙ + β × B_π(i₁i₂...iₙ)
```

Where `π` is a user-specified permutation, and `α`, `β` are scalar coefficients.

**Note**: This crate uses a fork of the upstream HPTT library ([springer13/hptt](https://github.com/springer13/hptt)) maintained at [ultimatile/hptt](https://github.com/ultimatile/hptt) with fixes for modern C++ compilers.

## Installation

```bash
cargo add hptt
```

Or add this to your `Cargo.toml`:

```toml
[dependencies]
hptt = "0.4"
```

## Usage

```rust
use hptt::{transpose_f64, MemoryOrder};

// Transpose a 3D tensor from shape [2, 3, 4] to [3, 2, 4]
// Permutation [1, 0, 2] swaps the first two dimensions

let input = vec![1.0f64; 2 * 3 * 4];
let mut output = vec![0.0f64; 3 * 2 * 4];

transpose_f64(
    &[1, 0, 2],           // permutation
    1.0,                  // alpha
    &input,
    &[2, 3, 4],           // shape
    0.0,                  // beta (0.0 = overwrite, 1.0 = accumulate)
    &mut output,
    1,                    // num_threads
    MemoryOrder::RowMajor,
)?;
```

## API

### Types

- `MemoryOrder` - Memory layout order (`RowMajor` or `ColumnMajor`)

### Dense Transpose

- `transpose_f64` - Double-precision (f64) tensor transpose
- `transpose_f32` - Single-precision (f32) tensor transpose
- `transpose_c64` - Double-precision complex (Complex64) tensor transpose
- `transpose_c32` - Single-precision complex (Complex32) tensor transpose

Complex variants accept a `conj: bool` parameter for conjugate transpose (dagger operation).

### Sub-tensor Transpose

For transposing sub-tensors within larger allocations without copying:

- `transpose_f64_sub`, `transpose_f32_sub`, `transpose_c64_sub`, `transpose_c32_sub`

These accept additional `outer_size_a` and `outer_size_b` parameters specifying the allocated (padded) dimensions of the input and output buffers.

### Error Handling

Returns `Result<(), hptt::Error>` with the following error types:

- `DimensionMismatch` - Permutation length doesn't match shape length
- `InvalidPermutation` - Invalid permutation (not a valid permutation of 0..n-1)
- `BufferSizeMismatch` - Buffer size doesn't match tensor size
- `ValueOutOfRange` - Value does not fit in C API integer range
- `ElementCountOverflow` - Tensor element count overflowed
- `OuterSizeLengthMismatch` - Outer size length doesn't match shape length
- `OuterSizeTooSmall` - Outer size is smaller than shape size for a dimension
- `NumThreadsZero` - `num_threads` was 0 (must be >= 1)

The `Error` enum is marked `#[non_exhaustive]`; downstream matches must include a catch-all arm.

## Building from Source

### Requirements

- Rust toolchain (edition 2024)
- CMake 3.7+
- C++ compiler with C++11 support
- OpenMP (optional, for multi-threading)

### Build

```bash
git clone --recursive https://github.com/ultimatile/hptt-rs
cd hptt-rs
cargo build --release
```

The build script will automatically compile the vendored HPTT library using CMake.

## License

- Rust bindings: MIT OR Apache-2.0
- HPTT library: BSD 3-Clause (see `vendor/hptt/LICENSE.txt`)

## References

- Original HPTT: <https://github.com/springer13/hptt>
- Forked HPTT (used by this crate): <https://github.com/ultimatile/hptt>
- HPTT Paper: ["HPTT: A High-Performance Tensor Transposition C++ Library"](https://arxiv.org/abs/1704.04374)
