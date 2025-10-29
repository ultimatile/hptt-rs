use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let hptt_src = manifest_dir.join("vendor/hptt");

    // Build HPTT using CMake
    let dst = cmake::Config::new(&hptt_src)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_CXX_VISIBILITY_PRESET", "hidden")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();

    // Link against the built library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=hptt");

    // Link against C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=dylib=c++");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Link against OpenMP (platform-dependent)
    // OpenMP is optional - HPTT can run single-threaded without it
    #[cfg(target_os = "macos")]
    {
        // Try to find libomp from Homebrew
        let omp_paths = vec![
            "/opt/homebrew/opt/libomp/lib",
            "/opt/homebrew/lib",
            "/usr/local/opt/libomp/lib",
            "/usr/local/lib",
        ];

        let mut found = false;
        for path in omp_paths {
            let lib_path = PathBuf::from(path).join("libomp.dylib");
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", path);
                println!("cargo:rustc-link-lib=dylib=omp");
                found = true;
                eprintln!("Found OpenMP at: {}", path);
                break;
            }
        }

        if !found {
            eprintln!("Warning: OpenMP not found, HPTT will run single-threaded");
        }
    }

    #[cfg(target_os = "linux")]
    {
        // On Linux, typically provided by GCC/Clang
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    // Rerun if vendor/hptt changes
    println!("cargo:rerun-if-changed=vendor/hptt");
}
