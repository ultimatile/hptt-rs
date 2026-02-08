use std::env;
#[cfg(target_os = "linux")]
use std::fs;
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
        link_openmp_linux();
    }

    // Rerun if vendor/hptt changes
    println!("cargo:rerun-if-changed=vendor/hptt");
}

#[cfg(target_os = "linux")]
fn has_library_in(path: &str, prefixes: &[&str]) -> bool {
    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };

    entries
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .any(|name| prefixes.iter().any(|prefix| name.starts_with(prefix)))
}

#[cfg(target_os = "linux")]
fn link_openmp_linux() {
    let search_paths = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/lib",
        "/lib64",
    ];

    let cxx = env::var("CXX").unwrap_or_default();
    let prefer_omp = cxx.contains("clang");

    // If the compiler is clang, prefer libomp. Otherwise prefer libgomp.
    let preference = if prefer_omp {
        [("omp", ["libomp.so"]), ("gomp", ["libgomp.so"])]
    } else {
        [("gomp", ["libgomp.so"]), ("omp", ["libomp.so"])]
    };

    for (lib, prefixes) in preference {
        if let Some(path) = search_paths
            .iter()
            .find(|path| has_library_in(path, &prefixes))
        {
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-lib=dylib={}", lib);
            eprintln!("Found OpenMP runtime ({}): {}", lib, path);
            return;
        }
    }

    // As a fallback, try GCC's runtime because it is common on Linux.
    println!("cargo:rustc-link-lib=dylib=gomp");
    eprintln!("Warning: OpenMP runtime not found in common paths, falling back to libgomp");
}
