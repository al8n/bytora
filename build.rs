use std::{env, fmt::Write, fs, path::PathBuf};

const MAX_HANDLE_BYTES: usize = 8192;

// Typenum names every integer through U1024. Append low bits to those names
// for larger sizes, including the ones absent from its const-generic mapping.
fn typenum(bytes: usize) -> String {
  if bytes <= 1024 {
    format!("typenum::U{bytes}")
  } else {
    format!(
      "typenum::UInt<{}, typenum::B{}>",
      typenum(bytes / 2),
      bytes % 2
    )
  }
}

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rustc-check-cfg=cfg(paguro_atomic)");
  let alloc =
    env::var_os("CARGO_FEATURE_STD").is_some() || env::var_os("CARGO_FEATURE_ALLOC").is_some();
  let native_cas = env::var("CARGO_CFG_TARGET_HAS_ATOMIC")
    .unwrap_or_default()
    .split(',')
    .any(|width| width == "ptr");
  if alloc && (native_cas || env::var_os("CARGO_FEATURE_EXTRA_PLATFORMS").is_some()) {
    println!("cargo:rustc-cfg=paguro_atomic");
  }

  let word = match env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() {
    Ok("32") => 4,
    Ok("64") => 8,
    other => panic!("unsupported target pointer width: {other:?}"),
  };
  let mut sizes = format!(
    "/// The largest supported total handle size, in bytes.\n\
     pub const MAX_HANDLE_BYTES: usize = {MAX_HANDLE_BYTES};\n"
  );
  for bytes in (word..=MAX_HANDLE_BYTES).step_by(word) {
    let cursor = if bytes <= 8 {
      0
    } else if bytes <= 128 {
      1
    } else {
      3
    };
    writeln!(sizes, "size!({bytes}, {}, {cursor});", typenum(bytes))
      .expect("writing to a String cannot fail");
  }

  let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
  fs::write(out_dir.join("sizes.rs"), sizes).expect("write generated sizes");
}
