use std::{env, fmt::Write, fs, path::PathBuf};

const MAX_HANDLE_BYTES: usize = 8192;

fn main() {
  println!("cargo:rerun-if-changed=build.rs");

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
    let align = if bytes == 4 { "u32" } else { "usize" };
    writeln!(sizes, "size!({bytes}, {cursor}, {align});").expect("writing to a String cannot fail");
  }

  let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
  fs::write(out_dir.join("sizes.rs"), sizes).expect("write generated sizes");
}
