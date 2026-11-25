//! A deterministic driver for instruction and allocator counts. It is not a criterion bench.
//!
//! `opcount <case> <k>` runs one case `k` times in a loop and exits 0. Run under
//! `valgrind --tool=callgrind` with two values of `k`, the difference of the two counts divided by
//! the difference of the two `k` is the cost of one iteration with the process's fixed cost
//! cancelled. `opcount <case>@setup <k>` runs the same loop with only the setup, so that it can be
//! subtracted too. `opcount --list` prints every case name.
//!
//! A case is `<label>-<op>-<len>`, or `<label>-<op>-<len>c<capacity>` when the input vector has
//! spare capacity. The labels use the handle types that `budgets.rs` calls `n_0_handle_32`,
//! `n_16_handle_16`, `n_24_handle_24` and `n_32_handle_32`, and `bytes::Bytes`. The operation
//! bodies are `budgets.rs`'s own, at its native stride (no padding), so that a count corresponds to
//! a criterion row. The setup is per iteration, inside the loop:
//!
//! - `from_vec`: the setup builds the input `Vec`; the routine adopts or copies it.
//! - `try_from_vec`: as `from_vec`, through `TryFrom<Vec<u8>>` (`try_from_shared_vec`, then
//!   `adopt_vec`), the constructor that only `N = 0` has. `Bytes` has no fallible or shared-vector
//!   constructor, so its nearest equivalent is `Bytes::from(Vec<u8>)`, the same body as its
//!   `from_vec`; the two `bytes` rows are a control for each other.
//! - `consume`: the setup builds the handle from a `Vec`; the routine turns it back into a `Vec`.
//! - `warmed_clone`, `range`: nothing per iteration; the source handle, already cloned once so that
//!   its owner exists, is built before the loop.
//!
//! Every iteration also drops what the routine produced, which criterion keeps out of its timing
//! and a count cannot. A `@setup` iteration drops what the setup produced. The difference of the
//! two is therefore the routine plus the output's drop less the setup product's drop.
//!
//! Under `cargo bench -- --test`, or any first argument starting with `-` other than `--list`,
//! every case and every `@setup` case runs once as a smoke check.

#[cfg(smol_slice_atomic)]
mod driver {
  use bytes::Bytes;
  use smol_slice::{InlineSize, Slice, SliceN, typenum};
  use std::{
    hint::black_box,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    process::ExitCode,
  };

  const CONTENTS: [u8; 64] = {
    let mut values = [0; 64];
    let mut i = 0;
    while i < values.len() {
      values[i] = i as u8;
      i += 1;
    }
    values
  };

  const LABELS: [&str; 5] = ["bytes", "n_0", "n_16", "n_24", "n_32"];
  const OPS: [&str; 4] = ["from_vec", "consume", "warmed_clone", "range"];
  const LENS: [usize; 3] = [0, 7, 64];
  const USAGE: &str = "usage: opcount <case> <k> | opcount <case>@setup <k> | opcount --list";

  fn input(len: usize, capacity: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(capacity);
    values.extend_from_slice(&CONTENTS[..len]);
    values
  }

  trait Handle: Clone + Sized {
    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
    fn range(&self, range: Range<usize>) -> Self;
    fn into_vec(self) -> Vec<u8>;
  }

  impl<S: InlineSize> Handle for Slice<u8, S> {
    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Self::try_from_vec(values)
    }
    fn range(&self, range: Range<usize>) -> Self {
      self.slice(range)
    }
    fn into_vec(self) -> Vec<u8> {
      self.try_into_vec().ok().expect("unique slice")
    }
  }

  impl Handle for Bytes {
    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from(values))
    }
    fn range(&self, range: Range<usize>) -> Self {
      self.slice(range)
    }
    fn into_vec(self) -> Vec<u8> {
      Vec::from(self)
    }
  }

  /// The constructor of `N = 0`, which is `Bytes`' shape: `TryFrom<Vec<T>>`.
  trait Adopt: Handle {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
  }

  impl Adopt for Slice<u8, typenum::U0> {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      <Self as TryFrom<Vec<u8>>>::try_from(values)
    }
  }

  impl Adopt for Bytes {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from(values))
    }
  }

  struct Case {
    label: &'static str,
    op: &'static str,
    len: usize,
    cap: usize,
  }

  impl Case {
    fn name(&self) -> String {
      if self.cap == self.len {
        format!("{}-{}-{}", self.label, self.op, self.len)
      } else {
        format!("{}-{}-{}c{}", self.label, self.op, self.len, self.cap)
      }
    }
  }

  fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for label in LABELS {
      for op in OPS {
        for len in LENS {
          cases.push(Case {
            label,
            op,
            len,
            cap: len,
          });
        }
        if op == "from_vec" || op == "consume" {
          cases.push(Case {
            label,
            op,
            len: 64,
            cap: 128,
          });
        }
      }
      if label == "bytes" || label == "n_0" {
        for len in LENS {
          cases.push(Case {
            label,
            op: "try_from_vec",
            len,
            cap: len,
          });
        }
      }
    }
    cases
  }

  // Each loop is a function of its own, so that every instantiation has a symbol to find in the
  // disassembly.

  #[inline(never)]
  fn run_from_vec<T: Handle>(len: usize, cap: usize, setup_only: bool, k: u64) {
    for _ in 0..k {
      let values = input(len, cap);
      if setup_only {
        drop(black_box(values));
      } else {
        drop(black_box(
          T::from_vec(black_box(values)).expect("representable input"),
        ));
      }
    }
  }

  #[inline(never)]
  fn run_try_from_vec<T: Adopt>(len: usize, cap: usize, setup_only: bool, k: u64) {
    for _ in 0..k {
      let values = input(len, cap);
      if setup_only {
        drop(black_box(values));
      } else {
        drop(black_box(
          T::try_from_vec(black_box(values)).expect("representable input"),
        ));
      }
    }
  }

  #[inline(never)]
  fn run_consume<T: Handle>(len: usize, cap: usize, setup_only: bool, k: u64) {
    for _ in 0..k {
      let mut slot = black_box(Some(
        T::from_vec(input(len, cap)).expect("representable input"),
      ));
      if !setup_only {
        drop(black_box(T::into_vec(black_box(
          slot.take().expect("unique input"),
        ))));
      }
      drop(slot);
    }
  }

  #[inline(never)]
  fn run_warmed_clone<T: Handle>(len: usize, setup_only: bool, k: u64) {
    let source = T::from_vec(input(len, len)).expect("representable input");
    let peer = source.clone();
    for _ in 0..k {
      if setup_only {
        let _ = black_box(&source);
      } else {
        drop(black_box(T::clone(black_box(&source))));
      }
    }
    let _ = black_box(&peer);
  }

  #[inline(never)]
  fn run_range<T: Handle>(len: usize, setup_only: bool, k: u64) {
    let bounds = if len > 1 { 1..len - 1 } else { 0..len };
    let source = T::from_vec(input(len, len)).expect("representable input");
    let peer = source.clone();
    drop(T::range(&source, bounds.clone()));
    for _ in 0..k {
      if setup_only {
        let _ = black_box(&source);
      } else {
        drop(black_box(T::range(
          black_box(&source),
          black_box(bounds.start)..black_box(bounds.end),
        )));
      }
    }
    let _ = black_box(&peer);
  }

  /// Returns false when the handle cannot represent the input.
  fn run<T: Handle>(case: &Case, setup_only: bool, k: u64) -> bool {
    match T::from_vec(input(case.len, case.cap)) {
      Ok(handle) => drop(handle),
      Err(_) => return false,
    }
    match case.op {
      "from_vec" => run_from_vec::<T>(case.len, case.cap, setup_only, k),
      "consume" => run_consume::<T>(case.len, case.cap, setup_only, k),
      "warmed_clone" => run_warmed_clone::<T>(case.len, setup_only, k),
      "range" => run_range::<T>(case.len, setup_only, k),
      other => panic!("unknown operation {other}"),
    }
    true
  }

  fn adopt<T: Adopt>(case: &Case, setup_only: bool, k: u64) -> bool {
    match T::try_from_vec(input(case.len, case.cap)) {
      Ok(handle) => drop(handle),
      Err(_) => return false,
    }
    run_try_from_vec::<T>(case.len, case.cap, setup_only, k);
    true
  }

  fn run_case(case: &Case, setup_only: bool, k: u64) -> bool {
    if case.op == "try_from_vec" {
      return match case.label {
        "bytes" => adopt::<Bytes>(case, setup_only, k),
        "n_0" => adopt::<SliceN<u8, 0>>(case, setup_only, k),
        other => panic!("no try_from_vec case for {other}"),
      };
    }
    match case.label {
      "bytes" => run::<Bytes>(case, setup_only, k),
      "n_0" => run::<SliceN<u8, 0>>(case, setup_only, k),
      "n_16" => run::<SliceN<u8, 16>>(case, setup_only, k),
      "n_24" => run::<SliceN<u8, 24>>(case, setup_only, k),
      "n_32" => run::<SliceN<u8, 32>>(case, setup_only, k),
      other => panic!("unknown label {other}"),
    }
  }

  fn usage(problem: &str) -> ExitCode {
    eprintln!("opcount: {problem}\n{USAGE}\ncases (each also as <case>@setup):");
    for case in cases() {
      println!("{}", case.name());
    }
    ExitCode::from(2)
  }

  fn smoke() -> ExitCode {
    let mut failed = 0usize;
    for case in cases() {
      for setup_only in [false, true] {
        match catch_unwind(AssertUnwindSafe(|| run_case(&case, setup_only, 1))) {
          Ok(true) => {}
          Ok(false) => eprintln!("opcount: {} unrepresentable; skipped", case.name()),
          Err(_) => {
            failed += 1;
            eprintln!("opcount: {} failed", case.name());
          }
        }
      }
    }
    if failed == 0 {
      ExitCode::SUCCESS
    } else {
      ExitCode::FAILURE
    }
  }

  pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--list") {
      for case in cases() {
        println!("{}", case.name());
      }
      return ExitCode::SUCCESS;
    }
    if args.first().is_some_and(|arg| arg.starts_with('-')) {
      return smoke();
    }
    let [name, k] = args.as_slice() else {
      return usage("expected <case> <k>");
    };
    let Ok(k) = k.parse::<u64>() else {
      return usage("<k> is not a number");
    };
    let (name, setup_only) = match name.strip_suffix("@setup") {
      Some(base) => (base, true),
      None => (name.as_str(), false),
    };
    let cases = cases();
    let Some(case) = cases.iter().find(|case| case.name() == name) else {
      return usage(&format!("unknown case {name}"));
    };
    if run_case(case, setup_only, k) {
      ExitCode::SUCCESS
    } else {
      eprintln!("opcount: {name} is unrepresentable by this handle");
      ExitCode::from(3)
    }
  }
}

#[cfg(smol_slice_atomic)]
fn main() -> std::process::ExitCode {
  driver::main()
}

// Without shared ownership there is nothing to count; `cargo bench -- --test` still passes.
#[cfg(not(smol_slice_atomic))]
fn main() -> std::process::ExitCode {
  let smoke = std::env::args()
    .nth(1)
    .is_some_and(|arg| arg.starts_with('-'));
  eprintln!("opcount: shared ownership (smol_slice_atomic) is unavailable on this build");
  if smoke {
    std::process::ExitCode::SUCCESS
  } else {
    std::process::ExitCode::from(2)
  }
}
