use bytes::Bytes;
use bytora::{AssocInlineSize, InlineSize, Vector, VectorN};
use criterion::{
  Bencher, BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use std::{
  hint::black_box,
  mem::size_of,
  time::{Duration, Instant},
};

const SHORT: usize = 8;
const LARGE_PAYLOAD: usize = 64 * 1024;
const BATCH_SIZE: usize = 64;

const fn payload() -> [u8; LARGE_PAYLOAD] {
  let mut bytes = [0; LARGE_PAYLOAD];
  let mut index = 0;
  while index < bytes.len() {
    bytes[index] = (index % 251) as u8;
    index += 1;
  }
  let mut index = 0;
  while index < SHORT {
    bytes[LARGE_PAYLOAD - SHORT + index] = bytes[index];
    index += 1;
  }
  bytes
}

static PAYLOAD: [u8; LARGE_PAYLOAD] = payload();

fn bench_operation<T>(b: &mut Bencher<'_>, mut operation: impl FnMut() -> T) {
  // Pre-touch the bounded output buffer before timing, including its large-handle pages.
  let mut outputs = Vec::with_capacity(BATCH_SIZE);
  for _ in 0..BATCH_SIZE {
    outputs.push(operation());
  }
  black_box(&outputs);
  outputs.clear();

  b.iter_custom(|iterations| {
    let mut elapsed = Duration::ZERO;
    let mut remaining = iterations;
    while remaining != 0 {
      let count = remaining.min(BATCH_SIZE as u64) as usize;
      let start = Instant::now();
      for _ in 0..count {
        outputs.push(operation());
      }
      black_box(&outputs);
      elapsed += start.elapsed();
      outputs.clear();
      remaining -= count as u64;
    }
    elapsed
  });
}

fn bench_clone<T: Clone>(b: &mut Bencher<'_>, source: &T) {
  bench_operation(b, || T::clone(black_box(source)));
}

fn bench_clone_from<T: Clone>(b: &mut Bencher<'_>, source: &T, empty: impl Fn() -> T) {
  // The timed operation replaces an empty destination; reset and drop the
  // resulting clones after timing, as in the clone-and-push benchmark.
  let mut outputs: Vec<T> = (0..BATCH_SIZE).map(|_| empty()).collect();
  for slot in &mut outputs {
    slot.clone_from(black_box(source));
  }
  black_box(&outputs);
  for slot in &mut outputs {
    *slot = empty();
  }

  b.iter_custom(|iterations| {
    let mut elapsed = Duration::ZERO;
    let mut remaining = iterations;
    while remaining != 0 {
      let count = remaining.min(BATCH_SIZE as u64) as usize;
      let start = Instant::now();
      for slot in &mut outputs[..count] {
        slot.clone_from(black_box(source));
      }
      black_box(&outputs);
      elapsed += start.elapsed();
      for slot in &mut outputs[..count] {
        *slot = empty();
      }
      remaining -= count as u64;
    }
    elapsed
  });
}

#[derive(Clone, Copy)]
enum CloneMethod {
  Push,
  From,
}

fn bench_clone_case<T: Clone>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  id: String,
  source: &T,
  method: CloneMethod,
  empty: fn() -> T,
) {
  group.bench_function(id, |b| match method {
    CloneMethod::Push => bench_clone(b, source),
    CloneMethod::From => bench_clone_from(b, source, empty),
  });
}

fn clone_preflight<T: Clone>(source: &T, method: CloneMethod, empty: fn() -> T) -> T {
  match method {
    CloneMethod::Push => source.clone(),
    CloneMethod::From => {
      let mut cloned = empty();
      cloned.clone_from(source);
      cloned
    }
  }
}

fn empty_vector<S: InlineSize>() -> Vector<u8, S> {
  Vector::<u8, S>::try_from_slice(&[]).unwrap()
}

fn vector_bytes<const N: usize>(source: &VectorN<u8, N>) -> &[u8]
where
  [u8; N]: AssocInlineSize,
{
  source.as_slice()
}

fn bytes_bytes(source: &Bytes) -> &[u8] {
  source.as_ref()
}

#[allow(
  clippy::too_many_arguments,
  reason = "one registration per case, and every argument is a distinct fact about it"
)]
fn register_case<T: Clone>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  backing: &str,
  label: &str,
  source: &T,
  empty: fn() -> T,
  as_bytes: fn(&T) -> &[u8],
  contents: &[u8],
  external_pointer: Option<*const u8>,
) {
  assert_eq!(as_bytes(source), contents);
  if let Some(pointer) = external_pointer {
    assert_eq!(as_bytes(source).as_ptr(), pointer);
  }
  for (operation, method) in [
    ("clone", CloneMethod::Push),
    ("clone_from", CloneMethod::From),
  ] {
    let cloned = clone_preflight(source, method, empty);
    assert_eq!(as_bytes(&cloned), contents);
    if let Some(pointer) = external_pointer {
      assert_eq!(as_bytes(&cloned).as_ptr(), pointer);
    } else {
      assert_ne!(as_bytes(&cloned).as_ptr(), as_bytes(source).as_ptr());
    }
    drop(cloned);
    bench_clone_case(
      group,
      format!("{operation}/{backing}/{label}"),
      source,
      method,
      empty,
    );
  }
}

fn assert_layout<const N: usize>()
where
  [u8; N]: AssocInlineSize,
{
  let expected = if N == 0 { size_of::<Bytes>() } else { N };
  assert_eq!(size_of::<VectorN<u8, N>>(), expected);
  assert_eq!(size_of::<Option<VectorN<u8, N>>>(), expected);
}

fn budget<const N: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
where
  [u8; N]: AssocInlineSize,
{
  let label = format!("n_{N}_handle_{}", size_of::<VectorN<u8, N>>());
  let empty = empty_vector::<<[u8; N] as AssocInlineSize>::Size>;
  let contents = &PAYLOAD[..7];
  if let Some(source) = VectorN::<u8, N>::try_from_slice(contents) {
    register_case(
      group,
      "inline",
      &label,
      &source,
      empty,
      vector_bytes::<N>,
      contents,
      None,
    );
  } else {
    assert!(VectorN::<u8, N>::INLINE_CAPACITY < contents.len());
    eprintln!("storage: inline/{label} unsupported; skipped");
  }

  if let Some(source) = VectorN::<u8, N>::try_from_static(&PAYLOAD) {
    register_case(
      group,
      "static",
      &label,
      &source,
      empty,
      vector_bytes::<N>,
      &PAYLOAD,
      Some(PAYLOAD.as_ptr()),
    );
  } else {
    eprintln!("storage: static/{label} unsupported; skipped");
  }

  assert!(PAYLOAD.len() > VectorN::<u8, N>::INLINE_CAPACITY);
  let values = PAYLOAD.to_vec();
  let pointer = values.as_ptr();
  match VectorN::<u8, N>::try_from_vec(values) {
    Ok(owner) => {
      assert_eq!(owner.as_slice(), &PAYLOAD);
      assert_eq!(owner.as_slice().as_ptr(), pointer);
      let source = owner.clone();
      assert_eq!(source.as_slice(), &PAYLOAD);
      assert_eq!(source.as_slice().as_ptr(), pointer);
      register_case(
        group,
        "heap",
        &label,
        &source,
        empty,
        vector_bytes::<N>,
        &PAYLOAD,
        Some(pointer),
      );
      black_box(&owner);
    }
    Err(values) => {
      assert_eq!(values, &PAYLOAD);
      assert_eq!(values.as_ptr(), pointer);
      eprintln!("storage: heap/{label} unsupported; skipped");
    }
  }
}

fn bytes_baselines(group: &mut BenchmarkGroup<'_, WallTime>) {
  let source = Bytes::from_static(&PAYLOAD);
  register_case(
    group,
    "static",
    "bytes",
    &source,
    Bytes::new,
    bytes_bytes,
    &PAYLOAD,
    Some(PAYLOAD.as_ptr()),
  );

  let values = PAYLOAD.to_vec();
  let pointer = values.as_ptr();
  let owner = Bytes::from(values);
  assert_eq!(owner.as_ref(), &PAYLOAD);
  assert_eq!(owner.as_ptr(), pointer);
  let source = owner.clone();
  assert_eq!(source.as_ref(), &PAYLOAD);
  assert_eq!(source.as_ptr(), pointer);
  register_case(
    group,
    "heap",
    "bytes",
    &source,
    Bytes::new,
    bytes_bytes,
    &PAYLOAD,
    Some(pointer),
  );
  black_box(&owner);
}

fn benchmarks(c: &mut Criterion) {
  assert_eq!(size_of::<usize>(), 8);
  assert_eq!(size_of::<Bytes>(), 4 * size_of::<usize>());
  assert_eq!(size_of::<Option<Bytes>>(), size_of::<Bytes>());
  macro_rules! each_budget {
    ($action:ident $(, $arg:ident)?) => {
      $action!($($arg)?; 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128);
    };
  }
  macro_rules! check_sizes {
    (; $($n:literal),+ $(,)?) => { $(assert_layout::<$n>();)+ };
  }
  macro_rules! register_sizes {
    ($group:ident; $($n:literal),+ $(,)?) => { $(budget::<$n>(&mut $group);)+ };
  }
  each_budget!(check_sizes);
  let mut group = c.benchmark_group("storage");
  each_budget!(register_sizes, group);
  bytes_baselines(&mut group);
  group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
