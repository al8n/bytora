use bytes::{Buf, Bytes};
use criterion::{
  Bencher, BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use smol_slice::{InlineSize, Slice, SliceN};
use std::{
  hint::black_box,
  mem::size_of,
  time::{Duration, Instant},
};

const SHORT: usize = 8;
const MEDIUM: usize = 64;
const LARGE_HANDLE: usize = 4096;
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

fn bench_clone<T: Clone>(b: &mut Bencher<'_>, source: &T) {
  // Pre-touch the bounded output buffer before timing, including its large-handle pages.
  let mut outputs = Vec::with_capacity(BATCH_SIZE);
  for _ in 0..BATCH_SIZE {
    outputs.push(T::clone(source));
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
        outputs.push(T::clone(black_box(source)));
      }
      black_box(&outputs);
      elapsed += start.elapsed();
      outputs.clear();
      remaining -= count as u64;
    }
    elapsed
  });
}

fn inline<S: InlineSize>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  make: fn(&[u8]) -> Option<Slice<u8, S>>,
) {
  for len in [SHORT, Slice::<u8, S>::INLINE_CAPACITY] {
    let source = make(&PAYLOAD[..len]).unwrap();
    assert_eq!(source.as_slice(), &PAYLOAD[..len]);
    let id = format!(
      "slice_inline/handle_{}/len_{len}",
      size_of::<Slice<u8, S>>()
    );
    group.bench_function(id, |b| bench_clone(b, &source));
  }
}

fn advanced_input(total: usize, remaining: usize) -> Vec<u8> {
  let mut input = vec![0; total];
  input[total - remaining..].copy_from_slice(&PAYLOAD[..remaining]);
  input
}

fn inline_advanced(group: &mut BenchmarkGroup<'_, WallTime>) {
  let capacity = SliceN::<u8, LARGE_HANDLE>::INLINE_CAPACITY;
  let input = advanced_input(capacity, SHORT);
  let mut source = SliceN::<u8, LARGE_HANDLE>::try_from_slice(&input).unwrap();
  source.advance(capacity - SHORT);
  assert_eq!(source.as_slice(), &PAYLOAD[..SHORT]);
  let id = format!("slice_inline_advanced/handle_{LARGE_HANDLE}/from_{capacity}/len_{SHORT}");
  group.bench_function(id, |b| bench_clone(b, &source));
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
fn slice_heap(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Slice<u8>>();
  for len in [MEDIUM, LARGE_PAYLOAD] {
    let owner = Slice::<u8>::try_from_vec(PAYLOAD[..len].to_vec()).unwrap();
    let shared = owner.clone();
    assert_eq!(shared.as_slice(), &PAYLOAD[..len]);
    let id = format!("slice_heap_shared/handle_{handle_size}/len_{len}");
    group.bench_function(id, |b| bench_clone(b, &shared));
    black_box(&owner);
  }

  for len in [SHORT, Slice::<u8>::INLINE_CAPACITY, MEDIUM] {
    let owner = Slice::<u8>::try_from_vec(advanced_input(LARGE_HANDLE, len)).unwrap();
    let mut shared = owner.clone();
    shared.advance(LARGE_HANDLE - len);
    assert_eq!(shared.as_slice(), &PAYLOAD[..len]);
    let id =
      format!("slice_heap_shared_advanced/handle_{handle_size}/from_{LARGE_HANDLE}/len_{len}");
    group.bench_function(id, |b| bench_clone(b, &shared));
    black_box(&owner);
  }

  let owner = SliceN::<u8, LARGE_HANDLE>::try_from_vec(PAYLOAD.to_vec()).unwrap();
  let mut shared = owner.clone();
  assert_eq!(shared.as_slice(), &PAYLOAD[..]);
  let id = format!("slice_heap_shared/handle_{LARGE_HANDLE}/len_{LARGE_PAYLOAD}");
  group.bench_function(id, |b| bench_clone(b, &shared));

  shared.advance(LARGE_PAYLOAD - SHORT);
  assert_eq!(shared.as_slice(), &PAYLOAD[..SHORT]);
  let id =
    format!("slice_heap_shared_advanced/handle_{LARGE_HANDLE}/from_{LARGE_PAYLOAD}/len_{SHORT}");
  group.bench_function(id, |b| bench_clone(b, &shared));
  black_box(&owner);
}

fn slice_static(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Slice<u8>>();
  for len in [SHORT, Slice::<u8>::INLINE_CAPACITY, MEDIUM, LARGE_PAYLOAD] {
    let contents = &PAYLOAD[..len];
    let source = Slice::<u8>::from_static(contents);
    assert_eq!(source.as_slice(), contents);
    assert_eq!(source.as_slice().as_ptr(), contents.as_ptr());
    let cloned = source.clone();
    assert_eq!(cloned.as_slice(), contents);
    assert_eq!(cloned.as_slice().as_ptr(), contents.as_ptr());
    let id = format!("slice_static/handle_{handle_size}/len_{len}");
    group.bench_function(id, |b| bench_clone(b, &source));
  }
}

fn bytes_baselines(group: &mut BenchmarkGroup<'_, WallTime>, len: usize) {
  let contents = &PAYLOAD[..len];
  let owner = Bytes::copy_from_slice(contents);
  // The first clone promotes the allocation; the measured source uses shared storage directly.
  let shared = owner.clone();
  let static_bytes = Bytes::from_static(contents);
  assert_eq!(shared.as_ref(), contents);
  assert_eq!(static_bytes.as_ref(), contents);

  let handle_size = size_of::<Bytes>();
  group.bench_function(
    format!("bytes_heap_shared/handle_{handle_size}/len_{len}"),
    |b| {
      bench_clone(b, &shared);
    },
  );
  group.bench_function(
    format!("bytes_static/handle_{handle_size}/len_{len}"),
    |b| {
      bench_clone(b, &static_bytes);
    },
  );
  black_box(&owner);
}

fn bytes_advanced(group: &mut BenchmarkGroup<'_, WallTime>, len: usize) {
  let owner = Bytes::from(advanced_input(LARGE_HANDLE, len));
  let mut shared = owner.clone();
  shared.advance(LARGE_HANDLE - len);
  assert_eq!(shared.as_ref(), &PAYLOAD[..len]);
  let id = format!(
    "bytes_heap_shared_advanced/handle_{}/from_{LARGE_HANDLE}/len_{len}",
    size_of::<Bytes>()
  );
  group.bench_function(id, |b| bench_clone(b, &shared));
  black_box(&owner);
}

fn clone_benches(c: &mut Criterion) {
  let mut group = c.benchmark_group("clone");
  inline(&mut group, Slice::<u8>::try_from_slice);
  inline(&mut group, SliceN::<u8, 256>::try_from_slice);
  inline(&mut group, SliceN::<u8, LARGE_HANDLE>::try_from_slice);
  inline_advanced(&mut group);
  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  slice_heap(&mut group);
  slice_static(&mut group);

  for len in [
    SHORT,
    Slice::<u8>::INLINE_CAPACITY,
    MEDIUM,
    SliceN::<u8, 256>::INLINE_CAPACITY,
    SliceN::<u8, LARGE_HANDLE>::INLINE_CAPACITY,
    LARGE_PAYLOAD,
  ] {
    bytes_baselines(&mut group, len);
  }
  for len in [SHORT, Slice::<u8>::INLINE_CAPACITY, MEDIUM] {
    bytes_advanced(&mut group, len);
  }
  group.finish();
}

criterion_group!(benches, clone_benches);
criterion_main!(benches);
