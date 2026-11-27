use bytes::{Buf, Bytes};
use criterion::{
  Bencher, BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use smol_bytes::{Bytes as SmolBytes, compact::Bytes as CompactSmolBytes};
use smol_slice::{InlineSize, Slice, SliceN};
use std::{
  hint::black_box,
  mem::size_of,
  ops::Range,
  time::{Duration, Instant},
};

const SHORT: usize = 8;
const MEDIUM: usize = 64;
const MATCHED_HANDLE: usize = 32;
const LARGE_HANDLE: usize = 4096;
const LARGE_PAYLOAD: usize = 64 * 1024;
const BATCH_SIZE: usize = 64;
const SMALL_LENGTHS: [usize; 13] = [0, 1, 7, 8, 14, 15, 16, 22, 23, 24, 30, 31, 32];

const _: () = {
  assert!(size_of::<SliceN<u8, 32>>() == 32);
  assert!(size_of::<SliceN<u8, 64>>() == 64);
  assert!(size_of::<SmolBytes>() == 64);
  assert!(size_of::<CompactSmolBytes>() == 64);
};
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<Bytes>() == 32);

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

fn unique_lengths(lengths: impl IntoIterator<Item = usize>) -> Vec<usize> {
  let mut unique = Vec::new();
  for len in lengths {
    if !unique.contains(&len) {
      unique.push(len);
    }
  }
  unique
}

fn inline<S: InlineSize>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  make: fn(&[u8]) -> Option<Slice<u8, S>>,
) {
  let capacity = Slice::<u8, S>::INLINE_CAPACITY;
  let mut lengths = vec![SHORT, capacity];
  if size_of::<Slice<u8, S>>() <= MATCHED_HANDLE {
    lengths.extend(SMALL_LENGTHS);
  }
  if size_of::<Slice<u8, S>>() == 64 {
    lengths.extend([30, 62]);
  }
  for len in unique_lengths(lengths) {
    if len > capacity {
      continue;
    }
    let source = make(&PAYLOAD[..len]).unwrap();
    assert_eq!(source.as_slice(), &PAYLOAD[..len]);
    let cloned = source.clone();
    assert_eq!(cloned.as_slice(), &PAYLOAD[..len]);
    if len != 0 {
      assert_ne!(cloned.as_slice().as_ptr(), source.as_slice().as_ptr());
    }
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
fn slice_heap<S: InlineSize>(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Slice<u8, S>>();
  if handle_size < 3 * size_of::<usize>() {
    return;
  }
  let capacity = Slice::<u8, S>::INLINE_CAPACITY;
  let lengths = [MEDIUM, LARGE_PAYLOAD, capacity + 1]
    .into_iter()
    .chain(SMALL_LENGTHS);
  for len in unique_lengths(lengths)
    .into_iter()
    .filter(|&len| len > capacity)
  {
    let values = PAYLOAD[..len].to_vec();
    let data = values.as_ptr();
    let owner = Slice::<u8, S>::try_from_vec(values).unwrap();
    let shared = owner.clone();
    assert_eq!(shared.as_slice(), &PAYLOAD[..len]);
    assert_eq!(shared.as_slice().as_ptr(), data);
    let id = format!("slice_heap_shared/handle_{handle_size}/len_{len}");
    group.bench_function(id, |b| bench_clone(b, &shared));
    black_box(&owner);
  }

  let mut lengths = vec![SHORT, capacity, 0, 1, capacity + 1, MATCHED_HANDLE];
  if handle_size == size_of::<Slice<u8>>() {
    lengths.push(MEDIUM);
  }
  for len in unique_lengths(lengths) {
    let owner = Slice::<u8, S>::try_from_vec(advanced_input(LARGE_HANDLE, len)).unwrap();
    let data = owner.as_slice().as_ptr();
    let mut shared = owner.clone();
    shared.advance(LARGE_HANDLE - len);
    assert_eq!(shared.as_slice(), &PAYLOAD[..len]);
    assert_eq!(
      shared.as_slice().as_ptr(),
      data.wrapping_add(LARGE_HANDLE - len)
    );
    assert_eq!(
      shared.clone().as_slice().as_ptr(),
      shared.as_slice().as_ptr()
    );
    let id =
      format!("slice_heap_shared_advanced/handle_{handle_size}/from_{LARGE_HANDLE}/len_{len}");
    group.bench_function(id, |b| bench_clone(b, &shared));
    black_box(&owner);
  }
}

fn slice_wide<S: InlineSize>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  method: CloneMethod,
  include_full: bool,
) {
  let handle_size = size_of::<Slice<u8, S>>();
  let source = Slice::<u8, S>::try_from_static(&PAYLOAD).unwrap();
  {
    let cloned = clone_preflight(&source, method, empty_slice::<S>);
    assert_eq!(cloned.as_slice(), &PAYLOAD[..]);
    assert_eq!(cloned.as_slice().as_ptr(), PAYLOAD.as_ptr());
  }
  if include_full {
    let id = format!("slice_static/handle_{handle_size}/len_{LARGE_PAYLOAD}");
    bench_clone_case(group, id, &source, method, empty_slice::<S>);
  }
  let mut advanced = source.clone();
  advanced.advance(LARGE_PAYLOAD - SHORT);
  {
    let cloned = clone_preflight(&advanced, method, empty_slice::<S>);
    assert_eq!(cloned.as_slice(), &PAYLOAD[..SHORT]);
    assert_eq!(
      cloned.as_slice().as_ptr(),
      PAYLOAD.as_ptr().wrapping_add(LARGE_PAYLOAD - SHORT)
    );
  }
  let id = format!("slice_static_advanced/handle_{handle_size}/from_{LARGE_PAYLOAD}/len_{SHORT}");
  bench_clone_case(group, id, &advanced, method, empty_slice::<S>);

  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  {
    assert!(LARGE_PAYLOAD > Slice::<u8, S>::INLINE_CAPACITY);
    let values = PAYLOAD.to_vec();
    let data = values.as_ptr();
    let owner = Slice::<u8, S>::try_from_vec(values).unwrap();
    let mut shared = owner.clone();
    assert_eq!(shared.as_slice(), &PAYLOAD[..]);
    assert_eq!(shared.as_slice().as_ptr(), data);
    {
      let cloned = clone_preflight(&shared, method, empty_slice::<S>);
      assert_eq!(cloned.as_slice(), &PAYLOAD[..]);
      assert_eq!(cloned.as_slice().as_ptr(), data);
    }
    if include_full {
      let id = format!("slice_heap_shared/handle_{handle_size}/len_{LARGE_PAYLOAD}");
      bench_clone_case(group, id, &shared, method, empty_slice::<S>);
    }

    shared.advance(LARGE_PAYLOAD - SHORT);
    {
      let cloned = clone_preflight(&shared, method, empty_slice::<S>);
      assert_eq!(cloned.as_slice(), &PAYLOAD[..SHORT]);
      assert_eq!(
        cloned.as_slice().as_ptr(),
        data.wrapping_add(LARGE_PAYLOAD - SHORT)
      );
    }
    let id =
      format!("slice_heap_shared_advanced/handle_{handle_size}/from_{LARGE_PAYLOAD}/len_{SHORT}");
    bench_clone_case(group, id, &shared, method, empty_slice::<S>);
    black_box(&owner);
  }
}

fn slice_static<S: InlineSize>(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Slice<u8, S>>();
  if Slice::<u8, S>::try_from_static(&PAYLOAD[..0]).is_none() {
    return;
  }
  let capacity = Slice::<u8, S>::INLINE_CAPACITY;
  let lengths = [SHORT, capacity, capacity + 1, MEDIUM, LARGE_PAYLOAD]
    .into_iter()
    .chain(SMALL_LENGTHS);
  for len in unique_lengths(lengths) {
    let contents = &PAYLOAD[..len];
    let source = Slice::<u8, S>::try_from_static(contents).unwrap();
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
  let static_bytes = Bytes::from_static(contents);
  assert_eq!(static_bytes.as_ref(), contents);
  assert_eq!(static_bytes.as_ptr(), contents.as_ptr());

  let handle_size = size_of::<Bytes>();
  if len == 0 {
    let empty = Bytes::new();
    group.bench_function(format!("bytes_empty/handle_{handle_size}/len_0"), |b| {
      bench_clone(b, &empty);
    });
  } else {
    let owner = Bytes::copy_from_slice(contents);
    // The first clone promotes the allocation; the measured source uses shared storage directly.
    let shared = owner.clone();
    assert_eq!(shared.as_ref(), contents);
    assert_eq!(shared.as_ptr(), owner.as_ptr());
    group.bench_function(
      format!("bytes_heap_shared/handle_{handle_size}/len_{len}"),
      |b| {
        bench_clone(b, &shared);
      },
    );
    black_box(&owner);
  }
  group.bench_function(
    format!("bytes_static/handle_{handle_size}/len_{len}"),
    |b| {
      bench_clone(b, &static_bytes);
    },
  );
}

fn bytes_advanced(group: &mut BenchmarkGroup<'_, WallTime>, len: usize) {
  let owner = Bytes::from(advanced_input(LARGE_HANDLE, len));
  let mut shared = owner.clone();
  shared.advance(LARGE_HANDLE - len);
  assert_eq!(shared.as_ref(), &PAYLOAD[..len]);
  assert_eq!(
    shared.as_ptr(),
    owner.as_ptr().wrapping_add(LARGE_HANDLE - len)
  );
  assert_eq!(shared.clone().as_ptr(), shared.as_ptr());
  let id = format!(
    "bytes_heap_shared_advanced/handle_{}/from_{LARGE_HANDLE}/len_{len}",
    size_of::<Bytes>()
  );
  group.bench_function(id, |b| bench_clone(b, &shared));
  black_box(&owner);
}

fn bytes_wide_baselines(group: &mut BenchmarkGroup<'_, WallTime>, method: CloneMethod) {
  let handle_size = size_of::<Bytes>();
  let source = Bytes::from_static(&PAYLOAD);
  {
    let cloned = clone_preflight(&source, method, Bytes::new);
    assert_eq!(cloned.as_ref(), &PAYLOAD[..]);
    assert_eq!(cloned.as_ptr(), PAYLOAD.as_ptr());
  }
  let id = format!("bytes_static/handle_{handle_size}/len_{LARGE_PAYLOAD}");
  bench_clone_case(group, id, &source, method, Bytes::new);

  let owner = Bytes::copy_from_slice(&PAYLOAD);
  let shared = owner.clone();
  {
    let cloned = clone_preflight(&shared, method, Bytes::new);
    assert_eq!(cloned.as_ref(), &PAYLOAD[..]);
    assert_eq!(cloned.as_ptr(), owner.as_ptr());
  }
  let id = format!("bytes_heap_shared/handle_{handle_size}/len_{LARGE_PAYLOAD}");
  bench_clone_case(group, id, &shared, method, Bytes::new);
  black_box(&owner);
}

fn bytes_wide_advanced(group: &mut BenchmarkGroup<'_, WallTime>, method: CloneMethod) {
  let handle_size = size_of::<Bytes>();
  let mut source = Bytes::from_static(&PAYLOAD);
  source.advance(LARGE_PAYLOAD - SHORT);
  {
    let cloned = clone_preflight(&source, method, Bytes::new);
    assert_eq!(cloned.as_ref(), &PAYLOAD[..SHORT]);
    assert_eq!(
      cloned.as_ptr(),
      PAYLOAD.as_ptr().wrapping_add(LARGE_PAYLOAD - SHORT)
    );
  }
  let id = format!("bytes_static_advanced/handle_{handle_size}/from_{LARGE_PAYLOAD}/len_{SHORT}");
  bench_clone_case(group, id, &source, method, Bytes::new);

  let owner = Bytes::copy_from_slice(&PAYLOAD);
  let mut shared = owner.clone();
  shared.advance(LARGE_PAYLOAD - SHORT);
  {
    let cloned = clone_preflight(&shared, method, Bytes::new);
    assert_eq!(cloned.as_ref(), &PAYLOAD[..SHORT]);
    assert_eq!(
      cloned.as_ptr(),
      owner.as_ptr().wrapping_add(LARGE_PAYLOAD - SHORT)
    );
  }
  let id =
    format!("bytes_heap_shared_advanced/handle_{handle_size}/from_{LARGE_PAYLOAD}/len_{SHORT}");
  bench_clone_case(group, id, &shared, method, Bytes::new);
  black_box(&owner);
}

fn smol_bytes_preflight<T: Clone + AsRef<[u8]>>(
  source: &T,
  method: CloneMethod,
  empty: fn() -> T,
  is_inline: fn(&T) -> bool,
  contents: &[u8],
  backing: Option<*const u8>,
) {
  let cloned = clone_preflight(source, method, empty);
  assert_eq!(source.as_ref(), contents);
  assert_eq!(cloned.as_ref(), contents);
  assert_eq!(is_inline(source), backing.is_none());
  assert_eq!(is_inline(&cloned), backing.is_none());
  if let Some(pointer) = backing {
    assert_eq!(source.as_ref().as_ptr(), pointer);
    assert_eq!(cloned.as_ref().as_ptr(), pointer);
  } else {
    assert_ne!(source.as_ref().as_ptr(), cloned.as_ref().as_ptr());
  }
}

fn smol_bytes_wide<T: Clone + Buf + AsRef<[u8]> + From<Vec<u8>>>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  method: CloneMethod,
  compact: bool,
  from_static: fn(&'static [u8]) -> T,
  is_inline: fn(&T) -> bool,
  empty: fn() -> T,
) {
  let handle_size = size_of::<T>();
  let strategy = if compact { "compact" } else { "shared" };
  if !compact {
    for len in [SHORT, 30, 62] {
      let source = from_static(&PAYLOAD[..len]);
      smol_bytes_preflight(&source, method, empty, is_inline, &PAYLOAD[..len], None);
      let id = format!("smol_bytes_{strategy}_inline/handle_{handle_size}/len_{len}");
      bench_clone_case(group, id, &source, method, empty);
    }
  }

  // Long static data remains zero-copy, despite being stored in the heap variant.
  let static_source = from_static(&PAYLOAD);
  let values = PAYLOAD.to_vec();
  let data = values.as_ptr();
  let owner = T::from(values);
  // Use the first clone as the measured source, after promotion to shared storage.
  let shared = owner.clone();
  for (source, data, origin) in [
    (&static_source, PAYLOAD.as_ptr(), "static"),
    (&shared, data, "heap"),
  ] {
    smol_bytes_preflight(source, method, empty, is_inline, &PAYLOAD, Some(data));
    let full_kind = if origin == "heap" {
      "heap_shared"
    } else {
      origin
    };
    let id = format!("smol_bytes_{strategy}_{full_kind}/handle_{handle_size}/len_{LARGE_PAYLOAD}");
    bench_clone_case(group, id, source, method, empty);

    let mut advanced = source.clone();
    advanced.advance(LARGE_PAYLOAD - SHORT);
    let pointer = data.wrapping_add(LARGE_PAYLOAD - SHORT);
    if compact {
      assert_ne!(advanced.as_ref().as_ptr(), pointer);
    }
    smol_bytes_preflight(
      &advanced,
      method,
      empty,
      is_inline,
      &PAYLOAD[..SHORT],
      (!compact).then_some(pointer),
    );
    let kind = if compact {
      format!("inline_after_{origin}_advanced")
    } else {
      format!("{full_kind}_advanced")
    };
    let id =
      format!("smol_bytes_{strategy}_{kind}/handle_{handle_size}/from_{LARGE_PAYLOAD}/len_{SHORT}");
    bench_clone_case(group, id, &advanced, method, empty);
  }
  black_box(&owner);
}

fn smol_bytes_wide_benches(group: &mut BenchmarkGroup<'_, WallTime>, method: CloneMethod) {
  smol_bytes_wide(
    group,
    method,
    false,
    SmolBytes::from_static,
    SmolBytes::is_inline,
    SmolBytes::new,
  );
  smol_bytes_wide(
    group,
    method,
    true,
    CompactSmolBytes::from_static,
    CompactSmolBytes::is_inline,
    CompactSmolBytes::new,
  );
}

fn slice_inline_clone_from<S: InlineSize>(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Slice<u8, S>>();
  for len in [SHORT, 30, 62] {
    if len > Slice::<u8, S>::INLINE_CAPACITY {
      continue;
    }
    let source = Slice::<u8, S>::try_from_slice(&PAYLOAD[..len]).unwrap();
    let cloned = clone_preflight(&source, CloneMethod::From, empty_slice::<S>);
    assert_eq!(source.as_slice(), &PAYLOAD[..len]);
    assert_eq!(cloned.as_slice(), &PAYLOAD[..len]);
    assert_ne!(cloned.as_slice().as_ptr(), source.as_slice().as_ptr());
    let id = format!("slice_inline/handle_{handle_size}/len_{len}");
    bench_clone_case(group, id, &source, CloneMethod::From, empty_slice::<S>);
  }
}

fn bytes_short_clone_from(group: &mut BenchmarkGroup<'_, WallTime>) {
  let handle_size = size_of::<Bytes>();
  for len in [SHORT, 30, 62] {
    let owner = Bytes::copy_from_slice(&PAYLOAD[..len]);
    let shared = owner.clone();
    let cloned = clone_preflight(&shared, CloneMethod::From, Bytes::new);
    assert_eq!(shared.as_ref(), &PAYLOAD[..len]);
    assert_eq!(cloned.as_ref(), &PAYLOAD[..len]);
    assert_eq!(shared.as_ptr(), owner.as_ptr());
    assert_eq!(cloned.as_ptr(), owner.as_ptr());
    let id = format!("bytes_heap_shared/handle_{handle_size}/len_{len}");
    bench_clone_case(group, id, &shared, CloneMethod::From, Bytes::new);
    black_box(&owner);
  }
}

fn array_copy<const N: usize>(group: &mut BenchmarkGroup<'_, WallTime>, method: CloneMethod) {
  let source = std::array::from_fn::<_, N, _>(|index| PAYLOAD[index]);
  let cloned = clone_preflight(&source, method, || [0; N]);
  assert_eq!(cloned, source);
  bench_clone_case(
    group,
    format!("array_copy/handle_{N}"),
    &source,
    method,
    || [0; N],
  );
}

fn clone_benches(c: &mut Criterion) {
  let mut group = c.benchmark_group("clone");
  inline(&mut group, SliceN::<u8, 8>::try_from_slice);
  inline(&mut group, SliceN::<u8, 16>::try_from_slice);
  inline(&mut group, Slice::<u8>::try_from_slice);
  if size_of::<Slice<u8>>() != 24 {
    inline(&mut group, SliceN::<u8, 24>::try_from_slice);
  }
  inline(&mut group, SliceN::<u8, MATCHED_HANDLE>::try_from_slice);
  inline(&mut group, SliceN::<u8, 64>::try_from_slice);
  inline(&mut group, SliceN::<u8, 128>::try_from_slice);
  inline(&mut group, SliceN::<u8, 256>::try_from_slice);
  inline(&mut group, SliceN::<u8, LARGE_HANDLE>::try_from_slice);
  inline_advanced(&mut group);
  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  {
    slice_heap::<typenum::U16>(&mut group);
    slice_heap::<typenum::U24>(&mut group);
    slice_heap::<typenum::U32>(&mut group);
    #[cfg(target_pointer_width = "32")]
    slice_heap::<typenum::U12>(&mut group);
  }
  slice_static::<typenum::U16>(&mut group);
  slice_static::<typenum::U24>(&mut group);
  slice_static::<typenum::U32>(&mut group);
  #[cfg(target_pointer_width = "32")]
  slice_static::<typenum::U12>(&mut group);
  // Full 32-byte cases were already registered above; add only advanced windows.
  slice_wide::<typenum::U32>(&mut group, CloneMethod::Push, false);
  slice_wide::<typenum::U64>(&mut group, CloneMethod::Push, true);
  slice_wide::<typenum::U128>(&mut group, CloneMethod::Push, true);
  slice_wide::<typenum::U4096>(&mut group, CloneMethod::Push, true);

  let lengths = [
    SHORT,
    Slice::<u8>::INLINE_CAPACITY,
    Slice::<u8>::INLINE_CAPACITY + 1,
    MEDIUM,
    SliceN::<u8, 64>::INLINE_CAPACITY,
    SliceN::<u8, 128>::INLINE_CAPACITY,
    SliceN::<u8, 256>::INLINE_CAPACITY,
    SliceN::<u8, LARGE_HANDLE>::INLINE_CAPACITY,
    62,
    LARGE_PAYLOAD,
  ]
  .into_iter()
  .chain(SMALL_LENGTHS);
  for len in unique_lengths(lengths) {
    bytes_baselines(&mut group, len);
  }
  let lengths = [
    SHORT,
    Slice::<u8>::INLINE_CAPACITY,
    Slice::<u8>::INLINE_CAPACITY + 1,
    MEDIUM,
  ]
  .into_iter()
  .chain([0, 1, 7, 14, 15, 22, 23, 30, 31, MATCHED_HANDLE]);
  for len in unique_lengths(lengths) {
    bytes_advanced(&mut group, len);
  }
  bytes_wide_advanced(&mut group, CloneMethod::Push);
  smol_bytes_wide_benches(&mut group, CloneMethod::Push);
  array_copy::<32>(&mut group, CloneMethod::Push);
  array_copy::<64>(&mut group, CloneMethod::Push);
  group.finish();
}

fn empty_slice<S: InlineSize>() -> Slice<u8, S> {
  Slice::<u8, S>::try_from_slice(&[]).unwrap()
}

fn clone_from_benches(c: &mut Criterion) {
  let mut group = c.benchmark_group("clone_from");
  // Compact handles serve as controls for the wider handles.
  slice_wide::<typenum::U24>(&mut group, CloneMethod::From, true);
  slice_wide::<typenum::U32>(&mut group, CloneMethod::From, true);
  slice_wide::<typenum::U64>(&mut group, CloneMethod::From, true);
  slice_wide::<typenum::U128>(&mut group, CloneMethod::From, true);
  slice_wide::<typenum::U4096>(&mut group, CloneMethod::From, true);
  bytes_wide_baselines(&mut group, CloneMethod::From);
  bytes_wide_advanced(&mut group, CloneMethod::From);
  slice_inline_clone_from::<typenum::U32>(&mut group);
  slice_inline_clone_from::<typenum::U64>(&mut group);
  bytes_short_clone_from(&mut group);
  smol_bytes_wide_benches(&mut group, CloneMethod::From);
  array_copy::<32>(&mut group, CloneMethod::From);
  array_copy::<64>(&mut group, CloneMethod::From);
  group.finish();
}

fn bench_range<T, O>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  kind: &str,
  source: &T,
  total: usize,
  range: Range<usize>,
  make_window: impl Fn(&T, Range<usize>) -> O,
) {
  let start = range.start;
  let end = range.end;
  let len = end - start;
  let id = format!("{kind}/handle_{MATCHED_HANDLE}/from_{total}/start_{start}/len_{len}");
  group.bench_function(id, |b| {
    bench_operation(b, || {
      make_window(black_box(source), black_box(start)..black_box(end))
    });
  });
}

fn range_static(group: &mut BenchmarkGroup<'_, WallTime>, total: usize, start: usize, end: usize) {
  let contents = &PAYLOAD[..total];
  let slice = SliceN::<u8, MATCHED_HANDLE>::try_from_static(contents).unwrap();
  let bytes = Bytes::from_static(contents);
  let expected = &contents[start..end];
  let slice_window = slice.slice(start..end);
  let bytes_window = bytes.slice(start..end);
  assert_eq!(slice.as_slice().as_ptr(), contents.as_ptr());
  assert_eq!(bytes.as_ptr(), contents.as_ptr());
  assert_eq!(slice_window.as_slice(), expected);
  assert_eq!(bytes_window.as_ref(), expected);
  assert_eq!(slice_window.as_slice().as_ptr(), expected.as_ptr());
  assert_eq!(bytes_window.as_ptr(), expected.as_ptr());
  bench_range(group, "slice_static", &slice, total, start..end, |s, r| {
    s.slice(r)
  });
  bench_range(group, "bytes_static", &bytes, total, start..end, |s, r| {
    s.slice(r)
  });
}

fn range_inline(group: &mut BenchmarkGroup<'_, WallTime>, start: usize, end: usize) {
  let contents = &PAYLOAD[..SHORT];
  let slice = SliceN::<u8, MATCHED_HANDLE>::try_from_slice(contents).unwrap();
  let window = slice.slice(start..end);
  assert_eq!(slice.as_slice(), contents);
  assert_eq!(window.as_slice(), &contents[start..end]);
  assert_ne!(
    window.as_slice().as_ptr(),
    slice.as_slice()[start..end].as_ptr()
  );
  bench_range(group, "slice_inline", &slice, SHORT, start..end, |s, r| {
    s.slice(r)
  });
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
fn range_heap(group: &mut BenchmarkGroup<'_, WallTime>, start: usize, end: usize) {
  let contents = &PAYLOAD[..MEDIUM];
  let values = contents.to_vec();
  let data = values.as_ptr();
  let owner = SliceN::<u8, MATCHED_HANDLE>::try_from_vec(values).unwrap();
  let slice = owner.clone();
  let window = slice.slice(start..end);
  assert_eq!(slice.as_slice(), contents);
  assert_eq!(slice.as_slice().as_ptr(), data);
  assert_eq!(window.as_slice(), &contents[start..end]);
  assert_eq!(window.as_slice().as_ptr(), data.wrapping_add(start));
  bench_range(
    group,
    "slice_heap_shared",
    &slice,
    MEDIUM,
    start..end,
    |s, r| s.slice(r),
  );
  black_box(&owner);
}

fn range_bytes_heap(
  group: &mut BenchmarkGroup<'_, WallTime>,
  total: usize,
  start: usize,
  end: usize,
) {
  let contents = &PAYLOAD[..total];
  let owner = Bytes::copy_from_slice(contents);
  let bytes = owner.clone();
  let window = bytes.slice(start..end);
  assert_eq!(bytes.as_ref(), contents);
  assert_eq!(bytes.as_ptr(), owner.as_ptr());
  assert_eq!(window.as_ref(), &contents[start..end]);
  assert_eq!(window.as_ptr(), owner.as_ptr().wrapping_add(start));
  bench_range(
    group,
    "bytes_heap_shared",
    &bytes,
    total,
    start..end,
    |s, r| s.slice(r),
  );
  black_box(&owner);
}

fn range_benches(c: &mut Criterion) {
  if size_of::<Bytes>() != MATCHED_HANDLE {
    return;
  }
  let mut group = c.benchmark_group("slice_range");
  range_inline(&mut group, 2, 6);
  range_static(&mut group, SHORT, 2, 6);
  range_static(&mut group, LARGE_PAYLOAD, 2, 6);
  range_static(
    &mut group,
    LARGE_PAYLOAD,
    LARGE_PAYLOAD / 2 - 2,
    LARGE_PAYLOAD / 2 + 2,
  );
  range_static(&mut group, LARGE_PAYLOAD, 2, LARGE_PAYLOAD - 2);
  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  range_heap(&mut group, 2, 6);
  range_bytes_heap(&mut group, SHORT, 2, 6);
  range_bytes_heap(&mut group, MEDIUM, 2, 6);
  group.finish();
}

criterion_group!(benches, clone_benches, clone_from_benches, range_benches);
criterion_main!(benches);
