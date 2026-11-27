use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(smol_slice_atomic)]
mod ownership {
  use bytes::Bytes;
  use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
  use smol_slice::{InlineSize, Slice, SliceN};
  use std::{
    hint::black_box,
    mem::size_of,
    sync::{
      Arc,
      atomic::{AtomicUsize, Ordering},
    },
  };

  const LEN: usize = 64;
  const CONTENTS: [u8; LEN] = [0x5a; LEN];
  const OUTPUT_BYTES: usize = 32;
  const BATCH: BatchSize = BatchSize::NumIterations(64);
  type DefaultSlice = Slice<u8>;
  type WideSlice = SliceN<u8, 32>;

  // All construction and clone results use the same output stride, including
  // the default three-word Slice. Round trips all return a Vec directly.
  const _: () = {
    assert!(size_of::<DefaultSlice>() <= OUTPUT_BYTES);
    assert!(size_of::<WideSlice>() == OUTPUT_BYTES);
    assert!(size_of::<Bytes>() <= OUTPUT_BYTES);
    assert!(
      size_of::<(DefaultSlice, [u8; OUTPUT_BYTES - size_of::<DefaultSlice>()])>() == OUTPUT_BYTES
    );
    assert!(size_of::<(WideSlice, [u8; OUTPUT_BYTES - size_of::<WideSlice>()])>() == OUTPUT_BYTES);
    assert!(size_of::<(Bytes, [u8; OUTPUT_BYTES - size_of::<Bytes>()])>() == OUTPUT_BYTES);
    assert!(DefaultSlice::INLINE_CAPACITY < LEN);
    assert!(WideSlice::INLINE_CAPACITY < LEN);
  };

  #[derive(Clone, Copy)]
  enum Capacity {
    Exact,
    Spare,
  }

  impl Capacity {
    fn name(self) -> &'static str {
      match self {
        Self::Exact => "exact",
        Self::Spare => "spare",
      }
    }
  }

  fn input(capacity: Capacity) -> Vec<u8> {
    let requested = match capacity {
      Capacity::Exact => LEN,
      Capacity::Spare => LEN * 2,
    };
    let mut values = Vec::with_capacity(requested);
    values.extend_from_slice(&CONTENTS);
    values
  }

  fn check_input(values: &[u8], capacity: usize, kind: Capacity) {
    assert_eq!(values, CONTENTS);
    match kind {
      Capacity::Exact => assert_eq!(capacity, LEN),
      Capacity::Spare => assert!(capacity > LEN),
    }
  }

  fn slice_preflight<S: InlineSize>(kind: Capacity) {
    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    check_input(&values, capacity, kind);
    let slice = Slice::<u8, S>::try_from_vec(values).unwrap();
    assert_eq!(slice.as_slice(), CONTENTS);
    assert_eq!(slice.as_slice().as_ptr(), pointer);
    let values = slice.try_into_vec().ok().expect("unique slice");
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, CONTENTS);

    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let slice = Slice::<u8, S>::try_from_vec(values).unwrap();
    let cloned = slice.clone();
    assert_eq!(cloned.as_slice(), CONTENTS);
    assert_eq!(cloned.as_slice().as_ptr(), pointer);
    let warmed = slice.clone();
    assert_eq!(warmed.as_slice(), CONTENTS);
    assert_eq!(warmed.as_slice().as_ptr(), pointer);
    drop(warmed);
    let slice = slice.try_into_vec().expect_err("slice is shared");
    drop(cloned);
    let values = slice.try_into_vec().ok().expect("slice is unique again");
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, CONTENTS);
  }

  fn bytes_preflight(kind: Capacity) {
    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    check_input(&values, capacity, kind);
    let bytes = Bytes::from(values);
    assert_eq!(bytes.as_ref(), CONTENTS);
    assert_eq!(bytes.as_ptr(), pointer);
    assert!(bytes.is_unique());
    let values = Vec::from(bytes);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, CONTENTS);

    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let bytes = Bytes::from(values);
    let cloned = bytes.clone();
    assert_eq!(cloned.as_ref(), CONTENTS);
    assert_eq!(cloned.as_ptr(), pointer);
    assert!(!bytes.is_unique());
    let warmed = bytes.clone();
    assert_eq!(warmed.as_ref(), CONTENTS);
    assert_eq!(warmed.as_ptr(), pointer);
    drop(warmed);
    drop(bytes);
    assert!(cloned.is_unique());
    let values = Vec::from(cloned);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, CONTENTS);
  }

  #[derive(Debug)]
  struct DropProbe(Arc<AtomicUsize>);

  impl Drop for DropProbe {
    fn drop(&mut self) {
      self.0.fetch_add(1, Ordering::Relaxed);
    }
  }

  fn drop_preflight<S: InlineSize>() {
    let drops = Arc::new(AtomicUsize::new(0));
    let values: Vec<_> = (0..LEN).map(|_| DropProbe(drops.clone())).collect();
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let slice = Slice::<DropProbe, S>::try_from_shared_vec(values).expect("shared slice fits");
    let values = slice.try_into_vec().ok().expect("unique slice");
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(values);
    assert_eq!(drops.load(Ordering::Relaxed), LEN);

    let values: Vec<_> = (0..LEN).map(|_| DropProbe(drops.clone())).collect();
    let slice = Slice::<DropProbe, S>::try_from_shared_vec(values).expect("shared slice fits");
    let cloned = slice.clone();
    drop(slice);
    assert_eq!(drops.load(Ordering::Relaxed), LEN);
    drop(cloned);
    assert_eq!(drops.load(Ordering::Relaxed), LEN * 2);
  }

  fn cases<T: Clone, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    kind: Capacity,
    from_vec: impl Fn(Vec<u8>) -> T,
    into_vec: impl Fn(T) -> Vec<u8>,
  ) {
    assert_eq!(size_of::<(T, [u8; PAD])>(), OUTPUT_BYTES);
    group.bench_function(
      BenchmarkId::new(format!("construct/{label}"), kind.name()),
      |b| {
        b.iter_batched(
          || input(kind),
          |values| black_box((from_vec(black_box(values)), [0u8; PAD])),
          BATCH,
        );
      },
    );
    group.bench_function(
      BenchmarkId::new(format!("round_trip/{label}"), kind.name()),
      |b| {
        b.iter_batched(
          || input(kind),
          |values| black_box(into_vec(from_vec(black_box(values)))),
          BATCH,
        );
      },
    );
    group.bench_function(
      BenchmarkId::new(format!("first_clone/{label}"), kind.name()),
      |b| {
        b.iter_batched_ref(
          || from_vec(input(kind)),
          |source| black_box((black_box(&*source).clone(), [0u8; PAD])),
          BATCH,
        );
      },
    );

    // Keep the first clone alive and measure later clones of the original
    // Vec-adopting handle, which may retain a different dispatch path.
    let original = from_vec(input(kind));
    let peer = original.clone();
    group.bench_function(
      BenchmarkId::new(format!("warmed_original/{label}"), kind.name()),
      |b| {
        b.iter_batched_ref(
          || &original,
          |source| black_box((black_box(*source).clone(), [0u8; PAD])),
          BATCH,
        );
      },
    );
    black_box(&peer);
  }

  pub fn benchmarks(c: &mut Criterion) {
    for kind in [Capacity::Exact, Capacity::Spare] {
      slice_preflight::<<DefaultSlice as SliceSize>::Size>(kind);
      slice_preflight::<<WideSlice as SliceSize>::Size>(kind);
      bytes_preflight(kind);
    }
    drop_preflight::<<DefaultSlice as SliceSize>::Size>();
    drop_preflight::<<WideSlice as SliceSize>::Size>();

    let mut group = c.benchmark_group("lazy_vec");
    for kind in [Capacity::Exact, Capacity::Spare] {
      cases::<DefaultSlice, { OUTPUT_BYTES - size_of::<DefaultSlice>() }>(
        &mut group,
        "slice_default",
        kind,
        |values| DefaultSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
      );
      cases::<WideSlice, { OUTPUT_BYTES - size_of::<WideSlice>() }>(
        &mut group,
        "slice_32",
        kind,
        |values| WideSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
      );
      cases::<Bytes, { OUTPUT_BYTES - size_of::<Bytes>() }>(
        &mut group,
        "bytes",
        kind,
        Bytes::from,
        Vec::from,
      );
    }
    group.finish();
  }

  // Extract the sealed type-level size from each public Slice alias.
  trait SliceSize {
    type Size: InlineSize;
  }

  impl<T, S: InlineSize> SliceSize for Slice<T, S> {
    type Size = S;
  }
}

fn benchmarks(c: &mut Criterion) {
  #[cfg(smol_slice_atomic)]
  ownership::benchmarks(c);
  #[cfg(not(smol_slice_atomic))]
  let _ = c;
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
