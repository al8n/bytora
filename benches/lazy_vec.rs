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
    time::{Duration, Instant},
  };

  const LEN: usize = 64;
  const CONTENTS: [u8; LEN] = [0x5a; LEN];
  const SHORT_LEN: usize = 7;
  const OUTPUT_BYTES: usize = 32;
  const BATCH: BatchSize = BatchSize::NumIterations(64);
  const DIRECT_BATCH: usize = 64;
  type DefaultSlice = Slice<u8>;
  type ZeroSlice = Slice<u8, typenum::U0>;
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
    OneSpare,
    Spare,
  }

  impl Capacity {
    fn name(self) -> &'static str {
      match self {
        Self::Exact => "exact",
        Self::OneSpare => "one_spare",
        Self::Spare => "spare",
      }
    }
  }

  fn input(capacity: Capacity) -> Vec<u8> {
    let requested = match capacity {
      Capacity::Exact => LEN,
      Capacity::OneSpare => LEN + 1,
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
      Capacity::OneSpare => assert_eq!(capacity, LEN + 1),
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
    assert!(slice.is_unique());
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
    assert_eq!(size_of::<(Option<T>, [u8; PAD])>(), OUTPUT_BYTES);
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
    // The vector and unique full handle are prepared outside the timed section.
    // Taking from the Option has the same input stride and timing boundary for
    // each type; Criterion drops the emptied slots and returned vectors afterward.
    group.bench_function(
      BenchmarkId::new(format!("unique_consume/{label}"), kind.name()),
      |b| {
        b.iter_batched_ref(
          || (Some(from_vec(input(kind))), [0u8; PAD]),
          |slot| black_box(into_vec(black_box(slot.0.take().expect("unique input")))),
          BATCH,
        );
      },
    );
    // Measure by-value consumption separately from Option slot extraction.
    // Input preparation, drain cleanup, and output drops remain outside timing.
    group.bench_function(
      BenchmarkId::new(format!("direct_consume/{label}"), kind.name()),
      |b| {
        b.iter_custom(|iterations| {
          let mut elapsed = Duration::ZERO;
          let mut inputs = Vec::with_capacity(DIRECT_BATCH);
          let mut outputs: Vec<Vec<u8>> = Vec::with_capacity(DIRECT_BATCH);
          outputs.resize_with(DIRECT_BATCH, Vec::new);
          black_box(&outputs);
          outputs.clear();
          let mut remaining = iterations;
          while remaining != 0 {
            let count = remaining.min(DIRECT_BATCH as u64) as usize;
            inputs.extend((0..count).map(|_| (from_vec(input(kind)), [0u8; PAD])));
            let mut pending = inputs.drain(..);
            let start = Instant::now();
            for (source, _) in pending.by_ref() {
              outputs.push(black_box(into_vec(black_box(source))));
            }
            black_box(&outputs);
            elapsed += start.elapsed();
            drop(pending);
            outputs.clear();
            remaining -= count as u64;
          }
          elapsed
        });
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

  fn short_input() -> Vec<u8> {
    let mut values = Vec::with_capacity(SHORT_LEN);
    values.extend_from_slice(&CONTENTS[..SHORT_LEN]);
    values
  }

  fn short_construct<T, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: String,
    from_vec: impl Fn(Vec<u8>) -> T,
  ) {
    assert_eq!(size_of::<(T, [u8; PAD])>(), OUTPUT_BYTES);
    group.bench_function(id, |b| {
      b.iter_batched(
        short_input,
        |values| black_box((from_vec(black_box(values)), [0u8; PAD])),
        BATCH,
      );
    });
  }

  // The seven-byte u8 input can use the two-word Vec representation when
  // its pointer slot meets the same atomic alignment requirements as the library.
  fn short_vec_is_adopted<S: InlineSize>() -> bool {
    #[cfg(feature = "extra-platforms")]
    type AtomicPtr = portable_atomic::AtomicPtr<()>;
    #[cfg(not(feature = "extra-platforms"))]
    type AtomicPtr = std::sync::atomic::AtomicPtr<()>;

    let bytes = size_of::<Slice<u8, S>>();
    let word = size_of::<usize>();
    bytes >= 3 * word
      || (bytes == 2 * word
        && align_of::<*const ()>() >= 4
        && align_of::<usize>() >= align_of::<AtomicPtr>()
        && size_of::<*mut ()>() == size_of::<AtomicPtr>()
        && size_of::<*mut ()>() % align_of::<AtomicPtr>() == 0)
  }

  fn short_slice_cases<S: InlineSize, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
  ) {
    let values = short_input();
    assert_eq!(values.capacity(), SHORT_LEN);
    let pointer = values.as_ptr();
    let slice = Slice::<u8, S>::try_from_vec(values).unwrap();
    assert_eq!(slice.as_slice(), &CONTENTS[..SHORT_LEN]);
    assert!(slice.is_unique());
    if short_vec_is_adopted::<S>() {
      assert_eq!(slice.as_slice().as_ptr(), pointer);
      let values = slice.try_into_vec().ok().expect("unique slice");
      assert_eq!(values.as_ptr(), pointer);
      assert_eq!(values.capacity(), SHORT_LEN);
      assert_eq!(values, &CONTENTS[..SHORT_LEN]);
    } else {
      assert_ne!(slice.as_slice().as_ptr(), pointer);
    }
    short_construct::<Slice<u8, S>, PAD>(group, format!("from_vec/{label}"), |values| {
      Slice::<u8, S>::try_from_vec(values).unwrap()
    });

    let values = short_input();
    let pointer = values.as_ptr();
    let shared = Slice::<u8, S>::try_from_shared_vec(values);
    if short_vec_is_adopted::<S>() {
      let shared = shared.expect("external slice fits");
      assert_eq!(shared.as_slice(), &CONTENTS[..SHORT_LEN]);
      assert_eq!(shared.as_slice().as_ptr(), pointer);
      assert!(shared.is_unique());
      let values = shared.try_into_vec().ok().expect("unique slice");
      assert_eq!(values.as_ptr(), pointer);
      assert_eq!(values.capacity(), SHORT_LEN);
      short_construct::<Slice<u8, S>, PAD>(group, format!("from_shared_vec/{label}"), |values| {
        Slice::<u8, S>::try_from_shared_vec(values).unwrap()
      });
    } else {
      let values = shared.err().expect("external slice does not fit");
      assert_eq!(values, &CONTENTS[..SHORT_LEN]);
      assert_eq!(values.capacity(), SHORT_LEN);
    }
  }

  fn short_benchmarks(c: &mut Criterion) {
    let values = short_input();
    assert_eq!(values.capacity(), SHORT_LEN);
    let pointer = values.as_ptr();
    let bytes = Bytes::from(values);
    assert_eq!(bytes.as_ref(), &CONTENTS[..SHORT_LEN]);
    assert_eq!(bytes.as_ptr(), pointer);
    assert!(bytes.is_unique());
    let values = Vec::from(bytes);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), SHORT_LEN);

    let mut group = c.benchmark_group("short_vec");
    short_slice_cases::<typenum::U8, { OUTPUT_BYTES - size_of::<SliceN<u8, 8>>() }>(
      &mut group, "slice_8",
    );
    short_slice_cases::<typenum::U16, { OUTPUT_BYTES - size_of::<SliceN<u8, 16>>() }>(
      &mut group, "slice_16",
    );
    short_slice_cases::<
      <DefaultSlice as SliceSize>::Size,
      { OUTPUT_BYTES - size_of::<DefaultSlice>() },
    >(&mut group, "slice_default");
    short_slice_cases::<typenum::U32, { OUTPUT_BYTES - size_of::<WideSlice>() }>(
      &mut group, "slice_32",
    );
    #[cfg(target_pointer_width = "32")]
    {
      short_slice_cases::<typenum::U20, { OUTPUT_BYTES - size_of::<SliceN<u8, 20>>() }>(
        &mut group, "slice_20",
      );
      short_slice_cases::<typenum::U24, { OUTPUT_BYTES - size_of::<SliceN<u8, 24>>() }>(
        &mut group, "slice_24",
      );
      short_slice_cases::<typenum::U28, { OUTPUT_BYTES - size_of::<SliceN<u8, 28>>() }>(
        &mut group, "slice_28",
      );
    }
    short_construct::<Bytes, { OUTPUT_BYTES - size_of::<Bytes>() }>(
      &mut group,
      "from_vec/bytes".to_owned(),
      Bytes::from,
    );
    group.finish();
  }

  pub fn benchmarks(c: &mut Criterion) {
    for kind in [Capacity::Exact, Capacity::OneSpare, Capacity::Spare] {
      slice_preflight::<<DefaultSlice as SliceSize>::Size>(kind);
      slice_preflight::<typenum::U0>(kind);
      slice_preflight::<<WideSlice as SliceSize>::Size>(kind);
      #[cfg(target_pointer_width = "32")]
      {
        slice_preflight::<typenum::U16>(kind);
        slice_preflight::<typenum::U20>(kind);
        slice_preflight::<typenum::U24>(kind);
        slice_preflight::<typenum::U28>(kind);
      }
      bytes_preflight(kind);
    }
    drop_preflight::<<DefaultSlice as SliceSize>::Size>();
    drop_preflight::<typenum::U0>();
    drop_preflight::<<WideSlice as SliceSize>::Size>();

    let mut group = c.benchmark_group("lazy_vec");
    for kind in [Capacity::Exact, Capacity::OneSpare, Capacity::Spare] {
      cases::<DefaultSlice, { OUTPUT_BYTES - size_of::<DefaultSlice>() }>(
        &mut group,
        "slice_default",
        kind,
        |values| DefaultSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
      );
      cases::<ZeroSlice, { OUTPUT_BYTES - size_of::<ZeroSlice>() }>(
        &mut group,
        "slice_zero",
        kind,
        |values| ZeroSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
      );
      cases::<WideSlice, { OUTPUT_BYTES - size_of::<WideSlice>() }>(
        &mut group,
        "slice_32",
        kind,
        |values| WideSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
      );
      #[cfg(target_pointer_width = "32")]
      {
        cases::<SliceN<u8, 16>, { OUTPUT_BYTES - size_of::<SliceN<u8, 16>>() }>(
          &mut group,
          "slice_16",
          kind,
          |values| SliceN::<u8, 16>::try_from_vec(values).unwrap(),
          |slice| slice.try_into_vec().ok().expect("unique slice"),
        );
        cases::<SliceN<u8, 20>, { OUTPUT_BYTES - size_of::<SliceN<u8, 20>>() }>(
          &mut group,
          "slice_20",
          kind,
          |values| SliceN::<u8, 20>::try_from_vec(values).unwrap(),
          |slice| slice.try_into_vec().ok().expect("unique slice"),
        );
        cases::<SliceN<u8, 24>, { OUTPUT_BYTES - size_of::<SliceN<u8, 24>>() }>(
          &mut group,
          "slice_24",
          kind,
          |values| SliceN::<u8, 24>::try_from_vec(values).unwrap(),
          |slice| slice.try_into_vec().ok().expect("unique slice"),
        );
        cases::<SliceN<u8, 28>, { OUTPUT_BYTES - size_of::<SliceN<u8, 28>>() }>(
          &mut group,
          "slice_28",
          kind,
          |values| SliceN::<u8, 28>::try_from_vec(values).unwrap(),
          |slice| slice.try_into_vec().ok().expect("unique slice"),
        );
      }
      cases::<Bytes, { OUTPUT_BYTES - size_of::<Bytes>() }>(
        &mut group,
        "bytes",
        kind,
        Bytes::from,
        Vec::from,
      );
    }
    group.finish();
    short_benchmarks(c);
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
