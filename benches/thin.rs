use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(smol_slice_atomic, target_pointer_width = "64"))]
mod ownership {
  use bytes::Bytes;
  use criterion::{
    BatchSize, Bencher, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime,
  };
  use smol_slice::{SliceN, ThinSlice};
  use std::{hint::black_box, mem::size_of, ops::Range};

  const BATCH: BatchSize = BatchSize::NumIterations(64);
  const CONTENTS: [u8; 64] = {
    let mut values = [0; 64];
    let mut i = 0;
    while i < values.len() {
      values[i] = i as u8;
      i += 1;
    }
    values
  };

  fn input(len: usize, capacity: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(capacity);
    values.extend_from_slice(&CONTENTS[..len]);
    assert_eq!((values.len(), values.capacity()), (len, capacity));
    values
  }

  trait Handle: Clone + Sized {
    const ADOPTS_VEC: bool;

    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
    fn as_bytes(&self) -> &[u8];
    fn is_unique(&self) -> bool;
    fn range(&self, range: Range<usize>) -> Self;
    fn into_vec(self) -> Vec<u8>;
  }

  impl Handle for ThinSlice<u8> {
    const ADOPTS_VEC: bool = true;

    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from_vec(values))
    }
    fn as_bytes(&self) -> &[u8] {
      self.as_slice()
    }
    fn is_unique(&self) -> bool {
      Self::is_unique(self)
    }
    fn range(&self, range: Range<usize>) -> Self {
      self.slice(range)
    }
    fn into_vec(self) -> Vec<u8> {
      self.try_into_vec().ok().expect("unique slice")
    }
  }

  impl Handle for SliceN<u8, 8> {
    const ADOPTS_VEC: bool = false;

    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Self::try_from_vec(values)
    }
    fn as_bytes(&self) -> &[u8] {
      self.as_slice()
    }
    fn is_unique(&self) -> bool {
      Self::is_unique(self)
    }
    fn range(&self, range: Range<usize>) -> Self {
      self.slice(range)
    }
    fn into_vec(self) -> Vec<u8> {
      self.try_into_vec().ok().expect("unique slice")
    }
  }

  impl Handle for Bytes {
    const ADOPTS_VEC: bool = true;

    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from(values))
    }
    fn as_bytes(&self) -> &[u8] {
      self.as_ref()
    }
    fn is_unique(&self) -> bool {
      Self::is_unique(self)
    }
    fn range(&self, range: Range<usize>) -> Self {
      self.slice(range)
    }
    fn into_vec(self) -> Vec<u8> {
      Vec::from(self)
    }
  }

  fn bench(
    group: &mut BenchmarkGroup<'_, WallTime>,
    operation: &str,
    label: &str,
    len: usize,
    capacity: usize,
    routine: impl FnMut(&mut Bencher<'_>),
  ) {
    group.bench_function(
      BenchmarkId::new(
        format!("{operation}/{label}"),
        format!("len_{len}_cap_{capacity}"),
      ),
      routine,
    );
  }

  fn cases<T: Handle>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    len: usize,
    capacity: usize,
  ) {
    let values = input(len, capacity);
    let pointer = values.as_ptr();
    let source = match T::from_vec(values) {
      Ok(source) => source,
      Err(values) => {
        assert_eq!((label, len), ("inline8", 64));
        assert_eq!(values, &CONTENTS[..len]);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!((values.len(), values.capacity()), (len, capacity));
        eprintln!("thin: {label}/len_{len}_cap_{capacity} unrepresentable; skipped");
        return;
      }
    };
    assert_eq!(source.as_bytes(), &CONTENTS[..len]);
    assert!(source.is_unique());
    if T::ADOPTS_VEC {
      assert_eq!(source.as_bytes().as_ptr(), pointer);
    }
    let peer = source.clone();
    assert_eq!(source.as_bytes(), &CONTENTS[..len]);
    assert_eq!(peer.as_bytes(), &CONTENTS[..len]);
    if T::ADOPTS_VEC {
      assert_eq!(peer.as_bytes().as_ptr(), pointer);
    }

    let bounds = 1..len - 1;
    let values = input(len, capacity);
    let fresh_pointer = values.as_ptr();
    let fresh = T::from_vec(values).expect("representable input");
    assert!(fresh.is_unique());
    let window = fresh.range(bounds.clone());
    let nested = window.range(1..window.as_bytes().len() - 1);
    assert_eq!(fresh.as_bytes(), &CONTENTS[..len]);
    assert_eq!(window.as_bytes(), &CONTENTS[bounds.clone()]);
    assert_eq!(nested.as_bytes(), &CONTENTS[2..len - 2]);
    if T::ADOPTS_VEC {
      assert_eq!(fresh.as_bytes().as_ptr(), fresh_pointer);
      assert_eq!(window.as_bytes().as_ptr(), fresh_pointer.wrapping_add(1));
      assert_eq!(nested.as_bytes().as_ptr(), fresh_pointer.wrapping_add(2));
    }
    drop((nested, window, fresh));

    let values = input(len, capacity);
    let pointer = values.as_ptr();
    let unique = T::from_vec(values).expect("representable input");
    assert!(unique.is_unique());
    if T::ADOPTS_VEC {
      assert_eq!(unique.as_bytes().as_ptr(), pointer);
    }
    let mut slot = Some(unique);
    let values = T::into_vec(slot.take().expect("unique input"));
    assert_eq!(values, &CONTENTS[..len]);
    assert!(values.capacity() >= len);
    if T::ADOPTS_VEC {
      assert_eq!(values.as_ptr(), pointer);
      assert_eq!(values.capacity(), capacity);
    }
    drop(values);

    bench(group, "from_vec", label, len, capacity, |b| {
      b.iter_batched(
        || input(len, capacity),
        |values| black_box(T::from_vec(black_box(values)).expect("representable input")),
        BATCH,
      );
    });
    bench(group, "first_clone", label, len, capacity, |b| {
      b.iter_batched_ref(
        || T::from_vec(input(len, capacity)).expect("representable input"),
        |input| black_box(T::clone(black_box(input))),
        BATCH,
      );
    });
    bench(group, "warmed_clone", label, len, capacity, |b| {
      b.iter_batched_ref(
        || &source,
        |source| black_box(T::clone(black_box(*source))),
        BATCH,
      );
    });
    bench(group, "first_slice", label, len, capacity, |b| {
      b.iter_batched_ref(
        || T::from_vec(input(len, capacity)).expect("representable input"),
        |input| {
          black_box(T::range(
            black_box(input),
            black_box(bounds.start)..black_box(bounds.end),
          ))
        },
        BATCH,
      );
    });
    bench(group, "warmed_slice", label, len, capacity, |b| {
      b.iter_batched_ref(
        || &source,
        |source| {
          black_box(T::range(
            black_box(*source),
            black_box(bounds.start)..black_box(bounds.end),
          ))
        },
        BATCH,
      );
    });
    bench(group, "consume", label, len, capacity, |b| {
      b.iter_batched_ref(
        || Some(T::from_vec(input(len, capacity)).expect("representable input")),
        |slot| black_box(T::into_vec(black_box(slot.take().expect("unique input")))),
        BATCH,
      );
    });
    black_box(&peer);
  }

  pub fn benchmarks(c: &mut Criterion) {
    assert_eq!(size_of::<ThinSlice<u8>>(), size_of::<usize>());
    assert_eq!(size_of::<Option<ThinSlice<u8>>>(), size_of::<usize>());
    assert_eq!(size_of::<SliceN<u8, 8>>(), size_of::<usize>());
    assert_eq!(size_of::<Option<SliceN<u8, 8>>>(), size_of::<usize>());
    let mut group = c.benchmark_group("thin");
    for (len, capacity) in [(7, 7), (64, 64), (64, 65)] {
      cases::<ThinSlice<u8>>(&mut group, "thin", len, capacity);
      cases::<SliceN<u8, 8>>(&mut group, "inline8", len, capacity);
      cases::<Bytes>(&mut group, "bytes", len, capacity);
    }
    group.finish();
  }
}

fn benchmarks(c: &mut Criterion) {
  #[cfg(all(smol_slice_atomic, target_pointer_width = "64"))]
  ownership::benchmarks(c);
  #[cfg(not(all(smol_slice_atomic, target_pointer_width = "64")))]
  let _ = c;
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
