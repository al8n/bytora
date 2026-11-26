use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(bytora_atomic)]
mod ownership {
  use bytes::Bytes;
  use bytora::{AssocInlineSize, InlineSize, Vector, VectorN};
  use criterion::{
    BatchSize, Bencher, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime,
  };
  use std::{hint::black_box, mem::size_of, ops::Range};

  const STRIDE: usize = 128;
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
    values
  }

  trait Handle: Clone + Sized {
    fn from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
    fn as_bytes(&self) -> &[u8];
    fn is_unique(&self) -> bool;
    fn range(&self, range: Range<usize>) -> Self;
    fn into_vec(self) -> Vec<u8>;
  }

  impl<S: InlineSize> Handle for Vector<u8, S> {
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
      self.try_into_vec().ok().expect("unique vector")
    }
  }

  impl Handle for Bytes {
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

  fn cases<T: Handle, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    len: usize,
    capacity: usize,
  ) {
    let stride = if PAD == 0 { size_of::<T>() } else { STRIDE };
    assert_eq!(size_of::<(T, [u8; PAD])>(), stride);
    assert_eq!(size_of::<(Option<T>, [u8; PAD])>(), stride);
    let values = input(len, capacity);
    let pointer = values.as_ptr();
    assert_eq!((values.len(), values.capacity()), (len, capacity));
    let source = match T::from_vec(values) {
      Ok(source) => source,
      Err(values) => {
        assert_eq!(values, &CONTENTS[..len]);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!((values.len(), values.capacity()), (len, capacity));
        eprintln!("stages: {label}/len_{len}_cap_{capacity} unrepresentable; skipped");
        return;
      }
    };
    assert_eq!(source.as_bytes(), &CONTENTS[..len]);
    assert!(source.is_unique());
    let peer = source.clone();
    assert_eq!(peer.as_bytes(), &CONTENTS[..len]);
    if source.as_bytes().as_ptr() == pointer {
      assert_eq!(peer.as_bytes().as_ptr(), pointer);
    }
    let bounds = 1..len - 1;
    let values = input(len, capacity);
    let fresh_pointer = values.as_ptr();
    let fresh = T::from_vec(values).expect("preflight accepted this input");
    assert!(fresh.is_unique());
    let window = fresh.range(bounds.clone());
    assert_eq!(window.as_bytes(), &CONTENTS[bounds.clone()]);
    if fresh.as_bytes().as_ptr() == fresh_pointer {
      assert_eq!(
        window.as_bytes().as_ptr(),
        fresh_pointer.wrapping_add(bounds.start)
      );
    }
    drop(window);
    drop(fresh);

    // Inline storage may copy; an adopted vector must retain all its raw parts.
    let values = input(len, capacity);
    let pointer = values.as_ptr();
    let unique = T::from_vec(values).expect("preflight accepted this input");
    let adopted = unique.as_bytes().as_ptr() == pointer;
    assert!(unique.is_unique());
    let mut slot = (Some(unique), [0u8; PAD]);
    let values = T::into_vec(slot.0.take().expect("unique input"));
    assert_eq!(values, &CONTENTS[..len]);
    assert_eq!(values.len(), len);
    assert!(values.capacity() >= len);
    if adopted {
      assert_eq!(values.as_ptr(), pointer);
      assert_eq!(values.capacity(), capacity);
    }

    bench(group, "from_vec", label, len, capacity, |b| {
      b.iter_batched(
        || input(len, capacity),
        |values| {
          black_box((
            T::from_vec(black_box(values)).expect("representable input"),
            [0u8; PAD],
          ))
        },
        BATCH,
      );
    });
    bench(group, "first_clone", label, len, capacity, |b| {
      b.iter_batched_ref(
        || {
          (
            T::from_vec(input(len, capacity)).expect("representable input"),
            [0u8; PAD],
          )
        },
        |input| black_box((T::clone(black_box(&input.0)), [0u8; PAD])),
        BATCH,
      );
    });
    bench(group, "warmed_clone", label, len, capacity, |b| {
      b.iter_batched_ref(
        || &source,
        |source| black_box((T::clone(black_box(*source)), [0u8; PAD])),
        BATCH,
      );
    });
    bench(group, "first_slice", label, len, capacity, |b| {
      b.iter_batched_ref(
        || {
          (
            T::from_vec(input(len, capacity)).expect("representable input"),
            [0u8; PAD],
          )
        },
        |input| {
          black_box((
            T::range(
              black_box(&input.0),
              black_box(bounds.start)..black_box(bounds.end),
            ),
            [0u8; PAD],
          ))
        },
        BATCH,
      );
    });
    bench(group, "warmed_slice", label, len, capacity, |b| {
      b.iter_batched_ref(
        || &source,
        |source| {
          black_box((
            T::range(
              black_box(*source),
              black_box(bounds.start)..black_box(bounds.end),
            ),
            [0u8; PAD],
          ))
        },
        BATCH,
      );
    });
    bench(group, "consume", label, len, capacity, |b| {
      b.iter_batched_ref(
        || {
          (
            Some(T::from_vec(input(len, capacity)).expect("representable input")),
            [0u8; PAD],
          )
        },
        |slot| black_box(T::into_vec(black_box(slot.0.take().expect("unique input")))),
        BATCH,
      );
    });
    black_box(&peer);
  }

  fn assert_layout<const N: usize>()
  where
    [u8; N]: AssocInlineSize,
  {
    let expected = if N == 0 { size_of::<Bytes>() } else { N };
    assert_eq!(size_of::<VectorN<u8, N>>(), expected);
    assert_eq!(size_of::<Option<VectorN<u8, N>>>(), expected);
  }

  fn budget<const N: usize, const PAD: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
  where
    [u8; N]: AssocInlineSize,
  {
    let label = format!("n_{N}_handle_{}", size_of::<VectorN<u8, N>>());
    for (len, capacity) in [(7, 7), (64, 64), (64, 65), (64, 128)] {
      cases::<VectorN<u8, N>, PAD>(group, &label, len, capacity);
    }
  }

  pub fn benchmarks(c: &mut Criterion) {
    macro_rules! each_budget {
      ($action:ident $(, $arg:ident)?) => {
        $action!($($arg)?; 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128);
        #[cfg(target_pointer_width = "32")]
        $action!($($arg)?; 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124);
      };
    }
    macro_rules! check_sizes {
      (; $($n:literal),+ $(,)?) => { $(assert_layout::<$n>();)+ };
    }
    macro_rules! register_native_sizes {
      ($group:ident; $($n:literal),+ $(,)?) => {
        $(budget::<$n, 0>(&mut $group);)+
      };
    }
    assert_eq!(size_of::<Bytes>(), 4 * size_of::<usize>());
    each_budget!(check_sizes);
    let mut group = c.benchmark_group("stages_native_stride");
    each_budget!(register_native_sizes, group);
    for (len, capacity) in [(7, 7), (64, 64), (64, 65), (64, 128)] {
      cases::<Bytes, 0>(&mut group, "bytes", len, capacity);
    }
    group.finish();
  }
}

fn benchmarks(c: &mut Criterion) {
  #[cfg(bytora_atomic)]
  ownership::benchmarks(c);
  #[cfg(not(bytora_atomic))]
  let _ = c;
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
