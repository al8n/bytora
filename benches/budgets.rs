use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(bytora_atomic)]
mod ownership {
  use bytes::Bytes;
  use bytora::{AssocInlineSize, InlineSize, Slice, SliceN};
  use criterion::{
    BatchSize, Bencher, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime,
  };
  use std::{hint::black_box, mem::size_of, ops::Range};

  // Padded control uses the same input Option and handle-output stride.
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

  fn input(len: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(len);
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

  impl<S: InlineSize> Handle for Slice<u8, S> {
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

  // `N = 0`'s own constructor, `TryFrom<Vec<T>>`, which reaches `adopt_vec` through
  // `try_from_shared_vec`. `Bytes` has no fallible or shared-vector constructor: its nearest
  // equivalent is `Bytes::from(Vec<u8>)`, the same body as its `from_vec`.
  trait Adopt: Handle {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
  }

  impl Adopt for Slice<u8, bytora::typenum::U0> {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      <Self as TryFrom<Vec<u8>>>::try_from(values)
    }
  }

  impl Adopt for Bytes {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from(values))
    }
  }

  fn bench(
    group: &mut BenchmarkGroup<'_, WallTime>,
    operation: &str,
    label: &str,
    len: usize,
    routine: impl FnMut(&mut Bencher<'_>),
  ) {
    group.bench_function(
      BenchmarkId::new(format!("{operation}/{label}"), format!("len_{len}")),
      routine,
    );
  }

  fn cases<T: Handle, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    len: usize,
  ) {
    let stride = if PAD == 0 { size_of::<T>() } else { STRIDE };
    assert_eq!(size_of::<(T, [u8; PAD])>(), stride);
    assert_eq!(size_of::<(Option<T>, [u8; PAD])>(), stride);
    let values = input(len);
    let pointer = values.as_ptr();
    assert_eq!(values.capacity(), len);
    let source = match T::from_vec(values) {
      Ok(source) => source,
      Err(values) => {
        assert_eq!(values, &CONTENTS[..len]);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.capacity(), len);
        eprintln!("budgets: {label}/len_{len} unrepresentable; skipped");
        return;
      }
    };
    assert_eq!(source.as_bytes(), &CONTENTS[..len]);
    if len != 0 {
      assert!(source.is_unique());
    }
    let peer = source.clone();
    assert_eq!(peer.as_bytes(), &CONTENTS[..len]);
    let bounds = if len > 1 { 1..len - 1 } else { 0..len };
    let window = source.range(bounds.clone());
    assert_eq!(window.as_bytes(), &CONTENTS[bounds.clone()]);
    drop(window);

    // Separately check the unique consuming path and retained allocation when
    // construction adopted the Vec; inline storage is equally valid here.
    let values = input(len);
    let pointer = values.as_ptr();
    let unique = T::from_vec(values).expect("preflight accepted this input");
    let adopted = len != 0 && unique.as_bytes().as_ptr() == pointer;
    if len != 0 {
      assert!(unique.is_unique());
    }
    let mut slot = (Some(unique), [0u8; PAD]);
    let values = T::into_vec(slot.0.take().expect("unique input"));
    assert_eq!(values, &CONTENTS[..len]);
    assert!(values.capacity() >= len);
    if adopted {
      assert_eq!(values.as_ptr(), pointer);
      assert_eq!(values.capacity(), len);
    }

    bench(group, "from_vec", label, len, |b| {
      b.iter_batched(
        || input(len),
        |values| {
          black_box((
            T::from_vec(black_box(values)).expect("representable input"),
            [0u8; PAD],
          ))
        },
        BATCH,
      );
    });
    bench(group, "warmed_clone", label, len, |b| {
      b.iter_batched_ref(
        || &source,
        |source| black_box((T::clone(black_box(*source)), [0u8; PAD])),
        BATCH,
      );
    });
    bench(group, "range", label, len, |b| {
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
    bench(group, "consume", label, len, |b| {
      b.iter_batched_ref(
        || {
          (
            Some(T::from_vec(input(len)).expect("representable input")),
            [0u8; PAD],
          )
        },
        |slot| black_box(T::into_vec(black_box(slot.0.take().expect("unique input")))),
        BATCH,
      );
    });
    black_box(&peer);
  }

  fn adopt_cases<T: Adopt, const PAD: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    len: usize,
  ) {
    let stride = if PAD == 0 { size_of::<T>() } else { STRIDE };
    assert_eq!(size_of::<(T, [u8; PAD])>(), stride);
    assert_eq!(size_of::<(Option<T>, [u8; PAD])>(), stride);
    let source = match T::try_from_vec(input(len)) {
      Ok(source) => source,
      Err(values) => {
        assert_eq!(values, &CONTENTS[..len]);
        eprintln!("budgets: try_from_vec {label}/len_{len} unrepresentable; skipped");
        return;
      }
    };
    assert_eq!(source.as_bytes(), &CONTENTS[..len]);
    if len != 0 {
      assert!(source.is_unique());
    }
    drop(source);

    bench(group, "try_from_vec", label, len, |b| {
      b.iter_batched(
        || input(len),
        |values| {
          black_box((
            T::try_from_vec(black_box(values)).expect("representable input"),
            [0u8; PAD],
          ))
        },
        BATCH,
      );
    });
  }

  // `N = 0` and `Bytes` are the same size, so one padding serves both.
  fn adoption<const PAD: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let label = format!("n_0_handle_{}", size_of::<SliceN<u8, 0>>());
    for len in [0, 7, 64] {
      adopt_cases::<SliceN<u8, 0>, PAD>(group, &label, len);
      adopt_cases::<Bytes, PAD>(group, "bytes", len);
    }
  }

  fn assert_layout<const N: usize>()
  where
    [u8; N]: AssocInlineSize,
  {
    let expected = if N == 0 { size_of::<Bytes>() } else { N };
    assert_eq!(size_of::<SliceN<u8, N>>(), expected);
    assert_eq!(size_of::<Option<SliceN<u8, N>>>(), expected);
  }

  fn budget<const N: usize, const PAD: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
  where
    [u8; N]: AssocInlineSize,
  {
    let label = format!("n_{N}_handle_{}", size_of::<SliceN<u8, N>>());
    for len in [7, 64] {
      cases::<SliceN<u8, N>, PAD>(group, &label, len);
    }
    if N == 0 {
      cases::<SliceN<u8, N>, PAD>(group, &label, 0);
    }
    if N == size_of::<usize>() {
      cases::<SliceN<u8, N>, PAD>(group, &label, 3);
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
    macro_rules! register_sizes {
      ($group:ident; $($n:literal),+ $(,)?) => {
        $(budget::<$n, { STRIDE - size_of::<SliceN<u8, $n>>() }>(&mut $group);)+
      };
    }
    macro_rules! register_native_sizes {
      ($group:ident; $($n:literal),+ $(,)?) => {
        $(budget::<$n, 0>(&mut $group);)+
      };
    }
    // Validate every physical layout before the first timing is started.
    each_budget!(check_sizes);
    let mut group = c.benchmark_group("budgets_padded_128");
    each_budget!(register_sizes, group);
    for len in [0, 3, 7, 64] {
      cases::<Bytes, { STRIDE - size_of::<Bytes>() }>(&mut group, "bytes", len);
    }
    adoption::<{ STRIDE - size_of::<Bytes>() }>(&mut group);
    group.finish();

    // Native stride exposes real array density instead of padding all handles.
    let mut group = c.benchmark_group("budgets_native_stride");
    each_budget!(register_native_sizes, group);
    for len in [0, 3, 7, 64] {
      cases::<Bytes, 0>(&mut group, "bytes", len);
    }
    adoption::<0>(&mut group);
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
