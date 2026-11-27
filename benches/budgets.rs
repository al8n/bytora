use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(bytora_atomic)]
mod ownership {
  use bytes::Bytes;
  use bytora::{
    AssocInlineSize, DefaultSize, InlineSize, LocalVector, Vector, VectorMut, VectorN,
    typenum::Unsigned,
  };
  use criterion::{
    BatchSize, Bencher, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime,
  };
  use std::{hint::black_box, mem::size_of, ops::Range, rc::Rc};

  // A timed result is written to a stack temporary or an output slot whose alignment would
  // otherwise be only the ABI's; on some cores its address modulo 32 selects the speed of the
  // timed loop. Wrapping each result pins it to a 32-byte boundary, for every handle alike.
  #[repr(C, align(32))]
  struct Aligned<T>(T);

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
      self.try_into_vec().expect("unique vector")
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

  // `N = 0`'s own constructor, `From<Vec<T>>`, which reaches `adopt_wide_vec` as `try_from_vec`
  // does. The row keeps the name it had when the constructor was `TryFrom<Vec<T>>`. `Bytes` has no
  // fallible or shared-vector constructor: its nearest equivalent is `Bytes::from(Vec<u8>)`, the
  // same body as its `from_vec`.
  trait Adopt: Handle {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>>;
  }

  impl Adopt for Vector<u8, bytora::typenum::U0> {
    fn try_from_vec(values: Vec<u8>) -> Result<Self, Vec<u8>> {
      Ok(Self::from(values))
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
          black_box(Aligned((
            T::from_vec(black_box(values)).expect("representable input"),
            [0u8; PAD],
          )))
        },
        BATCH,
      );
    });
    bench(group, "warmed_clone", label, len, |b| {
      b.iter_batched_ref(
        || &source,
        |source| black_box(Aligned((T::clone(black_box(*source)), [0u8; PAD]))),
        BATCH,
      );
    });
    bench(group, "range", label, len, |b| {
      b.iter_batched_ref(
        || &source,
        |source| {
          black_box(Aligned((
            T::range(
              black_box(*source),
              black_box(bounds.start)..black_box(bounds.end),
            ),
            [0u8; PAD],
          )))
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
        |slot| {
          black_box(Aligned(T::into_vec(black_box(
            slot.0.take().expect("unique input"),
          ))))
        },
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
          black_box(Aligned((
            T::try_from_vec(black_box(values)).expect("representable input"),
            [0u8; PAD],
          )))
        },
        BATCH,
      );
    });
  }

  // `N = 0` and `Bytes` are the same size, so one padding serves both.
  fn adoption<const PAD: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let label = format!("n_0_handle_{}", size_of::<VectorN<u8, 0>>());
    for len in [0, 7, 64] {
      adopt_cases::<VectorN<u8, 0>, PAD>(group, &label, len);
      adopt_cases::<Bytes, PAD>(group, "bytes", len);
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

  fn budget<const N: usize, const PAD: usize>(group: &mut BenchmarkGroup<'_, WallTime>)
  where
    [u8; N]: AssocInlineSize,
  {
    let label = format!("n_{N}_handle_{}", size_of::<VectorN<u8, N>>());
    for len in [7, 64] {
      cases::<VectorN<u8, N>, PAD>(group, &label, len);
    }
    if N == 0 {
      cases::<VectorN<u8, N>, PAD>(group, &label, 0);
    }
    if N == size_of::<usize>() {
      cases::<VectorN<u8, N>, PAD>(group, &label, 3);
    }
  }

  // The `try_*` twins against their panicking forms.
  //
  // One pair of rows per twin, on its success path: the panicking form and then the twin, with the
  // same input. An id's operation is `{form}/{method}`, `{form}` being the name of the panicking
  // method, so that the summary prints the two rows of a pair in adjacent columns. A twin's result
  // is unwrapped with `expect`. A group times one handle type at the default budget, and every
  // input holds `TWIN_LEN` elements: a shared heap window for `Vector` and `LocalVector`, and for
  // `VectorMut` a heap buffer with room for as many again, so that no append grows it.
  const TWIN_LEN: usize = 64;
  const TWIN_AT: usize = TWIN_LEN / 2;
  const TWIN_RANGE: Range<usize> = 1..TWIN_LEN - 1;
  static TWIN_BYTES: [u8; TWIN_LEN] = CONTENTS;

  fn twin_label<H>() -> String {
    format!(
      "n_{}_handle_{}",
      <DefaultSize as Unsigned>::USIZE,
      size_of::<H>()
    )
  }

  fn twin(
    group: &mut BenchmarkGroup<'_, WallTime>,
    form: &str,
    method: &str,
    label: &str,
    routine: impl FnMut(&mut Bencher<'_>),
  ) {
    bench(group, &format!("{form}/{method}"), label, TWIN_LEN, routine);
  }

  // The timed closure and every row's operation carry `#[inline(always)]`. Criterion calls the
  // routine from two sites, so without the pin the compiler inlines a closure or leaves it out
  // by its size, and a twin's closure, larger by its `expect` path, is left out before its
  // panicking form's is. A pair then differs by one call, which a 4 ns row shows as tens of
  // percent.
  //
  // Times `operation` on a borrowed source, as `range` does.
  fn on_source<T: ?Sized, O>(b: &mut Bencher<'_>, source: &T, mut operation: impl FnMut(&T) -> O) {
    b.iter_batched_ref(
      || source,
      #[inline(always)]
      |source| black_box(Aligned(operation(black_box(*source)))),
      BATCH,
    );
  }

  // Times `operation` on a fresh input from `setup`. The input, like the output, drops after the
  // timing, as in `consume`.
  fn on_input<I, O>(
    b: &mut Bencher<'_>,
    setup: impl FnMut() -> I,
    mut operation: impl FnMut(&mut I) -> O,
  ) {
    b.iter_batched_ref(
      setup,
      #[inline(always)]
      |input| black_box(Aligned(operation(black_box(input)))),
      BATCH,
    );
  }

  fn vector_twins(c: &mut Criterion) {
    let label = twin_label::<Vector<u8>>();
    let bytes: &[u8] = &TWIN_BYTES;
    let values = input(TWIN_LEN);
    let pointer = values.as_ptr();
    let source = Vector::<u8>::try_from_vec(values).expect("the default budget adopts a vector");
    assert_eq!(source.as_slice().as_ptr(), pointer);
    // The first clone shares the allocation, so that no row pays for it.
    let peer = source.clone();
    let subset = &source.as_slice()[TWIN_RANGE];
    let mut group = c.benchmark_group("budgets_twins_vector");
    twin(&mut group, "advance", "advance", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.advance(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "advance", "try_advance", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.try_advance(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(
      &mut group,
      "copy_from_slice",
      "copy_from_slice",
      &label,
      |b| {
        on_source(
          b,
          bytes,
          #[inline(always)]
          |values| Vector::<u8>::copy_from_slice(values),
        );
      },
    );
    twin(
      &mut group,
      "copy_from_slice",
      "try_copy_from_slice",
      &label,
      |b| {
        on_source(
          b,
          bytes,
          #[inline(always)]
          |values| {
            Vector::<u8>::try_copy_from_slice(values).expect("shared storage holds the bytes")
          },
        );
      },
    );
    twin(&mut group, "slice", "slice", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| source.slice(black_box(TWIN_RANGE.start)..black_box(TWIN_RANGE.end)),
      );
    });
    twin(&mut group, "slice", "try_slice", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| {
          source
            .try_slice(black_box(TWIN_RANGE.start)..black_box(TWIN_RANGE.end))
            .expect("in bounds")
        },
      );
    });
    twin(&mut group, "slice_ref", "slice_ref", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| source.slice_ref(black_box(subset)),
      );
    });
    twin(&mut group, "slice_ref", "try_slice_ref", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| {
          source
            .try_slice_ref(black_box(subset))
            .expect("inside the source")
        },
      );
    });
    twin(&mut group, "split_off", "split_off", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.split_off(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "split_off", "try_split_off", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.try_split_off(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(&mut group, "split_to", "split_to", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.split_to(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "split_to", "try_split_to", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.try_split_to(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    group.finish();
    black_box(&peer);
  }

  // A heap buffer of `TWIN_LEN` bytes with room for as many again.
  fn twin_buffer() -> VectorMut<u8> {
    let mut values = Vec::with_capacity(2 * TWIN_LEN);
    values.extend_from_slice(&TWIN_BYTES);
    VectorMut::<u8>::try_from_vec(values).expect("the default budget adopts a vector")
  }

  // `twin_buffer` with the range to copy beside it. The bounds come from the batch, not from two
  // `black_box` slots in the timed closure. Two adjacent slots are reloaded as one pair, and an
  // Apple core stalls on a pair that straddles a 16-byte boundary behind its two stores, which
  // made the two forms differ by where the compiler happened to put the slots.
  fn twin_range_buffer() -> (VectorMut<u8>, Range<usize>) {
    (twin_buffer(), black_box(0..TWIN_LEN))
  }

  fn vector_mut_twins(c: &mut Criterion) {
    let label = twin_label::<VectorMut<u8>>();
    let bytes: &[u8] = &TWIN_BYTES;
    let buffer = twin_buffer();
    assert_eq!(buffer.as_slice(), bytes);
    assert!(buffer.capacity() >= 2 * TWIN_LEN);
    drop(buffer);
    let mut group = c.benchmark_group("budgets_twins_vector_mut");
    twin(&mut group, "advance", "advance", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.advance(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "advance", "try_advance", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.try_advance(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(&mut group, "extend", "extend", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.extend(black_box(bytes).iter().copied()),
      );
    });
    twin(&mut group, "extend", "try_extend", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| {
          buffer
            .try_extend(black_box(bytes).iter().copied())
            .expect("room for the bytes")
        },
      );
    });
    twin(
      &mut group,
      "extend_from_slice",
      "extend_from_slice",
      &label,
      |b| {
        on_input(
          b,
          twin_buffer,
          #[inline(always)]
          |buffer| buffer.extend_from_slice(black_box(bytes)),
        );
      },
    );
    twin(
      &mut group,
      "extend_from_slice",
      "try_extend_from_slice",
      &label,
      |b| {
        on_input(
          b,
          twin_buffer,
          #[inline(always)]
          |buffer| {
            buffer
              .try_extend_from_slice(black_box(bytes))
              .expect("room for the bytes")
          },
        );
      },
    );
    twin(
      &mut group,
      "extend_from_within",
      "extend_from_within",
      &label,
      |b| {
        on_input(
          b,
          twin_range_buffer,
          #[inline(always)]
          |(buffer, range)| buffer.extend_from_within(range.clone()),
        );
      },
    );
    twin(
      &mut group,
      "extend_from_within",
      "try_extend_from_within",
      &label,
      |b| {
        on_input(
          b,
          twin_range_buffer,
          #[inline(always)]
          |(buffer, range)| {
            buffer
              .try_extend_from_within(range.clone())
              .expect("room for the bytes")
          },
        );
      },
    );
    twin(&mut group, "reserve", "reserve", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.reserve(black_box(TWIN_LEN)),
      );
    });
    twin(&mut group, "reserve", "try_reserve", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| {
          buffer
            .try_reserve(black_box(TWIN_LEN))
            .expect("room for the bytes")
        },
      );
    });
    twin(&mut group, "resize", "resize", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.resize(black_box(2 * TWIN_LEN), black_box(7)),
      );
    });
    twin(&mut group, "resize", "try_resize", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| {
          buffer
            .try_resize(black_box(2 * TWIN_LEN), black_box(7))
            .expect("room for the bytes")
        },
      );
    });
    twin(&mut group, "split_off", "split_off", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.split_off(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "split_off", "try_split_off", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.try_split_off(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(&mut group, "split_to", "split_to", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.split_to(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "split_to", "try_split_to", &label, |b| {
      on_input(
        b,
        twin_buffer,
        #[inline(always)]
        |buffer| buffer.try_split_to(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(&mut group, "with_capacity", "with_capacity", &label, |b| {
      on_input(
        b,
        || (),
        #[inline(always)]
        |_| VectorMut::<u8>::with_capacity(black_box(TWIN_LEN)),
      );
    });
    twin(
      &mut group,
      "with_capacity",
      "try_with_capacity",
      &label,
      |b| {
        on_input(
          b,
          || (),
          #[inline(always)]
          |_| VectorMut::<u8>::try_with_capacity(black_box(TWIN_LEN)).expect("allocates"),
        );
      },
    );
    twin(&mut group, "zeroed", "zeroed", &label, |b| {
      on_input(
        b,
        || (),
        #[inline(always)]
        |_| VectorMut::<u8>::zeroed(black_box(TWIN_LEN)),
      );
    });
    twin(&mut group, "zeroed", "try_zeroed", &label, |b| {
      on_input(
        b,
        || (),
        #[inline(always)]
        |_| VectorMut::<u8>::try_zeroed(black_box(TWIN_LEN)).expect("allocates"),
      );
    });
    group.finish();
  }

  fn local_vector_twins(c: &mut Criterion) {
    let label = twin_label::<LocalVector<u8>>();
    let source = LocalVector::<u8>::from_rc(Rc::from(&TWIN_BYTES[..]));
    assert_eq!(source.as_slice(), &TWIN_BYTES[..]);
    let mut group = c.benchmark_group("budgets_twins_local_vector");
    twin(&mut group, "advance", "advance", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.advance(black_box(TWIN_AT)),
      );
    });
    twin(&mut group, "advance", "try_advance", &label, |b| {
      on_input(
        b,
        || source.clone(),
        #[inline(always)]
        |handle| handle.try_advance(black_box(TWIN_AT)).expect("in bounds"),
      );
    });
    twin(&mut group, "slice", "slice", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| source.slice(black_box(TWIN_RANGE.start)..black_box(TWIN_RANGE.end)),
      );
    });
    twin(&mut group, "slice", "try_slice", &label, |b| {
      on_source(
        b,
        &source,
        #[inline(always)]
        |source| {
          source
            .try_slice(black_box(TWIN_RANGE.start)..black_box(TWIN_RANGE.end))
            .expect("in bounds")
        },
      );
    });
    group.finish();
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
        $(budget::<$n, { STRIDE - size_of::<VectorN<u8, $n>>() }>(&mut $group);)+
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

    vector_twins(c);
    vector_mut_twins(c);
    local_vector_twins(c);
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
