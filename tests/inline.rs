use smol_slice::{
  InlineSize, MAX_HANDLE_BYTES, Size, Slice, SliceMut, SliceMutOf, SliceOf, SupportedBytes,
};

#[test]
fn generic_size_witnesses_work_with_both_handles() {
  fn read<T: Copy, S: InlineSize>(values: &[T]) -> Option<SliceOf<T, S>> {
    assert_eq!(size_of::<SliceOf<T, S>>(), S::BYTES);
    SliceOf::try_from_slice(values)
  }

  fn write<T: Copy, S: InlineSize>(values: &[T]) -> Option<SliceMutOf<T, S>> {
    assert_eq!(size_of::<SliceMutOf<T, S>>(), S::BYTES);
    SliceMutOf::try_from_slice(values)
  }

  fn read_with_const<const BYTES: usize>(values: &[u8]) -> Option<Slice<u8, BYTES>>
  where
    Size<BYTES>: InlineSize,
  {
    read::<u8, Size<BYTES>>(values)
  }

  let mut read: Slice<u8, 24> = read::<u8, Size<24>>(b"abc").unwrap();
  read.advance(1);
  let read: SliceOf<u8, Size<24>> = read;
  assert_eq!(read.as_slice(), b"bc");

  let mut write: SliceMutOf<u8, Size<24>> = write::<u8, Size<24>>(b"abc").unwrap();
  write.advance(1);
  write.as_mut_slice().make_ascii_uppercase();
  let write: SliceMut<u8, 24> = write;
  assert_eq!(write.as_slice(), b"BC");
  assert_eq!(read_with_const::<24>(b"size").unwrap().as_slice(), b"size");

  let numeric: Slice<u8> =
    SliceOf::<u8, Size<{ 3 * size_of::<usize>() }>>::try_from_slice(b"a").unwrap();
  let generic: SliceOf<u8, Size<{ 3 * size_of::<usize>() }>> = numeric;
  let numeric_mut: SliceMut<u8> =
    SliceMutOf::<u8, Size<{ 3 * size_of::<usize>() }>>::try_from_slice(b"a").unwrap();
  let generic_mut: SliceMutOf<u8, Size<{ 3 * size_of::<usize>() }>> = numeric_mut;
  assert_eq!(generic.as_slice(), b"a");
  assert_eq!(generic_mut.as_slice(), b"a");
}

#[test]
fn supported_alignment_type_is_accessible_to_consumers() {
  fn word_aligned<B: SupportedBytes<Align = usize>>() {}
  word_aligned::<[(); 8]>();
  word_aligned::<[(); 16]>();
  word_aligned::<[(); 2056]>();
  word_aligned::<[(); 4104]>();
  word_aligned::<[(); MAX_HANDLE_BYTES]>();
  #[cfg(target_pointer_width = "32")]
  {
    fn four_byte_aligned<B: SupportedBytes<Align = u32>>() {}
    four_byte_aligned::<[(); 4]>();
    word_aligned::<[(); 12]>();
  }
  assert_eq!(MAX_HANDLE_BYTES, 8192);
}

#[test]
fn supported_bytes_remains_an_object_safe_marker() {
  let witness: Size<24> = [(); 24];
  let _: &dyn SupportedBytes<Align = usize, Payload = [u8; 22], Cursor = [u8; 1]> = &witness;

  trait Bytes {
    const BYTES: usize;
  }
  impl Bytes for Size<24> {
    const BYTES: usize = 7;
  }
  fn legacy_bytes<S: SupportedBytes + Bytes>() -> usize {
    S::BYTES
  }
  assert_eq!(legacy_bytes::<Size<24>>(), 7);
}

#[test]
fn downstream_transparent_types_use_the_same_layout_as_u16() {
  #[repr(transparent)]
  #[derive(Clone, Copy, Debug, Eq, PartialEq)]
  struct Wrapped(u16);

  #[repr(transparent)]
  struct NonCopy(u16);

  fn check_layout<T>() {
    assert_eq!(size_of::<Slice<T, MAX_HANDLE_BYTES>>(), MAX_HANDLE_BYTES);
    assert_eq!(
      size_of::<Option<Slice<T, MAX_HANDLE_BYTES>>>(),
      MAX_HANDLE_BYTES
    );
    assert_eq!(size_of::<SliceMut<T, MAX_HANDLE_BYTES>>(), MAX_HANDLE_BYTES);
    assert_eq!(
      size_of::<Option<SliceMut<T, MAX_HANDLE_BYTES>>>(),
      MAX_HANDLE_BYTES
    );
    assert_eq!(
      align_of::<Slice<T, MAX_HANDLE_BYTES>>(),
      align_of::<usize>()
    );
    assert_eq!(
      align_of::<SliceMut<T, MAX_HANDLE_BYTES>>(),
      align_of::<usize>()
    );
  }

  check_layout::<u16>();
  check_layout::<Wrapped>();
  check_layout::<NonCopy>();
  assert_eq!(NonCopy(7).0, 7);
  assert_eq!(Slice::<u16, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(Slice::<Wrapped, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(SliceMut::<Wrapped, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(Slice::<NonCopy, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(SliceMut::<NonCopy, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);

  let source = core::array::from_fn::<_, 4095, _>(|i| Wrapped(i as u16));
  let snapshot = source;
  assert!(Slice::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  assert!(SliceMut::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  let mut read = Slice::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source[..4094]).unwrap();
  let mut write = SliceMut::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source[..4094]).unwrap();
  read.advance(1);
  write.advance(1024);
  let read = core::hint::black_box(Some(read)).unwrap();
  let mut write = core::hint::black_box(Some(write)).unwrap();
  assert_eq!(read.as_slice(), &snapshot[1..4094]);
  assert_eq!(write.as_slice(), &snapshot[1024..4094]);
  write.as_mut_slice()[0] = Wrapped(u16::MAX);
  assert_eq!(write.as_slice()[0], Wrapped(u16::MAX));
  assert_eq!(read.as_slice(), &snapshot[1..4094]);
  assert_eq!(source, snapshot);
}

#[test]
fn handles_own_their_copied_elements() {
  let mut source = [2, 3, 5, 7];
  let mut read = Slice::<u32, 24>::try_from_slice(&source).unwrap();
  let mut write = SliceMut::<u32, 24>::try_from_slice(&source).unwrap();
  source.fill(0);
  read.advance(2);
  write.advance(1);
  write.as_mut_slice()[1] = 11;
  assert_eq!(read.as_slice(), &[5, 7]);
  assert_eq!(write.as_slice(), &[3, 11, 7]);
}

#[test]
fn failed_advances_leave_the_window_intact() {
  fn check<const BYTES: usize>(source: &[u8], start: usize)
  where
    [(); BYTES]: SupportedBytes,
  {
    let mut values = SliceMut::<u8, BYTES>::try_from_slice(source).unwrap();
    values.advance(start);
    for count in [values.len() + 1, usize::MAX] {
      assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.advance(count))).is_err()
      );
      assert_eq!(values.as_mut_slice(), &source[start..]);
    }
  }
  check::<8>(b"abcdefg", 4);
  check::<24>(b"abcdefg", 4);
  let wide: [u8; 521] = core::array::from_fn(|i| i as u8);
  for start in [254, 255, 256, 257, 511, 512, 521] {
    check::<536>(&wide, start);
  }
}
