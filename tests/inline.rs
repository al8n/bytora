use smol_slice::{
  AssocInlineSize, InlineSize, MAX_HANDLE_BYTES, Slice, SliceMut, SliceMutN, SliceN,
  typenum::{Sum, U0, U8, U24, U4096, Unsigned},
};

#[test]
fn generic_sizes_work_with_both_handles() {
  fn size<S: Unsigned + core::fmt::Debug>() -> usize {
    S::USIZE
  }
  fn read<T: Copy, S: InlineSize>(values: &[T]) -> Option<Slice<T, S>> {
    assert_eq!(size_of::<Slice<T, S>>(), S::HANDLE_BYTES);
    Slice::try_from_slice(values)
  }
  fn write<T: Copy, S: InlineSize>(values: &[T]) -> Option<SliceMut<T, S>> {
    assert_eq!(size_of::<SliceMut<T, S>>(), S::HANDLE_BYTES);
    SliceMut::try_from_slice(values)
  }

  assert_eq!(size::<U0>(), 0);
  assert_eq!(size::<U24>(), 24);
  assert!(read::<u8, U0>(&[]).unwrap().is_empty());
  assert!(write::<u8, U0>(&[]).unwrap().is_empty());
  let mut read: SliceN<u8, 24> = read::<_, U24>(b"abc").unwrap();
  read.advance(1);
  let read: Slice<u8, U24> = read;
  assert_eq!(read.as_slice(), b"bc");

  let mut write: SliceMutN<u8, 24> = write::<_, U24>(b"abc").unwrap();
  write.advance(1);
  write.as_mut_slice().make_ascii_uppercase();
  let write: SliceMut<u8, U24> = write;
  assert_eq!(write.as_slice(), b"BC");
}

#[test]
fn zero_budget_has_no_inline_elements_and_an_aligned_const_empty_window() {
  #[repr(align(128))]
  struct Aligned(core::marker::PhantomData<core::cell::Cell<u8>>);
  let empty = const { SliceN::<Aligned, 0>::empty() };
  assert_eq!(SliceN::<Aligned, 0>::INLINE_CAPACITY, 0);
  assert_eq!(SliceN::<u8, 0>::INLINE_CAPACITY, 0);
  assert_eq!(SliceN::<(), 0>::INLINE_CAPACITY, 0);
  assert!(SliceN::<(), 0>::try_from_slice(&[()]).is_none());
  assert_eq!(size_of::<Slice<Aligned, U0>>(), 3 * size_of::<usize>());
  assert_eq!(size_of::<Option<SliceN<Aligned, 0>>>(), U0::HANDLE_BYTES);
  assert!(empty.clone().slice(..).is_empty());
  assert!(empty.as_slice().as_ptr().is_aligned());
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
#[test]
fn zero_budget_shares_the_original_atomics_across_threads() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  let counters = vec![AtomicUsize::new(3), AtomicUsize::new(7)];
  let ptr = counters.as_ptr();
  let values: Slice<AtomicUsize, U0> = SliceN::<AtomicUsize, 0>::try_from(counters).unwrap();
  assert_eq!(values.as_slice().as_ptr(), ptr);
  let mut tail = values.clone();
  tail.advance(1);
  std::thread::spawn(move || {
    assert_eq!(tail.slice(..1).as_slice()[0].fetch_add(5, Relaxed), 7);
  })
  .join()
  .unwrap();
  assert_eq!(values.as_slice()[0].load(Relaxed), 3);
  assert_eq!(values.as_slice()[1].load(Relaxed), 12);
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
#[test]
fn zero_budget_keeps_all_owned_elements_until_the_last_empty_window_drops() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  #[repr(align(64))]
  struct CountDrop<'a>(&'a AtomicUsize);
  impl Drop for CountDrop<'_> {
    fn drop(&mut self) {
      self.0.fetch_add(1, Relaxed);
    }
  }
  let drops = [const { AtomicUsize::new(0) }; 3];
  let owned: Vec<_> = drops.iter().map(CountDrop).collect();
  let ptr = owned.as_ptr();
  let whole = SliceN::<_, 0>::try_from(owned).ok().unwrap();
  assert_eq!(whole.as_slice().as_ptr(), ptr);
  let mut tail = whole.slice(1..);
  assert_eq!(tail.as_slice().as_ptr(), ptr.wrapping_add(1));
  tail.advance(2);
  let mut last = SliceN::<_, 0>::empty();
  last.clone_from(&tail);
  drop(whole);
  drop(tail);
  assert!(last.is_empty());
  assert_eq!(last.as_slice().as_ptr(), ptr.wrapping_add(3));
  assert!(drops.iter().all(|count| count.load(Relaxed) == 0));
  std::thread::scope(|scope| scope.spawn(move || drop(last)).join().unwrap());
  assert!(drops.iter().all(|count| count.load(Relaxed) == 1));
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
#[test]
fn zero_budget_owned_vectors_respect_the_external_length_limit() {
  let max = isize::MAX as usize;
  let mut largest = SliceN::<(), 0>::try_from(vec![(); max]).unwrap();
  assert_eq!(largest.len(), max);
  largest.advance(max - 1);
  assert_eq!(largest.clone().as_slice(), &[()]);
  let oversized = vec![(); max + 1];
  let original = (oversized.as_ptr(), oversized.len(), oversized.capacity());
  let returned = SliceN::<(), 0>::try_from(oversized).err().unwrap();
  assert_eq!(
    (returned.as_ptr(), returned.len(), returned.capacity()),
    original
  );
}

#[test]
fn const_sizes_match_typenum_arithmetic_across_the_full_range() {
  fn read<const N: usize>(values: &[u8]) -> Option<SliceN<u8, N>>
  where
    [u8; N]: AssocInlineSize,
  {
    Slice::try_from_slice(values)
  }

  // Typenum computes this size but does not name it or map it from a const.
  type Budget = Sum<U4096, U8>;
  let typed: Slice<u8, Budget> = read::<4104>(b"size").unwrap();
  let numeric: SliceN<u8, 4104> = typed;
  assert_eq!(numeric.as_slice(), b"size");
  let typed: SliceMut<u8, Budget> = SliceMutN::<u8, 4104>::try_from_slice(b"size").unwrap();
  let mut numeric: SliceMutN<u8, 4104> = typed;
  numeric.as_mut_slice()[0] = b'S';
  assert_eq!(numeric.as_slice(), b"Size");
  assert_eq!(size_of::<Option<Slice<u8, Budget>>>(), 4104);
  assert_eq!(size_of::<Option<SliceMut<u8, Budget>>>(), 4104);
}

#[test]
fn default_handles_have_the_three_word_size_type() {
  const THREE_WORDS: usize = 3 * size_of::<usize>();
  let read: Slice<u8> = SliceN::<u8, THREE_WORDS>::try_from_slice(b"a").unwrap();
  let write: SliceMut<u8> = SliceMutN::<u8, THREE_WORDS>::try_from_slice(b"a").unwrap();
  let read: SliceN<u8, THREE_WORDS> = read;
  let write: SliceMutN<u8, THREE_WORDS> = write;
  assert_eq!(read.as_slice(), b"a");
  assert_eq!(write.as_slice(), b"a");
}

#[test]
fn downstream_transparent_types_use_the_same_layout_as_u16() {
  #[repr(transparent)]
  #[derive(Clone, Copy, Debug, Eq, PartialEq)]
  struct Wrapped(u16);

  #[repr(transparent)]
  struct NonCopy(u16);

  fn check_layout<T>() {
    assert_eq!(size_of::<SliceN<T, MAX_HANDLE_BYTES>>(), MAX_HANDLE_BYTES);
    assert_eq!(
      size_of::<Option<SliceN<T, MAX_HANDLE_BYTES>>>(),
      MAX_HANDLE_BYTES
    );
    assert_eq!(
      size_of::<SliceMutN<T, MAX_HANDLE_BYTES>>(),
      MAX_HANDLE_BYTES
    );
    assert_eq!(
      size_of::<Option<SliceMutN<T, MAX_HANDLE_BYTES>>>(),
      MAX_HANDLE_BYTES
    );
    assert_eq!(
      align_of::<SliceN<T, MAX_HANDLE_BYTES>>(),
      align_of::<usize>()
    );
    assert_eq!(
      align_of::<SliceMutN<T, MAX_HANDLE_BYTES>>(),
      align_of::<usize>()
    );
  }

  check_layout::<u16>();
  check_layout::<Wrapped>();
  check_layout::<NonCopy>();
  assert_eq!(NonCopy(7).0, 7);
  assert_eq!(SliceN::<u16, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(SliceN::<Wrapped, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(
    SliceMutN::<Wrapped, MAX_HANDLE_BYTES>::INLINE_CAPACITY,
    4094
  );
  assert_eq!(SliceN::<NonCopy, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
  assert_eq!(
    SliceMutN::<NonCopy, MAX_HANDLE_BYTES>::INLINE_CAPACITY,
    4094
  );

  let source = core::array::from_fn::<_, 4095, _>(|i| Wrapped(i as u16));
  let snapshot = source;
  assert!(SliceN::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  assert!(SliceMutN::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  let mut read = SliceN::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source[..4094]).unwrap();
  let mut write = SliceMutN::<Wrapped, MAX_HANDLE_BYTES>::try_from_slice(&source[..4094]).unwrap();
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
  let mut read = SliceN::<u32, 24>::try_from_slice(&source).unwrap();
  let mut write = SliceMutN::<u32, 24>::try_from_slice(&source).unwrap();
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
    [u8; BYTES]: AssocInlineSize,
  {
    let mut values = SliceMutN::<u8, BYTES>::try_from_slice(source).unwrap();
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

#[test]
fn owned_ranges_validate_both_bounds_against_the_current_window() {
  use core::ops::Bound::{Excluded, Included, Unbounded};

  let inline = SliceN::<u8, 32>::try_from_slice(b"abcdef").unwrap();
  let external = SliceN::<u8, 32>::try_from_static(b"abcdef").unwrap();
  for mut source in [inline, external] {
    source.advance(2);
    for bounds in [
      (Included(3), Excluded(2)),
      (Included(5), Unbounded),
      (Unbounded, Included(4)),
      (Unbounded, Excluded(usize::MAX)),
      (Excluded(usize::MAX), Unbounded),
      (Unbounded, Included(usize::MAX)),
    ] {
      assert!(std::panic::catch_unwind(|| source.slice(bounds)).is_err());
      assert_eq!(source.as_slice(), b"cdef");
    }
    assert_eq!(source.slice(..).as_slice(), b"cdef");
    assert_eq!(source.slice(1..=2).as_slice(), b"de");
    assert!(source.slice((Excluded(3), Unbounded)).is_empty());
    assert!(source.slice(..0).is_empty());
  }
}
