use smol_slice::{InlineSize, Slice, SliceN};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn length_queries_are_const_for_static_windows_and_initialized_empty_handles() {
  const fn measure<T, S: InlineSize>(slice: &Slice<T, S>) -> (usize, bool) {
    (slice.len(), slice.is_empty())
  }
  static SHORT: Slice<u8> = Slice::from_static(b"xyz");
  static WIDE: SliceN<u8, 8192> = SliceN::<_, 8192>::try_from_static(b"wide").unwrap();
  static ZSTS: [(); isize::MAX as usize] = [(); isize::MAX as usize];
  static LARGE: SliceN<(), 0> = SliceN::<_, 0>::try_from_static(&ZSTS).unwrap();
  static TINY_EMPTY: SliceN<u8, 8> = SliceN::<_, 8>::new();
  static ZERO_EMPTY: SliceN<u8, 0> = SliceN::<_, 0>::empty();
  const fn lengths() -> [(usize, bool); 5] {
    [
      measure(&SHORT),
      measure(&WIDE),
      measure(&LARGE),
      measure(&TINY_EMPTY),
      measure(&ZERO_EMPTY),
    ]
  }
  #[cfg(not(smol_slice_atomic))]
  const LENGTHS: [(usize, bool); 5] = lengths();
  #[cfg(not(smol_slice_atomic))]
  let result = LENGTHS;
  #[cfg(smol_slice_atomic)]
  let result = lengths();
  const LOCAL_LEN: usize = {
    let handle = Slice::<u8>::from_static(b"abc");
    let len = handle.len();
    #[cfg(smol_slice_atomic)]
    core::mem::forget(handle);
    len
  };
  assert_eq!(LOCAL_LEN, 3);
  assert_eq!(
    result,
    [
      (3, false),
      (4, false),
      (isize::MAX as usize, false),
      (0, true),
      (0, true)
    ]
  );
}

#[test]
fn new_is_const_and_accepts_non_copy_elements_for_every_handle_size() {
  #[repr(align(64))]
  struct NonCopy(core::cell::Cell<u8>);

  let zero = const { SliceN::<NonCopy, 0>::new() };
  let tiny = const { SliceN::<NonCopy, 8>::new() };
  let default = const { Slice::<NonCopy>::new() };
  let wide = const { SliceN::<NonCopy, 64>::new() };

  for empty in [
    zero.as_slice(),
    tiny.as_slice(),
    default.as_slice(),
    wide.as_slice(),
  ] {
    assert!(empty.is_empty());
    assert!(empty.as_ptr().is_aligned());
  }
  assert!(SliceN::<NonCopy, 8>::default().is_empty());
  let value = NonCopy(core::cell::Cell::new(7));
  assert_eq!(value.0.get(), 7);
}

#[test]
fn slice_ref_accepts_subsets_and_rejects_nonempty_out_of_range_slices() {
  fn check<S: InlineSize>(data: Slice<u8, S>) {
    let view = data.as_slice();
    assert_eq!(data.slice_ref(&view[1..4]).as_slice(), b"bcd");
    assert!(data.slice_ref(&[]).is_empty());
    assert!(catch_unwind(AssertUnwindSafe(|| data.slice_ref(b"other"))).is_err());
  }

  check(Slice::<u8>::try_from_slice(b"abcde").unwrap());
  static VALUES: &[u8] = b"abcdef";
  let shared = Slice::from_static(&VALUES[1..5]);
  let inside = &VALUES[2..4];
  let result = shared.slice_ref(inside);
  assert_eq!(result.as_slice(), b"cd");
  assert_eq!(result.as_slice().as_ptr(), inside.as_ptr());
  assert!(catch_unwind(AssertUnwindSafe(|| shared.slice_ref(&VALUES[4..6]))).is_err());
  assert!(catch_unwind(AssertUnwindSafe(|| shared.slice_ref(&VALUES[..2]))).is_err());
}

#[test]
fn slice_ref_supports_non_copy_elements_and_element_offsets() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  static VALUES: [AtomicUsize; 4] = [
    AtomicUsize::new(3),
    AtomicUsize::new(5),
    AtomicUsize::new(7),
    AtomicUsize::new(11),
  ];
  let data = SliceN::<AtomicUsize, 0>::try_from_static(&VALUES).unwrap();
  let selected = &data.as_slice()[1..3];
  let result = data.slice_ref(selected);
  assert_eq!(result.len(), 2);
  assert_eq!(result.as_slice().as_ptr(), selected.as_ptr());
  assert_eq!(result.as_slice()[0].load(Relaxed), 5);
  assert_eq!(result.as_slice()[1].load(Relaxed), 7);
}

#[test]
fn slice_ref_uses_a_checked_prefix_for_zero_sized_elements() {
  #[derive(Clone, Copy)]
  #[repr(align(64))]
  struct Aligned;
  #[repr(C)]
  struct Holder {
    marker: Aligned,
    byte: u8,
  }

  let data = Slice::<Aligned>::try_from_slice(&[Aligned; 4]).unwrap();
  let view = data.as_slice();
  assert_eq!(data.slice_ref(&view[2..4]).len(), 2);
  let short = data.slice(..2);
  assert!(catch_unwind(AssertUnwindSafe(|| short.slice_ref(view))).is_err());

  let holder = Holder {
    marker: Aligned,
    byte: 1,
  };
  let unrelated = core::slice::from_ref(&holder.marker);
  assert_ne!(unrelated.as_ptr(), view.as_ptr());
  assert!(catch_unwind(AssertUnwindSafe(|| data.slice_ref(unrelated))).is_err());
  assert!(data.slice_ref(&unrelated[..0]).is_empty());
}

#[test]
fn splits_cover_all_boundaries_without_changing_a_rejected_request() {
  static VALUES: &[u8] = b"abcdef";
  for at in 0..=VALUES.len() {
    let mut head = Slice::from_static(VALUES);
    let tail = head.split_off(at);
    assert_eq!(head.as_slice(), &VALUES[..at]);
    assert_eq!(tail.as_slice(), &VALUES[at..]);
    if !head.is_empty() {
      assert_eq!(head.as_slice().as_ptr(), VALUES.as_ptr());
    }
    if !tail.is_empty() {
      assert_eq!(tail.as_slice().as_ptr(), VALUES[at..].as_ptr());
    }

    let mut tail = Slice::<u8>::try_from_slice(VALUES).unwrap();
    let head = tail.split_to(at);
    assert_eq!(head.as_slice(), &VALUES[..at]);
    assert_eq!(tail.as_slice(), &VALUES[at..]);
  }

  let mut data = Slice::from_static(VALUES);
  assert!(catch_unwind(AssertUnwindSafe(|| data.split_off(VALUES.len() + 1))).is_err());
  assert_eq!(data.as_slice(), VALUES);
  assert!(catch_unwind(AssertUnwindSafe(|| data.split_to(usize::MAX))).is_err());
  assert_eq!(data.as_slice(), VALUES);
}

#[test]
fn truncate_and_clear_follow_the_current_window() {
  let mut data = Slice::<u8>::try_from_slice(b"abcdef").unwrap();
  data.advance(2);
  data.truncate(usize::MAX);
  assert_eq!(data.as_slice(), b"cdef");
  data.truncate(2);
  assert_eq!(data.as_slice(), b"cd");
  data.clear();
  assert!(data.is_empty());
  data.truncate(0);
  assert!(data.is_empty());
}

#[test]
fn inline_truncation_preserves_advanced_windows_in_each_cursor_encoding() {
  fn check<S: InlineSize>(skip: usize, keep: usize) {
    let values: Vec<u8> = (0..Slice::<u8, S>::INLINE_CAPACITY)
      .map(|i| (i ^ (i >> 8)) as u8)
      .collect();
    let mut slice = Slice::<u8, S>::try_from_slice(&values).unwrap();
    slice.advance(skip);
    let start = slice.as_slice().as_ptr();
    slice.truncate(keep);
    slice.truncate(usize::MAX);
    assert_eq!(slice.as_slice(), &values[skip..skip + keep]);
    assert_eq!(slice.as_slice().as_ptr(), start);
    let suffix = slice.split_off(keep / 2);
    assert_eq!(slice.as_slice(), &values[skip..skip + keep / 2]);
    assert_eq!(suffix.as_slice(), &values[skip + keep / 2..skip + keep]);
    slice.truncate(0);
    assert!(slice.is_empty());
  }
  check::<typenum::U8>(2, 4);
  check::<typenum::U128>(39, 70);
  check::<typenum::U8192>(507, 518);
}

#[test]
fn copied_values_use_inline_capacity_and_report_unsupported_overflow() {
  let source = [1u8, 2, 3];
  let copy = Slice::<u8>::copy_from_slice(&source);
  assert_eq!(copy.as_slice(), &source);
  assert!(SliceN::<u8, 8>::try_copy_from_slice(&[9; 64]).is_none());
  assert!(catch_unwind(|| SliceN::<u8, 8>::copy_from_slice(&[9; 64])).is_err());
}

#[cfg(smol_slice_atomic)]
#[test]
fn copied_values_spill_and_reject_unrepresentable_zst_lengths() {
  let source = vec![7u8; Slice::<u8>::INLINE_CAPACITY + 3];
  let original_ptr = source.as_ptr();
  let copy = Slice::<u8>::copy_from_slice(&source);
  assert_eq!(copy.as_slice(), source.as_slice());
  assert_ne!(copy.as_slice().as_ptr(), original_ptr);
  let tail = copy.slice(2..);
  assert_eq!(tail.as_slice().as_ptr(), copy.as_slice()[2..].as_ptr());
  let zero_budget = SliceN::<u8, 0>::try_copy_from_slice(&[3, 4, 5]).unwrap();
  assert_eq!(zero_budget.as_slice(), &[3, 4, 5]);

  let too_long = &[(); isize::MAX as usize + 1];
  assert!(SliceN::<(), 0>::try_copy_from_slice(too_long).is_none());
}

#[cfg(not(smol_slice_atomic))]
#[test]
fn core_only_copy_does_not_require_send_or_sync() {
  #[derive(Clone, Copy)]
  struct Local(*mut ());

  let source = [Local(core::ptr::null_mut())];
  assert!(
    Slice::<Local>::copy_from_slice(&source).as_slice()[0]
      .0
      .is_null()
  );
}

#[cfg(smol_slice_atomic)]
#[test]
fn owned_split_edges_and_clear_release_exactly_the_expected_owner() {
  use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
  use std::sync::Arc;

  struct Dropper(Arc<AtomicUsize>);
  impl Drop for Dropper {
    fn drop(&mut self) {
      self.0.fetch_add(1, SeqCst);
    }
  }
  fn tracked(counter: &Arc<AtomicUsize>) -> SliceN<Dropper, 0> {
    SliceN::<Dropper, 0>::try_from(vec![Dropper(counter.clone()), Dropper(counter.clone())])
      .ok()
      .unwrap()
  }

  let drops = Arc::new(AtomicUsize::new(0));
  let mut data = tracked(&drops);
  let empty = data.split_to(0);
  drop(empty);
  assert_eq!(drops.load(SeqCst), 0);
  let all = data.split_off(0);
  drop(data);
  assert_eq!(drops.load(SeqCst), 0);
  drop(all);
  assert_eq!(drops.load(SeqCst), 2);

  let mut data = tracked(&drops);
  let empty = data.split_off(data.len());
  drop(empty);
  assert_eq!(drops.load(SeqCst), 2);
  let all = data.split_to(data.len());
  drop(data);
  assert_eq!(drops.load(SeqCst), 2);
  drop(all);
  assert_eq!(drops.load(SeqCst), 4);

  let mut data = tracked(&drops);
  data.truncate(1);
  assert_eq!(drops.load(SeqCst), 4);
  data.clear();
  assert_eq!(drops.load(SeqCst), 6);

  let data = tracked(&drops);
  let mut empty_window = data.slice(..0);
  drop(data);
  assert_eq!(drops.load(SeqCst), 6);
  empty_window.clear();
  assert_eq!(drops.load(SeqCst), 8);

  let mut data = tracked(&drops);
  let suffix = data.split_off(1);
  data.clear();
  assert_eq!(drops.load(SeqCst), 8);
  drop(suffix);
  assert_eq!(drops.load(SeqCst), 10);
}
