use super::{Slice, SliceMut};

#[test]
fn full_inline_windows_move_and_advance() {
  fn check<const N: usize>() {
    let source = core::array::from_fn::<_, N, _>(|i| i as u8);
    let mut bytes = Slice::<u8, N>::try_from_slice(&source).unwrap();
    bytes.advance(1);
    let mut moved = core::hint::black_box(Some(bytes)).unwrap();
    assert_eq!(moved.len(), N - 1);
    assert_eq!(moved.as_slice(), &source[1..]);
    moved.advance(N - 1);
    moved.advance(0);
    assert!(moved.is_empty());
    assert!(moved.as_slice().is_empty());
  }
  check::<22>();
  check::<30>();
  check::<62>();
  check::<126>();
}

#[test]
fn over_aligned_elements_remain_aligned_after_move_and_advance() {
  #[repr(align(64))]
  #[derive(Copy, Clone, PartialEq, Debug)]
  struct Aligned(u8);

  let mut values = SliceMut::<Aligned, 3>::try_from_slice(&[Aligned(1), Aligned(2)]).unwrap();
  let base = values.as_slice().as_ptr();
  assert!(base.is_aligned());
  values.advance(1);
  assert_eq!(values.as_slice().as_ptr(), base.wrapping_add(1));
  values.as_mut_slice()[0] = Aligned(7);
  let mut moved = core::hint::black_box(values);
  assert_eq!(moved.as_slice(), &[Aligned(7)]);
  assert!(moved.as_mut_slice().as_mut_ptr().is_aligned());
  moved.advance(1);
  assert!(moved.as_mut_slice().is_empty());
}

#[test]
fn empty_and_zero_sized_elements_have_valid_ranges() {
  #[repr(align(64))]
  #[derive(Clone, Copy)]
  struct AlignedZst;

  let mut empty = SliceMut::<AlignedZst, 0>::try_from_slice(&[]).unwrap();
  assert!(empty.is_empty());
  empty.advance(0);
  assert!(empty.as_mut_slice().as_mut_ptr().is_aligned());
  assert!(Slice::<AlignedZst, 0>::try_from_slice(&[AlignedZst]).is_none());

  let mut zsts = SliceMut::<AlignedZst, 3>::try_from_slice(&[AlignedZst; 3]).unwrap();
  zsts.advance(2);
  assert_eq!(zsts.as_mut_slice().len(), 1);
  assert!(zsts.as_mut_slice().as_mut_ptr().is_aligned());
}

#[test]
fn capacity_limits_are_checked_before_encoding() {
  assert_eq!(Slice::<u8, 0>::INLINE_CAPACITY, 0);
  assert_eq!(SliceMut::<u8, 30>::INLINE_CAPACITY, 30);
  assert_eq!(Slice::<u8, 256>::INLINE_CAPACITY, 126);
  assert!(Slice::<u8, 22>::try_from_slice(&[0; 23]).is_none());
  assert!(SliceMut::<u8, 256>::try_from_slice(&[0; 127]).is_none());
  assert!(Slice::<u8, 256>::try_from_slice(&[0; 256]).is_none());
  assert_eq!(
    Slice::<u8, 256>::try_from_slice(&[0; 126]).unwrap().len(),
    126
  );
}

#[test]
#[should_panic(expected = "advance past end")]
fn advance_checks_the_whole_count_before_narrowing() {
  let mut values = SliceMut::<u8, 1>::try_from_slice(&[7]).unwrap();
  values.advance(usize::MAX);
}

#[test]
fn reference_elements_keep_their_provenance() {
  let numbers = [2, 3];
  let mut refs = SliceMut::<&i32, 4>::try_from_slice(&[&numbers[0], &numbers[1]]).unwrap();
  refs.advance(1);
  assert_eq!(*refs.as_slice()[0], 3);
  refs.as_mut_slice()[0] = &numbers[0];
  let moved = core::hint::black_box(refs);
  assert!(core::ptr::eq(moved.as_slice()[0], &numbers[0]));
  assert_eq!(*moved.as_slice()[0], 2);
}
