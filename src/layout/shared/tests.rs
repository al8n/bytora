use super::*;
use crate::SliceN;
use core::mem::ManuallyDrop;
use std::vec;

#[test]
fn vector_construction_uses_inline_capacity_and_returns_unusable_input() {
  let short = vec![3u8; SliceN::<u8, 8>::INLINE_CAPACITY];
  let inline = SliceN::<u8, 8>::try_from_vec(short).unwrap();
  assert!(inline.layout.is_inline());
  assert_eq!(inline.as_slice(), &[3; 7]);
  let input = vec![7u8; 8];
  let (ptr, cap) = (input.as_ptr(), input.capacity());
  let returned = SliceN::<u8, 8>::try_from_vec(input).err().unwrap();
  assert_eq!(returned, [7; 8]);
  assert_eq!((returned.as_ptr(), returned.capacity()), (ptr, cap));
}

#[test]
fn shared_clones_retain_one_owner_each_and_release_the_last_reference() {
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let len = SliceN::<u8, BYTES>::INLINE_CAPACITY + 1;
    let input: Vec<u8> = (0..len).map(|i| i as u8).collect();
    let base = input.as_ptr();
    let mut original = SliceN::<u8, BYTES>::try_from_vec(input).unwrap();
    assert_eq!(original.as_slice().as_ptr(), base);
    let owner = original.layout.external_prefix().owner.cast::<Vec<u8>>();
    // SAFETY: Borrow the original handle's live Arc without consuming its strong count.
    let arc = ManuallyDrop::new(unsafe { Arc::from_raw(owner) });
    let weak = Arc::downgrade(&arc);
    assert_eq!(weak.strong_count(), 1);
    original.advance(10);
    let mut peer = original.clone();
    assert_eq!(peer.as_slice().as_ptr(), base.wrapping_add(10));
    peer.advance(3);
    assert_eq!(original.as_slice()[0], 10);
    let mut last = peer.clone();
    assert_eq!(weak.strong_count(), 3);
    peer.advance(peer.len());
    assert!(peer.as_slice().is_empty());
    drop(original);
    drop(peer);
    assert_eq!(weak.strong_count(), 1);
    assert_eq!(last.as_slice()[0], 13);
    last.advance(last.len());
    let empty = last.clone();
    drop(last);
    assert!(empty.is_empty());
    assert_eq!(weak.strong_count(), 1);
    drop(empty);
    assert!(weak.upgrade().is_none());
  }
  check::<24>();
  check::<4104>();
}

#[test]
fn shared_storage_supports_overaligned_elements_and_zero_sized_windows() {
  #[repr(align(64))]
  #[derive(Clone, Copy, Debug)]
  struct Aligned(u8);
  let aligned = Slice::<Aligned>::try_from_vec(vec![Aligned(17)]).unwrap();
  assert!(aligned.as_slice().as_ptr().is_aligned());
  assert_eq!(aligned.clone().as_slice()[0].0, 17);

  let max = isize::MAX as usize;
  let mut zsts = Slice::<()>::try_from_vec(vec![(); max]).ok().unwrap();
  assert_eq!(zsts.as_slice().len(), max);
  zsts.advance(max - 1);
  let last = zsts.clone();
  zsts.advance(1);
  assert_eq!(last.as_slice(), &[()]);
  assert!(zsts.as_slice().is_empty());
  let too_long = Slice::<()>::try_from_vec(vec![(); max + 1]).err().unwrap();
  assert_eq!(too_long.len(), max + 1);
}

#[cfg(feature = "std")]
#[test]
fn shared_windows_can_cross_threads_and_reject_invalid_advances() {
  let number = 42;
  let original = Slice::<&i32>::try_from_vec(vec![&number; 64]).unwrap();
  let mut peer = original.clone();
  std::thread::scope(|scope| {
    let borrowed = &original;
    let worker = scope.spawn(move || {
      peer.advance(3);
      assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| peer.advance(62))).is_err());
      assert_eq!(peer.len(), 61);
      assert!(ptr::eq(peer.as_slice()[0], borrowed.as_slice()[0]));
      assert_eq!(borrowed.len(), 64);
      peer
    });
    let returned = worker.join().unwrap();
    assert_eq!(*returned.as_slice()[0], 42);
    assert!(ptr::eq(returned.as_slice()[0], &number));
  });
}
