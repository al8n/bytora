use super::*;
use core::cell::Cell;
use std::vec;

#[test]
fn shared_rc_and_local_elements_outlive_the_original_window() {
  struct Item<'a>(&'a Cell<usize>, &'a Cell<usize>);
  impl Drop for Item<'_> {
    fn drop(&mut self) {
      self.1.set(self.1.get() + 1);
    }
  }
  let value = Cell::new(3);
  let dropped = Cell::new(0);
  let owner: Rc<[_]> = (0..5).map(|_| Item(&value, &dropped)).collect();
  let base = owner.as_ptr();
  let weak = Rc::downgrade(&owner);
  let whole = LocalVector::from_rc(owner.clone());
  let mut middle = whole.slice(1..4);
  let mut peer = middle.clone();
  assert_eq!(Rc::strong_count(&owner), 4);
  middle.advance(1);
  middle.truncate(1);
  peer.truncate(2);
  assert_eq!(middle.as_slice().as_ptr(), base.wrapping_add(2));
  assert_eq!(peer.as_slice().as_ptr(), base.wrapping_add(1));
  assert_eq!(Rc::strong_count(&owner), 4);
  middle.as_slice()[0].0.set(9);
  assert_eq!(owner[2].0.get(), 9);
  drop((owner, whole, middle));
  assert_eq!(dropped.get(), 0);
  assert_eq!(weak.strong_count(), 1);
  let empty = peer.slice(peer.len()..);
  assert!(empty.is_empty());
  assert_eq!(empty.as_slice().as_ptr(), base.wrapping_add(3));
  drop(peer);
  let original = empty.into_original_rc();
  assert_eq!((original.as_ptr(), original.len()), (base, 5));
  drop(original);
  assert_eq!(dropped.get(), 5);
  assert!(weak.upgrade().is_none());
}

#[test]
fn byte_budgets_preserve_sizes_niches_and_rejected_inputs() {
  const WORD: usize = size_of::<usize>();
  assert_eq!(size_of::<Option<LocalVectorN<u8, 32>>>(), 32);
  let values: Rc<[u8]> = Rc::from([1, 2, 3]);
  let default: LocalVector<u8> = LocalVectorN::<_, { 4 * WORD }>::try_from(values.clone()).unwrap();
  let _: LocalVectorN<u8, { 4 * WORD }> = default;
  let rejected = LocalVectorN::<u8, { 2 * WORD }>::try_from(values.clone())
    .err()
    .unwrap();
  assert!(Rc::ptr_eq(&values, &rejected));
  let wide = LocalVectorN::<u8, 8192>::try_from(rejected).unwrap();
  let sub = wide.slice(1..);
  assert_eq!(sub.as_slice(), &[2, 3]);
  assert!(Rc::ptr_eq(&values, &sub.into_original_rc()));
}

#[test]
fn large_zst_ranges_recover_the_full_owner_and_survive_shared_metadata() {
  let len = COUNT_MASK + 6;
  let values: Rc<[()]> = Rc::from(vec![(); len]);
  let original = LocalVector::from_rc(values.clone());
  let far = original.slice(COUNT_MASK + 1..);
  let mut near = original.clone();
  near.truncate(1);
  assert!(matches!(near.backing(), Backing::Indirect(_)));
  assert!(matches!(far.backing(), Backing::Indirect(_)));
  let mut nested = far.slice(1..4);
  nested.advance(1);
  assert_eq!(nested.len(), 2);
  assert_eq!(nested.as_slice().as_ptr(), values.as_ptr());
  assert!(Rc::ptr_eq(&values, &nested.into_original_rc()));
  drop((original, near));
  assert!(Rc::ptr_eq(&values, &far.into_original_rc()));
  assert_eq!(Rc::strong_count(&values), 1);

  let oversized: Rc<[()]> = Rc::from(vec![(); isize::MAX as usize + 1]);
  let rejected = LocalVector::<()>::try_from_rc(oversized.clone())
    .err()
    .unwrap();
  assert!(Rc::ptr_eq(&oversized, &rejected));
}

#[test]
fn over_aligned_rc_ranges_keep_the_element_pointer_including_empty_ranges() {
  #[repr(align(64))]
  struct Aligned(u8);
  let original: Rc<[Aligned]> = Rc::from([Aligned(1), Aligned(2), Aligned(3)]);
  let base = original.as_ptr();
  let whole = LocalVector::from_rc(original);
  let mut tail = whole.slice(1..);
  assert_eq!(tail.as_slice().as_ptr(), base.wrapping_add(1));
  assert_eq!(tail.as_slice()[0].0, 2);
  tail.advance(tail.len());
  assert_eq!(tail.as_slice().as_ptr(), base.wrapping_add(3));
  drop(whole);
  let original = tail.into_original_rc();
  assert_eq!(original.as_ptr(), base);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn a_real_overflowing_rc_range_keeps_its_original_byte_allocation() {
  let values: Rc<[u8]> = Rc::from(vec![7; COUNT_MASK + 5]);
  let whole = LocalVector::from_rc(values.clone());
  let far = whole.slice(COUNT_MASK + 1..);
  assert!(matches!(far.backing(), Backing::Indirect(_)));
  assert_eq!(
    far.as_slice().as_ptr(),
    values.as_ptr().wrapping_add(COUNT_MASK + 1)
  );
  assert_eq!(far.as_slice(), &[7; 4]);
  drop(whole);
  assert!(Rc::ptr_eq(&values, &far.into_original_rc()));
}
