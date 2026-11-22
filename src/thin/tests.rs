use super::{Arc, ThinSlice};
use core::{
  ops::Bound,
  sync::atomic::{AtomicUsize, Ordering::Relaxed},
};
use std::{string::String, vec, vec::Vec};

#[test]
#[cfg(target_pointer_width = "64")]
fn handle_and_option_are_eight_bytes() {
  assert_eq!(size_of::<ThinSlice<String>>(), 8);
  assert_eq!(size_of::<Option<ThinSlice<String>>>(), 8);
}

#[test]
fn vector_round_trip_preserves_pointer_and_capacity() {
  for (len, capacity) in [(4, 4), (4, 12), (0, 12), (0, 0)] {
    let mut values = Vec::with_capacity(capacity);
    values.extend(0..len);
    let ptr = values.as_ptr();
    let capacity = values.capacity();
    let slice = ThinSlice::from_vec(values);
    assert!(slice.is_unique());
    assert_eq!(slice.len(), len);
    assert_eq!(slice.is_empty(), len == 0);
    let mut slot = Some(slice);
    let values = slot.take().unwrap().try_into_vec().ok().unwrap();
    assert!(slot.is_none());
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, (0..len).collect::<Vec<_>>());
  }
}

#[test]
fn full_range_shares_the_existing_descriptor() {
  let root = ThinSlice::from_vec(vec![1, 2, 3, 4]);
  let full = root.slice(..);
  assert!(Arc::ptr_eq(&root.0, &full.0));
  let window = root.slice(1..3);
  let window_full = window.slice(..);
  assert!(Arc::ptr_eq(&window.0, &window_full.0));
  assert_eq!(window_full.as_slice(), &[2, 3]);
  let empty = root.slice(2..2);
  let empty_full = empty.slice(..);
  assert!(Arc::ptr_eq(&empty.0, &empty_full.0));
}

#[test]
fn shared_extraction_failure_preserves_root_and_nested_window() {
  let root = ThinSlice::from_vec(vec![0, 1, 2, 3, 4, 5]);
  let window = root.slice(1..5);
  let nested = window.slice(1..=2);
  let peer = nested.clone();
  assert!(!root.is_unique());
  assert!(!nested.is_unique());
  let root = root.try_into_vec().err().unwrap();
  let nested = nested.try_into_vec().err().unwrap();
  assert_eq!(root.as_slice(), &[0, 1, 2, 3, 4, 5]);
  assert_eq!(nested.as_slice(), &[2, 3]);
  drop(root);
  drop(window);
  assert_eq!(peer.as_slice(), &[2, 3]);
  assert!(!nested.is_unique());
  drop(peer);
  assert!(nested.is_unique());
  assert_eq!(nested.try_into_vec().ok().unwrap(), vec![2, 3]);
}

#[test]
fn unique_window_remains_intact_when_the_root_is_shared() {
  let root = ThinSlice::from_vec(vec![0, 1, 2, 3]);
  let window = root.slice(1..3);
  let window = window.try_into_vec().err().unwrap();
  assert_eq!(window.as_slice(), &[1, 2]);
  assert_eq!(root.as_slice(), &[0, 1, 2, 3]);
  drop(root);
  assert!(window.is_unique());
  assert_eq!(window.try_into_vec().ok().unwrap(), vec![1, 2]);
}

#[test]
fn unique_proper_ranges_reuse_the_original_allocation() {
  for (start, end) in [(0, 3), (2, 5), (1, 4), (0, 0), (3, 3), (5, 5)] {
    let mut values = Vec::with_capacity(12);
    values.extend(0..5);
    let ptr = values.as_ptr();
    let capacity = values.capacity();
    let root = ThinSlice::from_vec(values);
    let view = root.slice(start..end);
    drop(root);
    assert!(view.is_unique());
    let values = view.try_into_vec().ok().unwrap();
    assert_eq!(values.as_ptr(), ptr);
    assert_eq!(values.capacity(), capacity);
    assert_eq!(values, (start..end).collect::<Vec<_>>());
  }
}

struct Count<'a> {
  id: usize,
  drops: &'a [AtomicUsize],
}

impl Drop for Count<'_> {
  fn drop(&mut self) {
    self.drops[self.id].fetch_add(1, Relaxed);
  }
}

#[test]
fn non_copy_elements_drop_only_when_hidden_or_returned() {
  let drops = [const { AtomicUsize::new(0) }; 5];
  let values = (0..5).map(|id| Count { id, drops: &drops }).collect();
  let root = ThinSlice::from_vec(values);
  let view = root.slice(1..4);
  let peer = view.clone();
  drop(root);
  drop(peer);
  assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [0; 5]);
  let returned = view.try_into_vec().ok().unwrap();
  assert_eq!(
    returned.iter().map(|item| item.id).collect::<Vec<_>>(),
    [1, 2, 3]
  );
  assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1, 0, 0, 0, 1]);
  drop(returned);
  assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1; 5]);
}

#[test]
fn zero_sized_non_copy_elements_are_recovered_and_dropped_once() {
  static DROPS: AtomicUsize = AtomicUsize::new(0);
  struct ZeroSized;
  impl Drop for ZeroSized {
    fn drop(&mut self) {
      DROPS.fetch_add(1, Relaxed);
    }
  }

  assert_eq!(size_of::<ZeroSized>(), 0);
  DROPS.store(0, Relaxed);
  let values: Vec<_> = (0..5).map(|_| ZeroSized).collect();
  let ptr = values.as_ptr();
  let capacity = values.capacity();
  let root = ThinSlice::from_vec(values);
  let parent = root.slice(1..5);
  let view = parent.slice(..3);
  drop(root);
  drop(parent);
  let values = view.try_into_vec().ok().unwrap();
  assert_eq!(values.as_ptr(), ptr);
  assert_eq!(values.capacity(), capacity);
  assert_eq!(values.len(), 3);
  assert_eq!(DROPS.load(Relaxed), 2);
  drop(values);
  assert_eq!(DROPS.load(Relaxed), 5);
}

#[cfg(feature = "std")]
#[test]
fn panicking_hidden_element_drops_every_element_once() {
  struct Panics<'a> {
    id: usize,
    panic_at: usize,
    drops: &'a [AtomicUsize],
  }

  impl Drop for Panics<'_> {
    fn drop(&mut self) {
      self.drops[self.id].fetch_add(1, Relaxed);
      assert_ne!(self.id, self.panic_at, "intentional destructor panic");
    }
  }

  for panic_at in [0, 5] {
    let drops = [const { AtomicUsize::new(0) }; 6];
    let values = (0..6)
      .map(|id| Panics {
        id,
        panic_at,
        drops: &drops,
      })
      .collect();
    let root = ThinSlice::from_vec(values);
    let view = root.slice(1..5);
    drop(root);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| view.try_into_vec()));
    assert!(result.is_err());
    assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1; 6]);
  }
}

#[cfg(feature = "std")]
#[test]
fn views_share_across_scoped_threads() {
  let root = ThinSlice::from_vec(vec![0, 1, 2, 3, 4]);
  std::thread::scope(|scope| {
    let view = root.slice(1..4);
    scope.spawn(move || {
      assert_eq!(view.as_slice(), &[1, 2, 3]);
      assert!(!view.is_unique());
      assert!(view.try_into_vec().is_err());
    });
  });
  assert!(root.is_unique());
}

#[test]
#[should_panic(expected = "slice range out of bounds")]
fn reversed_range_panics() {
  ThinSlice::from_vec(vec![1, 2]).slice((Bound::Included(2), Bound::Excluded(1)));
}

#[test]
#[should_panic(expected = "slice range out of bounds")]
fn out_of_bounds_range_panics() {
  ThinSlice::from_vec(vec![1, 2]).slice(..3);
}

#[test]
#[should_panic(expected = "slice end overflow")]
fn inclusive_end_overflow_panics() {
  ThinSlice::from_vec(vec![1, 2]).slice(..=usize::MAX);
}
