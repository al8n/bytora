use super::*;
use crate::SliceN;
use core::mem::ManuallyDrop;
use std::vec;

// The inline array moves with this !Sync owner. Its projection is valid
// only after the owner has reached its final allocation.
struct ArrayOwner {
  data: [u8; 12],
  projected: core::cell::Cell<bool>,
  drops: Arc<core::sync::atomic::AtomicUsize>,
}

impl ArrayOwner {
  fn new(drops: &Arc<core::sync::atomic::AtomicUsize>) -> Self {
    Self {
      data: *b"custom owner",
      projected: core::cell::Cell::new(false),
      drops: Arc::clone(drops),
    }
  }
}

impl AsRef<[u8]> for ArrayOwner {
  fn as_ref(&self) -> &[u8] {
    assert!(!self.projected.replace(true), "projected owner twice");
    &self.data
  }
}

impl Drop for ArrayOwner {
  fn drop(&mut self) {
    self
      .drops
      .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
  }
}

#[test]
fn opaque_owner_lives_until_the_final_window_and_clones_only_on_materialization() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let drops = Arc::new(AtomicUsize::new(0));
    let original = SliceN::<u8, BYTES>::try_from_owner(ArrayOwner::new(&drops))
      .ok()
      .unwrap();
    let base = original.as_slice().as_ptr();
    let mut peer = original.clone();
    peer.advance(7);
    drop(original);
    let last = peer.clone();
    let range = peer.slice(..2);
    let empty = range.slice(2..);
    drop(peer);
    assert_eq!(last.as_slice(), b"owner");
    assert_eq!(range.as_slice(), b"ow");
    assert_eq!(empty.as_slice().as_ptr(), base.wrapping_add(9));
    assert_eq!(range.into_mut().as_slice(), b"ow");
    drop(empty);
    assert_eq!(drops.load(Relaxed), 0);
    assert!(!last.is_unique());
    let last = last.try_into_mut().err().unwrap();
    let last = last.try_into_vec().err().unwrap();
    assert_eq!(last.as_slice().as_ptr(), base.wrapping_add(7));
    let copied = last.into_vec();
    assert_eq!(copied, b"owner");
    assert_eq!(drops.load(Relaxed), 1);
  }
  check::<{ 3 * size_of::<usize>() }>();
  check::<32>();
  check::<4104>();

  let mut huge = Slice::<()>::from_owner(vec![(); isize::MAX as usize]);
  let short = huge.slice(huge.len() - 1..);
  huge.advance(huge.len());
  assert!(!huge.is_unique());
  assert_eq!(short.clone().as_slice(), &[()]);
  assert!(short.try_into_mut().is_err());

  let drops = Arc::new(AtomicUsize::new(0));
  let returned = SliceN::<u8, 8>::try_from_owner(ArrayOwner::new(&drops))
    .err()
    .unwrap();
  assert!(!returned.projected.get());
  drop(returned);
  assert_eq!(drops.load(Relaxed), 1);
}

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
fn zero_budget_copy_vectors_share_short_inputs_and_empty_vectors_need_no_owner() {
  fn copy<T: Copy + Send + Sync, S: InlineSize>(values: Vec<T>) -> Result<Slice<T, S>, Vec<T>> {
    Slice::try_from_vec(values)
  }
  let short = vec![7u8];
  let ptr = short.as_ptr();
  let shared = copy::<_, typenum::U0>(short).unwrap();
  assert_eq!(shared.as_slice().as_ptr(), ptr);
  let empty = SliceN::<std::string::String, 0>::try_from(Vec::with_capacity(1)).unwrap();
  assert_eq!(empty.layout.tag, COMPACT_STATIC);
  assert!(!empty.layout.uses_external_owner());
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
    let short = last.slice(1..3);
    let nested = short.slice(1..);
    let empty_range = nested.slice(1..);
    assert_eq!(short.as_slice().as_ptr(), base.wrapping_add(14));
    assert_eq!(weak.strong_count(), 4);
    drop(short);
    last.advance(last.len());
    let empty = last.clone();
    drop(last);
    assert!(empty.is_empty());
    assert_eq!(weak.strong_count(), 3);
    drop(empty);
    assert_eq!(nested.as_slice(), &[15]);
    drop(nested);
    assert_eq!(weak.strong_count(), 1);
    assert!(empty_range.is_empty());
    assert_eq!(empty_range.as_slice().as_ptr(), base.wrapping_add(16));
    drop(empty_range);
    assert!(weak.upgrade().is_none());
  }
  check::<24>();
  check::<32>();
  check::<4104>();
}

#[test]
fn clone_from_preserves_windows_across_storage_kinds() {
  static DATA: &[u8] = b"static window";

  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let mut inline = SliceN::<u8, BYTES>::try_from_slice(b"inline").unwrap();
    inline.advance(1);
    let mut borrowed = SliceN::<u8, BYTES>::try_from_static(DATA).unwrap();
    borrowed.advance(2);

    let mut destination = inline.clone();
    destination.clone_from(&borrowed);
    assert_eq!(destination.as_slice(), &DATA[2..]);
    assert_eq!(destination.as_slice().as_ptr(), DATA[2..].as_ptr());
    destination.advance(1);
    assert_eq!(borrowed.as_slice(), &DATA[2..]);
    destination.clone_from(&inline);
    assert_eq!(destination.as_slice(), b"nline");
    destination.advance(1);
    assert_eq!(inline.as_slice(), b"nline");

    let len = SliceN::<u8, BYTES>::INLINE_CAPACITY + 4;
    let values = vec![7; len];
    let base = values.as_ptr();
    let mut shared = SliceN::<u8, BYTES>::try_from_vec(values).unwrap();
    shared.advance(2);
    let old_values = vec![9; len];
    let mut destination = SliceN::<u8, BYTES>::try_from_vec(old_values).unwrap();
    destination.clone_from(&shared);
    assert_eq!(destination.len(), len - 2);
    assert!(destination.as_slice().iter().all(|&value| value == 7));
    assert_eq!(destination.as_slice().as_ptr(), base.wrapping_add(2));

    let mut peer = shared.clone();
    peer.advance(1);
    destination.clone_from(&peer);
    drop(shared);
    drop(peer);
    assert_eq!(destination.len(), len - 3);
    assert!(destination.as_slice().iter().all(|&value| value == 7));
    assert_eq!(destination.as_slice().as_ptr(), base.wrapping_add(3));

    destination.clone_from(&borrowed);
    assert_eq!(destination.as_slice(), &DATA[2..]);
    assert_eq!(destination.as_slice().as_ptr(), DATA[2..].as_ptr());
  }

  let mut tiny = SliceN::<u8, 8>::try_from_slice(b"old").unwrap();
  let tiny_source = SliceN::<u8, 8>::try_from_slice(b"new").unwrap();
  tiny.clone_from(&tiny_source);
  assert_eq!(tiny.as_slice(), b"new");

  check::<24>();
  check::<32>();
  check::<64>();
  check::<128>();
  check::<4096>();
}

#[test]
fn clone_from_releases_the_old_owner_and_keeps_the_new_owner_live() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let old = Arc::new(AtomicUsize::new(0));
    let new = Arc::new(AtomicUsize::new(0));
    let mut destination = SliceN::<u8, BYTES>::try_from_owner(ArrayOwner::new(&old))
      .ok()
      .unwrap();
    let mut source = SliceN::<u8, BYTES>::try_from_owner(ArrayOwner::new(&new))
      .ok()
      .unwrap();
    source.advance(2);
    destination.clone_from(&source);
    assert_eq!(old.load(Relaxed), 1);
    assert_eq!(new.load(Relaxed), 0);
    assert_eq!(destination.as_slice().as_ptr(), source.as_slice().as_ptr());
    let mut peer = source.clone();
    peer.advance(1);
    destination.clone_from(&peer);
    drop(source);
    drop(peer);
    assert_eq!(new.load(Relaxed), 0);
    assert_eq!(destination.as_slice(), b"tom owner");
    drop(destination);
    assert_eq!(new.load(Relaxed), 1);
  }

  check::<24>();
  check::<32>();
  check::<64>();
  check::<128>();
  check::<4096>();
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
  let short = zsts.slice(max - 2..);
  assert_ne!(short.layout.tag, COMPACT_STATIC);
  zsts.advance(max - 1);
  assert_ne!(zsts.layout.tag, COMPACT_STATIC);
  let last = zsts.clone();
  zsts.advance(1);
  assert_eq!(last.as_slice(), &[()]);
  assert!(zsts.as_slice().is_empty());
  drop(zsts);
  assert_eq!(short.as_slice(), &[(), ()]);
  let too_long = Slice::<()>::try_from_vec(vec![(); max + 1]).err().unwrap();
  assert_eq!(too_long.len(), max + 1);
}

#[test]
fn zero_sized_frozen_windows_recover_logical_slots_and_rejoin() {
  let mut head = crate::SliceMutN::<_, 0>::from(vec![(); 6]);
  let tail = head.split_off(4);
  let mut frozen = head.freeze();
  frozen.advance(2);
  let mut head = frozen.try_into_mut().ok().unwrap();
  assert_eq!((head.len(), head.capacity()), (2, 2));
  head.try_unsplit(tail).ok().unwrap();
  assert_eq!(head.len(), 4);

  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  static DROPS: AtomicUsize = AtomicUsize::new(0);
  #[repr(align(64))]
  struct Count;
  impl Drop for Count {
    fn drop(&mut self) {
      DROPS.fetch_add(1, Relaxed);
    }
  }
  let values: Vec<Count> = (0..6).map(|_| Count).collect();
  let base = values.as_ptr();
  let mut head = crate::SliceMutN::<_, 0>::from(values);
  let tail = head.split_off(4);
  let whole = head.freeze();
  let mut nested = whole.slice(1..).slice(..3);
  nested.advance(1);
  assert_eq!(nested.clone().as_slice().as_ptr(), base);
  drop(whole);
  let mut head = nested.try_into_mut().ok().unwrap();
  assert_eq!((head.len(), head.capacity()), (2, 2));
  head.try_unsplit(tail).ok().unwrap();
  assert_eq!(DROPS.load(Relaxed), 0);
  let visible = head.into_vec();
  assert_eq!(visible.len(), 4);
  assert_eq!(DROPS.load(Relaxed), 2);
  drop(visible);
  assert_eq!(DROPS.load(Relaxed), 6);
}

#[test]
fn zero_sized_vector_recovers_large_offsets_and_read_only_owners_keep_their_pointer() {
  let max = isize::MAX as usize;
  let original = SliceN::<(), 0>::try_from_shared_vec(vec![(); max]).unwrap();
  let base = original.as_slice().as_ptr();
  let mut last = original.slice(max - 4..).slice(1..);
  last.advance(1);
  assert_eq!(last.as_slice().as_ptr(), base);
  drop(original);
  let last = last.try_into_mut().ok().unwrap();
  assert_eq!((last.len(), last.capacity()), (2, usize::MAX - (max - 2)));

  #[repr(align(64))]
  struct Aligned;
  #[repr(C)]
  struct Holder {
    slots: [Aligned; 5],
    byte: u8,
  }
  impl AsRef<[Aligned]> for Holder {
    fn as_ref(&self) -> &[Aligned] {
      &self.slots
    }
  }
  static STATIC: Holder = Holder {
    slots: [const { Aligned }; 5],
    byte: 7,
  };
  let borrowed = Slice::from_static(&STATIC.slots);
  assert_eq!(borrowed.as_slice().as_ptr(), STATIC.slots.as_ptr());
  let opaque = Slice::from_owner(Holder {
    slots: [const { Aligned }; 5],
    byte: 3,
  });
  for original in [borrowed, opaque] {
    let base = original.as_slice().as_ptr();
    assert!(base.is_aligned());
    let mut tail = original.slice(1..4);
    tail.advance(2);
    assert_eq!(tail.as_slice().as_ptr(), base);
    assert_eq!(tail.len(), 1);
    assert!(tail.try_into_mut().is_err());
  }
}

#[test]
fn unique_vector_extraction_preserves_allocation_and_drops_only_hidden_elements() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Count<'a>(&'a AtomicUsize);
  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.0.fetch_add(1, Relaxed);
    }
  }
  let drops = [const { AtomicUsize::new(0) }; 5];
  let mut values = Vec::with_capacity(12);
  values.extend(drops.iter().map(Count));
  let (base, capacity) = (values.as_ptr(), values.capacity());
  let original = SliceN::<_, 0>::try_from(values).ok().unwrap();
  let middle = original.slice(1..4);
  assert!(!middle.is_unique());
  let middle = middle.try_into_mut().err().unwrap();
  let middle = middle.try_into_vec().err().unwrap();
  drop(original);
  assert!(middle.is_unique());
  let values = middle.try_into_vec().ok().unwrap();
  assert_eq!(
    (values.as_ptr(), values.len(), values.capacity()),
    (base, 3, capacity)
  );
  assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1, 0, 0, 0, 1]);
  drop(values);
  assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1; 5]);
}

#[test]
fn static_and_shared_copy_fallbacks_and_unique_mutable_window() {
  let borrowed = Slice::<u8>::from_static(b"static");
  assert!(!borrowed.is_unique());
  assert!(borrowed.clone().try_into_mut().is_err());
  assert!(borrowed.clone().try_into_vec().is_err());
  assert_eq!(borrowed.slice(1..4).into_mut().as_slice(), b"tat");
  assert_eq!(borrowed.slice(2..5).into_vec(), b"ati");

  let input: Vec<u8> = (0..64).collect();
  let base = input.as_ptr();
  let mut original = SliceN::<_, 0>::try_from(input).unwrap();
  original.advance(4);
  let mut window = original.slice(2..6);
  assert_eq!(window.clone().into_vec(), [6, 7, 8, 9]);
  let mut copy = original.clone().into_mut();
  copy.as_mut_slice()[2] = 70;
  assert_eq!(window.as_slice()[0], 6);
  drop(original);
  window.advance(1);
  let mut unique = window.try_into_mut().ok().unwrap();
  assert_eq!(unique.as_slice().as_ptr(), base.wrapping_add(7));
  unique.as_mut_slice()[0] = 17;
  assert_eq!(unique.into_vec(), [17, 8, 9]);
}

#[test]
fn a_frozen_window_thaws_without_copying_while_its_mutable_sibling_is_live() {
  struct Item<'a>(&'a str);
  let local = std::string::String::from("scoped");
  let mut input = Vec::with_capacity(8);
  input.extend([Item("prefix"), Item(&local), Item("right")]);
  let base = input.as_ptr();
  let mut left = crate::SliceMutN::<_, 0>::from(input);
  let mut right = left.split_off(2);
  let frozen = left.freeze();
  let mut tail = frozen.slice(1..);
  assert!(!tail.is_unique());
  tail = tail.try_into_mut().err().unwrap();
  drop(frozen);
  let mut thawed = tail.try_into_mut().ok().unwrap();
  assert_eq!(thawed.as_slice().as_ptr(), base.wrapping_add(1));
  assert_eq!(thawed.as_slice()[0].0, "scoped");
  right.as_mut_slice()[0] = Item("separate");
  thawed.as_mut_slice()[0] = Item("mine");
  let moved = thawed.freeze().try_into_vec().ok().unwrap();
  assert_eq!(moved.len(), 1);
  assert_eq!(moved[0].0, "mine");
  assert_eq!(right.as_slice()[0].0, "separate");
}

#[test]
fn frozen_windows_can_shorten_element_lifetimes_before_recovering_ownership() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Count<'a>(&'a AtomicUsize);
  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.0.fetch_add(1, Relaxed);
    }
  }
  static ORIGINAL_DROPS: AtomicUsize = AtomicUsize::new(0);

  fn freeze_static() -> SliceN<Count<'static>, 0> {
    crate::SliceMutN::<_, 0>::from(vec![Count(&ORIGINAL_DROPS)]).freeze()
  }
  fn shorten<'a>(frozen: SliceN<Count<'static>, 0>, _: &'a AtomicUsize) -> SliceN<Count<'a>, 0> {
    frozen
  }

  let frozen_for_mut = freeze_static();
  let frozen_for_vec = freeze_static();
  let local_drops = AtomicUsize::new(0);
  let mut window = shorten(frozen_for_mut, &local_drops)
    .try_into_mut()
    .ok()
    .unwrap();
  window.as_mut_slice()[0] = Count(&local_drops);
  let mut vec = shorten(frozen_for_vec, &local_drops)
    .try_into_vec()
    .ok()
    .unwrap();
  vec[0] = Count(&local_drops);
  assert_eq!(ORIGINAL_DROPS.load(Relaxed), 2);
  assert_eq!(local_drops.load(Relaxed), 0);
  drop(window);
  drop(vec);
  assert_eq!(local_drops.load(Relaxed), 2);
}

#[cfg(feature = "std")]
#[test]
fn opaque_owner_cleans_up_a_panicking_projection_and_cross_thread_final_drop() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Panics(ArrayOwner);
  impl AsRef<[u8]> for Panics {
    fn as_ref(&self) -> &[u8] {
      let _ = self.0.as_ref();
      panic!("intentional projection panic");
    }
  }
  let drops = Arc::new(AtomicUsize::new(0));
  let owner = Panics(ArrayOwner::new(&drops));
  assert!(
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Slice::from_owner(owner))).is_err()
  );
  assert_eq!(drops.load(Relaxed), 1);
  let slice = Slice::from_owner(ArrayOwner::new(&drops));
  let tail = slice.slice(7..);
  drop(slice);
  std::thread::spawn(move || {
    assert_eq!(tail.clone().as_slice(), b"owner");
    assert!(tail.try_into_mut().is_err());
  })
  .join()
  .unwrap();
  assert_eq!(drops.load(Relaxed), 2);
}

#[cfg(feature = "std")]
#[test]
fn panic_during_extraction_releases_each_element_once() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Count<'a> {
    drops: &'a AtomicUsize,
    panic: bool,
  }
  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.drops.fetch_add(1, Relaxed);
      assert!(!self.panic, "intentional element destructor panic");
    }
  }
  for panic_at in [0, 3] {
    let drops = [const { AtomicUsize::new(0) }; 4];
    let values = drops
      .iter()
      .enumerate()
      .map(|(i, drops)| Count {
        drops,
        panic: i == panic_at,
      })
      .collect::<Vec<_>>();
    let original = SliceN::<_, 0>::try_from(values).ok().unwrap();
    let middle = original.slice(1..3);
    drop(original);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| middle.try_into_vec()));
    assert!(failure.is_err());
    assert_eq!(drops.each_ref().map(|n| n.load(Relaxed)), [1; 4]);
  }
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
