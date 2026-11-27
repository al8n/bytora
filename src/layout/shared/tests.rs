use super::*;
use crate::SliceN;
use core::mem::ManuallyDrop;
use std::vec;

#[test]
fn tagged_owner_dispatches_its_storage_operations_for_each_handle() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  static RETAINS: AtomicUsize = AtomicUsize::new(0);
  static RELEASES: AtomicUsize = AtomicUsize::new(0);
  static OPS: StorageVtable = StorageVtable {
    retain: Some(|owner| {
      assert_eq!(owner, OPS.tagged_owner());
      RETAINS.fetch_add(1, Relaxed);
    }),
    release: Some(|owner| {
      assert_eq!(owner, OPS.tagged_owner());
      RELEASES.fetch_add(1, Relaxed);
    }),
  };
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let data = b"static owner";
    let original: SliceN<u8, BYTES> = Slice {
      layout: Layout::try_external(data.as_ptr(), OPS.tagged_owner(), data.len()).unwrap(),
    };
    let mut peer = original.clone();
    peer.advance(7);
    drop(original);
    let last = peer.clone();
    let range = peer.slice(..2);
    let empty = range.slice(2..);
    drop(peer);
    assert_eq!(last.as_slice(), b"owner");
    assert_eq!(range.as_slice(), b"ow");
    assert_eq!(empty.as_slice().as_ptr(), data[9..].as_ptr());
  }
  check::<{ 3 * size_of::<usize>() }>();
  check::<32>();
  check::<4104>();
  {
    static UNITS: [(); isize::MAX as usize] = [(); isize::MAX as usize];
    let mut huge = Slice::<()> {
      layout: Layout::try_external(UNITS.as_ptr(), OPS.tagged_owner(), UNITS.len()).unwrap(),
    };
    // Shrinking across the high length byte must retain the custom owner.
    let short = huge.slice(UNITS.len() - 1..);
    huge.advance(UNITS.len());
    let _peer = short.clone();
  }
  assert_eq!(RETAINS.load(Relaxed), 14);
  assert_eq!(RELEASES.load(Relaxed), 18);
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
fn clone_from_retains_new_owner_and_releases_old_owner() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

  static OLD_LIVE: AtomicUsize = AtomicUsize::new(0);
  static NEW_LIVE: AtomicUsize = AtomicUsize::new(0);
  static STEP: AtomicUsize = AtomicUsize::new(0);
  static RETAIN_STEP: AtomicUsize = AtomicUsize::new(0);
  static RELEASE_STEP: AtomicUsize = AtomicUsize::new(0);
  static OLD_OPS: StorageVtable = StorageVtable {
    retain: Some(|owner| {
      assert_eq!(owner, OLD_OPS.tagged_owner());
      OLD_LIVE.fetch_add(1, Relaxed);
    }),
    release: Some(|owner| {
      assert_eq!(owner, OLD_OPS.tagged_owner());
      RELEASE_STEP.store(STEP.fetch_add(1, Relaxed), Relaxed);
      assert!(OLD_LIVE.fetch_sub(1, Relaxed) > 0);
    }),
  };
  static NEW_OPS: StorageVtable = StorageVtable {
    retain: Some(|owner| {
      assert_eq!(owner, NEW_OPS.tagged_owner());
      RETAIN_STEP.store(STEP.fetch_add(1, Relaxed), Relaxed);
      NEW_LIVE.fetch_add(1, Relaxed);
    }),
    release: Some(|owner| {
      assert_eq!(owner, NEW_OPS.tagged_owner());
      assert!(NEW_LIVE.fetch_sub(1, Relaxed) > 0);
    }),
  };
  static DATA: &[u8] = b"abcdefgh";

  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    OLD_LIVE.store(1, Relaxed);
    NEW_LIVE.store(1, Relaxed);
    STEP.store(0, Relaxed);
    let mut destination: SliceN<u8, BYTES> = Slice {
      layout: Layout::try_external(DATA.as_ptr(), OLD_OPS.tagged_owner(), DATA.len()).unwrap(),
    };
    let mut source: SliceN<u8, BYTES> = Slice {
      layout: Layout::try_external(DATA.as_ptr(), NEW_OPS.tagged_owner(), DATA.len()).unwrap(),
    };
    source.advance(2);
    destination.clone_from(&source);
    assert_eq!(OLD_LIVE.load(Relaxed), 0);
    assert_eq!(NEW_LIVE.load(Relaxed), 2);
    assert_eq!(RETAIN_STEP.load(Relaxed), 0);
    assert_eq!(RELEASE_STEP.load(Relaxed), 1);
    assert_eq!(destination.as_slice().as_ptr(), DATA[2..].as_ptr());

    let mut peer = source.clone();
    peer.advance(1);
    destination.clone_from(&peer);
    assert_eq!(NEW_LIVE.load(Relaxed), 3);
    drop(source);
    drop(peer);
    assert_eq!(NEW_LIVE.load(Relaxed), 1);
    assert_eq!(destination.as_slice(), &DATA[3..]);
    assert_eq!(destination.as_slice().as_ptr(), DATA[3..].as_ptr());
    drop(destination);
    assert_eq!(NEW_LIVE.load(Relaxed), 0);
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
