use smol_slice::SliceMutN;

#[test]
fn bounded_growth_split_and_reclaim_cover_inline_capacity() {
  type Tiny = SliceMutN<u8, { size_of::<usize>() }>;
  let capacity = Tiny::INLINE_CAPACITY;
  for len in 0..=capacity {
    for at in 0..=capacity {
      let mut left = const { Tiny::new() };
      assert_eq!(left.capacity(), capacity);
      left.extend_from_slice(&b"rustbuf"[..len]);
      let mut right = left.split_off(at);
      assert_eq!(left.capacity(), at);
      assert_eq!(right.capacity(), capacity - at);
      assert_eq!(left.as_slice(), &b"rustbuf"[..len.min(at)]);
      assert_eq!(right.as_slice(), &b"rustbuf"[len.min(at)..len]);

      let left_spare = left.spare_capacity_mut();
      for slot in left_spare {
        slot.write(b'L');
      }
      // SAFETY: All previously spare slots were just initialized.
      unsafe {
        left.set_len(at);
      }
      right.resize(capacity - at, b'R');
      left.unsplit(right);
      assert_eq!(left.len(), capacity);
      assert!(!left.try_reclaim(1));
      left.advance(capacity);
      assert!(left.try_reclaim(capacity));
      assert_eq!(left.capacity(), capacity);
    }
  }

  let mut tiny = Tiny::new();
  tiny.extend_from_slice(b"abc");
  assert!(
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tiny.reserve(usize::MAX))).is_err()
  );
  assert_eq!(tiny.as_slice(), b"abc");
}

#[test]
fn write_helpers_use_relative_ranges_and_reject_overflow_before_modification() {
  use core::ops::Bound::{Excluded, Included, Unbounded};
  let mut values = SliceMutN::<u16, 64>::new();
  values.extend_from_slice(&[1, 2, 3]);
  values.advance(1);
  values.extend_from_within(..=1);
  values.extend_from_within((Excluded(0), Included(1)));
  assert_eq!(values.as_slice(), &[2, 3, 2, 3, 3]);
  for bounds in [
    (Included(2), Excluded(1)),
    (Excluded(usize::MAX), Unbounded),
    (Unbounded, Included(usize::MAX)),
  ] {
    assert!(
      std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || values.extend_from_within(bounds)
      ))
      .is_err()
    );
    assert_eq!(values.as_slice(), &[2, 3, 2, 3, 3]);
  }
  let capacity = values.capacity();
  let data = values.split();
  assert_eq!(values.capacity(), capacity - 5);
  assert_eq!(data.capacity(), 5);
  assert_eq!(data.freeze().as_slice(), &[2, 3, 2, 3, 3]);
  assert_eq!(SliceMutN::<u8, 16>::zeroed(5).as_slice(), &[0; 5]);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn promotions_and_vector_recovery_preserve_allocations_and_spare_capacity() {
  type TwoWords = SliceMutN<u8, { 2 * size_of::<usize>() }>;
  let mut compact = TwoWords::new();
  compact.resize(TwoWords::INLINE_CAPACITY, 9);
  compact.extend_from_within(..);
  assert_eq!(compact.into_vec(), vec![9; 2 * TwoWords::INLINE_CAPACITY]);
  let mut allocated = Vec::<u8>::with_capacity(29);
  let base = allocated.as_ptr();
  let capacity = allocated.capacity();
  let mut handle = SliceMutN::<u8, 0>::from(allocated);
  assert_eq!(handle.capacity(), capacity);
  handle.extend_from_slice(b"abcdef");
  handle.advance(3);
  assert!(handle.try_reclaim(capacity - 3));
  assert_eq!(handle.as_slice().as_ptr(), base);
  allocated = handle.into_vec();
  assert_eq!(allocated, b"def");
  assert_eq!(allocated.as_ptr(), base);
  assert_eq!(allocated.capacity(), capacity);
}

#[cfg(smol_slice_atomic)]
#[test]
fn heap_splits_partition_unused_space_and_rejoin_without_moving_elements() {
  let mut data = Vec::with_capacity(32);
  data.extend_from_slice(b"abc");
  let base = data.as_ptr();
  let capacity = data.capacity();
  let mut a = SliceMutN::<_, 0>::from(data);
  let mut b = a.split_off(8);
  let mut c = b.split_off(5);
  assert_eq!(b.as_slice().as_ptr(), base.wrapping_add(8));
  assert_eq!(c.as_slice().as_ptr(), base.wrapping_add(13));
  assert_eq!(
    [a.capacity(), b.capacity(), c.capacity()],
    [8, 5, capacity - 13]
  );
  c.extend_from_slice(b"third");
  let c = a.try_unsplit(c).err().unwrap();
  let b = a.try_unsplit(b).err().unwrap();
  a.extend_from_slice(b"defgh");
  let mut b = b;
  b.extend_from_slice(b"ijklm");
  a.try_unsplit(b).ok().unwrap();
  a.try_unsplit(c).ok().unwrap();
  assert_eq!(a.as_slice(), b"abcdefghijklmthird");
  assert_eq!(a.capacity(), capacity);
  let joined = a.into_vec();
  assert_eq!(joined.as_ptr(), base);
}

#[cfg(smol_slice_atomic)]
#[test]
fn freeze_and_thaw_keep_disjoint_siblings_isolated_across_threads() {
  let mut raw = Vec::with_capacity(16);
  raw.extend(["one".to_owned(), "two".into(), "three".into()]);
  let base = raw.as_ptr();
  let mut input = SliceMutN::<_, 0>::from(raw);
  let mut tail = input.split_off(2);
  let frozen = input.freeze();
  let clone = frozen.slice(1..);
  let frozen = frozen.try_into_mut().err().unwrap();
  assert_eq!(clone.as_slice().as_ptr(), base.wrapping_add(1));
  std::thread::scope(|scope| {
    scope.spawn(|| {
      tail.as_mut_slice()[0].push('!');
      tail.resize(4, "extra".into());
    });
    assert_eq!(clone.as_slice(), &["two"]);
  });
  drop(clone);
  let mut head = frozen.try_into_mut().ok().unwrap();
  assert_eq!(head.as_slice().as_ptr(), base);
  assert_eq!(head.capacity(), 2);
  head.try_unsplit(tail).ok().unwrap();
  assert_eq!(
    head.as_slice(),
    &["one", "two", "three!", "extra", "extra", "extra"]
  );
  let joined = head.into_vec();
  assert_eq!(joined.as_ptr(), base);
}

#[cfg(smol_slice_atomic)]
#[test]
fn shared_reclaim_cannot_overwrite_a_live_sibling() {
  use core::cell::Cell;
  let data: Vec<_> = (0..8).map(Cell::new).collect();
  let base = data.as_ptr();
  let mut head = SliceMutN::<_, 0>::from(data);
  let mut tail = head.split_off(4);
  head.advance(2);
  assert!(head.try_reclaim(2));
  assert_eq!(head.as_slice().as_ptr(), base);
  assert!(!head.try_reclaim(3));
  std::thread::scope(|scope| {
    let worker = scope.spawn(move || {
      tail.as_mut_slice()[0].set(40);
      tail
    });
    head.resize(10, Cell::new(7));
    let tail = worker.join().unwrap();
    assert_eq!(tail.as_slice().as_ptr(), base.wrapping_add(4));
    assert_eq!(tail.as_slice()[0].get(), 40);
  });
  assert_eq!(
    head.as_slice().iter().map(Cell::get).collect::<Vec<_>>(),
    [2, 3, 7, 7, 7, 7, 7, 7, 7, 7]
  );
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn resize_and_reclaim_are_panic_safe_for_arbitrary_elements() {
  use std::{
    cell::Cell,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
  };
  struct Item {
    clones: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    panic_on_drop: bool,
  }
  impl Clone for Item {
    fn clone(&self) -> Self {
      let count = self.clones.get() + 1;
      self.clones.set(count);
      assert!(count != 2, "second clone");
      Self {
        clones: self.clones.clone(),
        drops: self.drops.clone(),
        panic_on_drop: false,
      }
    }
  }
  impl Drop for Item {
    fn drop(&mut self) {
      self.drops.set(self.drops.get() + 1);
      assert!(!self.panic_on_drop, "prefix drop");
    }
  }
  let clones = Rc::new(Cell::new(0));
  let drops = Rc::new(Cell::new(0));
  let item = |panic_on_drop| Item {
    clones: clones.clone(),
    drops: drops.clone(),
    panic_on_drop,
  };
  let mut handle = SliceMutN::<_, 0>::with_capacity(4);
  assert!(catch_unwind(AssertUnwindSafe(|| handle.resize(3, item(false)))).is_err());
  assert_eq!(handle.len(), 1);
  assert_eq!(drops.get(), 1);
  handle.clear();
  assert_eq!(drops.get(), 2);
  for (slot, flag) in handle
    .spare_capacity_mut()
    .iter_mut()
    .zip([true, false, false, false])
  {
    slot.write(item(flag));
  }
  // SAFETY: The first four slots were initialized directly above.
  unsafe {
    handle.set_len(4);
  }
  handle.advance(2);
  assert!(catch_unwind(AssertUnwindSafe(|| handle.try_reclaim(2))).is_err());
  assert_eq!(drops.get(), 4);
  assert_eq!(handle.len(), 2);
  assert!(handle.try_reclaim(2));
  drop(handle);
  assert_eq!(drops.get(), 6);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn thawing_a_generic_empty_cannot_create_noncopy_inline_elements() {
  use smol_slice::Slice;
  let mut buffer = Slice::<String>::new().try_into_mut().ok().unwrap();
  assert_eq!(buffer.capacity(), 0);
  buffer.resize(1, "owned".to_string());
  #[cfg(smol_slice_atomic)]
  {
    let original = buffer.as_slice().as_ptr();
    let frozen = buffer.try_freeze_shared().ok().unwrap();
    let clone = frozen.clone();
    assert_eq!(clone.as_slice().as_ptr(), original);
    assert!(frozen.try_into_mut().is_err());
    assert_eq!(clone.into_vec(), ["owned"]);
  }
}

#[cfg(smol_slice_atomic)]
#[test]
fn widening_immutable_then_thawing_moves_replacements_into_shared_storage() {
  use core::marker::PhantomData;
  use smol_slice::{Slice, SliceMut};

  struct Element<F>(u8, PhantomData<F>);
  impl Copy for Element<fn(&())> {}
  impl Clone for Element<fn(&())> {
    fn clone(&self) -> Self {
      *self
    }
  }

  let inputs = [Element::<fn(&())>(1, PhantomData), Element(2, PhantomData)];
  let original = Slice::<_>::try_from_slice(&inputs).unwrap();
  let wider: Slice<Element<fn(&'static ())>> = original;
  let mut mutable = wider.try_into_mut().ok().unwrap();
  mutable.as_mut_slice()[0] = Element(9, PhantomData);
  let mut suffix = mutable.split_off(1);
  suffix.as_mut_slice()[0].0 = 7;
  mutable.unsplit(suffix);
  let shared = mutable.try_freeze_shared().ok().unwrap();
  let clone = shared.clone();
  assert_eq!(shared.as_slice().as_ptr(), clone.as_slice().as_ptr());
  assert_eq!([clone.as_slice()[0].0, clone.as_slice()[1].0], [9, 7]);
  assert!(shared.try_into_mut().is_err());
  let moved = clone.try_into_vec().ok().unwrap();
  assert_eq!(moved.len(), 2);

  // Repeat with a positive mutable heap descriptor born with a non-Copy
  // element type. Its existing allocation and element drop owner survive.
  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };
  struct Count(Arc<AtomicUsize>);
  impl Drop for Count {
    fn drop(&mut self) {
      self.0.fetch_add(1, Ordering::Relaxed);
    }
  }
  let drops = Arc::new(AtomicUsize::new(0));
  let v = vec![Count(Arc::clone(&drops)), Count(Arc::clone(&drops))];
  let base = v.as_ptr();
  let owned = SliceMut::<_>::try_from_vec(v).ok().unwrap();
  let frozen = owned.try_freeze_shared().ok().unwrap();
  assert_eq!(frozen.as_slice().as_ptr(), base);
  let peer = frozen.clone();
  drop(frozen);
  assert_eq!(drops.load(Ordering::Relaxed), 0);
  drop(peer);
  assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn automatic_reserve_amortizes_sliding_full_windows() {
  const WIDTH: usize = if cfg!(miri) { 32 } else { 4096 };
  // With a live suffix the prefix may have to detach. Without one it can
  // grow the original allocation. Both paths must avoid copying every step.
  for keep_suffix in [false, true] {
    let whole = vec![0u8; if keep_suffix { 2 * WIDTH } else { WIDTH }];
    let mut buffer = SliceMutN::<_, 0>::from(whole);
    let sibling = keep_suffix.then(|| buffer.split_off(WIDTH));
    let mut moves = 0;
    for i in 0..2 * WIDTH {
      buffer.advance(1);
      let before = buffer.as_slice().as_ptr();
      buffer.extend_from_slice(&[(i % 251) as u8]);
      moves += usize::from(buffer.as_slice().as_ptr() != before);
    }
    let expected: Vec<_> = (WIDTH..2 * WIDTH).map(|i| (i % 251) as u8).collect();
    assert_eq!(buffer.as_slice(), expected);
    assert!(moves <= 4, "moved the active range {moves} times");
    if let Some(sibling) = sibling {
      assert_eq!(sibling.as_slice(), vec![0; WIDTH]);
    }
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn reserve_reclaims_near_maximum_zst_offsets_and_preserves_length_on_overflow() {
  let mut all = SliceMutN::<_, 0>::from(vec![(); usize::MAX]);
  all.advance(1);
  all.reserve(1);
  assert_eq!(all.len(), usize::MAX - 1);
  assert_eq!(all.spare_capacity_mut().len(), 1);
  all.extend_from_slice(&[()]);
  assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| all.reserve(1))).is_err());
  assert_eq!(all.len(), usize::MAX);
  assert_eq!(all.capacity(), usize::MAX);
}

#[cfg(smol_slice_atomic)]
#[test]
fn zero_sized_windows_use_indices_for_adjacency() {
  let mut head = SliceMutN::<_, 0>::from(vec![(); usize::MAX]);
  let mut mid = head.split_off(2);
  let tail = mid.split_off(3);
  let tail = head.try_unsplit(tail).err().unwrap();
  head.try_unsplit(mid).ok().unwrap();
  head.try_unsplit(tail).ok().unwrap();
  assert_eq!(head.len(), usize::MAX);
  let head = head.try_freeze().err().expect("immutable length limit");
  assert_eq!(head.into_vec().len(), usize::MAX);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn two_word_spill_keeps_its_buffer_on_failed_freeze() {
  let data = vec![9u8; 35];
  let base = data.as_ptr();
  let handle = SliceMutN::<_, { 2 * size_of::<usize>() }>::try_from_vec(data).unwrap();
  let handle = handle
    .try_freeze()
    .err()
    .expect("immutable external descriptor needs three words");
  let original = handle.into_vec();
  assert_eq!(original.as_ptr(), base);
}

#[cfg(smol_slice_atomic)]
#[test]
fn independently_coerced_siblings_drop_new_values_through_each_windows_current_type() {
  use core::{any::TypeId, marker::PhantomData};
  use std::sync::{Arc, Mutex};

  type AnyBorrow = fn(&());
  type StaticBorrow = fn(&'static ());
  struct Record<F: 'static> {
    id: u8,
    events: Arc<Mutex<Vec<(u8, TypeId)>>>,
    form: PhantomData<F>,
  }
  impl<F: 'static> Drop for Record<F> {
    fn drop(&mut self) {
      self
        .events
        .lock()
        .unwrap()
        .push((self.id, TypeId::of::<F>()));
    }
  }
  let events = Arc::new(Mutex::new(Vec::new()));
  let make = |id| Record {
    id,
    events: Arc::clone(&events),
    form: PhantomData::<AnyBorrow>,
  };
  let mut original = SliceMutN::<_, 0>::from(vec![make(0), make(1)]);
  // Higher-ranked function pointers are subtypes of the single-lifetime
  // function pointer. Both are 'static types but have distinct TypeIds.
  let mut shorter: SliceMutN<Record<StaticBorrow>, 0> = original.split_off(1);
  shorter.as_mut_slice()[0] = Record {
    id: 2,
    events: Arc::clone(&events),
    form: PhantomData,
  };
  let frozen = shorter.freeze();
  drop(original);
  drop(frozen.try_into_mut().ok().unwrap());
  assert_eq!(
    *events.lock().unwrap(),
    [
      (1, TypeId::of::<StaticBorrow>()),
      (0, TypeId::of::<AnyBorrow>()),
      (2, TypeId::of::<StaticBorrow>()),
    ]
  );
}
