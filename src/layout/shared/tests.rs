use super::*;
use crate::SliceN;
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
fn one_word_recovery_moves_the_visible_inline_elements() {
  use core::mem::MaybeUninit;

  type Word<T> = SliceN<T, { size_of::<usize>() }>;
  const BYTES: usize = size_of::<usize>() - 1;
  fn take<T: Copy>(input: &[T]) -> Vec<T> {
    Word::try_from_slice(input)
      .unwrap()
      .try_into_vec()
      .ok()
      .unwrap()
  }
  let input: [u8; BYTES] = core::array::from_fn(|i| i as u8);
  let full = Word::try_from_vec(input.to_vec()).unwrap();
  for start in 0..=BYTES {
    for end in start..=BYTES {
      let mut slot = Some(full.slice(start..end));
      let output = slot.take().unwrap().try_into_vec().ok().unwrap();
      assert!(slot.is_none());
      assert_eq!(output, input[start..end]);
    }
  }

  // Opaque fragments of a pointer can be stored inline. Reassemble the
  // original pointer after moving the fragments through the vector.
  let pointer = input.as_ptr();
  // SAFETY: A padding-free array of MaybeUninit bytes preserves pointer bytes.
  let mut bytes: [MaybeUninit<u8>; size_of::<*const u8>()] =
    unsafe { core::mem::transmute(pointer) };
  let part = take(&bytes[..BYTES]);
  bytes[..BYTES].copy_from_slice(&part);
  // SAFETY: All bytes came from the same live pointer and kept their offsets.
  let recovered: *const u8 = unsafe { core::mem::transmute(bytes) };
  // SAFETY: input is nonempty and still alive.
  assert_eq!(unsafe { *recovered }, input[0]);

  #[repr(align(2))]
  #[derive(Clone, Copy)]
  struct Padded(MaybeUninit<u8>);

  let mut input = [Padded(MaybeUninit::uninit()); Word::<Padded>::INLINE_CAPACITY];
  input[0].0.write(97);
  let output = take(&input);
  assert_eq!(output.len(), input.len());
  // SAFETY: The first field was initialized before the conversion.
  assert_eq!(unsafe { output[0].0.assume_init() }, 97);

  #[repr(align(256))]
  #[derive(Clone, Copy)]
  struct AlignedZst;

  let input = [AlignedZst; Word::<AlignedZst>::INLINE_CAPACITY];
  let output = take(&input);
  assert_eq!(output.len(), input.len());
  assert!(output.as_ptr().is_aligned());
}

#[test]
fn four_word_recovery_preserves_packed_references_and_inline_padding() {
  use core::mem::MaybeUninit;

  #[derive(Clone, Copy)]
  #[repr(C, packed)]
  struct Packed<'a> {
    byte: u8,
    first: &'a usize,
    second: &'a usize,
    unused: MaybeUninit<usize>,
  }

  #[derive(Clone, Copy)]
  #[repr(C)]
  struct Padded<'a> {
    byte: u8,
    value: &'a usize,
  }

  type FourWords<T> = SliceN<T, { 4 * size_of::<usize>() }>;
  let first = 17;
  let second = 29;
  let packed = Packed {
    byte: 37,
    first: &first,
    second: &second,
    unused: MaybeUninit::uninit(),
  };
  let mut slot = Some(FourWords::try_from_slice(&[packed]).unwrap());
  let taken = slot.take().unwrap();
  assert!(slot.is_none());
  let values = taken.try_into_vec().ok().unwrap();
  let left = values[0].first;
  let right = values[0].second;
  assert_eq!((values[0].byte, *left, *right), (37, 17, 29));
  assert!(core::ptr::eq(left, &first) && core::ptr::eq(right, &second));

  let padded = Padded {
    byte: 19,
    value: &second,
  };
  let values = FourWords::try_from_slice(&[padded])
    .unwrap()
    .try_into_vec()
    .ok()
    .unwrap();
  assert_eq!((values[0].byte, *values[0].value), (19, 29));
}

#[test]
fn a_failed_four_word_take_retains_the_original_window_and_allocation() {
  let values = vec![7u8, 8, 9, 10];
  let (base, capacity) = (values.as_ptr(), values.capacity());
  let source = SliceN::<_, { 4 * size_of::<usize>() }>::try_from_vec(values).unwrap();
  let peer = source.clone();
  let mut slot = Some(source);
  slot.as_mut().unwrap().advance(1);
  let original = slot.take().unwrap().try_into_vec().err().unwrap();
  assert!(slot.is_none());
  assert_eq!(original.as_slice(), &[8, 9, 10]);
  assert_eq!(original.as_slice().as_ptr(), base.wrapping_add(1));
  drop(peer);
  let values = original.try_into_vec().ok().unwrap();
  assert_eq!(values, [8, 9, 10]);
  assert_eq!((values.as_ptr(), values.capacity()), (base, capacity));
}

#[test]
fn unique_vector_owners_return_their_allocation_from_either_conversion() {
  static INPUT: [usize; 6] = [3, 5, 7, 11, 13, 17];

  fn check<S: InlineSize>() {
    // Word-sized elements keep the spare tail aligned for an embedded header.
    // An exact vector is a raw marker, one spare element boxes the header
    // and eight hold it in the allocation.
    for (spare, wide) in [
      (0, false),
      (0, true),
      (1, false),
      (1, true),
      (8, false),
      (8, true),
    ] {
      for (trim, fallible) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut values = Vec::with_capacity(INPUT.len() + spare);
        values.extend_from_slice(&INPUT);
        let (base, capacity) = (values.as_ptr(), values.capacity());
        // The spare count selects the owner encoding only at this exact capacity.
        assert_eq!(capacity, INPUT.len() + spare);
        let mut source = if wide {
          Slice::<usize, S>::try_from_vec(values)
        } else {
          Slice::<usize, S>::try_from_shared_vec(values)
        }
        .unwrap();
        let owner = source.layout.owner();
        let header = owner.addr().wrapping_sub(base.addr());
        match spare {
          0 => assert_eq!(
            vector::full_len(owner),
            vector::AVAILABLE.then_some(INPUT.len())
          ),
          1 => assert!(!vector::raw(owner) && header >= capacity * size_of::<usize>()),
          _ => assert!(!vector::raw(owner) && header < capacity * size_of::<usize>()),
        }
        let visible = if trim {
          source.advance(1);
          source.truncate(3);
          assert_eq!(vector::full_len(source.layout.owner()), None);
          &INPUT[1..4]
        } else {
          &INPUT[..]
        };
        let mut slot = Some(source);
        let taken = slot.take().unwrap();
        assert!(slot.is_none());
        let recovered = if fallible {
          taken.try_into_vec().ok().unwrap()
        } else {
          taken.into_vec()
        };
        assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
        assert_eq!(recovered, visible);
      }
    }
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  check::<typenum::U24>();
  check::<typenum::U32>();
  check::<typenum::U40>();
}

#[test]
fn shared_vector_owners_keep_the_handle_and_copy_until_the_last_peer() {
  static INPUT: [usize; 6] = [3, 5, 7, 11, 13, 17];

  fn check<S: InlineSize>() {
    for (spare, trim) in [
      (0, false),
      (0, true),
      (1, false),
      (1, true),
      (8, false),
      (8, true),
    ] {
      let mut values = Vec::with_capacity(INPUT.len() + spare);
      values.extend_from_slice(&INPUT);
      let (base, capacity) = (values.as_ptr(), values.capacity());
      // The spare count selects the owner encoding only at this exact capacity.
      assert_eq!(capacity, INPUT.len() + spare);
      let mut source = Slice::<usize, S>::try_from_shared_vec(values).unwrap();
      let (start, visible) = if trim {
        source.advance(1);
        source.truncate(3);
        (1, &INPUT[1..4])
      } else {
        (0, &INPUT[..])
      };
      let peer = source.clone();
      assert!(!vector::raw(source.layout.owner()));
      let mut slot = Some(source);
      let source = slot.take().unwrap().try_into_vec().err().unwrap();
      assert!(slot.is_none());
      assert_eq!(source.as_slice().as_ptr(), base.wrapping_add(start));
      let copied = source.into_vec();
      assert_eq!(copied, visible);
      // Both buffers are live, so the copy lies outside the original allocation.
      let offset = copied.as_ptr().addr().wrapping_sub(base.addr());
      assert!(offset >= capacity * size_of::<usize>());
      assert_eq!(peer.as_slice().as_ptr(), base.wrapping_add(start));
      let recovered = peer.into_vec();
      assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
      assert_eq!(recovered, visible);
    }
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  check::<typenum::U24>();
  check::<typenum::U32>();
  check::<typenum::U40>();
}

#[test]
fn frozen_windows_copy_while_shared_and_return_their_allocation_once_unique() {
  fn check<S: InlineSize>() {
    let mut values = Vec::with_capacity(8);
    values.extend_from_slice(b"abcdef");
    let (base, capacity) = (values.as_ptr(), values.capacity());
    let mutable = SliceMut::<u8, S>::try_from_vec(values).ok().unwrap();
    let mut frozen = mutable.try_freeze_shared().ok().unwrap();
    assert_eq!(frozen.as_slice().as_ptr(), base);
    frozen.advance(1);
    frozen.truncate(3);
    let peer = frozen.clone();
    let frozen = frozen.try_into_vec().err().unwrap();
    assert_eq!(frozen.as_slice().as_ptr(), base.wrapping_add(1));
    let copied = frozen.into_vec();
    assert_eq!(copied, b"bcd");
    assert!(copied.as_ptr().addr().wrapping_sub(base.addr()) >= capacity);
    let recovered = peer.try_into_vec().ok().unwrap();
    assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
    assert_eq!(recovered, b"bcd");
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  check::<typenum::U24>();
  check::<typenum::U32>();
  check::<typenum::U40>();
}

#[test]
fn static_and_opaque_owners_keep_the_handle_and_copy_on_demand() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  static DATA: &[u8] = b"static window";

  fn check<S: InlineSize>() {
    let mut borrowed = Slice::<u8, S>::try_from_static(DATA).unwrap();
    borrowed.advance(7);
    let borrowed = borrowed.try_into_vec().err().unwrap();
    assert_eq!(borrowed.as_slice().as_ptr(), DATA[7..].as_ptr());
    let copied = borrowed.into_vec();
    assert_eq!(copied, b"window");
    assert_ne!(copied.as_ptr(), DATA[7..].as_ptr());

    let drops = Arc::new(AtomicUsize::new(0));
    let mut opaque = Slice::<u8, S>::try_from_owner(ArrayOwner::new(&drops))
      .ok()
      .unwrap();
    let base = opaque.as_slice().as_ptr();
    opaque.advance(7);
    let opaque = opaque.try_into_vec().err().unwrap();
    assert_eq!(opaque.as_slice().as_ptr(), base.wrapping_add(7));
    assert_eq!(drops.load(Relaxed), 0);
    let copied = opaque.into_vec();
    assert_eq!(copied, b"owner");
    assert_eq!(drops.load(Relaxed), 1);
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  check::<typenum::U24>();
  check::<typenum::U32>();
  check::<typenum::U40>();
}

#[test]
fn inline_and_empty_windows_convert_into_a_fresh_vector() {
  fn inline<S: InlineSize>() {
    for fallible in [false, true] {
      let mut source = Slice::<u8, S>::try_from_slice(b"inline").unwrap();
      assert!(source.layout.is_inline());
      source.advance(1);
      let mut slot = Some(source);
      let taken = slot.take().unwrap();
      assert!(slot.is_none());
      let values = if fallible {
        taken.try_into_vec().ok().unwrap()
      } else {
        taken.into_vec()
      };
      assert_eq!(values, b"nline");
    }
  }
  fn empty<S: InlineSize>() {
    for source in [
      Slice::<u8, S>::new(),
      Slice::<u8, S>::try_from_vec(Vec::new()).unwrap(),
    ] {
      assert!(source.layout.is_inline());
      assert!(source.clone().into_vec().is_empty());
      assert!(source.try_into_vec().ok().unwrap().is_empty());
    }
    // An exact vector without an allocation is a raw marker of length zero.
    let unallocated = Slice::<u8, S>::try_from_shared_vec(Vec::new()).unwrap();
    assert_eq!(
      vector::full_len(unallocated.layout.owner()),
      vector::AVAILABLE.then_some(0)
    );
    let values = unallocated.try_into_vec().ok().unwrap();
    assert_eq!((values.len(), values.capacity()), (0, 0));
  }
  inline::<crate::DefaultSize>();
  inline::<typenum::U24>();
  inline::<typenum::U32>();
  inline::<typenum::U40>();
  empty::<typenum::U0>();
  empty::<crate::DefaultSize>();
  empty::<typenum::U24>();
  empty::<typenum::U32>();
  empty::<typenum::U40>();

  // The zero-budget constant is a static window, so it converts by copying.
  let unowned = Slice::<u8, typenum::U0>::empty();
  let unowned = unowned.try_into_vec().err().unwrap();
  assert!(unowned.into_vec().is_empty());
}

#[test]
fn vector_construction_uses_inline_capacity_and_returns_unusable_input() {
  type OneWord = SliceN<u8, { size_of::<usize>() }>;
  let short = vec![3u8; OneWord::INLINE_CAPACITY];
  let inline = OneWord::try_from_vec(short).unwrap();
  assert!(inline.layout.is_inline());
  assert_eq!(inline.as_slice(), &[3; OneWord::INLINE_CAPACITY]);
  let input = vec![7u8; OneWord::INLINE_CAPACITY + 1];
  let (ptr, cap) = (input.as_ptr(), input.capacity());
  let returned = OneWord::try_from_vec(input).err().unwrap();
  assert_eq!(returned, [7; OneWord::INLINE_CAPACITY + 1]);
  assert_eq!((returned.as_ptr(), returned.capacity()), (ptr, cap));
}

#[test]
fn zero_budget_vectors_keep_allocations_and_use_static_unallocated_empties() {
  fn copy<T: Copy + Send + Sync, S: InlineSize>(values: Vec<T>) -> Result<Slice<T, S>, Vec<T>> {
    Slice::try_from_vec(values)
  }
  let short = vec![7u8];
  let ptr = short.as_ptr();
  let shared = copy::<_, typenum::U0>(short).unwrap();
  assert_eq!(shared.as_slice().as_ptr(), ptr);
  let empty = SliceN::<std::string::String, 0>::try_from(Vec::new()).unwrap();
  assert_eq!(empty.layout.tag, COMPACT_STATIC);
  assert!(!empty.layout.uses_external_owner());
  assert!(empty.try_into_vec().is_err());
  let allocated = Vec::<std::string::String>::with_capacity(1);
  let (base, capacity) = (allocated.as_ptr(), allocated.capacity());
  let empty = SliceN::<_, 0>::try_from(allocated).unwrap();
  assert!(empty.is_empty());
  assert_eq!(empty.as_slice().as_ptr(), base);
  let recovered = empty.try_into_vec().ok().unwrap();
  assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
}

#[test]
fn allocated_copy_vectors_keep_their_buffers_at_short_and_empty_lengths() {
  fn check<S: InlineSize>() {
    for (len, capacity) in [(0, 1), (1, 1), (1, 2), (3, 64)] {
      let mut input = Vec::with_capacity(capacity);
      input.extend_from_slice(&b"abc"[..len]);
      let (base, capacity) = (input.as_ptr(), input.capacity());
      let source = Slice::<u8, S>::try_from_vec(input).unwrap();
      assert_eq!(source.as_slice(), &b"abc"[..len]);
      assert_eq!(source.as_slice().as_ptr(), base);
      let peer = source.clone();
      assert_eq!(peer.as_slice().as_ptr(), base);
      drop(peer);
      let recovered = source.try_into_vec().ok().unwrap();
      assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
    }
    let empty = Slice::<u8, S>::try_from_vec(Vec::new()).unwrap();
    assert!(empty.layout.is_inline());
    assert!(empty.is_empty() && empty.is_unique());
    let cloned = empty.clone();
    assert!(!cloned.layout.uses_external_owner());
    assert_eq!(cloned.try_into_mut().ok().unwrap().capacity(), 0);
    assert!(empty.try_into_vec().ok().unwrap().is_empty());
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  #[cfg(target_pointer_width = "32")]
  check::<typenum::U16>();
  check::<typenum::U32>();
  check::<typenum::U128>();
  check::<typenum::U136>();
  assert!(
    Slice::<()>::try_from_vec(vec![(); 2])
      .unwrap()
      .layout
      .is_inline()
  );
}

#[test]
fn unshared_vec_preserves_all_raw_parts_across_window_edits() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Count<'a>(&'a AtomicUsize);
  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.0.fetch_add(1, Relaxed);
    }
  }
  for thaw in [false, true] {
    let dropped = [const { AtomicUsize::new(0) }; 5];
    let mut values = Vec::with_capacity(dropped.len());
    values.extend(dropped.iter().map(Count));
    let (base, cap) = (values.as_ptr(), values.capacity());
    let mut source = SliceN::<_, 0>::try_from_shared_vec(values).ok().unwrap();
    source.advance(1);
    source.truncate(3);
    source.advance(1);
    assert_eq!(vector::raw(source.layout.owner()), vector::AVAILABLE);
    assert!(source.is_unique());
    assert_eq!(source.as_slice().as_ptr(), base.wrapping_add(2));
    assert_eq!(dropped.each_ref().map(|d| d.load(Relaxed)), [0; 5]);
    let visible = if thaw {
      let mutable = source.try_into_mut().ok().unwrap();
      assert_eq!(mutable.as_slice().as_ptr(), base.wrapping_add(2));
      mutable.into_vec()
    } else {
      source.try_into_vec().ok().unwrap()
    };
    assert_eq!(
      (visible.as_ptr(), visible.len(), visible.capacity()),
      (base, 2, cap)
    );
    assert_eq!(dropped.each_ref().map(|d| d.load(Relaxed)), [1, 1, 0, 0, 1]);
    drop(visible);
    assert_eq!(dropped.each_ref().map(|d| d.load(Relaxed)), [1; 5]);
  }
}

#[test]
fn exact_vectors_recover_their_allocation_before_and_after_trimming_or_promotion() {
  fn check<S: InlineSize>() {
    for (trim, share) in [(false, false), (false, true), (true, false), (true, true)] {
      let values = Vec::from(*b"abcdef");
      let (base, capacity) = (values.as_ptr(), values.capacity());
      assert_eq!(capacity, values.len());
      let mut source = Slice::<_, S>::try_from_shared_vec(values).unwrap();
      let owner = source.layout.owner();
      assert_eq!(
        vector::full_len(owner),
        vector::AVAILABLE.then_some(capacity)
      );
      source.advance(0);
      source.truncate(source.len());
      assert_eq!(source.layout.owner(), owner);
      if trim {
        source.advance(1);
        source.truncate(3);
        assert_eq!(vector::full_len(source.layout.owner()), None);
      }
      if share {
        let peer = source.clone();
        assert!(!vector::raw(source.layout.owner()));
        source = source.try_into_vec().unwrap_err();
        assert_eq!(source.as_slice(), peer.as_slice());
        drop(peer);
      }
      let values = source.try_into_vec().ok().unwrap();
      assert_eq!((values.as_ptr(), values.capacity()), (base, capacity));
      assert_eq!(
        values.as_slice(),
        if trim { &b"bcd"[..] } else { &b"abcdef"[..] }
      );
    }
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  check::<typenum::U32>();
}

#[test]
fn whole_vec_round_trip_preserves_even_an_empty_allocation() {
  for (len, capacity) in [(0, 0), (0, 1), (0, 32), (4, 4), (4, 5), (4, 32)] {
    let mut original = Vec::with_capacity(capacity);
    original.extend(0u32..len);
    let (base, capacity) = (original.as_ptr(), original.capacity());
    let source = Slice::<u32>::try_from_shared_vec(original).unwrap();
    assert_eq!(
      vector::raw(source.layout.owner()),
      vector::AVAILABLE && capacity == len as usize
    );
    let result = source.try_into_vec().ok().unwrap();
    assert_eq!(
      (result.as_ptr(), result.len(), result.capacity()),
      (base, len as usize, capacity)
    );
  }
}

#[cfg(feature = "std")]
#[test]
fn concurrent_first_vec_shares_keep_one_owner_and_the_borrowed_window() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  use std::sync::Barrier;

  struct Count<'a> {
    drops: &'a AtomicUsize,
    value: u8,
  }
  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.drops.fetch_add(1, Relaxed);
    }
  }

  fn exercise<S: InlineSize>(spare: usize, trim: bool)
  where
    for<'a> Slice<Count<'a>, S>: Send + Sync,
  {
    let drops = [const { AtomicUsize::new(0) }; 10];
    let mut input = Vec::with_capacity(10 + spare);
    input.extend(drops.iter().enumerate().map(|(value, drops)| Count {
      drops,
      value: value as u8,
    }));
    let (base, capacity) = (input.as_ptr(), input.capacity());
    let mut source = Slice::<_, S>::try_from_shared_vec(input).ok().unwrap();
    if trim {
      source.advance(1);
      source.truncate(8);
    }
    let (start, len) = if trim { (1, 8) } else { (0, 10) };
    assert_eq!(
      vector::full_len(source.layout.owner()),
      (vector::AVAILABLE && spare == 0 && !trim).then_some(10)
    );
    assert_eq!(
      vector::raw(source.layout.owner()),
      vector::AVAILABLE && spare == 0
    );
    let borrowed = source.as_slice();
    let barrier = Barrier::new(4);
    let peers = std::thread::scope(|threads| {
      let handles: Vec<_> = (0..4)
        .map(|i| {
          let (source, barrier) = (&source, &barrier);
          threads.spawn(move || {
            barrier.wait();
            match i {
              0 => source.clone(),
              1 => source.slice(2..5),
              2 => {
                let mut dest = Slice::new();
                dest.clone_from(source);
                dest
              }
              _ => source.slice(8..),
            }
          })
        })
        .collect();
      handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>()
    });
    let winner = source.layout.owner();
    assert!(!vector::raw(winner));
    assert!(peers.iter().all(|p| p.layout.owner() == winner));
    assert_eq!(borrowed[7].value, (start + 7) as u8);
    assert_eq!(peers[1].as_slice().as_ptr(), base.wrapping_add(start + 2));
    assert_eq!(peers[1].as_slice()[0].value, (start + 2) as u8);
    assert_eq!(peers[3].as_slice().as_ptr(), base.wrapping_add(start + 8));
    assert_eq!(peers[3].len(), len - 8);
    assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [0; 10]);
    let source = source.try_into_vec().err().unwrap();
    assert_eq!(source.as_slice().as_ptr(), base.wrapping_add(start));
    assert_eq!(source.len(), len);
    assert_eq!(source.as_slice()[0].value, start as u8);
    drop(peers);
    let result = source.try_into_vec().ok().unwrap();
    assert_eq!(
      (result.as_ptr(), result.len(), result.capacity()),
      (base, len, capacity)
    );
    assert_eq!(result[0].value, start as u8);
    assert_eq!(
      drops.each_ref().map(|d| d.load(Relaxed)),
      core::array::from_fn(|i| usize::from(trim && (i == 0 || i == 9)))
    );
    drop(result);
    assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [1; 10]);
  }

  type ThreeWords = <[u8; 3 * size_of::<usize>()] as crate::AssocInlineSize>::Size;
  for spare in [0, 1, 8] {
    exercise::<ThreeWords>(spare, true);
    exercise::<crate::DefaultSize>(spare, true);
    exercise::<typenum::U8192>(spare, true);
  }
  exercise::<ThreeWords>(0, false);
  exercise::<crate::DefaultSize>(0, false);
  exercise::<typenum::U0>(0, false);
}

#[test]
fn shared_clones_retain_one_owner_each_and_release_the_last_reference() {
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  struct Item<'a>(usize, &'a AtomicUsize);
  impl Drop for Item<'_> {
    fn drop(&mut self) {
      self.1.fetch_add(1, Relaxed);
    }
  }
  fn refs<S: InlineSize>(source: &Slice<Item<'_>, S>) -> usize {
    let owner = source.layout.owner();
    assert!(!vector::raw(owner));
    // SAFETY: This test only passes a live, directly owned Vec handle.
    unsafe { vector::Vector::<Item<'_>>::strong_count(owner) }
  }
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let drops = AtomicUsize::new(0);
    let len = 32;
    let input: Vec<_> = (0..len).map(|n| Item(n, &drops)).collect();
    let base = input.as_ptr();
    let mut original = SliceN::<_, BYTES>::try_from_shared_vec(input).ok().unwrap();
    assert_eq!(original.as_slice().as_ptr(), base);
    assert_eq!(vector::raw(original.layout.owner()), vector::AVAILABLE);
    assert!(original.is_unique());
    original.advance(10);
    let mut peer = original.clone();
    assert_eq!(refs(&peer), 2);
    assert_eq!(peer.as_slice().as_ptr(), base.wrapping_add(10));
    peer.advance(3);
    assert_eq!(original.as_slice()[0].0, 10);
    let mut last = peer.clone();
    assert_eq!(refs(&last), 3);
    peer.advance(peer.len());
    assert!(peer.as_slice().is_empty());
    drop(original);
    drop(peer);
    assert_eq!(refs(&last), 1);
    assert_eq!(last.as_slice()[0].0, 13);
    let short = last.slice(1..3);
    let nested = short.slice(1..);
    let empty_range = nested.slice(1..);
    assert_eq!(short.as_slice().as_ptr(), base.wrapping_add(14));
    assert_eq!(refs(&short), 4);
    drop(short);
    last.advance(last.len());
    let empty = last.clone();
    drop(last);
    assert!(empty.is_empty());
    assert_eq!(refs(&empty), 3);
    drop(empty);
    assert_eq!(nested.as_slice().len(), 1);
    assert_eq!(nested.as_slice()[0].0, 15);
    drop(nested);
    assert_eq!(refs(&empty_range), 1);
    assert!(empty_range.is_empty());
    assert_eq!(empty_range.as_slice().as_ptr(), base.wrapping_add(16));
    assert_eq!(drops.load(Relaxed), 0);
    drop(empty_range);
    assert_eq!(drops.load(Relaxed), len);
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

#[cfg(feature = "std")]
#[test]
fn wide_clone_from_keeps_the_new_owner_when_the_old_owner_panics() {
  struct PanicsOnDrop;
  impl AsRef<[u8]> for PanicsOnDrop {
    fn as_ref(&self) -> &[u8] {
      b"old"
    }
  }
  impl Drop for PanicsOnDrop {
    fn drop(&mut self) {
      panic!("intentional owner destructor panic");
    }
  }

  let mut destination = SliceN::<u8, 128>::try_from_owner(PanicsOnDrop)
    .ok()
    .unwrap();
  let values = vec![1, 2, 3];
  let base = values.as_ptr();
  let source = SliceN::<u8, 128>::try_from_vec(values).unwrap();
  assert!(
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      destination.clone_from(&source);
    }))
    .is_err()
  );
  drop(source);
  let recovered = destination.try_into_vec().ok().unwrap();
  assert_eq!(recovered.as_ptr(), base);
  assert_eq!(recovered, [1, 2, 3]);
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
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: crate::AssocInlineSize,
  {
    let drops = [const { AtomicUsize::new(0) }; 5];
    let mut values = Vec::with_capacity(12);
    values.extend(drops.iter().map(Count));
    let (base, capacity) = (values.as_ptr(), values.capacity());
    let original = SliceN::<_, BYTES>::try_from_shared_vec(values)
      .ok()
      .unwrap();
    let middle = original.slice(1..4);
    assert_eq!(middle.as_slice().as_ptr(), base.wrapping_add(1));
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
  check::<0>();
  check::<{ 3 * size_of::<usize>() }>();
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
  for (panic_at, share, spare) in [
    (0, false, 0),
    (3, false, 0),
    (0, true, 0),
    (3, true, 0),
    (0, true, 1),
    (3, true, 1),
    (0, true, 8),
    (3, true, 8),
  ] {
    let drops = [const { AtomicUsize::new(0) }; 4];
    let mut values = Vec::with_capacity(drops.len() + spare);
    values.extend(drops.iter().enumerate().map(|(i, drops)| Count {
      drops,
      panic: i == panic_at,
    }));
    let mut original = SliceN::<_, 0>::try_from(values).ok().unwrap();
    let middle = if share {
      let middle = original.slice(1..3);
      drop(original);
      middle
    } else {
      original.advance(1);
      original.truncate(2);
      original
    };
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
