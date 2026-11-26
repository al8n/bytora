use super::vector::{self, AtomicPtr, VecHeader};
use super::{HandleWord, InlineSize, Layout, Slice, SliceMut, Window};
use core::{
  mem::{ManuallyDrop, MaybeUninit},
  num::NonZeroU8,
  ops::Range,
  ptr, slice,
  sync::atomic::Ordering,
};
use std::{boxed::Box, vec::Vec};

// The last address byte is always nonzero. The low field is the visible
// length. A full descriptor stores the original pointer's low bit. A window
// stores its start instead. Limiting the original length
// to MAX_LEN ensures every subsequent subwindow fits without another owner.
const LEN_BITS: u32 = usize::BITS / 2 - 1;
const MAX_LEN: usize = (1 << LEN_BITS) - 1;
// Both full tags lie outside the two inline cursor encodings. Exact even
// buffers with 1..=64 elements use the contiguous 0x80..=0xbf range.
const EVEN_FULL: u8 = 0x20;
const ODD_FULL: u8 = 0x21;
const SHORT: u8 = 0x80;
const WINDOW: usize = 3 << (usize::BITS - 2);

fn start(meta: usize) -> usize {
  if meta & WINDOW != WINDOW {
    0
  } else {
    (meta >> LEN_BITS) & MAX_LEN
  }
}

fn window(start: usize, len: usize) -> usize {
  debug_assert!(start <= MAX_LEN && len <= MAX_LEN - start);
  WINDOW | (start << LEN_BITS) | len
}

fn raw_base<T>(state: *mut (), meta: usize) -> *mut T {
  state
    .wrapping_byte_sub(usize::from(
      meta >> super::super::TAG_SHIFT != ODD_FULL as usize,
    ))
    .cast()
}

impl<T, S: InlineSize> Layout<T, S> {
  pub(crate) fn fits_two_word_vec(len: usize) -> bool {
    Self::TWO_WORD_VEC && vector::AVAILABLE && size_of::<T>() != 0 && len <= MAX_LEN
  }

  pub(super) fn is_two_word_vec(&self) -> bool {
    Self::TWO_WORD_VEC && !Self::inline_tag(self.tag)
  }

  const fn two_word_meta(&self) -> usize {
    // SAFETY: This external descriptor initializes the entire second word.
    // It is disjoint from the atomic first word and spans the trailing tag.
    usize::from_le(unsafe { ptr::from_ref(self).cast::<usize>().add(1).read() })
  }

  pub(in crate::layout) const fn two_word_len(&self) -> usize {
    #[cfg(target_pointer_width = "32")]
    type Half = u16;
    #[cfg(target_pointer_width = "64")]
    type Half = u32;
    let words = ptr::from_ref(self).cast::<usize>();
    // SAFETY: The initialized second word stores length in its low field.
    // Reading only that field keeps it independent of the final niche byte.
    (Half::from_le(unsafe { words.add(1).cast::<Half>().read() }) as usize) & MAX_LEN
  }

  fn two_word_slot(&self) -> &AtomicPtr<()> {
    // SAFETY: The first pointer is aligned and initialized in the payload's
    // UnsafeCell. Adoption checked vector::AVAILABLE. All shared accesses to
    // this slot are atomic, including on handles that have already promoted.
    unsafe { AtomicPtr::from_ptr(self.payload.as_ptr().cast_mut().cast()) }
  }

  fn exclusive_state(&mut self) -> *mut () {
    // SAFETY: Exclusive access excludes concurrent promotion. The payload
    // contains the complete, initialized pointer including its provenance.
    unsafe { self.payload.as_mut_ptr().cast::<*mut ()>().read() }
  }

  fn two_words(state: *mut (), meta: usize) -> Self {
    debug_assert!(Self::TWO_WORD_VEC && vector::AVAILABLE);
    let mut layout = Self::empty_inline();
    // SAFETY: The first initialized pointer fits wholly in the aligned
    // payload. The metadata write spans the second word and initializes the
    // final address byte to a valid external tag on either endian.
    unsafe {
      layout.payload.as_mut_ptr().cast::<*mut ()>().write(state);
      ptr::from_mut(&mut layout)
        .cast::<usize>()
        .add(1)
        .write(meta.to_le());
    }
    layout
  }

  pub(super) fn two_word_vec<const INLINE_EMPTY: bool>(mut values: Vec<T>) -> Self {
    let len = values.len();
    if values.capacity() != len {
      return Self::two_words(vector::place_header(values).1.cast_mut(), window(0, len));
    }
    let base = values.as_mut_ptr().cast::<()>();
    let adjustment = usize::from(base.addr() & 1 == 0);
    let state = base.wrapping_byte_add(adjustment);
    let tag = if INLINE_EMPTY && len == 0 {
      // The fully initialized zero length supplies every inline cursor byte.
      Self::empty_inline().tag.get()
    } else if (1..=64).contains(&len) && adjustment != 0 {
      SHORT + (len - 1) as u8
    } else {
      EVEN_FULL | (base.addr() & 1) as u8
    };
    let meta = ((tag as usize) << super::super::TAG_SHIFT) | len;
    let layout = Self::two_words(state, meta);
    core::mem::forget(values);
    layout
  }

  fn two_word_header(state: *mut (), meta: usize, strong: usize) -> VecHeader<T> {
    let len = meta & MAX_LEN;
    VecHeader::new(raw_base(state, meta), len, len, strong)
  }

  pub(super) fn two_word_unique(&self) -> bool {
    let state = self.two_word_slot().load(Ordering::Acquire);
    // SAFETY: An even pointer identifies the header pinned by this handle.
    state.addr() & 1 != 0 || unsafe { VecHeader::<T>::strong_count(state) == 1 }
  }

  pub(super) fn two_word_slice(&self) -> &[T] {
    let meta = self.two_word_meta();
    let state = self.two_word_slot().load(Ordering::Acquire);
    let base = if state.addr() & 1 != 0 {
      raw_base(state, meta)
    } else {
      // SAFETY: Acquire observes a fully initialized live header.
      unsafe { (*state.cast::<VecHeader<T>>()).base }
    };
    // SAFETY: The checked, immutable window stays within the original
    // allocation, which remains pinned across a concurrent promotion.
    unsafe { slice::from_raw_parts(base.add(start(meta)), meta & MAX_LEN) }
  }

  pub(super) fn share_two_word(&self, range: Range<usize>) -> Self {
    let meta = self.two_word_meta();
    let new_meta = window(start(meta) + range.start, range.len());
    let slot = self.two_word_slot();
    let state = slot.load(Ordering::Acquire);
    let owner = if state.addr() & 1 != 0 {
      // SAFETY: The candidate covers the entire input Vec with one reference
      // each for the borrowed source and the new peer. Raw state can only be
      // replaced by an aligned header. The source pins the winner on a race.
      unsafe { vector::publish(slot, state, Self::two_word_header(state, meta, 2)) }
    } else {
      // SAFETY: This source pins a published header. Retain it for the peer.
      unsafe { VecHeader::<T>::retain(state) };
      state.cast_const()
    };
    Self::two_words(owner.cast_mut(), new_meta)
  }

  pub(super) fn keep_two_word(&mut self, range: Range<usize>) {
    let meta = self.two_word_meta();
    let new_meta = window(start(meta) + range.start, range.len());
    let state = self.exclusive_state();
    let owner = if state.addr() & 1 != 0 {
      Box::into_raw(Box::new(Self::two_word_header(state, meta, 1))).cast()
    } else {
      state
    };
    // Move this handle's existing ownership into its narrower descriptor.
    *self = Self::two_words(owner, new_meta);
  }

  unsafe fn take_two_word_original(&mut self, meta: usize) -> Option<Vec<T>> {
    let state = self.exclusive_state();
    if state.addr() & 1 != 0 {
      // SAFETY: The caller suppresses Drop and transfers the original Vec.
      // Raw state is always a full descriptor with the actual input capacity.
      let len = meta & MAX_LEN;
      return Some(unsafe { Vec::from_raw_parts(raw_base(state, meta), len, len) });
    }
    // SAFETY: Only a successful take consumes this strong reference.
    unsafe { VecHeader::<T>::try_take(state) }
  }

  pub(super) unsafe fn drop_two_word(&mut self) {
    let state = self.exclusive_state();
    if state.addr() & 1 != 0 {
      // SAFETY: The caller transfers the unpromoted Vec exactly once.
      let meta = self.two_word_meta();
      let len = meta & MAX_LEN;
      unsafe { drop(Vec::<T>::from_raw_parts(raw_base(state, meta), len, len)) };
    } else {
      // SAFETY: The descriptor transfers its owned reference to the header.
      unsafe { VecHeader::<T>::release(state) };
    }
  }
}

impl<T, S: InlineSize> Slice<T, S> {
  #[inline]
  pub(super) fn take_two_word_vec(self) -> Result<Vec<T>, Self> {
    let mut this = ManuallyDrop::new(self);
    let tag = this.layout.tag.get();
    let exact = if tag & 0xc0 == SHORT || tag & 0xfe == EVEN_FULL {
      let state = this.layout.exclusive_state();
      (state.addr() & 1 != 0).then_some(state)
    } else {
      None
    };
    let (data, len, capacity) = if let Some(state) = exact {
      let (data, len) = if tag < SHORT {
        let adjustment = usize::from(tag == EVEN_FULL);
        (
          state.wrapping_byte_sub(adjustment),
          this.layout.two_word_len(),
        )
      } else {
        (
          state.wrapping_byte_sub(1),
          usize::from(tag) - usize::from(SHORT - 1),
        )
      };
      (data.cast(), len, len)
    } else if tag == Layout::<T, S>::empty_inline().tag.get() {
      // The initial inline tag describes a zero endpoint and owns no elements.
      return Ok(Vec::new());
    } else {
      let words = ptr::from_ref(&*this).cast::<HandleWord>();
      // SAFETY: The first word, remaining body bytes and niche tag together
      // transfer one owned handle. A failing take restores that handle.
      let values = unsafe {
        Self::take_two_words(words.read_unaligned(), words.add(1).cast(), this.layout.tag)
      };
      let Some(values) = values else {
        return Err(ManuallyDrop::into_inner(this));
      };
      let mut values = ManuallyDrop::new(values);
      (values.as_mut_ptr(), values.len(), values.capacity())
    };
    // SAFETY: Both paths return every original Vec part, transfer the sole
    // ownership to this Vec and suppress the old descriptor's destructor.
    Ok(unsafe { Vec::from_raw_parts(data, len, capacity) })
  }

  #[inline(never)]
  unsafe fn take_two_words(first: HandleWord, tail: *const u8, tag: NonZeroU8) -> Option<Vec<T>> {
    let mut copy = MaybeUninit::<Self>::uninit();
    let dest = copy.as_mut_ptr().cast::<u8>();
    // SAFETY: The caller supplies the entire live handle. Bytewise copying
    // preserves partial provenance and any uninitialized inline padding.
    let handle = unsafe {
      dest.cast::<HandleWord>().write(first);
      ptr::copy_nonoverlapping(
        tail,
        dest.add(size_of::<HandleWord>()),
        size_of::<HandleWord>() - 1,
      );
      (&raw mut (*copy.as_mut_ptr()).layout.tag).write(tag);
      copy.assume_init()
    };
    if handle.layout.is_inline() {
      return Some(handle.take_inline_vec());
    }
    let mut this = ManuallyDrop::new(handle);
    let meta = this.layout.two_word_meta();
    // SAFETY: On failure the caller resumes owning the same reference.
    let mut original = unsafe { this.layout.take_two_word_original(meta) }?;
    if meta & WINDOW == WINDOW && original.len() != (meta & MAX_LEN) {
      let start = start(meta);
      original.truncate(start + (meta & MAX_LEN));
      drop(original.drain(..start));
    }
    Some(original)
  }

  pub(super) fn take_two_word_mut(self) -> Result<SliceMut<T, S>, Self> {
    let mut this = ManuallyDrop::new(self);
    let meta = this.layout.two_word_meta();
    // SAFETY: On success, the mutable window owns the complete original
    // allocation. It preserves both the allocation and the visible range.
    let Some(original) = (unsafe { this.layout.take_two_word_original(meta) }) else {
      return Err(ManuallyDrop::into_inner(this));
    };
    let start = start(meta);
    let range = start..start + (meta & MAX_LEN);
    Ok(SliceMut::from_window(
      Window::from_vec(original).keep_range(range),
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SliceN;
  use std::vec;

  type TwoWord<T> = SliceN<T, { 2 * size_of::<usize>() }>;

  #[test]
  fn two_word_full_take_preserves_the_input_allocation() {
    if !vector::AVAILABLE {
      return;
    }
    for (len, spare) in [
      (1, 0),
      (7, 0),
      (31, 0),
      (32, 0),
      (64, 0),
      (65, 0),
      (127, 0),
      (4096, 0),
      (64, 1),
      (64, 64),
    ] {
      let mut values = Vec::with_capacity(len + spare);
      values.extend(0..len);
      let (base, capacity) = (values.as_ptr(), values.capacity());
      let mut slice = TwoWord::try_from_vec(values).unwrap();
      assert_eq!(slice.layout.exclusive_state().addr() & 1 != 0, spare == 0);
      slice.advance(0);
      slice.truncate(len);
      assert!(slice.is_unique());
      assert_eq!(slice.as_slice().as_ptr(), base);
      let mut slot = Some(slice);
      let returned = core::hint::black_box(slot.take().unwrap())
        .try_into_vec()
        .ok()
        .unwrap();
      assert!(slot.is_none());
      assert_eq!((returned.as_ptr(), returned.capacity()), (base, capacity));
      assert!(returned.into_iter().eq(0..len));
    }
  }

  #[test]
  fn two_word_vectors_stay_raw_at_exact_capacity_and_take_one_placed_header_otherwise() {
    if !vector::AVAILABLE {
      return;
    }
    fn check<T>(element: T, shared: bool)
    where
      T: Copy + core::fmt::Debug + PartialEq + Send + Sync,
    {
      for len in [0, 1, 64, 65] {
        for spare in [0, 1, 3, 4, 9] {
          for recover in [false, true] {
            let mut values = Vec::with_capacity(len + spare);
            values.resize(len, element);
            let (base, capacity) = (values.as_ptr(), values.capacity());
            // The spare count selects the owner only at this exact capacity.
            assert_eq!(capacity, len + spare);
            let mut source = if shared {
              TwoWord::<T>::try_from_shared_vec(values)
            } else {
              TwoWord::<T>::try_from_vec(values)
            }
            .unwrap();
            if capacity == 0 && !shared {
              // The copying constructor has no allocation to keep.
              assert!(source.layout.is_inline() && source.is_empty());
              assert!(source.try_into_vec().ok().unwrap().is_empty());
              continue;
            }
            assert!(source.is_unique());
            assert_eq!((source.as_slice().as_ptr(), source.len()), (base, len));
            assert!(source.as_slice().iter().all(|value| *value == element));
            let meta = source.layout.two_word_meta();
            let state = source.layout.exclusive_state();
            if spare == 0 {
              // An exact vector is its own owner. The odd state is its pointer,
              // and the tag holds a short length or the pointer's parity.
              let tag = if base.addr() & 1 != 0 {
                ODD_FULL
              } else if (1..=64).contains(&len) {
                SHORT + (len - 1) as u8
              } else {
                EVEN_FULL
              };
              assert_eq!((state.addr() & 1, source.layout.tag.get()), (1, tag));
              assert_eq!(
                (raw_base::<T>(state, meta).cast_const(), meta & MAX_LEN),
                (base, len)
              );
            } else {
              // Any spare capacity takes a header, and the window spans the vector.
              assert_eq!(state.addr() & 1, 0);
              assert_eq!(
                (meta & WINDOW, start(meta), meta & MAX_LEN),
                (WINDOW, 0, len)
              );
              // SAFETY: An even state is the live header this handle owns.
              let (held, strong) = unsafe {
                (
                  (*state.cast::<VecHeader<T>>()).base,
                  VecHeader::<T>::strong_count(state),
                )
              };
              assert_eq!((held, strong), (base, 1));
              let offset = state.addr().wrapping_sub(base.addr());
              if spare * size_of::<T>() >= size_of::<VecHeader<T>>() {
                // Nothing pads a word-aligned tail.
                assert!(align_of::<T>() >= align_of::<VecHeader<T>>());
                assert_eq!(offset, len * size_of::<T>());
              } else {
                // A box lies outside every byte of the vector's allocation.
                assert!(offset >= capacity * size_of::<T>());
              }
            }
            if !recover {
              continue;
            }
            let mut recovered = source.try_into_vec().ok().unwrap();
            assert_eq!(
              (recovered.as_ptr(), recovered.len(), recovered.capacity()),
              (base, len, capacity)
            );
            // The capacity that held a header is the vector's to fill again.
            recovered.resize(capacity, element);
            assert_eq!(recovered.as_ptr(), base);
            assert!(recovered.iter().all(|value| *value == element));
          }
        }
      }
    }
    const { assert!(size_of::<[usize; 4]>() == size_of::<VecHeader<[usize; 4]>>()) };
    for shared in [false, true] {
      check(usize::MAX / 3, shared);
      // One spare element of this size holds the header itself.
      check([usize::MAX / 3; 4], shared);
      // A byte vector's pointer can be odd, and none of these spare counts
      // gives it room for a header.
      check(0x55u8, shared);
    }
  }

  #[test]
  fn two_word_inline_take_keeps_a_pointer_split_across_the_words() {
    #[derive(Clone, Copy)]
    #[repr(C, packed)]
    struct Packed<'a> {
      byte: u8,
      reference: &'a u8,
    }
    let value = 53;
    let handle = TwoWord::try_from_slice(&[Packed {
      byte: 9,
      reference: &value,
    }])
    .unwrap();
    let mut slot = Some(handle);
    let recovered = core::hint::black_box(slot.take().unwrap())
      .try_into_vec()
      .ok()
      .unwrap();
    let reference = recovered[0].reference;
    assert!(ptr::eq(reference, &value));
    assert_eq!((recovered[0].byte, *reference), (9, 53));
  }

  #[test]
  fn two_word_edited_ranges_thaw_at_their_original_offset() {
    if !vector::AVAILABLE {
      return;
    }
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct Padded<'a> {
      n: u8,
      reference: &'a u32,
    }

    let referent = 19;
    let mut original = Vec::with_capacity(32);
    original.extend((0..6).map(|n| Padded {
      n,
      reference: &referent,
    }));
    let (base, capacity) = (original.as_ptr(), original.capacity());
    let mut view = TwoWord::try_from_vec(original).unwrap();
    view.advance(1);
    view.truncate(4);
    let before = view.as_slice();
    let mut middle = view.slice(1..4);
    assert_eq!(before[1].n, 2);
    assert!(ptr::eq(before[1].reference, &referent));
    let middle_base = base.wrapping_add(2);
    assert_eq!(middle.as_slice().as_ptr(), middle_base);
    middle.truncate(2);
    assert!(middle.try_into_vec().is_err());
    let suffix = view.split_off(1);
    let suffix = suffix.try_into_mut().err().unwrap();
    drop(view);
    let mut writable = suffix.try_into_mut().ok().unwrap();
    assert_eq!(writable.as_slice().as_ptr(), middle_base);
    assert_eq!(writable.capacity(), capacity - 2);
    writable.as_mut_slice()[1].n = 42;
    let recovered = writable.into_vec();
    assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
    assert_eq!(
      recovered.iter().map(|v| v.n).collect::<Vec<_>>(),
      [2, 42, 4]
    );
  }

  #[cfg(feature = "std")]
  #[test]
  fn two_word_first_shares_keep_live_borrows_and_one_owner() {
    if !vector::AVAILABLE {
      return;
    }
    let mut original = Vec::with_capacity(64);
    original.extend(0..64);
    let (base, capacity) = (original.as_ptr(), original.capacity());
    let source = TwoWord::try_from_vec(original).unwrap();
    let borrow = source.as_slice();
    let barrier = std::sync::Barrier::new(3);
    let (whole, middle, empty) = std::thread::scope(|threads| {
      let clone = threads.spawn(|| {
        barrier.wait();
        source.clone()
      });
      let slice = threads.spawn(|| {
        barrier.wait();
        source.slice(5..9)
      });
      barrier.wait();
      let empty = source.slice(64..);
      (clone.join().unwrap(), slice.join().unwrap(), empty)
    });
    assert_eq!((borrow[63], whole.as_slice().as_ptr()), (63, base));
    assert_eq!(empty.as_slice().as_ptr(), base.wrapping_add(64));
    assert!(!middle.is_unique());
    let mut middle = middle.try_into_vec().err().unwrap();
    drop((whole, empty, source));
    assert!(middle.is_unique());
    middle.advance(1);
    let recovered = middle.try_into_vec().ok().unwrap();
    assert_eq!((recovered.as_ptr(), recovered.capacity()), (base, capacity));
    assert_eq!(recovered, [6, 7, 8]);
  }

  #[test]
  fn two_word_metadata_preserves_maximum_ranges_and_both_pointer_parities() {
    assert_eq!(start(window(MAX_LEN, 0)), MAX_LEN);
    assert_eq!(start(window(MAX_LEN - 1, 1)), MAX_LEN - 1);
    assert_eq!(window(0, MAX_LEN) & MAX_LEN, MAX_LEN);
    #[repr(align(2))]
    struct Pair([u8; 2]);
    let mut pair = Pair([13, 17]);
    let even = pair.0.as_mut_ptr();
    let odd = even.wrapping_add(1);
    let shift = super::super::super::TAG_SHIFT;
    assert_eq!(
      raw_base::<u8>(odd.cast(), (EVEN_FULL as usize) << shift),
      even
    );
    assert_eq!(raw_base::<u8>(odd.cast(), (SHORT as usize) << shift), even);
    assert_eq!(
      raw_base::<u8>(odd.cast(), (ODD_FULL as usize) << shift),
      odd
    );
  }

  #[test]
  fn two_word_empty_and_zst_inputs_retain_the_inline_rules() {
    let empty = TwoWord::<u8>::try_from_vec(Vec::new()).unwrap();
    assert!(empty.layout.is_inline() && empty.is_empty());
    if vector::AVAILABLE {
      let values = Vec::<u8>::with_capacity(3);
      let base = values.as_ptr();
      let empty = TwoWord::try_from_vec(values).unwrap();
      assert!(empty.is_empty() && empty.is_unique());
      assert_eq!(empty.as_slice().as_ptr(), base);
      let returned = empty.try_into_vec().ok().unwrap();
      assert_eq!((returned.as_ptr(), returned.capacity()), (base, 3));
    }
    let count = TwoWord::<()>::INLINE_CAPACITY;
    assert!(
      TwoWord::try_from_vec(vec![(); count])
        .unwrap()
        .layout
        .is_inline()
    );
    assert_eq!(
      TwoWord::try_from_vec(vec![(); count + 1])
        .err()
        .unwrap()
        .len(),
      count + 1
    );
  }

  #[test]
  fn two_word_forced_vec_and_box_move_noncopy_and_drop_hidden_elements_once() {
    use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    if !vector::AVAILABLE {
      return;
    }
    struct Counted<'a>(&'a AtomicUsize);
    impl Drop for Counted<'_> {
      fn drop(&mut self) {
        self.0.fetch_add(1, Relaxed);
      }
    }
    for mode in [0, 1, 64, usize::MAX] {
      let counts: [AtomicUsize; 5] = core::array::from_fn(|_| AtomicUsize::new(0));
      let spare = if mode == usize::MAX { 0 } else { mode };
      let mut values = Vec::with_capacity(5 + spare);
      values.extend(counts.iter().map(Counted));
      let (base, capacity) = (values.as_ptr(), values.capacity());
      let source = if mode == usize::MAX {
        TwoWord::try_from_boxed_slice(values.into_boxed_slice())
          .ok()
          .unwrap()
      } else {
        TwoWord::try_from_shared_vec(values).ok().unwrap()
      };
      let view = source.slice(1..4);
      let view = view.try_into_mut().err().unwrap();
      drop(source);
      let returned = view.try_into_vec().ok().unwrap();
      assert_eq!(
        (returned.as_ptr(), returned.capacity(), returned.len()),
        (base, capacity, 3)
      );
      assert_eq!(counts.each_ref().map(|v| v.load(Relaxed)), [1, 0, 0, 0, 1]);
      drop(returned);
      assert!(counts.iter().all(|v| v.load(Relaxed) == 1));
    }
    for empty in [Vec::<Counted<'_>>::new(), Vec::with_capacity(7)] {
      let (base, capacity) = (empty.as_ptr(), empty.capacity());
      let source = TwoWord::try_from_shared_vec(empty).ok().unwrap();
      let peer = source.clone();
      drop(source);
      let returned = peer.try_into_vec().ok().unwrap();
      assert_eq!(
        (returned.as_ptr(), returned.capacity(), returned.len()),
        (base, capacity, 0)
      );
    }
    assert!(TwoWord::try_from_shared_vec(vec![()]).is_err());
    assert!(TwoWord::try_from_boxed_slice(vec![()].into_boxed_slice()).is_err());
  }

  #[cfg(feature = "std")]
  #[test]
  fn two_word_panicking_hidden_drop_releases_the_rest_once() {
    use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    if !vector::AVAILABLE {
      return;
    }
    struct Element<'a> {
      count: &'a AtomicUsize,
      panic: bool,
    }
    impl Drop for Element<'_> {
      fn drop(&mut self) {
        self.count.fetch_add(1, Relaxed);
        if self.panic {
          panic!("hidden tail");
        }
      }
    }
    for thaw in [false, true] {
      let counts: [AtomicUsize; 4] = core::array::from_fn(|_| AtomicUsize::new(0));
      let items = counts
        .iter()
        .enumerate()
        .map(|(i, count)| Element {
          count,
          panic: i == 2,
        })
        .collect();
      let mut source = TwoWord::try_from_shared_vec(items).ok().unwrap();
      source.truncate(2);
      let result = std::panic::catch_unwind(|| {
        if thaw {
          drop(source.try_into_mut());
        } else {
          drop(source.try_into_vec());
        }
      });
      assert!(result.is_err());
      assert!(counts.iter().all(|v| v.load(Relaxed) == 1));
    }
  }

  #[test]
  fn two_word_thaw_and_refreeze_retain_the_visible_address_and_reject_shared_windows() {
    if !vector::AVAILABLE {
      return;
    }
    for offset in [0, 2] {
      let values = b"1234567".to_vec();
      let (base, capacity) = (values.as_ptr(), values.capacity());
      let source = TwoWord::try_from_vec(values).unwrap();
      let mut mutable = source.try_into_mut().ok().unwrap();
      mutable.advance(offset);
      mutable.as_mut_slice()[0] = b'*';
      let frozen = mutable.try_freeze().ok().unwrap();
      assert_eq!(frozen.as_slice().as_ptr(), base.wrapping_add(offset));
      assert_eq!(frozen.as_slice()[0], b'*');
      let mut mutable = frozen.try_into_mut().ok().unwrap();
      let suffix = mutable.split_off(2);
      let mut mutable = mutable.try_freeze().err().unwrap();
      assert_eq!(mutable.as_slice().as_ptr(), base.wrapping_add(offset));
      mutable.as_mut_slice()[1] = b'!';
      let suffix = suffix.try_freeze().err().unwrap();
      assert_eq!(suffix.as_slice().as_ptr(), base.wrapping_add(offset + 2));
      drop(suffix);
      let frozen = mutable.try_freeze().ok().unwrap();
      assert_eq!(frozen.as_slice(), b"*!");
      let returned = frozen.try_into_vec().ok().unwrap();
      assert_eq!((returned.as_ptr(), returned.capacity()), (base, capacity));
    }
    let mut head =
      crate::SliceMutN::<_, { 2 * size_of::<usize>() }>::try_from_vec(b"abcd".to_vec()).unwrap();
    let tail = head.split_off(2);
    let tail_base = tail.as_slice().as_ptr();
    drop(head);
    let tail = tail.try_freeze().err().unwrap();
    assert_eq!(tail.as_slice().as_ptr(), tail_base);
  }

  #[cfg(target_pointer_width = "32")]
  #[test]
  fn two_word_length_and_spare_boundaries_keep_the_original() {
    if !vector::AVAILABLE {
      return;
    }
    for spare in [0, 1, 64] {
      let mut values = Vec::with_capacity(MAX_LEN + spare);
      values.resize(MAX_LEN, 7u8);
      let (base, capacity) = (values.as_ptr(), values.capacity());
      let mut source = TwoWord::try_from_vec(values).unwrap();
      source.advance(MAX_LEN - 1);
      let empty = source.slice(1..);
      assert_eq!(empty.as_slice().as_ptr(), base.wrapping_add(MAX_LEN));
      drop(source);
      let recovered = empty.try_into_vec().ok().unwrap();
      assert_eq!(
        (recovered.as_ptr(), recovered.capacity(), recovered.len()),
        (base, capacity, 0)
      );
    }
    let values = vec![1u8; MAX_LEN + 1];
    let (base, capacity) = (values.as_ptr(), values.capacity());
    let returned = TwoWord::try_from_vec(values).err().unwrap();
    assert_eq!(
      (returned.as_ptr(), returned.capacity(), returned.len()),
      (base, capacity, MAX_LEN + 1)
    );
  }
}
