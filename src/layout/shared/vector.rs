use super::Arc;
#[cfg(not(feature = "extra-platforms"))]
pub(super) use core::sync::atomic::AtomicPtr;
use core::sync::atomic::Ordering;
use core::{mem::ManuallyDrop, ptr};
#[cfg(feature = "extra-platforms")]
pub(super) use portable_atomic::AtomicPtr;
use std::vec::Vec;

// Direct Arc pointers end in 00, tagged managed pointers and the static
// sentinel end in 01. A raw Vec stores three element counts and the tag 11.
// The data pointer stays unmodified, including for an odd-address allocation.
const TAG: usize = 3;
const BITS: u32 = (usize::BITS - 2) / 3;
const MASK: usize = (1 << BITS) - 1;

// Some 32-bit ABIs align pointers and handles to two bytes, while their
// atomic pointers require four. Keep those handles on the eager Arc path.
pub(super) const AVAILABLE: bool = align_of::<*const ()>() >= 4
  && align_of::<usize>() >= align_of::<AtomicPtr<()>>()
  && size_of::<*mut ()>() == size_of::<AtomicPtr<()>>()
  && size_of::<*mut ()>() % align_of::<AtomicPtr<()>>() == 0;

pub(super) fn raw(owner: *const ()) -> bool {
  AVAILABLE && owner.addr() & TAG == TAG
}

pub(super) fn full_spare(owner: *const ()) -> Option<usize> {
  // The low fields are the tag, skipped prefix and hidden suffix. An untouched
  // raw window has exactly TAG in those bits, even with unused capacity.
  const SPARE_SHIFT: u32 = 2 + 2 * BITS;
  if AVAILABLE && owner.addr() & ((1 << SPARE_SHIFT) - 1) == TAG {
    Some(owner.addr() >> SPARE_SHIFT)
  } else {
    None
  }
}

fn encode(prefix: usize, suffix: usize, spare: usize) -> Option<*const ()> {
  if prefix > MASK || suffix > MASK || spare > MASK {
    return None;
  }
  let counts = prefix | (suffix << BITS) | (spare << (2 * BITS));
  Some(ptr::without_provenance((counts << 2) | TAG))
}

fn counts(owner: *const ()) -> (usize, usize, usize) {
  let fields = owner.addr() >> 2;
  (fields & MASK, (fields >> BITS) & MASK, fields >> (2 * BITS))
}

pub(super) fn initial<T>(values: &Vec<T>) -> Option<*const ()> {
  if !AVAILABLE || size_of::<T>() == 0 {
    return None;
  }
  encode(0, 0, values.capacity() - values.len())
}

pub(super) fn narrow(owner: *const (), front: usize, back: usize) -> Option<*const ()> {
  let (prefix, suffix, spare) = counts(owner);
  // Both additions are bounded by the original vector's initialized length.
  encode(prefix + front, suffix + back, spare)
}

// A direct Arc owns these raw parts. Speculative Arc candidates contain the
// same header under ManuallyDrop and have no ownership of the allocation.
pub(super) struct Vector<T> {
  pub(super) base: *const T,
  len: usize,
  capacity: usize,
}

// SAFETY: An owning header has Vec<T>'s ownership and thread-safety bounds.
// Its raw fields stay immutable while shared. Extraction consumes ownership.
unsafe impl<T: Send> Send for Vector<T> {}
unsafe impl<T: Sync> Sync for Vector<T> {}

impl<T> Vector<T> {
  pub(super) fn from_vec(values: Vec<T>) -> Self {
    let mut values = ManuallyDrop::new(values);
    Self {
      base: values.as_mut_ptr(),
      len: values.len(),
      capacity: values.capacity(),
    }
  }

  // The caller supplies the raw descriptor's original geometry and window.
  // Only consuming that descriptor, or winning its promotion, transfers
  // ownership. Until then this header must stay under ManuallyDrop.
  unsafe fn parts(owner: *const (), data: *const (), len: usize) -> ManuallyDrop<Self> {
    let (prefix, suffix, spare) = counts(owner);
    let len = prefix + len + suffix;
    ManuallyDrop::new(Self {
      base: data.cast::<T>().wrapping_sub(prefix),
      len,
      capacity: len + spare,
    })
  }

  pub(super) unsafe fn take_raw(owner: *const (), data: *const (), len: usize) -> Self {
    // SAFETY: The caller consumes the unpromoted descriptor.
    ManuallyDrop::into_inner(unsafe { Self::parts(owner, data, len) })
  }

  pub(super) fn into_vec(self) -> Vec<T> {
    let this = ManuallyDrop::new(self);
    // SAFETY: We own exactly the original buffer and its initialized elements.
    unsafe { Vec::from_raw_parts(this.base.cast_mut(), this.len, this.capacity) }
  }
}

impl<T> Drop for Vector<T> {
  fn drop(&mut self) {
    // SAFETY: Only the owning header is dropped. All other candidates
    // suppress this destructor, including when allocation unwinds.
    unsafe {
      drop(Vec::from_raw_parts(
        self.base.cast_mut(),
        self.len,
        self.capacity,
      ))
    };
  }
}

// Publish a complete Arc in one step. Readers can retain the winning Arc as
// soon as they observe it, including when an interrupt preempts a contender.
pub(super) unsafe fn promote<T>(
  slot: &AtomicPtr<()>,
  owner: *const (),
  data: *const (),
  len: usize,
) -> *const () {
  // SAFETY: The caller holds the source live, with this immutable geometry.
  let candidate = Arc::new(unsafe { Vector::<T>::parts(owner, data, len) });
  let candidate = Arc::into_raw(candidate);
  let published = candidate.cast::<()>().cast_mut();
  match slot.compare_exchange(
    owner.cast_mut(),
    published,
    Ordering::AcqRel,
    Ordering::Acquire,
  ) {
    Ok(_) => {
      // The source's ownership transfers to Arc<Vector<T>>. ManuallyDrop<T>
      // has T's size, alignment and validity, as required by Arc::from_raw.
      published
    }
    Err(winner) => {
      // SAFETY: This allocation was never published. Its inert header keeps
      // the winner's original vector entirely untouched.
      unsafe { drop(Arc::from_raw(candidate)) };
      winner
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Slice;

  #[test]
  fn geometry_leaves_an_odd_data_pointer_untagged() {
    #[repr(align(2))]
    struct Bytes([u8; 8]);
    let mut bytes = Bytes([0; 8]);
    let base = bytes.0.as_mut_ptr().wrapping_add(1);
    assert_eq!(base.addr() & 1, 1);
    let raw = encode(2, 1, 1).unwrap();
    // SAFETY: Use the three actual in-bounds ranges to inspect the geometry.
    // The inert header never takes ownership of the local array.
    let parts = unsafe { Vector::<u8>::parts(raw, base.wrapping_add(2).cast(), 3) };
    assert_eq!(
      (parts.base, parts.len, parts.capacity),
      (base.cast_const(), 6, 7)
    );
  }

  #[test]
  fn an_unrepresentable_count_preserves_the_original_allocation() {
    for trim_front in [false, true] {
      let input = std::vec![3u8; MASK + 4];
      let (base, capacity) = (input.as_ptr(), input.capacity());
      let mut source = Slice::<u8>::try_from_shared_vec(input).unwrap();
      assert_eq!(raw(source.layout.owner()), AVAILABLE);
      if trim_front {
        source.advance(MASK + 1);
      } else {
        source.truncate(3);
      }
      assert!(!raw(source.layout.owner()));
      let result = source.try_into_vec().ok().unwrap();
      assert_eq!((result.as_ptr(), result.capacity()), (base, capacity));
      assert_eq!(result, [3; 3]);
    }
    let mut values = Vec::with_capacity(MASK + 2);
    values.push(7u8);
    let (base, capacity) = (values.as_ptr(), values.capacity());
    let source = Slice::<u8>::try_from_shared_vec(values).unwrap();
    assert!(!raw(source.layout.owner()));
    let result = source.try_into_vec().ok().unwrap();
    assert_eq!((result.as_ptr(), result.capacity()), (base, capacity));
  }
}
