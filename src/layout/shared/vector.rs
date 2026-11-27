#[cfg(not(feature = "extra-platforms"))]
pub(super) use core::sync::atomic::AtomicPtr;
#[cfg(not(feature = "extra-platforms"))]
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use core::{mem::MaybeUninit, ptr};
#[cfg(feature = "extra-platforms")]
pub(super) use portable_atomic::AtomicPtr;
#[cfg(feature = "extra-platforms")]
use portable_atomic::AtomicUsize;
use std::{boxed::Box, vec::Vec};

// The low two owner bits distinguish real Vec headers (00), static or managed
// owners and Arc geometry (01), a full raw Vec (10), and raw Vec geometry (11).
// The data pointer stays unmodified, including for odd-address allocations.
const TAG_MASK: usize = 0b11;
const RAW_BIT: usize = 0b10;
const GEOMETRY_TAG: usize = RAW_BIT | 1;
const NO_HIDDEN_ELEMENTS: *const () = ptr::without_provenance(GEOMETRY_TAG);
const BITS: u32 = (usize::BITS - 2) / 2;
const MASK: usize = (1 << BITS) - 1;

// Every real Vector<T> contains a pointer, including headers placed in spare
// capacity. Four-byte pointer alignment leaves both low bits free. On some
// 32-bit ABIs the handle also provides less alignment than atomic pointers need.
pub(super) const AVAILABLE: bool = align_of::<*const ()>() >= 4
  && align_of::<usize>() >= align_of::<AtomicPtr<()>>()
  && size_of::<*mut ()>() == size_of::<AtomicPtr<()>>()
  && size_of::<*mut ()>() % align_of::<AtomicPtr<()>>() == 0;

// An exact-capacity vector has no hidden elements at adoption. Caching its
// length in the owner word lets consumption recover both raw parts from aligned
// pointer slots. Larger lengths still fit the ordinary zero-prefix/suffix form.
pub(super) const MAX_FULL_LEN: usize = usize::MAX >> 2;

pub(super) fn full_owner(len: usize) -> *const () {
  if len <= MAX_FULL_LEN {
    ptr::without_provenance((len << 2) | RAW_BIT)
  } else {
    NO_HIDDEN_ELEMENTS
  }
}

pub(super) fn full_len(owner: *const ()) -> Option<usize> {
  (AVAILABLE && owner.addr() & TAG_MASK == RAW_BIT).then_some(owner.addr() >> 2)
}

pub(super) fn raw(owner: *const ()) -> bool {
  AVAILABLE && owner.addr() & RAW_BIT != 0
}

fn encode(prefix: usize, suffix: usize) -> Option<*const ()> {
  if prefix > MASK || suffix > MASK {
    return None;
  }
  let counts = prefix | (suffix << BITS);
  Some(ptr::without_provenance((counts << 2) | GEOMETRY_TAG))
}

fn counts(owner: *const ()) -> (usize, usize) {
  if full_len(owner).is_some() {
    // The first trim switches from a cached full length to hidden-end counts.
    return (0, 0);
  }
  let fields = owner.addr() >> 2;
  (fields & MASK, fields >> BITS)
}

// Only the initialized length is capped at isize::MAX, even for ZSTs.
// Its spare high bit distinguishes an embedded header at final release.
const IN_TAIL: usize = 1 << (usize::BITS - 1);

// Every vector that is not adopted as a raw marker takes its header here.
// One call holds the allocation and the tail placement, so a constructor's
// exact-capacity path carries neither.
#[inline(never)]
pub(super) fn place_header<T>(values: Vec<T>) -> (*mut T, *const ()) {
  if single_spare_needs_box(&values) {
    box_vec(values)
  } else {
    from_vec(values)
  }
}

// One spare element cannot hold a header when the element itself is smaller.
#[inline]
fn single_spare_needs_box<T>(values: &Vec<T>) -> bool {
  size_of::<T>() < size_of::<Vector<T>>() && values.capacity() - values.len() == 1
}

#[inline(always)]
fn from_vec<T>(mut values: Vec<T>) -> (*mut T, *const ()) {
  let base = values.as_mut_ptr();
  let len = values.len();
  let capacity = values.capacity();
  let Some(header) = tail_header(base, len, capacity) else {
    return box_vec(values);
  };
  // SAFETY: The exclusive Vec owns these aligned bytes beyond all of its
  // initialized elements. Its raw base retains the original provenance.
  unsafe { header.write(Vector::new(base, len | IN_TAIL, capacity, 1)) };
  core::mem::forget(values);
  (base, header.cast())
}

#[inline(always)]
fn box_vec<T>(mut values: Vec<T>) -> (*mut T, *const ()) {
  // Keep the input live until allocation succeeds, including on unwind.
  let header = Box::<Vector<T>>::new_uninit();
  let base = values.as_mut_ptr();
  let owner = Box::into_raw(header).cast::<Vector<T>>();
  // SAFETY: The fresh box has the layout of the entire header.
  unsafe { owner.write(Vector::new(base, values.len(), values.capacity(), 1)) };
  core::mem::forget(values);
  (base, owner.cast())
}

fn tail_header<T>(base: *mut T, len: usize, capacity: usize) -> Option<*mut Vector<T>> {
  // Non-ZST Vec allocations are at most isize::MAX bytes. ZSTs contribute
  // zero bytes even though Vec reports usize::MAX capacity for them.
  let spare_bytes = (capacity - len) * size_of::<T>();
  let available = spare_bytes.checked_sub(size_of::<Vector<T>>())?;
  let tail = base.wrapping_add(len).cast::<u8>();
  let padding = tail.align_offset(align_of::<Vector<T>>());
  (padding <= available).then(|| tail.wrapping_add(padding).cast())
}

pub(super) fn narrow(owner: *const (), front: usize, back: usize) -> Option<*const ()> {
  if front == 0 && back == 0 {
    return Some(owner);
  }
  let (prefix, suffix) = counts(owner);
  // Both additions are bounded by the original vector's initialized length.
  encode(prefix + front, suffix + back)
}

// Only a published header owns the buffer. A losing promotion candidate
// contains inert raw parts and can be dropped without touching any T.
pub(super) struct Vector<T> {
  strong: AtomicUsize,
  pub(super) base: *const T,
  len: usize,
  capacity: usize,
}

impl<T> Vector<T> {
  pub(super) fn new(base: *const T, len: usize, capacity: usize, strong: usize) -> Self {
    Self {
      strong: AtomicUsize::new(strong),
      base,
      len,
      capacity,
    }
  }

  fn parts(owner: *const (), data: *const (), len: usize, strong: usize) -> Self {
    if owner == NO_HIDDEN_ELEMENTS || full_len(owner).is_some() {
      return Self::new(data.cast(), len, len, strong);
    }
    let (prefix, suffix) = counts(owner);
    let len = prefix + len + suffix;
    Self::new(data.cast::<T>().wrapping_sub(prefix), len, len, strong)
  }

  pub(super) unsafe fn take_raw(owner: *const (), data: *const (), len: usize) -> Vec<T> {
    // SAFETY: The caller consumes the unpromoted descriptor and all its elements.
    unsafe { Self::parts(owner, data, len, 1).into_vec() }
  }

  unsafe fn into_vec(self) -> Vec<T> {
    // SAFETY: The caller owns the original allocation, including hidden elements.
    unsafe { Vec::from_raw_parts(self.base.cast_mut(), self.len & !IN_TAIL, self.capacity) }
  }

  // Called exactly once with the unique or final strong reference. Move out
  // all four words before returning the entire allocation to Vec, so no
  // pointer to an embedded header is used after the tail becomes writable.
  unsafe fn finish(owner: *const ()) -> Vec<T> {
    let ptr = owner.cast_mut().cast::<Self>();
    // SAFETY: No other thread can access any part of this header.
    let header = unsafe { ptr.read() };
    if header.len & IN_TAIL == 0 {
      // SAFETY: A separate header was allocated as Box<Self>. The header has
      // been moved out and MaybeUninit has the same allocation layout.
      unsafe { drop(Box::from_raw(ptr.cast::<MaybeUninit<Self>>())) };
    }
    // SAFETY: This final reference owns the Vec, independent of header storage.
    unsafe { header.into_vec() }
  }

  #[inline]
  pub(super) unsafe fn retain(owner: *const ()) {
    // SAFETY: The source handle pins this initialized atomic.
    let old = unsafe {
      (*owner.cast::<Self>())
        .strong
        .fetch_add(1, Ordering::Relaxed)
    };
    if old > isize::MAX as usize {
      refcount_overflow();
    }
  }

  pub(super) unsafe fn strong_count(owner: *const ()) -> usize {
    // SAFETY: The caller owns a live reference. Acquire pairs with any prior
    // release of the buffer before a unique consumer reuses its elements.
    unsafe { (*owner.cast::<Self>()).strong.load(Ordering::Acquire) }
  }

  pub(super) unsafe fn try_take(owner: *const ()) -> Option<Vec<T>> {
    // SAFETY: The caller transfers a strong reference on success. No weak
    // pointers exist, and an exclusively consumed final handle cannot be cloned.
    if unsafe { Self::strong_count(owner) } == 1 {
      Some(unsafe { Self::finish(owner) })
    } else {
      None
    }
  }

  #[inline]
  pub(super) unsafe fn release(owner: *const ()) {
    // SAFETY: The handle transfers one live strong reference.
    let strong = unsafe { &(*owner.cast::<Self>()).strong };
    if strong.fetch_sub(1, Ordering::Release) == 1 {
      // All prior accesses happen before final destruction. The load also
      // exposes this acquire to thread sanitizers that do not model fences.
      strong.load(Ordering::Acquire);
      // SAFETY: This was the final strong reference, including across threads.
      drop(unsafe { Self::finish(owner) });
    }
  }
}

#[cold]
#[inline(never)]
extern "C" fn refcount_overflow() -> ! {
  #[cfg(feature = "std")]
  std::process::abort();
  // A panic cannot unwind across this ABI, including in alloc-only builds.
  #[cfg(not(feature = "std"))]
  panic!("vector reference count overflow")
}

pub(super) fn box_raw<T>(owner: *const (), data: *const (), len: usize) -> *const () {
  Box::into_raw(Box::new(Vector::<T>::parts(owner, data, len, 1))).cast()
}

// Publish both the source and peer counts in one step. Even if a contender
// is preempted, another clone can immediately retain the winning header.
pub(super) unsafe fn promote<T>(
  slot: &AtomicPtr<()>,
  owner: *const (),
  data: *const (),
  len: usize,
) -> *const () {
  // SAFETY: The candidate represents this source plus the caller's new peer.
  unsafe { publish(slot, owner, Vector::<T>::parts(owner, data, len, 2)) }
}

// The expected word belongs to the borrowed source. Its replacement must
// describe the same buffer with one reference for the source and one for the peer.
pub(super) unsafe fn publish<T>(
  slot: &AtomicPtr<()>,
  expected: *const (),
  candidate: Vector<T>,
) -> *const () {
  let candidate = Box::into_raw(Box::new(candidate));
  let published = candidate.cast::<()>();
  match slot.compare_exchange(
    expected.cast_mut(),
    published,
    Ordering::AcqRel,
    Ordering::Acquire,
  ) {
    Ok(_) => published,
    Err(winner) => {
      // SAFETY: The loser owns only its unpublished box. The borrowed source
      // pins the winning header and this clone needs its own reference.
      unsafe {
        drop(Box::from_raw(candidate));
        Vector::<T>::retain(winner);
      }
      winner
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Slice;

  #[test]
  fn full_length_markers_respect_the_encoding_and_alignment_limits() {
    let limit = usize::MAX >> 2;
    for len in [0, 1, limit - 1, limit, limit + 1, isize::MAX as usize] {
      let owner = full_owner(len);
      assert_eq!(raw(owner), AVAILABLE);
      assert_eq!(full_len(owner), (AVAILABLE && len <= limit).then_some(len));
      if len > limit {
        assert_eq!(owner, NO_HIDDEN_ELEMENTS);
      }
      if AVAILABLE {
        assert_eq!(counts(owner), (0, 0));
        let front = len.min(MASK);
        let back = (len - front).min(MASK);
        assert_eq!(counts(narrow(owner, front, back).unwrap()), (front, back));
      }
    }
    let header = Vector::<u8>::new(core::ptr::null(), 0, 0, 1);
    let owner = core::ptr::from_ref(&header).cast();
    assert!(!raw(owner));
    assert_eq!(full_len(owner), None);
  }

  #[test]
  fn geometry_leaves_an_odd_data_pointer_untagged() {
    #[repr(align(2))]
    struct Bytes([u8; 8]);
    let mut bytes = Bytes([0; 8]);
    let base = bytes.0.as_mut_ptr().wrapping_add(1);
    assert_eq!(base.addr() & 1, 1);
    let raw = encode(2, 1).unwrap();
    let parts = Vector::<u8>::parts(raw, base.wrapping_add(2).cast(), 3, 1);
    assert_eq!(
      (parts.base, parts.len, parts.capacity),
      (base.cast_const(), 6, 6)
    );
  }

  #[test]
  fn an_embedded_header_fits_only_in_aligned_spare_bytes_and_releases_them() {
    let bytes = size_of::<Vector<u8>>();
    let align = align_of::<Vector<u8>>();
    for len in (0..align).chain(core::iter::once(64)) {
      for spare in [1, bytes - 1, bytes, bytes + align - 1] {
        let mut values = Vec::with_capacity(len + spare);
        values.resize(len, 7u8);
        let (base, capacity) = (values.as_mut_ptr(), values.capacity());
        let spare = capacity - len;
        let tail = base.wrapping_add(len);
        let padding = tail.align_offset(align);
        let source = Slice::<u8>::try_from_shared_vec(values).unwrap();
        let owner = source.layout.owner();
        assert!(!raw(owner));
        if padding <= spare && bytes <= spare - padding {
          assert_eq!(owner, tail.wrapping_add(padding).cast());
        } else {
          // A separate box lies outside every byte of the original allocation.
          assert!(owner.addr().wrapping_sub(base.addr()) >= capacity);
        }
        let peer = source.slice(0..len);
        assert_eq!(owner, peer.layout.owner());
        assert_eq!(source.as_slice(), peer.as_slice());
        drop(source);
        let mut original = peer.try_into_vec().ok().unwrap();
        assert_eq!(
          (original.as_mut_ptr(), original.capacity()),
          (base, capacity)
        );
        original.resize(capacity, 0xaa);
        assert!(original[..len].iter().all(|&x| x == 7));
        assert!(original[len..].iter().all(|&x| x == 0xaa));
      }
    }
  }

  #[test]
  fn a_placed_header_records_the_vector_in_its_spare_capacity_or_in_a_box() {
    // Word-aligned elements leave the tail aligned, so the spare capacity holds
    // a header exactly when its bytes span one. A single spare element does so
    // only when the element is at least as large as the header.
    fn check<T: Copy>(element: T) {
      assert!(align_of::<T>() >= align_of::<Vector<T>>());
      for len in [0, 1, 6] {
        for spare in [0, 1, 3, 4, 9] {
          let mut values = Vec::with_capacity(len + spare);
          values.resize(len, element);
          let (base, capacity) = (values.as_mut_ptr(), values.capacity());
          // The spare count selects the placement only at this exact capacity.
          assert_eq!(capacity, len + spare);
          let (data, owner) = place_header(values);
          assert_eq!(data, base);
          assert!(!raw(owner));
          let header = owner.cast::<Vector<T>>();
          // SAFETY: The new header is live, and this test holds its only reference.
          let (recorded, parts, strong) = unsafe {
            (
              (*header).len,
              ((*header).base, (*header).capacity),
              Vector::<T>::strong_count(owner),
            )
          };
          assert_eq!((parts, strong), ((base.cast_const(), capacity), 1));
          let embedded = spare * size_of::<T>() >= size_of::<Vector<T>>();
          assert_eq!(recorded, if embedded { len | IN_TAIL } else { len });
          let offset = owner.addr().wrapping_sub(base.addr());
          if embedded {
            assert_eq!(offset, len * size_of::<T>());
          } else {
            // A separate box lies outside every byte of the original allocation.
            assert!(offset >= capacity * size_of::<T>());
          }
          // SAFETY: The only reference moves into the returned vector.
          let mut original = unsafe { Vector::<T>::try_take(owner) }.unwrap();
          assert_eq!(
            (original.as_mut_ptr(), original.len(), original.capacity()),
            (base, len, capacity)
          );
          // The bytes that held an embedded header are spare capacity again.
          original.resize(capacity, element);
          assert_eq!(original.as_mut_ptr(), base);
        }
      }
    }
    const { assert!(size_of::<[usize; 4]>() == size_of::<Vector<[usize; 4]>>()) };
    check(usize::MAX / 3);
    check([usize::MAX / 3; 4]);
    check([usize::MAX / 3; 5]);

    // A zero-sized vector has no bytes to spare, whatever its capacity.
    let slots = std::vec![(); 5];
    let base = slots.as_ptr();
    let (data, owner) = place_header(slots);
    assert_eq!(data.cast_const(), base);
    assert!(!raw(owner));
    let header = owner.cast::<Vector<()>>();
    // SAFETY: The boxed header is live, and its only reference moves into the
    // returned vector.
    let recovered = unsafe {
      assert_eq!(
        ((*header).base, (*header).len, (*header).capacity),
        (base, 5, usize::MAX)
      );
      Vector::<()>::try_take(owner)
    }
    .unwrap();
    assert_eq!((recovered.as_ptr(), recovered.len()), (base, 5));
  }

  #[test]
  fn raw_geometry_uses_both_fields_and_rejects_overflow() {
    let full = narrow(NO_HIDDEN_ELEMENTS, MASK, MASK).unwrap();
    assert_eq!(counts(full), (MASK, MASK));
    assert!(narrow(full, 1, 0).is_none());
    assert!(narrow(full, 0, 1).is_none());
  }

  // On 32-bit targets both geometry fields can overflow in a 32 KiB Vec.
  // The equivalent real allocation would require over two GiB on 64-bit.
  #[cfg(target_pointer_width = "32")]
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
  }
}
