use super::{Layout, owner};
use crate::{InlineSize, Slice};
use core::ptr;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;
#[cfg(target_has_atomic = "ptr")]
use std::sync::Arc;

// Arc geometry uses 101 and managed pointers use 001. Neither encoding can
// alias a direct Vec header (even) or a raw Vec marker (10 or 11).
const TAG: usize = 0b101;
const BITS: u32 = (usize::BITS - 3) / 2;
const MASK: usize = (1 << BITS) - 1;

pub(super) fn is_arc(owner: *const ()) -> bool {
  owner.addr() & 7 == TAG
}

fn ends(owner: *const ()) -> (usize, usize) {
  let fields = owner.addr() >> 3;
  (fields & MASK, fields >> BITS)
}

pub(super) fn narrow(owner: *const (), front: usize, back: usize) -> Option<*const ()> {
  let (prefix, suffix) = ends(owner);
  // Each sum is bounded by the original, checked slice length.
  let (prefix, suffix) = (prefix + front, suffix + back);
  if prefix > MASK || suffix > MASK {
    return None;
  }
  Some(ptr::without_provenance(
    TAG | (prefix << 3) | (suffix << (BITS + 3)),
  ))
}

fn original<T>(owner: *const (), data: *const (), len: usize) -> *const [T] {
  let (front, back) = ends(owner);
  // The data word retains the original Arc pointer's provenance. For ZSTs
  // the pointer stays fixed and only the counts change.
  ptr::slice_from_raw_parts(data.cast::<T>().wrapping_sub(front), front + len + back)
}

pub(super) unsafe fn retain<T>(owner: *const (), data: *const (), len: usize) {
  // SAFETY: The caller pins a live Arc with this geometry. Reconstruct exactly
  // the fat pointer supplied by Arc::into_raw, including its original length.
  unsafe { Arc::increment_strong_count(original::<T>(owner, data, len)) };
}

pub(super) unsafe fn release<T>(owner: *const (), data: *const (), len: usize) {
  // SAFETY: The caller transfers one live reference to this original Arc.
  unsafe { Arc::decrement_strong_count(original::<T>(owner, data, len)) };
}

pub(super) unsafe fn managed_copy<T>(owner: *const (), data: *const (), len: usize) -> *const () {
  let original = original::<T>(owner, data, len);
  // SAFETY: The source's Arc remains live. Transfer one additional reference
  // to the managed owner, keeping the source valid if allocation unwinds.
  let copy = unsafe {
    Arc::increment_strong_count(original);
    Arc::from_raw(original)
  };
  owner::manage_read_only::<T, _>(copy)
}

impl<T: Send + Sync, S: InlineSize> Slice<T, S> {
  /// Adopts an `Arc<[T]>` and keeps its elements at their original addresses.
  /// Adoption and cloning reuse the original reference count. Ranges and
  /// cursor changes store each hidden end of up to
  /// `(1 << ((usize::BITS - 3) / 2)) - 1` elements in the handle. A longer
  /// hidden end allocates an owner that subsequent views of that handle share.
  ///
  /// This backing grants shared access. Returns the original Arc if the
  /// handle is smaller than three words or the length exceeds `isize::MAX`.
  /// On targets without native pointer atomics, `extra-platforms` accepts
  /// the Arc supplied by `portable_atomic_util`.
  pub fn try_from_arc(values: Arc<[T]>) -> Result<Self, Arc<[T]>> {
    let len = values.len();
    if !Layout::<T, S>::FITS_EXTERNAL || len > isize::MAX as usize {
      return Err(values);
    }
    let data = Arc::into_raw(values).cast::<T>();
    Ok(Self {
      layout: Layout::external(
        data,
        ptr::without_provenance(TAG),
        len,
        core::num::NonZeroU8::new(super::EXTERNAL).unwrap(),
      ),
    })
  }
}

impl<T: Send + Sync, S: InlineSize> TryFrom<Arc<[T]>> for Slice<T, S> {
  type Error = Arc<[T]>;

  fn try_from(values: Arc<[T]>) -> Result<Self, Self::Error> {
    Self::try_from_arc(values)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SliceN;
  use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
  use std::{boxed::Box, vec};

  #[derive(Debug)]
  #[repr(align(64))]
  struct Count<'a>(&'a AtomicUsize);

  impl Drop for Count<'_> {
    fn drop(&mut self) {
      self.0.fetch_add(1, Relaxed);
    }
  }

  #[test]
  fn boxed_slices_preserve_noncopy_elements_and_return_the_original_allocation() {
    let drops = [const { AtomicUsize::new(0) }; 3];
    let boxed: Box<[_]> = drops.iter().map(Count).collect();
    let base = boxed.as_ptr();
    let source = Slice::<_>::try_from(boxed).unwrap();
    let peer = source.slice(1..);
    assert_eq!(peer.as_slice().as_ptr(), base.wrapping_add(1));
    drop(source);
    let values = peer.try_into_vec().ok().unwrap();
    assert_eq!(
      (values.as_ptr(), values.len(), values.capacity()),
      (base, 2, 3)
    );
    assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [1, 0, 0]);
    drop(values);
    assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [1; 3]);
  }

  #[test]
  fn arc_windows_hold_independent_native_references_and_accept_borrowed_elements() {
    fn check<S: InlineSize>() {
      let drops = [const { AtomicUsize::new(0) }; 4];
      let input: Arc<[_]> = drops.iter().map(Count).collect();
      let base = input.as_ptr();
      let mut first = Slice::<_, S>::try_from(input.clone()).unwrap();
      assert_eq!(first.as_slice().as_ptr(), base);
      let peer = first.clone();
      let mut middle = first.slice(1..3);
      assert_eq!(Arc::strong_count(&input), 4);
      first.advance(1);
      first.truncate(2);
      middle.advance(2);
      assert_eq!(middle.as_slice().as_ptr(), base.wrapping_add(3));
      assert_eq!(first.as_slice().as_ptr(), base.wrapping_add(1));
      assert!(!first.is_unique());
      let first = first.try_into_vec().err().unwrap();
      let first = first.try_into_mut().err().unwrap();
      assert_eq!(first.as_slice().len(), 2);
      drop((peer, first, middle));
      assert_eq!(Arc::strong_count(&input), 1);
      assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [0; 4]);
      drop(input);
      assert_eq!(drops.each_ref().map(|d| d.load(Relaxed)), [1; 4]);
    }
    check::<crate::DefaultSize>();
    check::<typenum::U32>();
    check::<typenum::U136>();
  }

  #[test]
  fn arc_windows_remain_covariant_after_adoption() {
    type Any = for<'a> fn(&'a str) -> &'a str;
    type Static = fn(&'static str) -> &'static str;
    fn identity(value: &str) -> &str {
      value
    }
    let input: Arc<[Any]> = Arc::from(vec![identity as Any].into_boxed_slice());
    let original = Slice::<Any>::try_from_arc(input.clone()).unwrap();
    let narrowed: Slice<Static> = original;
    assert_eq!(narrowed.slice(..).as_slice()[0]("slice"), "slice");
    drop(narrowed);
    assert_eq!(Arc::strong_count(&input), 1);
  }

  #[test]
  fn rejected_owners_keep_their_identity_and_empty_owners_keep_their_pointer() {
    type Tiny<T> = SliceN<T, { size_of::<usize>() }>;
    let boxed = vec![3].into_boxed_slice();
    let base = boxed.as_ptr();
    let returned = match Tiny::try_from_boxed_slice(boxed) {
      Ok(handle) => {
        assert!(cfg!(smol_slice_owner_table));
        handle.try_into_vec().ok().unwrap().into_boxed_slice()
      }
      Err(boxed) => {
        assert!(!cfg!(smol_slice_owner_table));
        boxed
      }
    };
    assert_eq!(returned.as_ptr(), base);
    let arc = Arc::<[u32]>::from(returned);
    let returned = Tiny::try_from_arc(arc.clone()).err().unwrap();
    assert!(Arc::ptr_eq(&returned, &arc));
    let huge: Box<[()]> = Box::new([(); isize::MAX as usize + 1]);
    let returned = Slice::<()>::try_from_boxed_slice(huge).err().unwrap();
    assert_eq!(returned.len(), isize::MAX as usize + 1);
    let huge = Arc::from(returned);
    let returned = Slice::<()>::try_from_arc(Arc::clone(&huge)).err().unwrap();
    assert!(Arc::ptr_eq(&huge, &returned));
    let empty = Arc::<[u32]>::from(vec![].into_boxed_slice());
    let source = Slice::<u32>::try_from_arc(empty.clone()).unwrap();
    assert_eq!(source.slice(..).as_slice().as_ptr(), empty.as_ptr());
    assert_eq!(Arc::strong_count(&empty), 2);
    let empty: Box<[u32]> = Box::new([]);
    let base = empty.as_ptr();
    let source = Slice::<u32>::try_from_boxed_slice(empty).unwrap();
    let recovered = source.try_into_vec().ok().unwrap();
    assert_eq!(recovered.as_ptr(), base);
  }

  #[test]
  fn overflowing_arc_ranges_retain_the_original_fat_pointer() {
    let boxed: Box<[()]> = Box::new([(); MASK + 3]);
    let arc = Arc::from(boxed);
    let mut source = Slice::<()>::try_from_arc(Arc::clone(&arc)).unwrap();
    let mut tail = source.slice(MASK + 1..);
    let mut head = source.slice(..1);
    assert_eq!(Arc::strong_count(&arc), 4);
    assert!(!is_arc(tail.layout.owner()));
    assert!(!is_arc(head.layout.owner()));
    tail.advance(1);
    head.truncate(0);
    assert_eq!(Arc::strong_count(&arc), 3);
    assert_eq!((tail.len(), head.len()), (1, 0));
    let mut from_back = source.clone();
    assert_eq!(Arc::strong_count(&arc), 4);
    source.advance(MASK + 1);
    from_back.truncate(1);
    assert_eq!((source.len(), from_back.len()), (2, 1));
    assert!(!is_arc(source.layout.owner()));
    assert!(!is_arc(from_back.layout.owner()));
    assert_eq!(source.as_slice().as_ptr(), arc.as_ptr());
    assert_eq!(Arc::strong_count(&arc), 4);
    drop((tail, head, source, from_back));
    assert_eq!(Arc::strong_count(&arc), 1);
  }
}
