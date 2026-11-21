#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
#[allow(
  unused_extern_crates,
  reason = "atomics-free targets only use alloc in layout assertions"
)]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

mod layout;
use core::ops::{Bound, RangeBounds};
use layout::Layout;
pub use layout::{AssocInlineSize, InlineSize, MAX_HANDLE_BYTES};
pub use typenum;
#[cfg(target_pointer_width = "32")]
use typenum::U12 as DefaultSize;
#[cfg(target_pointer_width = "64")]
use typenum::U24 as DefaultSize;

/// A slice with inline storage for `Copy` elements, static borrows, and optional heap storage.
///
/// `S` is the exact size in bytes of the handle and its `Option`. It defaults
/// to three machine words, the size of a `Vec<T>`. Use [`SliceN`] for a numeric byte count.
/// An inline clone copies the whole handle. Static and heap-backed clones
/// share their elements. Each clone has an independent cursor.
#[repr(transparent)]
pub struct Slice<T, S: InlineSize = DefaultSize> {
  layout: Layout<T, S>,
}

/// A [`Slice`] with a const-generic size of `N` bytes, including its cursor.
pub type SliceN<T, const N: usize> = Slice<T, <[u8; N] as AssocInlineSize>::Size>;

impl<T, S: InlineSize> Slice<T, S> {
  /// The maximum number of inline elements. Each zero-sized element counts as
  /// one payload byte. Nonzero-sized elements requiring greater alignment than
  /// the handle have capacity zero.
  pub const INLINE_CAPACITY: usize = Layout::<T, S>::INLINE_CAPACITY;

  /// Copies `values` inline. Returns `None` if their length exceeds [`Self::INLINE_CAPACITY`].
  pub fn try_from_slice(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Layout::try_inline(values).map(|layout| Self { layout })
  }

  /// Borrows the remaining elements.
  pub fn as_slice(&self) -> &[T] {
    if !self.layout.is_inline() {
      return self.shared_slice();
    }
    self.layout.as_slice()
  }

  /// Returns an independent view of a range of the remaining elements.
  ///
  /// Static and heap-backed ranges share their original storage, including
  /// empty ranges. Slicing an inline value copies the handle.
  ///
  /// # Panics
  ///
  /// Panics if the range starts after its end, exceeds [`Self::len`], or
  /// cannot be represented as a half-open range.
  #[track_caller]
  pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
    let start = match range.start_bound() {
      Bound::Included(&start) => start,
      Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
      Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
      Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
      Bound::Excluded(&end) => end,
      Bound::Unbounded => self.len(),
    };
    if !self.layout.is_inline() {
      return self.slice_shared(start..end);
    }
    Self {
      layout: self.layout.slice_inline(start..end),
    }
  }

  /// Returns the number of remaining elements.
  pub fn len(&self) -> usize {
    self.layout.len()
  }

  /// Returns whether the remaining slice is empty.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Skips the first `count` remaining elements.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds [`Self::len`].
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    if !self.layout.is_inline() {
      return self.advance_shared(count);
    }
    self.layout.advance(count);
  }
}

// Only the inline constructor copies elements, and it requires T: Copy.
// Cloning an external window duplicates its descriptor regardless of T.
impl<T, S: InlineSize> Clone for Slice<T, S> {
  #[inline]
  fn clone(&self) -> Self {
    if Layout::<T, S>::COMPACT_EXTERNAL {
      return self.clone_compact();
    }
    if !self.layout.is_inline() {
      return self.clone_shared();
    }
    Self {
      layout: self.layout.clone_inline(),
    }
  }

  #[inline]
  fn clone_from(&mut self, source: &Self) {
    // Keep the external assignment separate so wide shared descriptors can be
    // written without also moving the unused inline storage.
    if Layout::<T, S>::COMPACT_EXTERNAL || source.layout.is_inline() {
      *self = source.clone();
    } else {
      *self = source.clone_shared();
    }
  }
}

/// An owned slice whose inline `Copy` elements can be borrowed mutably.
///
/// `S` is the exact size in bytes of the handle and its `Option`. It defaults
/// to three machine words, the size of a `Vec<T>`. Use [`SliceMutN`] for a numeric byte count.
#[repr(transparent)]
pub struct SliceMut<T, S: InlineSize = DefaultSize> {
  layout: Layout<T, S>,
}

/// A [`SliceMut`] with a const-generic size of `N` bytes, including its cursor.
pub type SliceMutN<T, const N: usize> = SliceMut<T, <[u8; N] as AssocInlineSize>::Size>;

impl<T, S: InlineSize> SliceMut<T, S> {
  /// The maximum number of inline elements, as in [`Slice::INLINE_CAPACITY`].
  pub const INLINE_CAPACITY: usize = Slice::<T, S>::INLINE_CAPACITY;

  /// Copies `values` inline. Returns `None` if their length exceeds [`Self::INLINE_CAPACITY`].
  pub fn try_from_slice(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Layout::try_inline(values).map(|layout| Self { layout })
  }

  /// Borrows the remaining elements.
  pub fn as_slice(&self) -> &[T] {
    self.layout.as_slice()
  }

  /// Mutably borrows the remaining elements.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.layout.as_mut_slice()
  }

  /// Returns the number of remaining elements.
  pub fn len(&self) -> usize {
    self.layout.len()
  }

  /// Returns whether the remaining slice is empty.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Skips the first `count` remaining elements.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds [`Self::len`].
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    self.layout.advance(count);
  }
}

const _: () = {
  assert!(size_of::<Slice<u32>>() == 3 * size_of::<usize>());
  assert!(size_of::<SliceMut<u32>>() == 3 * size_of::<usize>());
  assert!(size_of::<Option<Slice<u32>>>() == 3 * size_of::<usize>());
  assert!(size_of::<Option<SliceMut<u32>>>() == 3 * size_of::<usize>());
};

#[cfg(any(feature = "std", feature = "alloc"))]
const _: () = {
  assert!(size_of::<Slice<u32>>() == size_of::<std::vec::Vec<u32>>());
  assert!(size_of::<SliceMut<u32>>() == size_of::<std::vec::Vec<u32>>());
  assert!(size_of::<Option<Slice<u32>>>() == size_of::<std::vec::Vec<u32>>());
  assert!(size_of::<Option<SliceMut<u32>>>() == size_of::<std::vec::Vec<u32>>());
};

#[cfg(test)]
mod tests;
