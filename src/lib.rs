#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

mod layout;
mod slice;
use core::ops::{Bound, Range, RangeBounds};
use layout::Layout;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use layout::local::{LocalSlice, LocalSliceN};
pub use layout::mutable::WritableSize;
pub use layout::{AssocInlineSize, InlineSize, MAX_HANDLE_BYTES};
pub use typenum;
#[cfg(target_pointer_width = "32")]
use typenum::U16 as DefaultSize;
#[cfg(target_pointer_width = "64")]
use typenum::U32 as DefaultSize;

/// A slice with inline storage for `Copy` elements, static borrows, and optional heap storage.
///
/// A positive `S` is the exact size in bytes of the handle and its `Option`. It
/// defaults to four machine words, the size of `bytes::Bytes` (16 bytes on
/// 32-bit targets and 32 bytes on 64-bit targets). [`typenum::U0`] has the same
/// size and disables inline elements.
/// Use [`SliceN`] for a numeric byte budget.
/// An inline clone copies the whole handle. Static and heap-backed clones
/// share their elements. Each clone has an independent cursor.
#[repr(transparent)]
pub struct Slice<T, S: InlineSize = DefaultSize> {
  layout: Layout<T, S>,
}

/// A [`Slice`] with byte budget `N`. Zero disables inline elements.
pub type SliceN<T, const N: usize> = Slice<T, <[u8; N] as AssocInlineSize>::Size>;

#[track_caller]
pub(crate) fn checked_range(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
  let start = match range.start_bound() {
    Bound::Included(&start) => start,
    Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
    Bound::Unbounded => 0,
  };
  let end = match range.end_bound() {
    Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
    Bound::Excluded(&end) => end,
    Bound::Unbounded => len,
  };
  assert!(start <= end && end <= len, "slice range out of bounds");
  start..end
}

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
    let len = self.len();
    let range = checked_range(range, len);
    if !self.layout.is_inline() {
      // SAFETY: The tag selects external storage and checked_range validated
      // both endpoints against this immutable window's length.
      return unsafe { self.slice_shared(range, len) };
    }
    Self {
      layout: self.layout.slice_inline(range),
    }
  }

  /// Returns the number of remaining elements.
  pub const fn len(&self) -> usize {
    self.layout.len()
  }

  /// Returns whether the remaining slice is empty.
  pub const fn is_empty(&self) -> bool {
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
    if Layout::<T, S>::COMPACT_EXTERNAL || Layout::<T, S>::SPARSE_CLONE {
      return self.clone_descriptor();
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
    #[cfg(paguro_atomic)]
    if Layout::<T, S>::SPARSE_CLONE {
      if source.layout.is_inline() {
        *self = Self {
          layout: source.layout.clone_inline(),
        };
      } else {
        self.replace_external_clone(source);
      }
      return;
    }
    // Keep the external assignment separate so wide shared descriptors can be
    // written without also moving the unused inline storage.
    if Layout::<T, S>::COMPACT_EXTERNAL || source.layout.is_inline() {
      *self = source.clone();
    } else {
      *self = source.clone_shared();
    }
  }
}

/// An owned slice with exclusive access to its elements.
///
/// Positive budgets store `Copy` elements inline. [`typenum::U0`] disables
/// inline elements and can own a vector of any element type with `alloc` or
/// `std`. The default and zero budgets use four machine words, the size of
/// `bytes::Bytes` (16 bytes on 32-bit targets and 32 bytes on 64-bit targets).
/// Use [`SliceMutN`] for a numeric byte budget.
pub type SliceMut<T, S = DefaultSize> =
  layout::mutable::Mutable<T, S, <S as InlineSize>::Mutable<T>>;

/// A [`SliceMut`] with byte budget `N`. Zero disables inline elements.
pub type SliceMutN<T, const N: usize> = SliceMut<T, <[u8; N] as AssocInlineSize>::Size>;

const _: () = {
  assert!(size_of::<Slice<u32>>() == 4 * size_of::<usize>());
  assert!(size_of::<SliceMut<u32>>() == 4 * size_of::<usize>());
  assert!(size_of::<Option<Slice<u32>>>() == 4 * size_of::<usize>());
  assert!(size_of::<Option<SliceMut<u32>>>() == 4 * size_of::<usize>());
};

#[cfg(test)]
mod tests;
