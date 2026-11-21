#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
#[allow(unused_extern_crates, reason = "the inline handles use only core")]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

mod layout;
use layout::Layout;
pub use layout::{AssocInlineSize, InlineSize, MAX_HANDLE_BYTES};
pub use typenum;
#[cfg(target_pointer_width = "32")]
use typenum::U12 as DefaultSize;
#[cfg(target_pointer_width = "64")]
use typenum::U24 as DefaultSize;

/// An owned slice with inline storage for `Copy` elements.
///
/// `S` is the exact size in bytes of the handle and its `Option`. It defaults
/// to three machine words, the size of a `Vec<T>`. Use [`SliceN`] for a numeric byte count.
/// Moving the handle moves its elements. Advancing changes only the cursor.
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
    self.layout.as_slice()
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

/// An owned slice whose inline `Copy` elements can be borrowed mutably.
///
/// `S` is the exact size in bytes of the handle and its `Option`. It defaults
/// to three machine words, the size of a `Vec<T>`. Use [`SliceMutN`] for a numeric byte count.
#[repr(transparent)]
pub struct SliceMut<T, S: InlineSize = DefaultSize> {
  inner: Slice<T, S>,
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
    Slice::try_from_slice(values).map(|inner| Self { inner })
  }

  /// Borrows the remaining elements.
  pub fn as_slice(&self) -> &[T] {
    self.inner.as_slice()
  }

  /// Mutably borrows the remaining elements.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.inner.layout.as_mut_slice()
  }

  /// Returns the number of remaining elements.
  pub fn len(&self) -> usize {
    self.inner.len()
  }

  /// Returns whether the remaining slice is empty.
  pub fn is_empty(&self) -> bool {
    self.inner.is_empty()
  }

  /// Skips the first `count` remaining elements.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds [`Self::len`].
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    self.inner.advance(count);
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
