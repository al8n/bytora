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

/// An owned slice with inline storage for `Copy` elements.
///
/// The inline cursor can address `min(N, 126)` elements. Moving the handle moves
/// those elements. Advancing it leaves them in place and changes the start of the
/// visible slice. All public constructors currently use inline storage.
#[repr(transparent)]
pub struct Slice<T, const N: usize> {
  layout: Layout<T, N>,
}

impl<T, const N: usize> Slice<T, N> {
  /// The number of elements the inline cursor can address, the smaller of `N` and 126.
  pub const INLINE_CAPACITY: usize = Layout::<T, N>::INLINE_CAPACITY;

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
/// The inline cursor can address `min(N, 126)` elements. All public constructors
/// currently use inline storage.
#[repr(transparent)]
pub struct SliceMut<T, const N: usize> {
  inner: Slice<T, N>,
}

impl<T, const N: usize> SliceMut<T, N> {
  /// The number of elements the inline cursor can address, the smaller of `N` and 126.
  pub const INLINE_CAPACITY: usize = Slice::<T, N>::INLINE_CAPACITY;

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

// Keep these contracts in library builds, including cross-compilation without tests.
#[cfg(target_pointer_width = "64")]
const _: () = {
  if align_of::<usize>() == 8 {
    assert!(size_of::<Slice<u8, 22>>() == 24);
    assert!(size_of::<Option<Slice<u8, 22>>>() == 24);
    assert!(size_of::<SliceMut<u8, 22>>() == 24);
    assert!(size_of::<Option<SliceMut<u8, 22>>>() == 24);
    assert!(size_of::<Slice<u8, 30>>() == 32);
    assert!(size_of::<Option<Slice<u8, 30>>>() == 32);
    assert!(size_of::<SliceMut<u8, 30>>() == 32);
    assert!(size_of::<Option<SliceMut<u8, 30>>>() == 32);
    assert!(size_of::<Slice<u8, 62>>() == 64);
    assert!(size_of::<Option<Slice<u8, 62>>>() == 64);
    assert!(size_of::<SliceMut<u8, 62>>() == 64);
    assert!(size_of::<Option<SliceMut<u8, 62>>>() == 64);
  }
};

#[cfg(test)]
mod tests;
