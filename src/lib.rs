#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc as std;

#[cfg(feature = "std")]
extern crate std;

pub mod error;
mod layout;
mod traits;
mod vector;
use core::ops::{Bound, Range, RangeBounds};
pub use error::{ByteEditError, OutOfBounds, RangeOutOfBounds, TryPutError, TryReserveError};
use layout::Layout;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use layout::local::{LocalVector, LocalVectorN};
pub use layout::mutable::WritableSize;
pub use layout::{AssocInlineSize, InlineSize, MAX_HANDLE_BYTES};
pub use typenum;

/// The default budget of the handles: four machine words, the size of `bytes::Bytes`.
///
/// It is [`typenum::U32`] on 64-bit targets and [`typenum::U16`] on 32-bit targets.
#[cfg(target_pointer_width = "32")]
pub type DefaultSize = typenum::U16;
/// The default budget of the handles: four machine words, the size of `bytes::Bytes`.
///
/// It is [`typenum::U32`] on 64-bit targets and [`typenum::U16`] on 32-bit targets.
#[cfg(target_pointer_width = "64")]
pub type DefaultSize = typenum::U32;

/// A vector with inline storage for `Copy` elements, static borrows, and optional heap storage.
///
/// A positive `S` is the exact size in bytes of the handle and its `Option`. It
/// defaults to four machine words, the size of `bytes::Bytes` (16 bytes on
/// 32-bit targets and 32 bytes on 64-bit targets). [`typenum::U0`] has the same
/// size and disables inline elements.
/// Use [`VectorN`] for a numeric byte budget.
/// An inline clone copies the whole handle. Static and heap-backed clones
/// share their elements. Each clone has an independent cursor.
#[repr(transparent)]
pub struct Vector<T, S: InlineSize = DefaultSize> {
  layout: Layout<T, S>,
}

/// A [`Vector`] with byte budget `N`. Zero disables inline elements.
pub type VectorN<T, const N: usize> = Vector<T, <[u8; N] as AssocInlineSize>::Size>;

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

// The ranges checked_range accepts, as a result. Each try_* twin of a ranged
// method checks with this and then calls the panicking method, whose own check
// can no longer fail, so the unsafe it guards stays behind that check.
pub(crate) fn try_checked_range(
  range: impl RangeBounds<usize>,
  len: usize,
) -> Result<Range<usize>, RangeOutOfBounds> {
  let start = match range.start_bound() {
    Bound::Included(&start) => Some(start),
    Bound::Excluded(&start) => start.checked_add(1),
    Bound::Unbounded => Some(0),
  };
  let end = match range.end_bound() {
    Bound::Included(&end) => end.checked_add(1),
    Bound::Excluded(&end) => Some(end),
    Bound::Unbounded => Some(len),
  };
  match (start, end) {
    (Some(start), Some(end)) if start <= end && end <= len => Ok(start..end),
    // An endpoint past usize::MAX is reported as usize::MAX.
    (start, end) => Err(RangeOutOfBounds::new(
      start.unwrap_or(usize::MAX),
      end.unwrap_or(usize::MAX),
      len,
    )),
  }
}

impl<T, S: InlineSize> Vector<T, S> {
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
  /// cannot be represented as a half-open range. [`Self::try_slice`] returns an
  /// error instead.
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

  /// Returns an independent view of a range of the remaining elements, or an
  /// error where [`Self::slice`] panics.
  pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Result<Self, RangeOutOfBounds> {
    let range = try_checked_range(range, self.len())?;
    Ok(self.slice(range))
  }

  /// Returns the number of remaining elements.
  pub const fn len(&self) -> usize {
    self.layout.len()
  }

  /// Returns whether the remaining vector is empty.
  pub const fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Skips the first `count` remaining elements.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds [`Self::len`]. [`Self::try_advance`] returns an
  /// error instead.
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    if !self.layout.is_inline() {
      return self.advance_shared(count);
    }
    self.layout.advance(count);
  }

  /// Skips the first `count` remaining elements, or returns an error where
  /// [`Self::advance`] panics.
  pub fn try_advance(&mut self, count: usize) -> Result<(), OutOfBounds> {
    let len = self.len();
    if count > len {
      return Err(OutOfBounds::new(count, len));
    }
    self.advance(count);
    Ok(())
  }
}

// Only the inline constructor copies elements, and it requires T: Copy.
// Cloning an external window duplicates its descriptor regardless of T.
impl<T, S: InlineSize> Clone for Vector<T, S> {
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
    #[cfg(bytora_atomic)]
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

/// An owned vector with exclusive access to its elements.
///
/// Positive budgets store `Copy` elements inline. [`typenum::U0`] disables
/// inline elements and can own a vector of any element type with `alloc` or
/// `std`. The default and zero budgets use four machine words, the size of
/// `bytes::Bytes` (16 bytes on 32-bit targets and 32 bytes on 64-bit targets).
/// Use [`VectorMutN`] for a numeric byte budget.
pub type VectorMut<T, S = DefaultSize> =
  layout::mutable::Mutable<T, S, <S as InlineSize>::Mutable<T>>;

/// A [`VectorMut`] with byte budget `N`. Zero disables inline elements.
pub type VectorMutN<T, const N: usize> = VectorMut<T, <[u8; N] as AssocInlineSize>::Size>;

/// An immutable byte handle: a [`Vector`] of `u8` with the [`DefaultSize`] budget.
///
/// The alias has no budget parameter, so `Bytes::new()` needs no type annotation.
/// [`BytesN`] names the other budgets.
pub type Bytes = Vector<u8>;

/// A [`Bytes`] with byte budget `N`. Zero disables inline elements.
pub type BytesN<const N: usize> = VectorN<u8, N>;

/// A mutable byte buffer: a [`VectorMut`] of `u8` with the [`DefaultSize`] budget.
///
/// The alias has no budget parameter, so `BytesMut::new()` and `BytesMut::with_capacity(n)` need
/// no type annotation. [`BytesMutN`] names the other budgets.
pub type BytesMut = VectorMut<u8>;

/// A [`BytesMut`] with byte budget `N`. Zero disables inline elements.
pub type BytesMutN<const N: usize> = VectorMutN<u8, N>;

const _: () = {
  assert!(size_of::<Vector<u32>>() == 4 * size_of::<usize>());
  assert!(size_of::<VectorMut<u32>>() == 4 * size_of::<usize>());
  assert!(size_of::<Option<Vector<u32>>>() == 4 * size_of::<usize>());
  assert!(size_of::<Option<VectorMut<u32>>>() == 4 * size_of::<usize>());
};

#[cfg(test)]
mod tests;
