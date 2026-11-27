//! The errors that the non-panicking `try_*` operations return in place of a panic.
//!
//! Each error carries the numbers that its check compared. They count elements, which are bytes
//! for [`Bytes`](crate::Bytes) and [`BytesMut`](crate::BytesMut).

use core::fmt;

/// An index or count that exceeds the length or capacity it was checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutOfBounds {
  /// The requested index or count.
  pub requested: usize,
  /// The length or capacity that bounds the request.
  pub available: usize,
}

impl OutOfBounds {
  /// Creates the error for a request of `requested` against `available`.
  #[inline]
  pub const fn new(requested: usize, available: usize) -> Self {
    Self {
      requested,
      available,
    }
  }
}

impl fmt::Display for OutOfBounds {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "index out of bounds: requested {} but only {} available",
      self.requested, self.available
    )
  }
}

impl core::error::Error for OutOfBounds {}

#[cfg(feature = "std")]
impl From<OutOfBounds> for std::io::Error {
  fn from(error: OutOfBounds) -> Self {
    Self::new(std::io::ErrorKind::InvalidInput, error)
  }
}

/// A range that is reversed, overflows `usize`, or ends past the length it was checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeOutOfBounds {
  /// The start of the requested range.
  pub start: usize,
  /// The end of the requested range.
  pub end: usize,
  /// The length that bounds the range.
  pub available: usize,
}

impl RangeOutOfBounds {
  /// Creates the error for the range `start..end` against `available`.
  #[inline]
  pub const fn new(start: usize, end: usize, available: usize) -> Self {
    Self {
      start,
      end,
      available,
    }
  }
}

impl fmt::Display for RangeOutOfBounds {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "range out of bounds: requested {}..{} but only {} available",
      self.start, self.end, self.available
    )
  }
}

impl core::error::Error for RangeOutOfBounds {}

#[cfg(feature = "std")]
impl From<RangeOutOfBounds> for std::io::Error {
  fn from(error: RangeOutOfBounds) -> Self {
    Self::new(std::io::ErrorKind::InvalidInput, error)
  }
}

/// A write of more elements than the storage can hold.
///
/// `available` is the most that the storage holds for this write: the room left in a bounded
/// buffer, or the longest input that a constructor accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TryPutError {
  /// The number of elements to write.
  pub requested: usize,
  /// The number of elements the storage can hold.
  pub available: usize,
}

impl fmt::Display for TryPutError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Not enough capacity remaining in buffer to write value (requested {} but only {} available)",
      self.requested, self.available
    )
  }
}

impl core::error::Error for TryPutError {}

#[cfg(feature = "std")]
impl From<TryPutError> for std::io::Error {
  fn from(error: TryPutError) -> Self {
    Self::other(error)
  }
}

/// The error of an operation that has to make room for more elements.
///
/// It tells three causes apart: a length past the limit of the element type, a handle that cannot
/// grow past its own storage, and an allocator that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TryReserveError {
  /// The new length overflows `usize`, or its elements would exceed `isize::MAX` bytes.
  CapacityOverflow,
  /// The handle cannot grow, and its storage has less room than the request.
  Bounded(TryPutError),
  /// The allocator could not provide the memory.
  AllocError,
}

impl fmt::Display for TryReserveError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::CapacityOverflow => f.write_str(
        "memory allocation failed because the computed capacity exceeded the collection's maximum",
      ),
      Self::Bounded(error) => fmt::Display::fmt(error, f),
      Self::AllocError => {
        f.write_str("memory allocation failed because the memory allocator returned an error")
      }
    }
  }
}

impl core::error::Error for TryReserveError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Bounded(error) => Some(error),
      Self::CapacityOverflow | Self::AllocError => None,
    }
  }
}

impl From<TryPutError> for TryReserveError {
  fn from(error: TryPutError) -> Self {
    Self::Bounded(error)
  }
}

/// The error of an edit that reads a range of the elements and then makes room for more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteEditError {
  /// The range is reversed, overflows `usize`, or ends past the visible elements.
  Range(RangeOutOfBounds),
  /// The handle could not make room for the result.
  Capacity(TryReserveError),
}

impl fmt::Display for ByteEditError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Range(error) => fmt::Display::fmt(error, f),
      Self::Capacity(error) => fmt::Display::fmt(error, f),
    }
  }
}

impl core::error::Error for ByteEditError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Range(error) => Some(error),
      Self::Capacity(error) => Some(error),
    }
  }
}

impl From<RangeOutOfBounds> for ByteEditError {
  fn from(error: RangeOutOfBounds) -> Self {
    Self::Range(error)
  }
}

impl From<TryReserveError> for ByteEditError {
  fn from(error: TryReserveError) -> Self {
    Self::Capacity(error)
  }
}
