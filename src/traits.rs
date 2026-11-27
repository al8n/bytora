//! The standard traits of `bytes::Bytes` and `bytes::BytesMut`, for both handles.
//!
//! Comparison, ordering and hashing follow the visible elements, as `[T]` does.

#[cfg(not(any(feature = "std", feature = "alloc")))]
use crate::TryPutError;
use crate::{InlineSize, Vector, VectorMut, WritableSize, layout::class};
use core::{
  borrow::{Borrow, BorrowMut},
  cmp::Ordering,
  fmt,
  hash::{Hash, Hasher},
  ops::{Deref, DerefMut},
  slice,
};
#[cfg(any(feature = "std", feature = "alloc"))]
use std::{string::String, vec::Vec};

impl<T, S: InlineSize> Deref for Vector<T, S> {
  type Target = [T];

  #[inline]
  fn deref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> AsRef<[T]> for Vector<T, S> {
  #[inline]
  fn as_ref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> Borrow<[T]> for Vector<T, S> {
  #[inline]
  fn borrow(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> Deref for VectorMut<T, S> {
  type Target = [T];

  #[inline]
  fn deref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> DerefMut for VectorMut<T, S> {
  #[inline]
  fn deref_mut(&mut self) -> &mut [T] {
    self.as_mut_slice()
  }
}

impl<T, S: InlineSize> AsRef<[T]> for VectorMut<T, S> {
  #[inline]
  fn as_ref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> AsMut<[T]> for VectorMut<T, S> {
  #[inline]
  fn as_mut(&mut self) -> &mut [T] {
    self.as_mut_slice()
  }
}

impl<T, S: InlineSize> Borrow<[T]> for VectorMut<T, S> {
  #[inline]
  fn borrow(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T, S: InlineSize> BorrowMut<[T]> for VectorMut<T, S> {
  #[inline]
  fn borrow_mut(&mut self) -> &mut [T] {
    self.as_mut_slice()
  }
}

// Both directions of `==` and `partial_cmp` between a handle and another sequence of its
// elements. `$view` turns `$value`, a reference to the other sequence, into a slice.
macro_rules! compare_elements {
  ($(#[$attr:meta])* $handle:ident, $other:ty, |$value:ident| $view:expr $(, const $n:ident)?) => {
    $(#[$attr])*
    impl<T: PartialEq, S: InlineSize $(, const $n: usize)?> PartialEq<$other> for $handle<T, S> {
      #[inline]
      fn eq(&self, other: &$other) -> bool {
        let $value = other;
        self.as_slice() == $view
      }
    }

    $(#[$attr])*
    impl<T: PartialEq, S: InlineSize $(, const $n: usize)?> PartialEq<$handle<T, S>> for $other {
      #[inline]
      fn eq(&self, other: &$handle<T, S>) -> bool {
        let $value = self;
        $view == other.as_slice()
      }
    }

    $(#[$attr])*
    impl<T: PartialOrd, S: InlineSize $(, const $n: usize)?> PartialOrd<$other> for $handle<T, S> {
      #[inline]
      fn partial_cmp(&self, other: &$other) -> Option<Ordering> {
        let $value = other;
        self.as_slice().partial_cmp($view)
      }
    }

    $(#[$attr])*
    impl<T: PartialOrd, S: InlineSize $(, const $n: usize)?> PartialOrd<$handle<T, S>> for $other {
      #[inline]
      fn partial_cmp(&self, other: &$handle<T, S>) -> Option<Ordering> {
        let $value = self;
        $view.partial_cmp(other.as_slice())
      }
    }
  };
}

// Both directions of `==` and `partial_cmp` between a byte handle and text, by its bytes.
macro_rules! compare_text {
  ($(#[$attr:meta])* $handle:ident, $other:ty) => {
    $(#[$attr])*
    impl<S: InlineSize> PartialEq<$other> for $handle<u8, S> {
      #[inline]
      fn eq(&self, other: &$other) -> bool {
        self.as_slice() == other.as_bytes()
      }
    }

    $(#[$attr])*
    impl<S: InlineSize> PartialEq<$handle<u8, S>> for $other {
      #[inline]
      fn eq(&self, other: &$handle<u8, S>) -> bool {
        self.as_bytes() == other.as_slice()
      }
    }

    $(#[$attr])*
    impl<S: InlineSize> PartialOrd<$other> for $handle<u8, S> {
      #[inline]
      fn partial_cmp(&self, other: &$other) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_bytes())
      }
    }

    $(#[$attr])*
    impl<S: InlineSize> PartialOrd<$handle<u8, S>> for $other {
      #[inline]
      fn partial_cmp(&self, other: &$handle<u8, S>) -> Option<Ordering> {
        self.as_bytes().partial_cmp(other.as_slice())
      }
    }
  };
}

// The traits that both handles implement the same way, through their visible elements.
macro_rules! element_traits {
  ($handle:ident) => {
    impl<T: PartialEq, S: InlineSize> PartialEq for $handle<T, S> {
      #[inline]
      fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
      }
    }

    impl<T: Eq, S: InlineSize> Eq for $handle<T, S> {}

    impl<T: PartialOrd, S: InlineSize> PartialOrd for $handle<T, S> {
      #[inline]
      fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
      }
    }

    impl<T: Ord, S: InlineSize> Ord for $handle<T, S> {
      #[inline]
      fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
      }
    }

    impl<T: Hash, S: InlineSize> Hash for $handle<T, S> {
      #[inline]
      fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
      }
    }

    compare_elements!($handle, [T], |value| value);
    compare_elements!($handle, &[T], |value| *value);
    compare_elements!($handle, [T; N], |value| value.as_slice(), const N);
    compare_elements!(
      #[cfg(any(feature = "std", feature = "alloc"))]
      $handle,
      Vec<T>,
      |value| value.as_slice()
    );
    compare_text!($handle, str);
    compare_text!($handle, &str);
    compare_text!(
      #[cfg(any(feature = "std", feature = "alloc"))]
      $handle,
      String
    );

    impl<S: InlineSize> fmt::LowerHex for $handle<u8, S> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.as_slice() {
          write!(f, "{byte:02x}")?;
        }
        Ok(())
      }
    }

    impl<S: InlineSize> fmt::UpperHex for $handle<u8, S> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.as_slice() {
          write!(f, "{byte:02X}")?;
        }
        Ok(())
      }
    }
  };
}

element_traits!(Vector);
element_traits!(VectorMut);

impl<T: PartialEq, S: InlineSize> PartialEq<VectorMut<T, S>> for Vector<T, S> {
  #[inline]
  fn eq(&self, other: &VectorMut<T, S>) -> bool {
    self.as_slice() == other.as_slice()
  }
}

impl<T: PartialEq, S: InlineSize> PartialEq<Vector<T, S>> for VectorMut<T, S> {
  #[inline]
  fn eq(&self, other: &Vector<T, S>) -> bool {
    self.as_slice() == other.as_slice()
  }
}

impl<T: PartialOrd, S: InlineSize> PartialOrd<VectorMut<T, S>> for Vector<T, S> {
  #[inline]
  fn partial_cmp(&self, other: &VectorMut<T, S>) -> Option<Ordering> {
    self.as_slice().partial_cmp(other.as_slice())
  }
}

impl<T: PartialOrd, S: InlineSize> PartialOrd<Vector<T, S>> for VectorMut<T, S> {
  #[inline]
  fn partial_cmp(&self, other: &Vector<T, S>) -> Option<Ordering> {
    self.as_slice().partial_cmp(other.as_slice())
  }
}

/// Formats the elements as a list, as `[T]` does.
///
/// For bytes that is a list of numbers, where `bytes::Bytes` prints `b"…"`. That text is one call
/// away through `Deref`, with `<[u8]>::escape_ascii`. Its escapes differ from those of `bytes`:
/// `\x00` where `bytes` prints `\0`, and `\'` where it prints `'`.
impl<T: fmt::Debug, S: InlineSize> fmt::Debug for Vector<T, S> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(self.as_slice(), f)
  }
}

/// Formats the elements as a list, as `[T]` does. The `Debug` of [`Vector`] says how to print
/// bytes as text.
impl<T: fmt::Debug, S: InlineSize> fmt::Debug for VectorMut<T, S> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(self.as_slice(), f)
  }
}

/// Appends the bytes of each text where `try_extend_from_slice` would, and returns [`fmt::Error`]
/// where it returns an error.
///
/// A write never panics on capacity. A text past the room of a buffer that cannot grow returns the
/// error, as `bytes::BytesMut` does past its `remaining_mut`, and that call appends nothing.
/// `write!` makes one call per piece, so after an error the pieces before the failed one stay.
///
/// One departure from `bytes`: an allocation that the allocator refuses returns [`fmt::Error`]
/// here, where `bytes` aborts.
impl<S: InlineSize> fmt::Write for VectorMut<u8, S> {
  #[inline]
  fn write_str(&mut self, text: &str) -> fmt::Result {
    self
      .try_extend_from_slice(text.as_bytes())
      .map_err(|_| fmt::Error)
  }
}

impl<'a, T, S: InlineSize> IntoIterator for &'a Vector<T, S> {
  type Item = &'a T;
  type IntoIter = slice::Iter<'a, T>;

  #[inline]
  fn into_iter(self) -> slice::Iter<'a, T> {
    self.as_slice().iter()
  }
}

impl<'a, T, S: InlineSize> IntoIterator for &'a VectorMut<T, S> {
  type Item = &'a T;
  type IntoIter = slice::Iter<'a, T>;

  #[inline]
  fn into_iter(self) -> slice::Iter<'a, T> {
    self.as_slice().iter()
  }
}

impl<'a, T, S: InlineSize> IntoIterator for &'a mut VectorMut<T, S> {
  type Item = &'a mut T;
  type IntoIter = slice::IterMut<'a, T>;

  #[inline]
  fn into_iter(self) -> slice::IterMut<'a, T> {
    self.as_mut_slice().iter_mut()
  }
}

/// Borrows the static slice without copying, as [`Vector::from_static`] does.
impl<T: Sync> From<&'static [T]> for Vector<T> {
  #[inline]
  #[track_caller]
  fn from(values: &'static [T]) -> Self {
    Self::from_static(values)
  }
}

/// Borrows the bytes of the static text without copying, as [`Vector::from_static`] does.
impl From<&'static str> for Vector<u8> {
  #[inline]
  fn from(text: &'static str) -> Self {
    Self::from_static(text.as_bytes())
  }
}

/// Keeps the string's allocation, as the conversion from `Vec<u8>` does.
#[cfg(bytora_atomic)]
impl From<String> for Vector<u8> {
  #[inline]
  fn from(text: String) -> Self {
    Self::from(text.into_bytes())
  }
}

/// Freezes the buffer, as its `freeze` method does, on each budget whose `freeze` cannot fail.
///
/// Those are the one-word budget, whose buffer never leaves its inline storage, and the budgets of
/// three or more words, the default and zero budgets among them, wherever every heap window
/// freezes: with shared ownership, which native pointer-width atomics or `extra-platforms`
/// provide, and without `alloc` and `std`, where no buffer has a heap window. The two-word budget,
/// `BytesN<16>` on 64-bit targets, freezes a heap window only as a unique vector prefix, so it has
/// no such conversion, and nor have the wider budgets with `alloc` or `std` but without shared
/// ownership. On every budget `try_freeze` freezes the buffer, or returns one it cannot freeze.
///
/// ```
/// use bytora::{BytesMutN, BytesN};
///
/// const FOUR_WORDS: usize = 4 * size_of::<usize>();
/// let frozen = BytesN::<FOUR_WORDS>::from(BytesMutN::<FOUR_WORDS>::new());
/// assert!(frozen.is_empty());
/// ```
///
/// ```compile_fail
/// use bytora::{BytesMutN, BytesN};
///
/// const TWO_WORDS: usize = 2 * size_of::<usize>();
/// let frozen = BytesN::<TWO_WORDS>::from(BytesMutN::<TWO_WORDS>::new());
/// ```
impl<S: WritableSize<u8> + class::InfallibleFreeze> From<VectorMut<u8, S>> for Vector<u8, S> {
  #[inline]
  #[track_caller]
  fn from(buffer: VectorMut<u8, S>) -> Self {
    buffer.freeze()
  }
}

/// Moves the elements into a vector, as the buffer's `into_vec` method does.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, S: InlineSize> From<VectorMut<T, S>> for Vec<T> {
  #[inline]
  fn from(buffer: VectorMut<T, S>) -> Self {
    buffer.into_vec()
  }
}

/// Copies the bytes into a new buffer, inline when they fit and into a heap window otherwise.
///
/// With `alloc` or `std`, every budget whose buffer can grow has this conversion: all but the
/// one-word budget, which cannot hold a heap window. That budget has neither this conversion nor
/// `TryFrom<&[u8]>`, and `try_copy_from_slice` returns an error past its inline capacity. Without
/// `alloc` and `std` no buffer grows, and every budget has `TryFrom<&[u8]>` in place of this
/// conversion. Where the allocator refuses the memory, this conversion panics or aborts as the
/// conversions of `Vec` do, and `try_copy_from_slice` returns the refusal.
///
/// ```
/// use bytora::BytesMutN;
///
/// const FOUR_WORDS: usize = 4 * size_of::<usize>();
/// let buffer = BytesMutN::<FOUR_WORDS>::from(&b"bytes"[..]);
/// assert_eq!(buffer, *b"bytes");
/// ```
///
/// ```compile_fail
/// use bytora::BytesMutN;
///
/// const ONE_WORD: usize = size_of::<usize>();
/// let buffer = BytesMutN::<ONE_WORD>::from(&b"bytes"[..]);
/// ```
#[cfg(any(feature = "std", feature = "alloc"))]
impl<S: WritableSize<u8> + class::Growable> From<&[u8]> for VectorMut<u8, S> {
  #[track_caller]
  fn from(bytes: &[u8]) -> Self {
    let mut buffer = Self::new();
    buffer.extend_from_slice(bytes);
    buffer
  }
}

/// Copies the bytes of the text into a new buffer, as the conversion from `&[u8]` does, on the
/// same budgets.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<S: WritableSize<u8> + class::Growable> From<&str> for VectorMut<u8, S> {
  #[inline]
  #[track_caller]
  fn from(text: &str) -> Self {
    Self::from(text.as_bytes())
  }
}

/// Copies the bytes into a new buffer's inline storage, as `try_copy_from_slice` does, or returns
/// a [`TryPutError`] when they exceed the inline capacity: its `requested` is their length and its
/// `available` the inline capacity.
///
/// Without `alloc` and `std` no buffer can grow, so every budget has this conversion in place of
/// `From<&[u8]>`, which only the budgets that can grow have, with either feature.
#[cfg(not(any(feature = "std", feature = "alloc")))]
impl<S: WritableSize<u8>> TryFrom<&[u8]> for VectorMut<u8, S> {
  type Error = TryPutError;

  fn try_from(bytes: &[u8]) -> Result<Self, TryPutError> {
    // Without an allocator a new buffer has only its inline storage, and the
    // room there is the one error.
    Self::try_copy_from_slice(bytes).map_err(|_| TryPutError {
      requested: bytes.len(),
      available: Self::INLINE_CAPACITY,
    })
  }
}

/// Copies the bytes of the text into a new buffer, as the conversion from `&[u8]` does.
#[cfg(not(any(feature = "std", feature = "alloc")))]
impl<S: WritableSize<u8>> TryFrom<&str> for VectorMut<u8, S> {
  type Error = TryPutError;

  #[inline]
  fn try_from(text: &str) -> Result<Self, TryPutError> {
    Self::try_from(text.as_bytes())
  }
}

/// Collects the elements into a [`VectorMut`] and freezes it.
///
/// # Panics
///
/// Panics when the buffer cannot hold the elements, as its `extend` does, or cannot freeze them:
/// heap storage needs shared ownership, which targets without pointer-width atomics get only with
/// the `extra-platforms` feature. `try_extend` and `try_freeze` on a new buffer report either
/// failure instead.
impl<T: Send + Sync, S: WritableSize<T>> FromIterator<T> for Vector<T, S> {
  fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
    iter.into_iter().collect::<VectorMut<T, S>>().freeze()
  }
}
