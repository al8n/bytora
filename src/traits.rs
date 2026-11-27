//! The standard traits of `bytes::Bytes` and `bytes::BytesMut`, for both handles.
//!
//! Comparison, ordering and hashing follow the visible elements, as `[T]` does.

use crate::{InlineSize, Vector, VectorMut, WritableSize};
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

/// Freezes the buffer, as its `freeze` method does.
///
/// # Panics
///
/// Panics where `freeze` panics: for a heap buffer that the immutable handle cannot represent.
/// `try_freeze` returns such a buffer instead.
impl<S: WritableSize<u8>> From<VectorMut<u8, S>> for Vector<u8, S> {
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

/// Copies the bytes into a new buffer, inline when they fit.
///
/// # Panics
///
/// Panics when the bytes exceed the inline capacity and the buffer cannot grow: with a one-word
/// budget, or without `alloc` and `std`.
impl<S: WritableSize<u8>> From<&[u8]> for VectorMut<u8, S> {
  #[track_caller]
  fn from(bytes: &[u8]) -> Self {
    let mut buffer = Self::new();
    buffer.extend_from_slice(bytes);
    buffer
  }
}

/// Copies the bytes of the text into a new buffer, as the conversion from `&[u8]` does.
///
/// # Panics
///
/// Panics where the conversion from `&[u8]` panics.
impl<S: WritableSize<u8>> From<&str> for VectorMut<u8, S> {
  #[inline]
  #[track_caller]
  fn from(text: &str) -> Self {
    Self::from(text.as_bytes())
  }
}

/// Collects the elements into a [`VectorMut`] and freezes it.
///
/// # Panics
///
/// Panics when the buffer cannot hold the elements, as its `extend` does, or cannot freeze them:
/// heap storage needs shared ownership, which targets without pointer-width atomics get only with
/// the `extra-platforms` feature.
impl<T: Send + Sync, S: WritableSize<T>> FromIterator<T> for Vector<T, S> {
  fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
    iter.into_iter().collect::<VectorMut<T, S>>().freeze()
  }
}
