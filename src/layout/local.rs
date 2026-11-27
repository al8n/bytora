use super::{EXTERNAL, Layout};
use crate::{
  AssocInlineSize, DefaultSize, InlineSize, OutOfBounds, RangeOutOfBounds, checked_range,
  try_checked_range,
};
use core::{
  marker::PhantomData,
  num::NonZeroU8,
  ops::{Range, RangeBounds},
  ptr, slice,
};
use std::rc::Rc;

/// A thread-local window into an `Rc<[T]>`. Clones and ranges share the
/// original elements and each keeps its own cursor. Elements can contain
/// borrowed data or use thread-local interior mutability.
///
/// The default handle and its `Option` each occupy four machine words, the
/// size of `bytes::Bytes` (16 bytes on 32-bit targets and 32 on 64-bit targets).
/// A positive `S` specifies the exact size in bytes. Zero also selects four words.
///
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<bytora::LocalVector<u8>>();
/// ```
///
/// ```compile_fail
/// fn sync<T: Sync>() {}
/// sync::<bytora::LocalVector<u8>>();
/// ```
#[repr(transparent)]
pub struct LocalVector<T, S: InlineSize = DefaultSize> {
  layout: Layout<T, S>,
  _owner: PhantomData<Rc<[T]>>,
}

/// A thread-local shared vector with byte budget `N`.
pub type LocalVectorN<T, const N: usize> = LocalVector<T, <[u8; N] as AssocInlineSize>::Size>;

// Odd owner words encode the hidden prefix and suffix of a directly held
// Rc<[T]>. Even words are real Rc::into_raw pointers to aligned metadata.
const DIRECT: *const () = ptr::without_provenance(1);
const COUNT_BITS: u32 = usize::BITS / 2 - 1;
const COUNT_MASK: usize = (1 << COUNT_BITS) - 1;

#[repr(align(2))]
struct Owner<T>(Rc<[T]>);

enum Backing<T> {
  Direct {
    original: *const [T],
    prefix: usize,
    suffix: usize,
  },
  Indirect(*const Owner<T>),
}

impl<T> LocalVector<T> {
  /// Takes a reference to the original `Rc` allocation in a four-word handle.
  ///
  /// # Panics
  ///
  /// Panics if the slice contains more than `isize::MAX` elements.
  #[track_caller]
  pub fn from_rc(values: Rc<[T]>) -> Self {
    Self::try_from_rc(values).unwrap_or_else(|_| panic!("Rc slice length exceeds isize::MAX"))
  }
}

impl<T, S: InlineSize> LocalVector<T, S> {
  /// Adopts an existing strong reference without copying or allocating.
  /// Returns the original `Rc` if the handle is smaller than three words
  /// or the slice contains more than `isize::MAX` elements.
  pub fn try_from_rc(values: Rc<[T]>) -> Result<Self, Rc<[T]>> {
    let len = values.len();
    if !Layout::<T, S>::FITS_EXTERNAL || len > isize::MAX as usize {
      return Err(values);
    }
    let original = Rc::into_raw(values);
    Ok(Self::window(original.cast(), DIRECT, len))
  }

  fn window(data: *const T, owner: *const (), len: usize) -> Self {
    Self {
      layout: Layout::external(data, owner, len, NonZeroU8::new(EXTERNAL).unwrap()),
      _owner: PhantomData,
    }
  }

  fn owner_word(&self) -> *const () {
    // SAFETY: LocalVector only constructs external descriptors. Its owner
    // word never changes through a shared reference.
    unsafe { (*self.layout.external_ptr()).owner }
  }

  fn backing(&self) -> Backing<T> {
    let owner = self.owner_word();
    let bits = owner.addr();
    if bits & 1 == 0 {
      return Backing::Indirect(owner.cast());
    }
    let prefix = (bits >> 1) & COUNT_MASK;
    let suffix = bits >> (1 + COUNT_BITS);
    let base = self.layout.data().cast::<T>().wrapping_sub(prefix);
    let len = prefix + self.len() + suffix;
    // This reconstitutes both the Rc::into_raw pointer and its original DST
    // metadata. For a ZST the pointer never moves and the counts still apply.
    Backing::Direct {
      original: ptr::slice_from_raw_parts(base, len),
      prefix,
      suffix,
    }
  }

  /// Borrows the elements of the current window.
  pub fn as_slice(&self) -> &[T] {
    // SAFETY: Every handle pins its original Rc and its window was checked.
    unsafe { slice::from_raw_parts(self.layout.data().cast(), self.len()) }
  }

  /// Returns the number of elements in the current window.
  pub const fn len(&self) -> usize {
    self.layout.external_len()
  }

  /// Returns whether the current window has no elements.
  pub const fn is_empty(&self) -> bool {
    self.len() == 0
  }

  // The range was checked against the current window. Normal adjustments
  // keep this handle's existing reference and only edit the descriptor.
  fn keep_range(&mut self, range: Range<usize>) {
    let data = self.layout.data().cast::<T>().wrapping_add(range.start);
    let owner = match self.backing() {
      Backing::Direct {
        original,
        prefix,
        suffix,
      } => {
        let prefix = prefix + range.start;
        let suffix = suffix + (self.len() - range.end);
        if prefix <= COUNT_MASK && suffix <= COUNT_MASK {
          ptr::without_provenance(1 | (prefix << 1) | (suffix << (1 + COUNT_BITS)))
        } else {
          // SAFETY: The old descriptor pins the original Rc. Keep its own
          // reference intact until we can replace the descriptor, including
          // if allocating the metadata for the new reference unwinds.
          let retained = unsafe {
            Rc::increment_strong_count(original);
            Rc::from_raw(original)
          };
          let header = Rc::into_raw(Rc::new(Owner(retained))).cast();
          *self = Self::window(data, header, range.len());
          return;
        }
      }
      Backing::Indirect(header) => header.cast(),
    };
    self.layout = Layout::external(data, owner, range.len(), NonZeroU8::new(EXTERNAL).unwrap());
  }

  /// Shares a subrange with an independent cursor. A range whose hidden
  /// ends exceed the compact counts allocates a shared metadata owner once.
  ///
  /// # Panics
  ///
  /// Panics if the range is outside the current window. [`Self::try_slice`]
  /// returns an error instead.
  #[track_caller]
  pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
    let range = checked_range(range, self.len());
    let mut peer = self.clone();
    peer.keep_range(range);
    peer
  }

  /// Shares a subrange with an independent cursor, or returns an error where
  /// [`Self::slice`] panics.
  pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Result<Self, RangeOutOfBounds> {
    let range = try_checked_range(range, self.len())?;
    Ok(self.slice(range))
  }

  /// Skips `count` elements at the front.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds the current length. [`Self::try_advance`]
  /// returns an error instead.
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    assert!(count <= self.len(), "advance past end");
    if count != 0 {
      self.keep_range(count..self.len());
    }
  }

  /// Skips `count` elements at the front, or returns an error where
  /// [`Self::advance`] panics.
  pub fn try_advance(&mut self, count: usize) -> Result<(), OutOfBounds> {
    let len = self.len();
    if count > len {
      return Err(OutOfBounds::new(count, len));
    }
    self.advance(count);
    Ok(())
  }

  /// Keeps the first `len` elements, or the whole window when it is shorter.
  pub fn truncate(&mut self, len: usize) {
    if len < self.len() {
      self.keep_range(0..len);
    }
  }

  /// Returns a strong reference to the entire original allocation, including
  /// any elements outside this window.
  pub fn into_original_rc(self) -> Rc<[T]> {
    let backing = self.backing();
    core::mem::forget(self);
    match backing {
      // SAFETY: The consumed descriptor transfers its strong reference.
      Backing::Direct { original, .. } => unsafe { Rc::from_raw(original) },
      Backing::Indirect(header) => {
        // SAFETY: Transfer this descriptor's reference to the metadata owner.
        let owned = unsafe { Rc::from_raw(header) };
        match Rc::try_unwrap(owned) {
          Ok(owner) => owner.0,
          Err(shared) => Rc::clone(&shared.0),
        }
      }
    }
  }
}

impl<T, S: InlineSize> TryFrom<Rc<[T]>> for LocalVector<T, S> {
  type Error = Rc<[T]>;

  fn try_from(values: Rc<[T]>) -> Result<Self, Self::Error> {
    Self::try_from_rc(values)
  }
}

impl<T, S: InlineSize> Clone for LocalVector<T, S> {
  fn clone(&self) -> Self {
    // SAFETY: The source pins the allocation that produced this raw pointer.
    // The new descriptor owns exactly the additional reference taken here.
    unsafe {
      match self.backing() {
        Backing::Direct { original, .. } => Rc::increment_strong_count(original),
        Backing::Indirect(header) => Rc::increment_strong_count(header),
      }
      if Layout::<T, S>::COMPACT_EXTERNAL {
        return ptr::read(self);
      }
    }
    Self::window(self.layout.data().cast(), self.owner_word(), self.len())
  }
}

impl<T, S: InlineSize> Drop for LocalVector<T, S> {
  fn drop(&mut self) {
    // SAFETY: This descriptor owns exactly one reference of the selected
    // type. Releasing the original Rc drops all of its initialized elements.
    unsafe {
      match self.backing() {
        Backing::Direct { original, .. } => Rc::decrement_strong_count(original),
        Backing::Indirect(header) => Rc::decrement_strong_count(header),
      }
    }
  }
}

#[cfg(test)]
mod tests;
