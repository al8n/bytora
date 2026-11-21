use super::{ExternalPrefix, InlineSize, Layout};
use crate::Slice;
use core::{ptr, slice};
use std::{sync::Arc, vec::Vec};

impl<T, S: InlineSize> Layout<T, S> {
  fn external_prefix_mut(&mut self) -> &mut ExternalPrefix {
    assert!(!self.is_inline(), "expected external storage");
    // SAFETY: The external tag guarantees an initialized packed prefix. We
    // have exclusive access to this handle's descriptor, independent of its peers.
    unsafe { &mut *self.payload.as_mut_ptr().cast::<ExternalPrefix>() }
  }
}

impl<T, S: InlineSize> Slice<T, S> {
  /// Takes a vector, copying it inline when it fits. Otherwise the handle keeps
  /// the vector's buffer and allocates one reference-counted owner. Cloning a
  /// shared handle increments its reference count and preserves its window.
  ///
  /// Returns the original vector if shared storage is needed and the handle is
  /// smaller than three machine words or the length exceeds `isize::MAX`.
  /// The shared constructor requires `Send + Sync` because clones can share
  /// their elements across threads. It is available with `alloc` or `std` on
  /// targets with pointer-width atomics.
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Copy + Send + Sync,
  {
    if let Some(inline) = Self::try_from_slice(&values) {
      return Ok(inline);
    }
    let Some(mut layout) = Layout::try_external(ptr::null(), ptr::null(), values.len()) else {
      return Err(values);
    };
    let prefix = layout.external_prefix_mut();
    let owner = Arc::new(values);
    prefix.data = owner.as_ptr().cast();
    prefix.owner = Arc::into_raw(owner).cast();
    Ok(Self { layout })
  }

  pub(crate) fn shared_slice(&self) -> &[T] {
    let data = self.layout.external_prefix().data.cast::<T>();
    // SAFETY: Each shared Slice holds a strong Arc<Vec<T>> for the entire borrow.
    // The buffer is immutable. data and len describe a checked window of that
    // vector, including a non-null, aligned pointer for empty or ZST windows.
    unsafe { slice::from_raw_parts(data, self.len()) }
  }

  #[track_caller]
  pub(crate) fn advance_shared(&mut self, count: usize) {
    let len = self.len();
    assert!(count <= len, "advance past end");
    let (low, high, tag) = Layout::<T, S>::split_external_len(len - count);
    let prefix = self.layout.external_prefix_mut();
    prefix.data = prefix.data.cast::<T>().wrapping_add(count).cast();
    prefix.len_low = low;
    self.layout.cursor.as_mut()[0] = high;
    self.layout.tag = tag;
  }

  pub(crate) fn clone_shared(&self) -> Self {
    let prefix = *self.layout.external_prefix();
    let layout = Layout::external(prefix, self.layout.cursor.as_ref()[0], self.layout.tag);
    // SAFETY: try_from_vec stored the pointer returned by Arc::into_raw. The
    // source Slice holds a strong reference. Give the new handle its own.
    unsafe { Arc::<Vec<T>>::increment_strong_count(prefix.owner.cast()) };
    Self { layout }
  }
}

impl<T, S: InlineSize> Drop for Slice<T, S> {
  #[inline]
  fn drop(&mut self) {
    if !self.layout.is_inline() {
      let owner = self.layout.external_prefix().owner.cast();
      // SAFETY: Each shared handle owns exactly one strong reference to this
      // Arc<Vec<T>>. Advancing changes only the data window and retains the owner.
      unsafe { Arc::<Vec<T>>::decrement_strong_count(owner) };
    }
  }
}

#[cfg(test)]
mod tests;
