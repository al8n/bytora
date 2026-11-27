use super::{COMPACT_STATIC, EXTERNAL, ExternalPrefix, InlineSize, Layout};
use crate::Slice;
use core::{num::NonZeroU8, ops::Range, ptr, slice};
#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
use std::{sync::Arc, vec::Vec};

// An untagged owner is an Arc<Vec<T>> data pointer, aligned to at least two
// bytes. A tagged owner points one byte into a storage operations table.
#[repr(C, align(2))]
struct StorageVtable {
  // A present callback receives this table's tagged owner pointer.
  retain: Option<unsafe fn(*const ())>,
  release: Option<unsafe fn(*const ())>,
}

impl StorageVtable {
  const fn tagged_owner(&'static self) -> *const () {
    ptr::from_ref(self).wrapping_byte_add(1).cast()
  }
}

static STATIC_VTABLE: StorageVtable = StorageVtable {
  retain: None,
  release: None,
};

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
unsafe fn vtable(owner: *const ()) -> Option<&'static StorageVtable> {
  if owner.addr() & 1 == 0 {
    None
  } else {
    // SAFETY: The caller supplies the owner of a live Slice. Tagged owners
    // are created by StorageVtable::tagged_owner from a static table.
    Some(unsafe { &*owner.wrapping_byte_sub(1).cast::<StorageVtable>() })
  }
}

impl<T, S: InlineSize> Layout<T, S> {
  const fn try_static(data: *const T, len: usize) -> Option<Self> {
    let Some(mut layout) = Self::try_external(data, STATIC_VTABLE.tagged_owner(), len) else {
      return None;
    };
    if Self::COMPACT_EXTERNAL && layout.tag.get() == EXTERNAL {
      layout.tag = COMPACT_STATIC;
    }
    Some(layout)
  }

  fn subwindow_tag(&self, new_tag: NonZeroU8) -> NonZeroU8 {
    if !Self::COMPACT_EXTERNAL || new_tag.get() != EXTERNAL {
      return new_tag;
    }
    // An external source with a zero high length byte keeps its kind when it
    // shrinks. Only a transition from a larger length needs to check the owner.
    if self.tag.get() <= EXTERNAL {
      return self.tag;
    }
    if ptr::eq(self.external_prefix().owner, STATIC_VTABLE.tagged_owner()) {
      COMPACT_STATIC
    } else {
      new_tag
    }
  }

  fn external_prefix_mut(&mut self) -> &mut ExternalPrefix {
    assert!(!self.is_inline(), "expected external storage");
    // SAFETY: The external tag guarantees an initialized packed prefix. We
    // have exclusive access to this handle's descriptor, independent of its peers.
    unsafe { &mut *self.payload.as_mut_ptr().cast::<ExternalPrefix>() }
  }
}

impl<T> Slice<T, typenum::U0> {
  /// Creates an empty three-word handle with zero inline capacity.
  /// This uses only `core` and works for every element type.
  pub const fn empty() -> Self {
    let data = ptr::NonNull::<T>::dangling().as_ptr();
    // A zero-length window needs only an aligned, non-null data pointer.
    Self {
      layout: Layout::try_static(data, 0).expect("a zero-budget handle fits a static window"),
    }
  }
}

/// Shares a vector's buffer with zero inline capacity. A nonempty input keeps
/// its original data pointer. Clones and ranges share those elements.
/// An empty vector yields [`Slice::empty`] without allocating a shared owner.
///
/// Returns the original vector if it contains more than `isize::MAX` elements.
/// `Send + Sync` allows the owning handles to be cloned and used across threads.
#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
impl<T: Send + Sync> TryFrom<Vec<T>> for Slice<T, typenum::U0> {
  type Error = Vec<T>;

  fn try_from(values: Vec<T>) -> Result<Self, Self::Error> {
    if values.is_empty() {
      return Ok(Self::empty());
    }
    Self::try_from_shared_vec(values)
  }
}

impl<T> Slice<T> {
  /// Borrows a static slice in the default three-word handle. Construction and
  /// cloning leave the elements in place and require no allocation or reference counts.
  ///
  /// # Panics
  ///
  /// Panics if the slice has more than `isize::MAX` elements. Only a slice of
  /// zero-sized elements can exceed this limit.
  #[track_caller]
  pub const fn from_static(values: &'static [T]) -> Self
  where
    T: Sync,
  {
    Self::try_from_static(values).expect("static slice length exceeds isize::MAX")
  }
}

impl<T, S: InlineSize> Slice<T, S> {
  /// Borrows a static slice without copying, including when its elements would
  /// fit inline. Returns `None` if the handle is smaller than three machine
  /// words or the slice has more than `isize::MAX` elements.
  ///
  /// This constructor works with just `core`. The elements can be shared
  /// across threads and need only implement `Sync`.
  ///
  /// ```compile_fail
  /// use core::cell::Cell;
  /// use smol_slice::Slice;
  ///
  /// fn handle(cells: &'static [Cell<u8>]) -> Slice<Cell<u8>> {
  ///     Slice::from_static(cells)
  /// }
  /// ```
  pub const fn try_from_static(values: &'static [T]) -> Option<Self>
  where
    T: Sync,
  {
    match Layout::try_static(values.as_ptr(), values.len()) {
      Some(layout) => Some(Self { layout }),
      None => None,
    }
  }

  /// Takes a vector, copying it inline when it fits. Otherwise the handle keeps
  /// the vector's buffer and allocates one reference-counted owner. Cloning a
  /// shared handle increments its reference count and preserves its window.
  ///
  /// Returns the original vector if shared storage is needed and the handle is
  /// smaller than three machine words or the length exceeds `isize::MAX`.
  /// The shared constructor requires `Send + Sync` because clones can share
  /// their elements across threads. It is available with `alloc` or `std` on
  /// targets with pointer-width atomics.
  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Copy + Send + Sync,
  {
    if let Some(inline) = Self::try_from_slice(&values) {
      return Ok(inline);
    }
    Self::try_from_shared_vec(values)
  }

  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  fn try_from_shared_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Send + Sync,
  {
    const { assert!(align_of::<Vec<T>>() >= 2) };
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
    // SAFETY: The storage is borrowed for 'static or kept alive by our strong
    // reference. data and len describe a checked window of the original slice,
    // including a non-null, aligned pointer for empty or ZST windows.
    unsafe { slice::from_raw_parts(data, self.len()) }
  }

  #[track_caller]
  pub(crate) fn slice_shared(&self, range: Range<usize>) -> Self {
    let window = &self.shared_slice()[range];
    let owner = self.layout.external_prefix().owner;
    let mut layout = Layout::try_external(window.as_ptr(), owner, window.len())
      .expect("a subrange fits its source descriptor");
    layout.tag = self.layout.subwindow_tag(layout.tag);
    #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
    {
      let retain = if Layout::<T, S>::COMPACT_EXTERNAL {
        layout.uses_external_owner()
      } else {
        !ptr::eq(owner, STATIC_VTABLE.tagged_owner())
      };
      if retain {
        // SAFETY: The source holds this external owner live. Its validated
        // subwindow keeps the same owner and needs its own reference.
        unsafe { Self::retain(owner) };
      }
    }
    Self { layout }
  }

  #[track_caller]
  pub(crate) fn advance_shared(&mut self, count: usize) {
    let len = self.len();
    assert!(count <= len, "advance past end");
    let (low, high, tag) = Layout::<T, S>::split_external_len(len - count);
    let tag = self.layout.subwindow_tag(tag);
    let prefix = self.layout.external_prefix_mut();
    prefix.data = prefix.data.cast::<T>().wrapping_add(count).cast();
    prefix.len_low = low;
    self.layout.cursor.as_mut()[0] = high;
    self.layout.tag = tag;
  }

  #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
  unsafe fn retain(owner: *const ()) {
    // SAFETY: The caller supplies the owner of a live Slice. A vtable operates
    // on its tagged owner. An untagged owner came directly from Arc::into_raw.
    unsafe {
      match vtable(owner) {
        Some(ops) => {
          if let Some(retain) = ops.retain {
            retain(owner);
          }
        }
        None => Arc::<Vec<T>>::increment_strong_count(owner.cast()),
      }
    }
  }

  // On AArch64 the call boundary keeps a compact handle's transfer together
  // across the conditional retain and the caller's subsequent drop.
  #[cfg_attr(
    all(
      target_arch = "aarch64",
      any(feature = "std", feature = "alloc"),
      target_has_atomic = "ptr"
    ),
    inline(never)
  )]
  pub(crate) fn clone_compact(&self) -> Self {
    #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
    if self.layout.uses_external_owner() {
      // SAFETY: This live external handle owns a reference to the backing storage.
      unsafe { Self::retain(self.layout.external_prefix().owner) };
    }
    // SAFETY: Only Copy elements can be constructed inline. A static window
    // can be duplicated freely, and every managed owner has been retained.
    // Reading Self preserves the window, pointer provenance and uninitialized bytes.
    unsafe { ptr::read(self) }
  }

  pub(crate) fn clone_shared(&self) -> Self {
    let prefix = *self.layout.external_prefix();
    let layout = Layout::external(prefix, self.layout.cursor.as_ref()[0], self.layout.tag);
    #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
    // SAFETY: The source handle keeps this window and its owner live.
    unsafe {
      Self::retain(prefix.owner)
    };
    Self { layout }
  }
}

#[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
impl<T, S: InlineSize> Drop for Slice<T, S> {
  #[inline]
  fn drop(&mut self) {
    if self.layout.uses_external_owner() {
      let owner = self.layout.external_prefix().owner;
      // SAFETY: Each handle holds one reference to its original owner. Advancing
      // changes only the window. Release exactly the reference held by this handle.
      unsafe {
        match vtable(owner) {
          Some(ops) => {
            if let Some(release) = ops.release {
              release(owner);
            }
          }
          None => Arc::<Vec<T>>::decrement_strong_count(owner.cast()),
        }
      }
    }
  }
}

#[cfg(all(
  test,
  any(feature = "std", feature = "alloc"),
  target_has_atomic = "ptr"
))]
mod tests;
