#[cfg(any(feature = "std", feature = "alloc"))]
use super::mutable::Window;
use super::{COMPACT_STATIC, EXTERNAL, ExternalPrefix, InlineSize, Layout};
use crate::{Slice, SliceMut};
use core::{mem::ManuallyDrop, num::NonZeroU8, ops::Range, ptr, slice};
#[cfg(all(smol_slice_atomic, feature = "extra-platforms"))]
pub(super) use portable_atomic_util::Arc;
#[cfg(all(smol_slice_atomic, not(feature = "extra-platforms")))]
pub(super) use std::sync::Arc;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

#[cfg(smol_slice_atomic)]
mod owner;

// Strong Arc pointers are non-null and aligned. No actual managed pointer
// tagged by adding one can have this address. The sentinel is never dereferenced.
const STATIC_OWNER: *const () = ptr::without_provenance(1);

impl<T, S: InlineSize> Layout<T, S> {
  const fn try_static(data: *const T, len: usize) -> Option<Self> {
    let Some(mut layout) = Self::try_external(data, STATIC_OWNER, len) else {
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
    if ptr::eq(self.external_prefix().owner, STATIC_OWNER) {
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
#[cfg(smol_slice_atomic)]
impl<T: Send + Sync> TryFrom<Vec<T>> for Slice<T, typenum::U0> {
  type Error = Vec<T>;

  fn try_from(values: Vec<T>) -> Result<Self, Self::Error> {
    if values.is_empty() {
      return Ok(Self::empty());
    }
    Self::try_from_shared_vec(values)
  }
}

/// Transfers a movable window or clones the visible elements of a shared one.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Clone, S: InlineSize> From<Slice<T, S>> for Vec<T> {
  fn from(slice: Slice<T, S>) -> Self {
    slice.into_vec()
  }
}

/// Transfers a mutable window or clones the visible elements of a shared one.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Clone, S: InlineSize> From<Slice<T, S>> for SliceMut<T, S> {
  fn from(slice: Slice<T, S>) -> Self {
    slice.into_mut()
  }
}

impl<T> Slice<T> {
  /// Borrows an owner's slice without copying and drops the owner after all
  /// clones and subranges are gone.
  ///
  /// # Panics
  ///
  /// Panics if the owner exposes more than `isize::MAX` elements.
  #[cfg(smol_slice_atomic)]
  #[track_caller]
  pub fn from_owner<O>(owner: O) -> Self
  where
    T: Sync,
    O: AsRef<[T]> + Send + 'static,
  {
    Self::try_from_owner(owner).unwrap_or_else(|_| panic!("owner slice length exceeds isize::MAX"))
  }

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
  /// Reports whether the current window can become mutable without copying.
  /// An inline handle owns its elements. A vector or frozen mutable window
  /// must have a single immutable owner. Static and custom owners are read-only.
  /// Concurrent clones can change the result immediately.
  pub fn is_unique(&self) -> bool {
    if self.layout.is_inline() {
      return true;
    }
    #[cfg(smol_slice_atomic)]
    {
      let owner = self.layout.external_prefix().owner;
      // SAFETY: This handle keeps the tagged owner and its table alive.
      unsafe { (owner::operations::<T>(owner).is_unique)(owner) }
    }
    #[cfg(not(smol_slice_atomic))]
    false
  }

  fn into_inline_mut(self) -> SliceMut<T, S> {
    debug_assert!(self.layout.is_inline());
    let this = ManuallyDrop::new(self);
    // SAFETY: Transfer the inline elements and cursor exactly once. The move
    // does not require the current T to be Copy after a subtype coercion.
    SliceMut::from_inline_layout(unsafe { ptr::read(&this.layout) })
  }

  #[cfg(smol_slice_atomic)]
  fn try_take<R>(
    self,
    select: impl FnOnce(&owner::StorageVtable<T>) -> owner::Take<R>,
  ) -> Result<R, Self> {
    let prefix = self.layout.external_prefix();
    let (owner, data, len) = (prefix.owner, prefix.data, self.len());
    // SAFETY: This handle owns the reference corresponding to this table.
    let operation = select(unsafe { owner::operations(owner) });
    let this = ManuallyDrop::new(self);
    // SAFETY: Suppress our release before transferring the strong reference.
    // A failing callback returns that reference untouched. A succeeding or
    // unwinding callback owns it, including when dropping skipped elements panics.
    match unsafe { operation(owner, data, len) } {
      Some(result) => Ok(result),
      None => Err(ManuallyDrop::into_inner(this)),
    }
  }

  /// Takes exclusive access to the current window without copying its elements.
  /// Returns the original handle if another immutable view shares the backing
  /// or the storage was supplied by a static slice or custom owner.
  ///
  /// Thawing an empty inline slice, including one created by [`Self::new`],
  /// gives the mutable handle zero capacity. Growing that handle requires a
  /// budget and feature set that support heap storage. `SliceMut::new`
  /// starts with writable inline capacity for positive budgets and `Copy` elements.
  pub fn try_into_mut(self) -> Result<SliceMut<T, S>, Self> {
    if self.layout.is_inline() {
      return Ok(self.into_inline_mut());
    }
    #[cfg(smol_slice_atomic)]
    return self.try_take(|ops| ops.into_mut).map(SliceMut::from_window);
    #[cfg(not(smol_slice_atomic))]
    Err(self)
  }

  /// Takes exclusive access to the current window, cloning its elements only
  /// when the backing is read-only or another immutable view shares it.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn into_mut(self) -> SliceMut<T, S>
  where
    T: Clone,
  {
    self
      .try_into_mut()
      .unwrap_or_else(|shared| SliceMut::from_window(Window::from_vec(shared.as_slice().to_vec())))
  }

  /// Moves the current window into a vector without cloning elements. An
  /// exclusively owned vector reuses its allocation. An inline window, or a
  /// disjoint segment whose allocation is still shared, can require a new vector.
  /// Returns the original handle if its elements cannot be moved.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
    if self.layout.is_inline() {
      return Ok(self.into_inline_mut().into_vec());
    }
    #[cfg(smol_slice_atomic)]
    return self.try_take(|ops| ops.into_vec);
    #[cfg(not(smol_slice_atomic))]
    Err(self)
  }

  /// Moves the current elements into a vector, cloning the visible window
  /// only if it cannot be moved out of its backing storage.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn into_vec(self) -> Vec<T>
  where
    T: Clone,
  {
    self
      .try_into_vec()
      .unwrap_or_else(|shared| shared.as_slice().to_vec())
  }

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
  /// targets with pointer-width atomics, or with the `extra-platforms` feature.
  #[cfg(smol_slice_atomic)]
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Copy + Send + Sync,
  {
    if let Some(inline) = Self::try_from_slice(&values) {
      return Ok(inline);
    }
    Self::try_from_shared_vec(values)
  }

  /// Keeps a vector's allocation in shared storage for any `Send + Sync`
  /// element type, including borrowed and non-`Clone` elements. This always
  /// uses a shared owner, even if the elements would fit inline.
  ///
  /// Returns the original vector if this handle is smaller than three words
  /// or the length exceeds `isize::MAX`.
  #[cfg(smol_slice_atomic)]
  pub fn try_from_shared_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Send + Sync,
  {
    const { assert!(align_of::<Vec<T>>() >= 2) };
    let Some(mut layout) = Layout::try_external(ptr::null(), ptr::null(), values.len()) else {
      return Err(values);
    };
    let prefix = layout.external_prefix_mut();
    let owner = Arc::new(values);
    prefix.data = owner::initial_data(owner.as_ptr()).cast();
    prefix.owner = Arc::into_raw(owner).cast();
    Ok(Self { layout })
  }

  #[cfg(smol_slice_atomic)]
  fn indexed_zst_base(&self) -> Option<owner::ZstBase> {
    if size_of::<T>() == 0 {
      let owner = self.layout.external_prefix().owner;
      // SAFETY: This external descriptor holds a live reference to its backing.
      unsafe { owner::operations::<T>(owner) }.zst_base
    } else {
      None
    }
  }

  pub(crate) fn shared_slice(&self) -> &[T] {
    let prefix = self.layout.external_prefix();
    let data = prefix.data;
    #[cfg(smol_slice_atomic)]
    let data = match self.indexed_zst_base() {
      // SAFETY: The callback belongs to this live Vec or frozen Window and
      // returns its original aligned ZST pointer, independent of slot offset.
      Some(base) => unsafe { base(prefix.owner) },
      None => data,
    };
    // SAFETY: The storage is borrowed for 'static or kept alive by our strong
    // reference. data and len describe a checked window of the original slice,
    // including a non-null, aligned pointer for empty or ZST windows.
    unsafe { slice::from_raw_parts(data.cast(), self.len()) }
  }

  // count has been checked against the remainder of this descriptor's window.
  fn subwindow_data(&self, count: usize) -> *const T {
    let data = self.layout.external_prefix().data;
    #[cfg(smol_slice_atomic)]
    if self.indexed_zst_base().is_some() {
      // This sum is bounded by the originally accepted length (isize::MAX).
      return ptr::without_provenance(data.addr() + count);
    }
    data.cast::<T>().wrapping_add(count)
  }

  #[track_caller]
  pub(crate) fn slice_shared(&self, range: Range<usize>) -> Self {
    assert!(
      range.start <= range.end && range.end <= self.len(),
      "slice range out of bounds"
    );
    let owner = self.layout.external_prefix().owner;
    let mut layout = Layout::try_external(self.subwindow_data(range.start), owner, range.len())
      .expect("a subrange fits its source descriptor");
    layout.tag = self.layout.subwindow_tag(layout.tag);
    #[cfg(smol_slice_atomic)]
    {
      let retain = if Layout::<T, S>::COMPACT_EXTERNAL {
        layout.uses_external_owner()
      } else {
        !ptr::eq(owner, STATIC_OWNER)
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
    let data = self.subwindow_data(count);
    let prefix = self.layout.external_prefix_mut();
    prefix.data = data.cast();
    prefix.len_low = low;
    self.layout.cursor.as_mut()[0] = high;
    self.layout.tag = tag;
  }

  pub(crate) fn truncate_shared(&mut self, len: usize) {
    debug_assert!(len <= self.len());
    let (low, high, tag) = Layout::<T, S>::split_external_len(len);
    let tag = self.layout.subwindow_tag(tag);
    self.layout.external_prefix_mut().len_low = low;
    self.layout.cursor.as_mut()[0] = high;
    self.layout.tag = tag;
  }

  #[cfg(smol_slice_atomic)]
  unsafe fn retain(owner: *const ()) {
    // SAFETY: The source holds this strong reference live and passes it to
    // the clone operation for the backing that produced it.
    unsafe {
      if !owner::retain_direct::<T>(owner) {
        (owner::managed_operations::<T>(owner).clone)(owner);
      }
    }
  }

  #[cfg(smol_slice_atomic)]
  #[inline(never)]
  unsafe fn clone_compact_managed(&self, owner: *const ()) -> Self {
    // Keeping the indirect call and the following copy together lets the
    // compact fast path tail-call this branch without saving its registers.
    // SAFETY: The caller supplies this handle's tagged managed owner. After
    // retaining it, both descriptors may own a reference to the same window.
    unsafe {
      (owner::managed_operations::<T>(owner).clone)(owner);
      ptr::read(self)
    }
  }

  // On AArch64 the call boundary keeps a compact handle's transfer together
  // across the conditional retain and the caller's subsequent drop.
  #[cfg_attr(all(target_arch = "aarch64", smol_slice_atomic), inline(never))]
  pub(crate) fn clone_compact(&self) -> Self {
    #[cfg(smol_slice_atomic)]
    if self.layout.uses_external_owner() {
      let owner = self.layout.external_prefix().owner;
      // SAFETY: This live external handle owns a reference to the backing storage.
      unsafe {
        if !owner::retain_direct::<T>(owner) {
          return self.clone_compact_managed(owner);
        }
      }
    }
    // SAFETY: An immutable inline window admits elements under a Copy proof
    // and never overwrites them. A static window can be duplicated freely,
    // and every managed owner has been retained.
    // Reading Self preserves the window, pointer provenance and uninitialized bytes.
    unsafe { ptr::read(self) }
  }

  pub(crate) fn clone_shared(&self) -> Self {
    let prefix = *self.layout.external_prefix();
    let layout = Layout::external(prefix, self.layout.cursor.as_ref()[0], self.layout.tag);
    #[cfg(smol_slice_atomic)]
    // SAFETY: The source handle keeps this window and its owner live.
    unsafe {
      (owner::operations::<T>(prefix.owner).clone)(prefix.owner)
    };
    Self { layout }
  }
}

#[cfg(smol_slice_atomic)]
impl<T, S: InlineSize> Drop for Slice<T, S> {
  #[inline]
  fn drop(&mut self) {
    if self.layout.uses_external_owner() {
      let owner = self.layout.external_prefix().owner;
      // SAFETY: Each handle holds one reference to its original owner. Advancing
      // changes only the window. Release exactly the reference held by this handle.
      unsafe { owner::release_owner::<T>(owner) };
    }
  }
}

#[cfg(all(test, smol_slice_atomic))]
mod tests;
