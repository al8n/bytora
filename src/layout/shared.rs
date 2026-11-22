#[cfg(any(feature = "std", feature = "alloc"))]
use super::mutable::Window;
use super::{COMPACT_STATIC, EXTERNAL, ExternalPrefix, InlineSize, Layout};
use crate::{Slice, SliceMut};
use core::{mem::ManuallyDrop, num::NonZeroU8, ops::Range, ptr, slice};
#[cfg(all(smol_slice_atomic, feature = "extra-platforms"))]
pub(super) use portable_atomic_util::Arc;
#[cfg(smol_slice_atomic)]
use std::boxed::Box;
#[cfg(all(smol_slice_atomic, not(feature = "extra-platforms")))]
pub(super) use std::sync::Arc;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

#[cfg(smol_slice_atomic)]
mod arc;
#[cfg(smol_slice_atomic)]
mod owner;
#[cfg(smol_slice_atomic)]
mod vector;

#[cfg(any(feature = "std", feature = "alloc"))]
type HandleWord = core::mem::MaybeUninit<[u8; size_of::<usize>()]>;

// Strong Arc pointers are non-null and aligned. No actual managed pointer
// tagged by adding one can have this address. The sentinel is never dereferenced.
const STATIC_OWNER: *const () = ptr::without_provenance(1);

impl<T, S: InlineSize> Layout<T, S> {
  const fn try_static(data: *const T, len: usize) -> Option<Self> {
    if !Self::FITS_EXTERNAL || len > isize::MAX as usize {
      return None;
    }
    let kind = if Self::COMPACT_EXTERNAL {
      COMPACT_STATIC
    } else {
      NonZeroU8::new(EXTERNAL).unwrap()
    };
    Some(Self::external(data, STATIC_OWNER, len, kind))
  }

  fn external_kind(owner: *const ()) -> NonZeroU8 {
    if !Self::COMPACT_EXTERNAL {
      return NonZeroU8::new(EXTERNAL).unwrap();
    }
    if ptr::eq(owner, STATIC_OWNER) {
      COMPACT_STATIC
    } else {
      NonZeroU8::new(EXTERNAL).unwrap()
    }
  }

  #[cfg(smol_slice_atomic)]
  fn owner_slot(&self) -> &vector::AtomicPtr<()> {
    let prefix = self.external_ptr().cast_mut();
    // SAFETY: The external descriptor guarantees that the whole field exists.
    let word = unsafe { (&raw mut (*prefix).owner).cast::<*mut ()>() };
    debug_assert!(word.cast::<vector::AtomicPtr<()>>().is_aligned());
    // SAFETY: The enclosing layout aligns this initialized pointer word. It
    // is inside a real UnsafeCell. Only the owner word can change through a
    // shared reference, and all accesses to a promotable slot are atomic.
    unsafe { vector::AtomicPtr::from_ptr(word) }
  }

  fn owner(&self) -> *const () {
    #[cfg(smol_slice_atomic)]
    if vector::AVAILABLE && self.uses_external_owner() {
      return self
        .owner_slot()
        .load(core::sync::atomic::Ordering::Acquire);
    }
    // SAFETY: A compact static descriptor is immutable. Platforms without
    // raw Vec owners never update the slot through a shared reference.
    unsafe { (*self.external_ptr()).owner }
  }

  #[cfg(not(smol_slice_atomic))]
  fn retain_owner(&self, _front: usize, _back: usize) -> *const () {
    self.owner()
  }

  #[cfg(smol_slice_atomic)]
  fn retain_owner(&self, front: usize, back: usize) -> *const () {
    let owner = self.owner();
    if vector::raw(owner) {
      // SAFETY: The borrowed descriptor pins the vector, and its geometry
      // and window remain immutable. Promotion also retains this new peer.
      return unsafe { vector::promote::<T>(self.owner_slot(), owner, self.data(), self.len()) };
    }
    // SAFETY: The source holds a live reference for this owner.
    unsafe {
      if owner::retain_direct::<T>(owner) {
        return owner;
      }
      if arc::is_arc(owner) {
        let (data, len) = (self.data(), self.external_len());
        if let Some(narrowed) = arc::narrow(owner, front, back) {
          arc::retain::<T>(owner, data, len);
          return narrowed;
        }
        return arc::managed_copy::<T>(owner, data, len);
      }
      (owner::managed_operations::<T>(owner).clone)(owner);
      owner
    }
  }

  fn narrow_owner(&mut self, _front: usize, _back: usize) -> *const () {
    let owner = self.external_prefix_mut().owner;
    #[cfg(smol_slice_atomic)]
    if vector::raw(owner) {
      let owner = vector::narrow(owner, _front, _back)
        .unwrap_or_else(|| vector::box_raw::<T>(owner, self.data(), self.len()));
      self.external_prefix_mut().owner = owner;
      return owner;
    }
    #[cfg(smol_slice_atomic)]
    if arc::is_arc(owner) {
      let narrowed = arc::narrow(owner, _front, _back).unwrap_or_else(|| {
        let (data, len) = (self.data(), self.external_len());
        // SAFETY: The new managed owner retains the original Arc before this
        // handle relinquishes its reference. Its elements stay live throughout.
        unsafe {
          let managed = arc::managed_copy::<T>(owner, data, len);
          arc::release::<T>(owner, data, len);
          managed
        }
      });
      self.external_prefix_mut().owner = narrowed;
      return narrowed;
    }
    owner
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

/// Shares a vector's buffer with zero inline capacity. An allocated input
/// keeps its original pointer and capacity, even when its length is zero.
/// Clones and ranges share the elements. An empty vector without an
/// allocation yields [`Slice::empty`].
///
/// Returns the original vector if it contains more than `isize::MAX` elements.
/// `Send + Sync` allows the owning handles to be cloned and used across threads.
#[cfg(smol_slice_atomic)]
impl<T: Send + Sync> TryFrom<Vec<T>> for Slice<T, typenum::U0> {
  type Error = Vec<T>;

  fn try_from(values: Vec<T>) -> Result<Self, Self::Error> {
    if values.capacity() == 0 || (size_of::<T>() == 0 && values.is_empty()) {
      return Ok(Self::empty());
    }
    Self::try_from_shared_vec(values)
  }
}

#[cfg(smol_slice_atomic)]
impl<T: Send + Sync, S: InlineSize> TryFrom<Box<[T]>> for Slice<T, S> {
  type Error = Box<[T]>;

  fn try_from(values: Box<[T]>) -> Result<Self, Self::Error> {
    Self::try_from_boxed_slice(values)
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
  /// must have a single immutable owner. Static slices, Arc slices and custom
  /// owners grant shared access only.
  /// Concurrent clones can change the result immediately.
  pub fn is_unique(&self) -> bool {
    if self.layout.is_inline() {
      return true;
    }
    #[cfg(smol_slice_atomic)]
    {
      let owner = self.layout.owner();
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
  unsafe fn take_external<R>(
    owner: *const (),
    data: *const (),
    len: usize,
    select: impl FnOnce(&owner::StorageVtable<T>) -> owner::Take<R>,
  ) -> Option<R> {
    // SAFETY: The caller transfers the reference corresponding to this table.
    let operation = select(unsafe { owner::operations(owner) });
    // SAFETY: The source's destructor is suppressed. Only a failing callback
    // leaves the reference untouched, including if a skipped element's Drop panics.
    unsafe { operation(owner, data, len) }
  }

  /// Takes exclusive access to the current window without copying its elements.
  /// Returns the original handle if another immutable view shares the backing
  /// or the storage was supplied by a static slice, Arc slice or custom owner.
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
    {
      let mut this = ManuallyDrop::new(self);
      let prefix = this.layout.external_prefix_mut();
      let (owner, data, len) = (prefix.owner, prefix.data, this.layout.external_len());
      // SAFETY: The reference is restored only if the take callback returns None.
      match unsafe { Self::take_external(owner, data, len, |ops| ops.into_mut) } {
        Some(window) => Ok(SliceMut::from_window(window)),
        None => Err(ManuallyDrop::into_inner(this)),
      }
    }
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
  #[inline]
  pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
    let mut this = ManuallyDrop::new(self);
    #[cfg(smol_slice_atomic)]
    let full_len = if vector::AVAILABLE && this.layout.uses_external_owner() {
      // SAFETY: The consuming handle has exclusive access to its owner slot.
      vector::full_len(unsafe { (*this.layout.external_ptr()).owner })
    } else {
      None
    };
    #[cfg(not(smol_slice_atomic))]
    let full_len = None;
    let (data, len, capacity) = if let Some(len) = full_len {
      (this.layout.data().cast_mut().cast(), len, len)
    } else {
      let values = if cfg!(smol_slice_atomic)
        && Layout::<T, S>::COMPACT_EXTERNAL
        && size_of::<Self>() == 4 * size_of::<HandleWord>()
      {
        let words = ptr::from_ref(&*this).cast::<HandleWord>();
        // SAFETY: Four words cover the owned descriptor. Only None leaves
        // ownership with this handle; success and unwind suppress its Drop.
        unsafe {
          Self::take_four_words(
            words.read_unaligned(),
            words.add(1).read_unaligned(),
            words.add(2).read_unaligned(),
            words.add(3).read_unaligned(),
          )
        }
      } else {
        // SAFETY: Only failure below reads the original handle again.
        unsafe { Self::take_backing_vec(&mut this) }
      };
      let Some(values) = values else {
        return Err(ManuallyDrop::into_inner(this));
      };
      let mut values = ManuallyDrop::new(values);
      (values.as_mut_ptr(), values.len(), values.capacity())
    };
    // SAFETY: A full raw marker carries the original length and has matching
    // capacity. For other backings the recovered vector supplies all three parts.
    Ok(unsafe { Vec::from_raw_parts(data, len, capacity) })
  }

  // Separate scalar arguments let the caller read a full raw vector's two
  // pointer words directly from an Option slot, even across Option::take.
  /// # Safety
  ///
  /// The words must be the complete representation of one live, owned Self.
  /// Only a failed take permits the caller to use the original handle again.
  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline(never)]
  unsafe fn take_four_words(
    first: HandleWord,
    second: HandleWord,
    third: HandleWord,
    fourth: HandleWord,
  ) -> Option<Vec<T>> {
    let words = [first, second, third, fourth];
    // SAFETY: The caller transfers exactly the bytes of one valid Self. The
    // four padding-free words preserve partial pointer provenance and every
    // uninitialized byte. Their array need not have the alignment of Self.
    let handle = unsafe { ptr::from_ref(&words).cast::<Self>().read_unaligned() };
    let mut this = ManuallyDrop::new(handle);
    // SAFETY: The copy carries the original handle's ownership. On failure
    // it remains with the caller and this copy's destructor stays suppressed.
    unsafe { Self::take_owned_window(&mut this) }
  }

  /// # Safety
  ///
  /// The caller must own a live descriptor and may use it again only after `None`.
  /// Success transfers ownership and unwinding may already have dropped elements.
  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline(never)]
  unsafe fn take_backing_vec(this: &mut ManuallyDrop<Self>) -> Option<Vec<T>> {
    // SAFETY: This transfers ownership under the same contract.
    unsafe { Self::take_owned_window(this) }
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline(always)]
  unsafe fn take_owned_window(this: &mut ManuallyDrop<Self>) -> Option<Vec<T>> {
    #[cfg(smol_slice_atomic)]
    if this.layout.uses_external_owner() {
      let owner = this.layout.external_prefix_mut().owner;
      let (data, len) = (this.layout.data(), this.layout.external_len());
      if owner.addr() & 1 == 0 && !vector::raw(owner) {
        // SAFETY: Even non-raw owners are real vector headers, and the
        // caller suppresses this handle's destructor on success or unwind.
        return unsafe { owner::direct_into_vec(owner, data, len) };
      }
      // SAFETY: The caller suppressed Drop. The callback owns a successful take
      // or unwind, while a failed take leaves the original reference untouched.
      return unsafe { Self::take_external(owner, data, len, |ops| ops.into_vec) };
    }
    if this.layout.is_inline() {
      // SAFETY: The caller has suppressed Drop and forgets this slot on success.
      Some(unsafe { ManuallyDrop::take(this) }.take_inline_vec())
    } else {
      None
    }
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline(never)]
  fn take_inline_vec(self) -> Vec<T> {
    self.into_inline_mut().into_vec()
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

  /// Takes a vector. A handle of at least three words keeps its allocation,
  /// including its pointer and capacity, even when the vector is short or
  /// empty. An empty vector without an allocation yields an empty inline
  /// handle. Zero-sized elements use inline storage when their count fits.
  /// Smaller handles copy fitting elements inline.
  ///
  /// Returns the original vector if shared storage is needed and the handle is
  /// smaller than three machine words or the length exceeds `isize::MAX`.
  /// The shared constructor requires `Send + Sync` because clones can share
  /// their elements across threads. It is available with `alloc` or `std` on
  /// targets with pointer-width atomics, or with the `extra-platforms` feature.
  #[cfg(smol_slice_atomic)]
  #[inline]
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Copy + Send + Sync,
  {
    if Layout::<T, S>::FITS_EXTERNAL {
      if size_of::<T>() != 0 {
        let empty = values.capacity() == 0;
        if !vector::AVAILABLE && empty {
          return Ok(Self::new());
        }
        let mut layout = Self::adopt_vec(values);
        if empty {
          // An exact, zero-capacity non-ZST Vec owns no allocation. Its zero
          // length also initializes the inline cursor to zero at every size.
          layout.tag = NonZeroU8::MIN;
        }
        return Ok(Self { layout });
      }
      if values.len() > Self::INLINE_CAPACITY {
        return Self::try_from_shared_vec(values);
      }
    }
    Self::try_from_slice(&values).ok_or(values)
  }

  /// Takes ownership of a boxed slice without copying its elements, including
  /// short and empty slices. An exclusively owned window can recover the
  /// original allocation through [`Self::try_into_vec`].
  ///
  /// Returns the original box if the handle is smaller than three words or
  /// the slice has more than `isize::MAX` elements.
  #[cfg(smol_slice_atomic)]
  pub fn try_from_boxed_slice(values: Box<[T]>) -> Result<Self, Box<[T]>>
  where
    T: Send + Sync,
  {
    if !Layout::<T, S>::FITS_EXTERNAL || values.len() > isize::MAX as usize {
      return Err(values);
    }
    Self::try_from_shared_vec(values.into_vec()).map_err(Vec::into_boxed_slice)
  }

  /// Keeps a vector's allocation in shared storage for any `Send + Sync`
  /// element type, including borrowed and non-`Clone` elements. This keeps
  /// the allocation even if the elements would fit inline. On eligible targets,
  /// exact-capacity vectors of non-zero-sized elements defer allocating an owner
  /// until first shared or until their hidden ends outgrow the descriptor.
  /// Vectors with spare capacity store their owner in the aligned tail of the
  /// allocation when it fits, or allocate a separate owner at construction.
  /// Zero-sized vectors keep an owner that records their original pointer and
  /// logical offsets.
  ///
  /// Returns the original vector if this handle is smaller than three words
  /// or the length exceeds `isize::MAX`.
  #[cfg(smol_slice_atomic)]
  #[inline]
  pub fn try_from_shared_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Send + Sync,
  {
    const { assert!(align_of::<vector::Vector<T>>() >= 2) };
    let len = values.len();
    // A non-ZST Vec's allocation already limits its length to isize::MAX.
    if !Layout::<T, S>::FITS_EXTERNAL || (size_of::<T>() == 0 && len > isize::MAX as usize) {
      return Err(values);
    }
    Ok(Self {
      layout: Self::adopt_vec(values),
    })
  }

  #[cfg(smol_slice_atomic)]
  #[inline(always)]
  fn adopt_vec(mut values: Vec<T>) -> Layout<T, S> {
    let len = values.len();
    let data = owner::initial_data(values.as_mut_ptr());
    let owner = if vector::AVAILABLE && size_of::<T>() != 0 && values.capacity() == len {
      core::mem::forget(values);
      vector::full_owner(len)
    } else {
      vector::from_vec(values)
    };
    Layout::external(data, owner, len, NonZeroU8::new(EXTERNAL).unwrap())
  }

  #[cfg(smol_slice_atomic)]
  fn indexed_zst_base(&self) -> Option<owner::ZstBase> {
    if size_of::<T>() == 0 {
      let owner = self.layout.owner();
      // SAFETY: This external descriptor holds a live reference to its backing.
      unsafe { owner::operations::<T>(owner) }.zst_base
    } else {
      None
    }
  }

  pub(crate) fn shared_slice(&self) -> &[T] {
    let data = self.layout.data();
    #[cfg(smol_slice_atomic)]
    let data = match self.indexed_zst_base() {
      // SAFETY: The callback belongs to this live Vec or frozen Window and
      // returns its original aligned ZST pointer, independent of slot offset.
      Some(base) => unsafe { base(self.layout.owner()) },
      None => data,
    };
    // SAFETY: The storage is borrowed for 'static or kept alive by our strong
    // reference. data and len describe a checked window of the original slice,
    // including a non-null, aligned pointer for empty or ZST windows.
    unsafe { slice::from_raw_parts(data.cast(), self.len()) }
  }

  // count has been checked against the remainder of this descriptor's window.
  fn subwindow_data(&self, count: usize) -> *const T {
    let data = self.layout.data();
    #[cfg(smol_slice_atomic)]
    if self.indexed_zst_base().is_some() {
      // This sum is bounded by the originally accepted length (isize::MAX).
      return ptr::without_provenance(data.addr() + count);
    }
    data.cast::<T>().wrapping_add(count)
  }

  #[track_caller]
  pub(crate) fn slice_shared(&self, range: Range<usize>) -> Self {
    let len = self.len();
    assert!(
      range.start <= range.end && range.end <= len,
      "slice range out of bounds"
    );
    let data = self.subwindow_data(range.start);
    let owner = self.layout.retain_owner(range.start, len - range.end);
    let layout = Layout::external(
      data,
      owner,
      range.len(),
      Layout::<T, S>::external_kind(owner),
    );
    Self { layout }
  }

  #[track_caller]
  pub(crate) fn advance_shared(&mut self, count: usize) {
    let len = self.len();
    assert!(count <= len, "advance past end");
    let data = self.subwindow_data(count);
    let owner = self.layout.narrow_owner(count, 0);
    self.layout.external_prefix_mut().data = data.cast();
    self
      .layout
      .set_external_len(len - count, Layout::<T, S>::external_kind(owner));
  }

  pub(crate) fn truncate_shared(&mut self, len: usize) {
    debug_assert!(len <= self.len());
    let owner = self.layout.narrow_owner(0, self.len() - len);
    self
      .layout
      .set_external_len(len, Layout::<T, S>::external_kind(owner));
  }

  #[cfg(smol_slice_atomic)]
  #[inline(always)]
  unsafe fn copy_compact_vec(&self, owner: *const ()) -> Self {
    let word = size_of::<*const ()>();
    let src = ptr::from_ref(self).cast::<u8>();
    let mut copy = core::mem::MaybeUninit::<Self>::uninit();
    let dst = copy.as_mut_ptr().cast::<u8>();
    // SAFETY: The caller retained this compact external descriptor's vector
    // header. Copy the immutable data word and bytes beyond its owner,
    // preserving the tag, padding, and provenance. Only the atomic load
    // accesses the original owner slot, including during concurrent promotion.
    unsafe {
      ptr::copy_nonoverlapping(src, dst, word);
      dst.add(word).cast::<*const ()>().write_unaligned(owner);
      ptr::copy_nonoverlapping(
        src.add(2 * word),
        dst.add(2 * word),
        size_of::<Self>() - 2 * word,
      );
      copy.assume_init()
    }
  }

  #[cfg(smol_slice_atomic)]
  #[inline(never)]
  unsafe fn clone_compact_tagged(&self, owner: *const ()) -> Self {
    if vector::raw(owner) {
      // SAFETY: The source pins this window and its raw geometry. Promotion
      // returns a reference for the peer, even when another clone wins.
      let owner = unsafe {
        vector::promote::<T>(
          self.layout.owner_slot(),
          owner,
          self.layout.data(),
          self.layout.external_len(),
        )
      };
      return unsafe { self.copy_compact_vec(owner) };
    }
    if arc::is_arc(owner) {
      // SAFETY: The enclosing handle keeps the original Arc alive.
      unsafe { arc::retain::<T>(owner, self.layout.data(), self.layout.external_len()) };
    } else if owner != STATIC_OWNER {
      // SAFETY: The caller supplied a tagged pointer and excluded raw geometry.
      unsafe { (owner::managed_operations::<T>(owner).clone)(owner) };
    }
    // SAFETY: Both managed and static descriptors have immutable owner slots.
    unsafe { ptr::read(self) }
  }

  // On AArch64 the call boundary keeps a compact handle's transfer together
  // across the conditional retain and the caller's subsequent drop.
  #[cfg_attr(all(target_arch = "aarch64", smol_slice_atomic), inline(never))]
  pub(crate) fn clone_compact(&self) -> Self {
    #[cfg(smol_slice_atomic)]
    if self.layout.uses_external_owner() {
      let owner = if vector::AVAILABLE {
        self
          .layout
          .owner_slot()
          .load(core::sync::atomic::Ordering::Acquire)
          .cast_const()
      } else {
        self.layout.owner()
      };
      // SAFETY: After excluding tagged owners and raw markers, only actual
      // vector headers remain. Acquire observes any published promotion.
      unsafe {
        if owner.addr() & 1 != 0 || vector::raw(owner) {
          return self.clone_compact_tagged(owner);
        }
        vector::Vector::<T>::retain(owner);
        return self.copy_compact_vec(owner);
      }
    }
    // SAFETY: An immutable inline window admits elements under a Copy proof
    // and never overwrites them. A static window can be duplicated freely,
    // and every managed owner has been retained.
    // Reading Self preserves the window, pointer provenance and uninitialized bytes.
    unsafe { ptr::read(self) }
  }

  pub(crate) fn clone_shared(&self) -> Self {
    let data = self.layout.data().cast();
    let len = self.layout.external_len();
    let owner = self.layout.retain_owner(0, 0);
    let layout = Layout::external(data, owner, len, Layout::<T, S>::external_kind(owner));
    Self { layout }
  }
}

#[cfg(smol_slice_atomic)]
impl<T, S: InlineSize> Drop for Slice<T, S> {
  #[inline(always)]
  fn drop(&mut self) {
    if self.layout.uses_external_owner() {
      let prefix = *self.layout.external_prefix_mut();
      let len = self.layout.external_len();
      // SAFETY: This handle transfers the external owner reference and the
      // window it describes. No concurrent promotion can access this slot.
      unsafe { owner::release_owner::<T>(prefix.owner, prefix.data, len) };
    }
  }
}

#[cfg(all(test, smol_slice_atomic))]
mod tests;
