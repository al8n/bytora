#[cfg(any(feature = "std", feature = "alloc"))]
use super::mutable::Window;
use super::{COMPACT_STATIC, EXTERNAL, ExternalPrefix, InlineSize, Layout};
use crate::{Vector, VectorMut};
use core::{mem::ManuallyDrop, num::NonZeroU8, ops::Range, ptr, slice};
#[cfg(all(bytora_atomic, feature = "extra-platforms"))]
pub(super) use portable_atomic_util::Arc;
#[cfg(bytora_atomic)]
use std::boxed::Box;
#[cfg(all(bytora_atomic, not(feature = "extra-platforms")))]
pub(super) use std::sync::Arc;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

#[cfg(bytora_atomic)]
mod arc;
#[cfg(bytora_atomic)]
mod owner;
#[cfg(bytora_atomic)]
mod two_word;
#[cfg(bytora_atomic)]
mod vector;

#[cfg(bytora_atomic)]
type HandleWord = core::mem::MaybeUninit<[u8; size_of::<usize>()]>;

// Vec's accessors supply the raw parts independently of its private field
// order. This record describes the initialized words of the resulting handle.
#[cfg(bytora_atomic)]
#[repr(C)]
struct WideVecWords {
  prefix: ExternalPrefix,
  len: usize,
  tag_word: usize,
}

#[cfg(bytora_atomic)]
impl WideVecWords {
  fn new<T>(data: *const T, owner: *const (), len: usize, tag: u8) -> Self {
    Self {
      prefix: ExternalPrefix {
        data: data.cast(),
        owner,
      },
      len,
      tag_word: ((tag as usize) << super::TAG_SHIFT).to_le(),
    }
  }
}

// Strong Arc pointers are non-null and aligned. No actual managed pointer
// tagged by adding one can have this address. The sentinel is never dereferenced.
const STATIC_OWNER: *const () = ptr::without_provenance(1);

impl<T, S: InlineSize> Layout<T, S> {
  #[cfg(bytora_atomic)]
  #[inline(always)]
  fn adopt_wide_vec(mut values: Vec<T>) -> Self {
    const {
      assert!(size_of::<WideVecWords>() == 4 * size_of::<usize>());
    }
    assert!(S::HANDLE_BYTES >= 4 * size_of::<usize>() && S::HANDLE_BYTES <= 128);
    assert!(vector::AVAILABLE && size_of::<T>() != 0);
    let len = values.len();
    let capacity = values.capacity();
    let (data, owner) = if capacity == len {
      if len == 0 {
        return Self::empty_inline();
      }
      let data = values.as_mut_ptr();
      core::mem::forget(values);
      (data, vector::full_owner(len))
    } else {
      vector::place_header(values)
    };
    let parts = WideVecWords::new(data, owner, len, EXTERNAL);
    let mut storage = core::mem::MaybeUninit::<Self>::uninit();
    let dest = storage.as_mut_ptr();
    let count = S::HANDLE_BYTES / size_of::<usize>();
    // SAFETY: Both records are repr(C), and verify_size checks the handle's
    // word alignment, cursor and tag offsets. The first three words hold the
    // typed pointer prefix and native length. At these sizes the final word
    // covers the complete cursor and tag, with a nonzero final address byte
    // on either endian. Any intervening words belong to the MaybeUninit payload.
    unsafe {
      if count == 4 {
        dest.cast::<WideVecWords>().write(parts);
      } else {
        dest.cast::<ExternalPrefix>().write(parts.prefix);
        dest.cast::<usize>().add(2).write(parts.len);
        dest.cast::<usize>().add(count - 1).write(parts.tag_word);
      }
      storage.assume_init()
    }
  }

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

  #[cfg(bytora_atomic)]
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
    #[cfg(bytora_atomic)]
    if vector::AVAILABLE && self.uses_external_owner() {
      return self
        .owner_slot()
        .load(core::sync::atomic::Ordering::Acquire);
    }
    // SAFETY: A compact static descriptor is immutable. Platforms without
    // raw Vec owners never update the slot through a shared reference.
    unsafe { (*self.external_ptr()).owner }
  }

  #[cfg(not(bytora_atomic))]
  unsafe fn retain_owner(
    &self,
    _data: *const (),
    _len: usize,
    _front: usize,
    _back: usize,
  ) -> *const () {
    self.owner()
  }

  /// # Safety
  ///
  /// `data` and `len` must be the current external descriptor's data word and
  /// visible length. Removing `front` and `back` elements must stay within it.
  #[cfg(bytora_atomic)]
  unsafe fn retain_owner(
    &self,
    data: *const (),
    len: usize,
    front: usize,
    back: usize,
  ) -> *const () {
    // SAFETY: The source pins this acquired owner and the caller supplies its window.
    unsafe { self.retain_loaded_owner(self.owner(), data, len, front, back) }
  }

  /// # Safety
  ///
  /// `owner` must come from this live external descriptor, with an acquire
  /// load if its slot can be promoted. `data` and `len` must be the source's
  /// data word and visible length. The ends omitted by `front` and `back`
  /// must fit together within that window.
  #[cfg(bytora_atomic)]
  #[inline]
  unsafe fn retain_loaded_owner(
    &self,
    owner: *const (),
    data: *const (),
    len: usize,
    front: usize,
    back: usize,
  ) -> *const () {
    // SAFETY: The borrowed source pins its immutable window. A raw marker can
    // occur only in its atomic owner slot. The full marker carries the original
    // length and capacity. Publication retains the peer even if another
    // borrower wins the race.
    unsafe {
      if let Some(full_len) = vector::full_len(owner) {
        return vector::publish(
          self.owner_slot(),
          owner,
          vector::VecHeader::<T>::new(data.cast(), full_len, full_len, 2),
        );
      }
      if vector::raw(owner) {
        return vector::promote::<T>(self.owner_slot(), owner, data, len);
      }
      if owner::retain_direct::<T>(owner) {
        return owner;
      }
      if arc::is_arc(owner) {
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
    #[cfg(bytora_atomic)]
    if vector::raw(owner) {
      let owner = vector::narrow(owner, _front, _back)
        .unwrap_or_else(|| vector::box_raw::<T>(owner, self.data(), self.len()));
      self.external_prefix_mut().owner = owner;
      return owner;
    }
    #[cfg(bytora_atomic)]
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

impl<T> Vector<T, typenum::U0> {
  /// Creates an empty four-word handle with zero inline capacity.
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
/// allocation yields an empty handle that owns nothing, as [`Vector::new`] does.
/// `Send + Sync` allows the owning handles to be cloned and used across threads.
///
/// # Panics
///
/// Panics if a vector of zero-sized elements is longer than `isize::MAX`, as
/// the conversions of the standard library panic on a capacity overflow.
/// [`Vector::try_from_shared_vec`] returns such a vector instead.
#[cfg(bytora_atomic)]
impl<T: Send + Sync> From<Vec<T>> for Vector<T, typenum::U0> {
  #[inline]
  #[track_caller]
  fn from(values: Vec<T>) -> Self {
    Self::adopt_four_word_vec(values)
  }
}

/// Shares a vector's buffer in the default handle without copying, even when the
/// elements would fit inline. An allocated input keeps its original pointer and
/// capacity, even when its length is zero. Clones and ranges share the elements.
/// An empty vector without an allocation yields an empty handle that owns
/// nothing, as [`Vector::new`] does. `Send + Sync` allows the owning handles to
/// be cloned and used across threads.
///
/// # Panics
///
/// Panics if a vector of zero-sized elements is longer than `isize::MAX`, as
/// the conversions of the standard library panic on a capacity overflow.
/// [`Vector::try_from_shared_vec`] returns such a vector instead.
#[cfg(bytora_atomic)]
impl<T: Send + Sync> From<Vec<T>> for Vector<T> {
  #[inline]
  #[track_caller]
  fn from(values: Vec<T>) -> Self {
    Self::adopt_four_word_vec(values)
  }
}

#[cfg(bytora_atomic)]
impl<T: Send + Sync, S: InlineSize> Vector<T, S> {
  // The conversion from `Vec<T>` at the two four-word budgets. Non-zero-sized
  // elements take the adoption path of `try_from_vec`, which needs no Copy.
  #[inline]
  #[track_caller]
  fn adopt_four_word_vec(values: Vec<T>) -> Self {
    if vector::AVAILABLE && size_of::<T>() != 0 {
      return Self {
        layout: Layout::adopt_wide_vec(values),
      };
    }
    if values.capacity() == 0 || (size_of::<T>() == 0 && values.is_empty()) {
      return Self::new();
    }
    match Self::try_from_shared_vec(values) {
      Ok(vector) => vector,
      Err(_) => vector_too_long(),
    }
  }
}

#[cfg(bytora_atomic)]
#[cold]
#[inline(never)]
#[track_caller]
fn vector_too_long() -> ! {
  panic!("vector length exceeds isize::MAX")
}

#[cfg(bytora_atomic)]
impl<T: Send + Sync, S: InlineSize> TryFrom<Box<[T]>> for Vector<T, S> {
  type Error = Box<[T]>;

  fn try_from(values: Box<[T]>) -> Result<Self, Self::Error> {
    Self::try_from_boxed_slice(values)
  }
}

/// Transfers a movable window or clones the visible elements of a shared one.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Clone, S: InlineSize> From<Vector<T, S>> for Vec<T> {
  fn from(vector: Vector<T, S>) -> Self {
    vector.into_vec()
  }
}

/// Transfers a mutable window or clones the visible elements of a shared one.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T: Clone, S: InlineSize> From<Vector<T, S>> for VectorMut<T, S> {
  fn from(vector: Vector<T, S>) -> Self {
    vector.into_mut()
  }
}

impl<T> Vector<T> {
  /// Borrows an owner's slice without copying and drops the owner after all
  /// clones and subranges are gone.
  ///
  /// # Panics
  ///
  /// Panics if the owner exposes more than `isize::MAX` elements.
  #[cfg(bytora_atomic)]
  #[track_caller]
  pub fn from_owner<O>(owner: O) -> Self
  where
    T: Sync,
    O: AsRef<[T]> + Send + 'static,
  {
    Self::try_from_owner(owner).unwrap_or_else(|_| panic!("owner slice length exceeds isize::MAX"))
  }

  /// Borrows a static slice in the default four-word handle. Construction and
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

impl<T, S: InlineSize> Vector<T, S> {
  /// Reports whether the current window can become mutable without copying.
  /// An inline handle owns its elements. A vector or frozen mutable window
  /// must have a single immutable owner. Static slices, Arc slices and custom
  /// owners grant shared access only.
  /// Concurrent clones can change the result immediately.
  pub fn is_unique(&self) -> bool {
    if self.layout.is_inline() {
      return true;
    }
    #[cfg(bytora_atomic)]
    {
      if Layout::<T, S>::TWO_WORD_VEC {
        return self.layout.two_word_unique();
      }
      let owner = self.layout.owner();
      // SAFETY: This handle keeps the tagged owner and its table alive.
      unsafe { (owner::operations::<T>(owner).is_unique)(owner) }
    }
    #[cfg(not(bytora_atomic))]
    false
  }

  fn into_inline_mut(self) -> VectorMut<T, S> {
    debug_assert!(self.layout.is_inline());
    let this = ManuallyDrop::new(self);
    // SAFETY: Transfer the inline elements and cursor exactly once. The move
    // does not require the current T to be Copy after a subtype coercion.
    VectorMut::from_inline_layout(unsafe { ptr::read(&this.layout) })
  }

  #[cfg(bytora_atomic)]
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
  /// Thawing an empty inline vector, including one created by [`Self::new`],
  /// gives the mutable handle zero capacity. Growing that handle requires a
  /// budget and feature set that support heap storage. `VectorMut::new`
  /// starts with writable inline capacity for positive budgets and `Copy` elements.
  pub fn try_into_mut(self) -> Result<VectorMut<T, S>, Self> {
    if self.layout.is_inline() {
      return Ok(self.into_inline_mut());
    }
    #[cfg(bytora_atomic)]
    {
      if Layout::<T, S>::TWO_WORD_VEC {
        return self.take_two_word_mut();
      }
      let mut this = ManuallyDrop::new(self);
      let prefix = this.layout.external_prefix_mut();
      let (owner, data, len) = (prefix.owner, prefix.data, this.layout.external_len());
      // SAFETY: The reference is restored only if the take callback returns None.
      match unsafe { Self::take_external(owner, data, len, |ops| ops.into_mut) } {
        Some(window) => Ok(VectorMut::from_window(window)),
        None => Err(ManuallyDrop::into_inner(this)),
      }
    }
    #[cfg(not(bytora_atomic))]
    Err(self)
  }

  /// Takes exclusive access to the current window, cloning its elements only
  /// when the backing is read-only or another immutable view shares it.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn into_mut(self) -> VectorMut<T, S>
  where
    T: Clone,
  {
    self
      .try_into_mut()
      .unwrap_or_else(|shared| VectorMut::from_window(Window::from_vec(shared.as_slice().to_vec())))
  }

  /// Moves the current window into a vector without cloning elements. An
  /// exclusively owned vector reuses its allocation. An inline window, or a
  /// disjoint segment whose allocation is still shared, can require a new vector.
  /// Returns the original handle if its elements cannot be moved.
  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline]
  pub fn try_into_vec(self) -> Result<Vec<T>, Self> {
    if S::HANDLE_BYTES == size_of::<usize>() {
      let capacity = Self::INLINE_CAPACITY;
      if self.layout.tag.get() == super::TINY_INLINE | capacity as u8 {
        let mut values = Vec::<T>::with_capacity(capacity);
        // SAFETY: This initial cursor covers the entire inline payload. Copy
        // exactly that many elements, including padding, to the fresh vector.
        // Empty and zero-sized windows have no bytes to transfer.
        unsafe {
          if capacity != 0 && size_of::<T>() != 0 {
            ptr::copy_nonoverlapping(
              self.layout.payload.as_ptr().cast(),
              values.as_mut_ptr(),
              capacity,
            );
          }
          values.set_len(capacity);
        }
        return Ok(values);
      }
      return Ok(self.take_one_word_box().into_vec());
    }
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      return self.take_two_word_vec();
    }
    let mut this = ManuallyDrop::new(self);
    #[cfg(bytora_atomic)]
    let full_len = if vector::AVAILABLE && this.layout.uses_external_owner() {
      // SAFETY: The consuming handle has exclusive access to its owner slot.
      vector::full_len(unsafe { (*this.layout.external_ptr()).owner })
    } else {
      None
    };
    #[cfg(not(bytora_atomic))]
    let full_len = None;
    let (data, len, capacity) = if let Some(len) = full_len {
      (this.layout.data().cast_mut().cast(), len, len)
    } else {
      if S::USIZE == 0 && this.layout.is_inline() {
        // A zero-budget inline window is empty for every element type.
        return Ok(Vec::new());
      }
      // SAFETY: Only failure below reads the original handle again.
      let values = unsafe { Self::take_backing_vec(&mut this) };
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

  #[cfg(any(feature = "std", feature = "alloc"))]
  #[inline(never)]
  fn take_one_word_box(self) -> std::boxed::Box<[T]> {
    let range = self.layout.inline_range();
    let len = range.len();
    let mut values = std::boxed::Box::<[T]>::new_uninit_slice(len);
    // SAFETY: The source is the initialized inline window and its entire
    // non-ZST payload fits in seven bytes. The fresh allocation has room for
    // len aligned elements. Dropping this inline handle never visits them.
    unsafe {
      if size_of::<T>() != 0 && len != 0 {
        let src = self.layout.payload.as_ptr().cast::<T>().add(range.start);
        super::copy_tiny(src.cast(), values.as_mut_ptr().cast(), len * size_of::<T>());
      }
      values.assume_init()
    }
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
    #[cfg(bytora_atomic)]
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
  /// use bytora::Vector;
  ///
  /// fn handle(cells: &'static [Cell<u8>]) -> Vector<Cell<u8>> {
  ///     Vector::from_static(cells)
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
  /// On targets with suitable pointer alignment, a two-word handle also
  /// keeps an allocated vector of non-zero-sized elements when the length fits
  /// in 31 bits on 64-bit targets or 15 bits on 32-bit targets. Other smaller
  /// handles copy fitting elements inline.
  ///
  /// Returns the original vector if neither representation can hold it.
  /// The shared constructor requires `Send + Sync` because clones can share
  /// their elements across threads. It is available with `alloc` or `std` on
  /// targets with pointer-width atomics, or with the `extra-platforms` feature.
  #[cfg(bytora_atomic)]
  #[inline]
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Copy + Send + Sync,
  {
    if Layout::<T, S>::fits_two_word_vec(0) {
      if !Layout::<T, S>::fits_two_word_vec(values.len()) {
        // An unrepresentable non-ZST vector cannot fit inline in two words either.
        return Err(values);
      }
      return Ok(Self {
        layout: Layout::two_word_vec::<true>(values),
      });
    }
    if Layout::<T, S>::FITS_EXTERNAL {
      if size_of::<T>() != 0 {
        if vector::AVAILABLE && S::HANDLE_BYTES >= 4 * size_of::<usize>() && S::HANDLE_BYTES <= 128
        {
          return Ok(Self {
            layout: Layout::adopt_wide_vec(values),
          });
        }
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
    if values.len() > Self::INLINE_CAPACITY {
      return Err(values);
    }
    Ok(Self {
      layout: Layout::copy_inline(&values),
    })
  }

  /// Takes ownership of a boxed slice without copying its elements, including
  /// short and empty slices. An exclusively owned window can recover the
  /// original allocation through [`Self::try_into_vec`].
  ///
  /// Uses the same sizes, element lengths and target requirements as
  /// [`Self::try_from_shared_vec`]. Returns the original box on failure.
  #[cfg(bytora_atomic)]
  pub fn try_from_boxed_slice(values: Box<[T]>) -> Result<Self, Box<[T]>>
  where
    T: Send + Sync,
  {
    if (!Layout::<T, S>::FITS_EXTERNAL && !Layout::<T, S>::fits_two_word_vec(values.len()))
      || values.len() > isize::MAX as usize
    {
      return Err(values);
    }
    Self::try_from_shared_vec(values.into_vec()).map_err(Vec::into_boxed_slice)
  }

  /// Keeps a vector's allocation in shared storage for any `Send + Sync`
  /// element type, including borrowed and non-`Clone` elements. This keeps
  /// the allocation even if the elements would fit inline. On eligible targets,
  /// exact-capacity vectors of non-zero-sized elements defer allocating an owner
  /// until first shared or until their hidden ends outgrow the descriptor.
  /// `Vec`s with spare capacity store their owner in the aligned tail of the
  /// allocation when it fits, or allocate a separate owner at construction.
  /// Zero-sized vectors keep an owner that records their original pointer and
  /// logical offsets.
  ///
  /// Handles of at least three words accept up to `isize::MAX` elements.
  /// On targets with suitable pointer alignment, two words can hold a vector
  /// of non-zero-sized elements with a length up to 2,147,483,647 on 64-bit
  /// targets or 32,767 on 32-bit targets. The first share or window adjustment
  /// of a two-word exact-capacity vector allocates its owner. Returns the
  /// original vector if its backing cannot be represented.
  #[cfg(bytora_atomic)]
  #[inline]
  pub fn try_from_shared_vec(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Send + Sync,
  {
    const { assert!(align_of::<vector::VecHeader<T>>() >= 2) };
    let len = values.len();
    if Layout::<T, S>::fits_two_word_vec(len) {
      return Ok(Self {
        layout: Layout::two_word_vec::<false>(values),
      });
    }
    // A non-ZST Vec's allocation already limits its length to isize::MAX.
    if !Layout::<T, S>::FITS_EXTERNAL || (size_of::<T>() == 0 && len > isize::MAX as usize) {
      return Err(values);
    }
    Ok(Self {
      layout: Self::adopt_vec(values),
    })
  }

  #[cfg(bytora_atomic)]
  #[inline(always)]
  fn adopt_vec(mut values: Vec<T>) -> Layout<T, S> {
    let len = values.len();
    let (data, owner) = if vector::AVAILABLE && size_of::<T>() != 0 && values.capacity() == len {
      let data = values.as_mut_ptr();
      core::mem::forget(values);
      (data, vector::full_owner(len))
    } else {
      vector::place_header(values)
    };
    Layout::external(
      owner::initial_data(data),
      owner,
      len,
      NonZeroU8::new(EXTERNAL).unwrap(),
    )
  }

  #[cfg(bytora_atomic)]
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
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      return self.layout.two_word_slice();
    }
    let data = self.layout.data();
    #[cfg(bytora_atomic)]
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
    #[cfg(bytora_atomic)]
    if self.indexed_zst_base().is_some() {
      // This sum is bounded by the originally accepted length (isize::MAX).
      return ptr::without_provenance(data.addr() + count);
    }
    data.cast::<T>().wrapping_add(count)
  }

  /// # Safety
  ///
  /// The source must be external, `len` its visible length, and `range` an
  /// ordered subrange of `0..len`.
  #[track_caller]
  pub(crate) unsafe fn slice_shared(&self, range: Range<usize>, len: usize) -> Self {
    debug_assert!(!self.layout.is_inline() && len == self.len());
    debug_assert!(range.start <= range.end && range.end <= len);
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      return Self {
        layout: self.layout.share_two_word(range),
      };
    }
    let data = self.subwindow_data(range.start);
    let kind = self.layout.tag;
    #[cfg(bytora_atomic)]
    if (Layout::<T, S>::PACKED_LEN || S::HANDLE_BYTES == 4 * size_of::<usize>())
      && vector::AVAILABLE
      && size_of::<T>() != 0
    {
      // SAFETY: The source is a three- or four-word external descriptor for
      // non-ZSTs, and the range is an ordered subrange of the visible length.
      let (owner, kind) = unsafe { self.retain_window(range.start, len - range.end, len) };
      return Self {
        layout: Layout::external(data, owner, range.end - range.start, kind),
      };
    }
    // SAFETY: The original data word and visible length belong to this source.
    // The checked range gives the number of elements omitted at each end.
    let owner = unsafe {
      self
        .layout
        .retain_owner(self.layout.data(), len, range.start, len - range.end)
    };
    // A packed tag also carries length bits. In wider handles the tag only
    // distinguishes static and owned storage, which retention preserves.
    let kind = if Layout::<T, S>::PACKED_LEN {
      Layout::<T, S>::external_kind(owner)
    } else {
      kind
    };
    let layout = Layout::external(data, owner, range.end - range.start, kind);
    Self { layout }
  }

  /// Retains this window's owner for a subwindow that omits `front` elements
  /// before it and `back` after it, and returns the new handle's owner word
  /// and kind. Nothing here returns a handle, so each caller builds the
  /// handle as words, last.
  ///
  /// # Safety
  ///
  /// The source must be a three- or four-word external descriptor for
  /// non-ZSTs, `len` its visible length, and `front + back <= len`.
  /// `vector::AVAILABLE` must hold: the owner slot's atomic load needs the
  /// alignment it guarantees.
  #[cfg(bytora_atomic)]
  #[inline(always)]
  unsafe fn retain_window(&self, front: usize, back: usize, len: usize) -> (*const (), NonZeroU8) {
    if !self.layout.uses_external_owner() {
      // A short static window has this tag in either descriptor size.
      return (STATIC_OWNER, COMPACT_STATIC);
    }
    let owner = self
      .layout
      .owner_slot()
      .load(core::sync::atomic::Ordering::Acquire);
    if owner.addr() & 0b11 != 0 {
      // SAFETY: The acquire observes this window's raw or tagged owner, and
      // the caller's omissions fit within the visible length.
      let owner = unsafe { self.retain_tagged_owner(owner, front, back, len) };
      let kind = if Layout::<T, S>::PACKED_LEN {
        // A long static window may now be short enough for its own tag.
        Layout::<T, S>::external_kind(owner)
      } else {
        NonZeroU8::new(EXTERNAL).unwrap()
      };
      return (owner, kind);
    }
    // SAFETY: Only an initialized, pinned Vec header has both low bits clear.
    unsafe { vector::VecHeader::<T>::retain(owner) };
    (owner.cast_const(), NonZeroU8::new(EXTERNAL).unwrap())
  }

  /// Whether split_off builds its external arm as words: four-word handles
  /// of 64-bit targets, for non-ZSTs, with atomics, the one class measured.
  /// Every other class keeps `slice` and `truncate` behind one call.
  #[cfg(bytora_atomic)]
  pub(crate) const WORD_SPLIT: bool = S::HANDLE_BYTES == 4 * size_of::<usize>()
    && size_of::<usize>() == 8
    && vector::AVAILABLE
    && size_of::<T>() != 0;

  /// `slice(at..)` then `truncate(at)` on an external source, with the
  /// suffix built from words before the truncation, so that an unwind out of
  /// it releases the suffix's reference, for split_off's arm that stays in the
  /// public method. The retain comes before the narrowing, as in `slice`
  /// then `truncate`, so a promotion it publishes is seen by the narrowing.
  ///
  /// # Safety
  ///
  /// [`Self::WORD_SPLIT`] must hold, the source must be external, `len` its
  /// visible length, and `0 < at < len`.
  #[cfg(bytora_atomic)]
  #[inline(always)]
  pub(crate) unsafe fn split_off_words(&mut self, at: usize, len: usize) -> Self {
    debug_assert!(Self::WORD_SPLIT && !self.layout.is_inline() && len == self.len());
    debug_assert!(0 < at && at < len);
    let data = self.subwindow_data(at);
    // SAFETY: The caller's, with nothing omitted at the back. WORD_SPLIT
    // makes the source a four-word descriptor whose owner slot is atomic.
    let (owner, kind) = unsafe { self.retain_window(at, 0, len) };
    // The suffix owns its reference from here. Its words wait in a guard, so
    // that an unwind out of the narrowing drops the suffix, as `slice` then
    // `truncate` would, while the handle itself is built only once the
    // narrowing returns: built before the call, it waits in memory, and its
    // last word is then stored in pieces that a whole-word reload cannot
    // forward.
    let suffix = PendingSuffix::<T, S> {
      data,
      owner,
      len: len - at,
      kind,
      _budget: core::marker::PhantomData,
    };
    self.truncate_shared(at);
    suffix.into_vector()
  }

  /// Retains a raw or tagged owner for a subwindow and returns the owner word
  /// of the new handle, which the caller builds as words, as it builds every
  /// other arm. A call that returned the handle would put the result in
  /// memory on every path.
  ///
  /// # Safety
  ///
  /// The source must be a three- or four-word external descriptor for non-ZSTs.
  /// Its promotable slot must have yielded `owner` with an acquire load and
  /// at least one of the two low bits set. `len` is the source's visible
  /// length, and the subwindow omits `front` elements before it and `back`
  /// after it.
  #[cfg(bytora_atomic)]
  #[inline(never)]
  unsafe fn retain_tagged_owner(
    &self,
    owner: *const (),
    front: usize,
    back: usize,
    len: usize,
  ) -> *const () {
    // SAFETY: The caller acquired this owner from the source's external
    // descriptor and checked both endpoints. Its elements have nonzero size.
    unsafe {
      self
        .layout
        .retain_loaded_owner(owner, self.layout.data(), len, front, back)
    }
  }

  #[track_caller]
  pub(crate) fn advance_shared(&mut self, count: usize) {
    let len = self.len();
    assert!(count <= len, "advance past end");
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      if count != 0 {
        self.layout.keep_two_word(count..len);
      }
      return;
    }
    let data = self.subwindow_data(count);
    let owner = self.layout.narrow_owner(count, 0);
    self.layout.external_prefix_mut().data = data.cast();
    self
      .layout
      .set_external_len(len - count, Layout::<T, S>::external_kind(owner));
  }

  pub(crate) fn truncate_shared(&mut self, len: usize) {
    debug_assert!(len <= self.len());
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      self.layout.keep_two_word(0..len);
      return;
    }
    let owner = self.layout.narrow_owner(0, self.len() - len);
    self
      .layout
      .set_external_len(len, Layout::<T, S>::external_kind(owner));
  }

  #[cfg(bytora_atomic)]
  #[inline(always)]
  unsafe fn copy_retained_descriptor(&self, owner: *const ()) -> Self {
    // On targets that inline the clone, rebuild a four-word descriptor from
    // its fields as separate words. A sixteen-byte copy of the length and
    // final words would be stored and reloaded as one value on its way to
    // the destination, behind the retain. AArch64 keeps the copy: its clone
    // returns through memory, and the caller reloads each sixteen-byte half
    // as one value.
    if cfg!(not(target_arch = "aarch64")) && S::HANDLE_BYTES == 4 * size_of::<usize>() {
      // SAFETY: The caller retained this external window for the new handle.
      return unsafe { self.rebuild_retained_descriptor(owner) };
    }
    // SAFETY: The caller retained this external window for the new handle.
    unsafe { self.copy_retained_words(owner) }
  }

  /// Builds the retained clone of a four-word descriptor from its fields. The
  /// final word of such a descriptor holds only the tag, so the result has
  /// every word that `copy_retained_words` copies.
  ///
  /// # Safety
  ///
  /// This handle must be external, and `owner` must own its window: the word
  /// in its owner slot, or the header that a promotion published there. The
  /// caller must have retained `owner` once for the returned handle, which
  /// releases it when dropped. A static owner needs no retain.
  #[cfg(bytora_atomic)]
  #[inline(always)]
  unsafe fn rebuild_retained_descriptor(&self, owner: *const ()) -> Self {
    Self {
      layout: Layout::external(
        self.layout.data().cast(),
        owner,
        self.layout.external_len(),
        self.layout.tag,
      ),
    }
  }

  /// Copies an external descriptor word by word, with `owner` in place of its
  /// owner word. That slot is the one word a concurrent promotion can replace,
  /// so the copy never reads it.
  ///
  /// # Safety
  ///
  /// This handle must be external, and `owner` must own its window: the word
  /// in its owner slot, or the header that a promotion published there. The
  /// caller must have retained `owner` once for the returned handle, which
  /// releases it when dropped. A static owner needs no retain.
  #[cfg(bytora_atomic)]
  #[inline(always)]
  unsafe fn copy_retained_words(&self, owner: *const ()) -> Self {
    let word = size_of::<usize>();
    let src = ptr::from_ref(self).cast::<u8>();
    let mut copy = core::mem::MaybeUninit::<Self>::uninit();
    let dst = copy.as_mut_ptr().cast::<u8>();
    // SAFETY: The caller has retained this external window. Its data, length
    // and final word (including the entire cursor and tag) are immutable.
    // The two-word prefix fits every such handle. Only the atomic load reads
    // the potentially promotable owner; its retained result replaces that slot.
    // The middle of a wide external payload may remain uninitialized.
    unsafe {
      ptr::copy_nonoverlapping(src, dst, word);
      dst.add(word).cast::<*const ()>().write_unaligned(owner);
      if Layout::<T, S>::COMPACT_EXTERNAL {
        ptr::copy_nonoverlapping(
          src.add(2 * word),
          dst.add(2 * word),
          size_of::<Self>() - 2 * word,
        );
      } else {
        ptr::copy_nonoverlapping(src.add(2 * word), dst.add(2 * word), word);
        let last = size_of::<Self>() - word;
        ptr::copy_nonoverlapping(src.add(last), dst.add(last), word);
      }
      copy.assume_init()
    }
  }

  #[cfg(bytora_atomic)]
  #[inline(never)]
  unsafe fn clone_tagged_owner(&self, owner: *const ()) -> Self {
    if vector::raw(owner) {
      // SAFETY: The source pins this window and its raw geometry. Promotion
      // returns a reference for the peer, even when another clone wins.
      let slot = self.layout.owner_slot();
      let data = self.layout.data();
      let owner = unsafe {
        if let Some(len) = vector::full_len(owner) {
          vector::publish(
            slot,
            owner,
            vector::VecHeader::<T>::new(data.cast(), len, len, 2),
          )
        } else {
          vector::promote::<T>(slot, owner, data, self.layout.external_len())
        }
      };
      return unsafe { self.copy_retained_descriptor(owner) };
    }
    if arc::is_arc(owner) {
      // SAFETY: The enclosing handle keeps the original Arc alive.
      unsafe { arc::retain::<T>(owner, self.layout.data(), self.layout.external_len()) };
    } else if owner != STATIC_OWNER {
      // SAFETY: The caller supplied a tagged pointer and excluded raw geometry.
      unsafe { (owner::managed_operations::<T>(owner).clone)(owner) };
    }
    // SAFETY: These owner slots are immutable. Compact descriptors fit in a
    // few words; wide descriptors only need the prefix, length and final word.
    unsafe {
      if Layout::<T, S>::COMPACT_EXTERNAL {
        ptr::read(self)
      } else {
        self.copy_retained_descriptor(owner)
      }
    }
  }

  // A call boundary keeps the inline payload copy separate from construction
  // of a sparse external descriptor in the caller's return slot on AArch64.
  // The owned arm is a call of its own there, so the plain path is a leaf: it
  // saves no register before its tag test.
  #[cfg_attr(all(target_arch = "aarch64", bytora_atomic), inline(never))]
  pub(crate) fn clone_descriptor(&self) -> Self {
    #[cfg(bytora_atomic)]
    if self.layout.uses_external_owner() {
      // SAFETY: The test above found an external owner reference.
      return unsafe { self.clone_owned() };
    }
    // SAFETY: An immutable inline window admits elements under a Copy proof
    // and never overwrites them. A static window can be duplicated freely,
    // and every managed owner has been retained.
    // Reading Self preserves the window, pointer provenance and uninitialized bytes.
    unsafe { ptr::read(self) }
  }

  /// Clones a handle that holds an external owner reference.
  ///
  /// Out of line on AArch64, so that clone_descriptor's plain path is a leaf
  /// that saves no register before its tag test. Inlined, this arm gave that
  /// path a frame on Linux, for its outline-atomic call, and on Windows, for
  /// the refcount overflow call. macOS's atomics are inline and its plain
  /// path was a leaf either way; the attribute keeps one shape on every
  /// AArch64 target. Inlined everywhere else.
  ///
  /// # Safety
  ///
  /// `self.layout.uses_external_owner()` must hold.
  #[cfg(bytora_atomic)]
  #[cfg_attr(target_arch = "aarch64", inline(never))]
  #[cfg_attr(not(target_arch = "aarch64"), inline(always))]
  unsafe fn clone_owned(&self) -> Self {
    // SAFETY: This restates the caller's obligation. Out of line, the
    // function cannot see the caller's tag test, so without it every owned
    // clone would test the tag again before reaching the owner.
    unsafe { core::hint::assert_unchecked(self.layout.uses_external_owner()) };
    let owner = if vector::AVAILABLE {
      self
        .layout
        .owner_slot()
        .load(core::sync::atomic::Ordering::Acquire)
        .cast_const()
    } else {
      self.layout.owner()
    };
    // SAFETY: The caller found an external owner reference. After excluding
    // tagged owners and raw markers, only actual vector headers remain.
    // Acquire observes any published promotion.
    unsafe {
      if owner.addr() & 1 != 0 || vector::raw(owner) {
        return self.clone_tagged_owner(owner);
      }
      vector::VecHeader::<T>::retain(owner);
      self.copy_retained_descriptor(owner)
    }
  }

  #[cfg(bytora_atomic)]
  #[inline(never)]
  pub(crate) fn replace_external_clone(&mut self, source: &Self) {
    let data = source.layout.data();
    let len = source.layout.external_len();
    // SAFETY: Cloning retains the source's complete external window.
    let owner = unsafe { source.layout.retain_owner(data, len, 0, 0) };
    let previous = if self.layout.uses_external_owner() {
      Some((
        *self.layout.external_prefix_mut(),
        self.layout.external_len(),
      ))
    } else {
      None
    };
    // A bare Layout has no destructor. Install the newly retained window first
    // so even an unwinding destructor leaves a valid destination.
    self.layout = Layout::external(data.cast(), owner, len, NonZeroU8::new(EXTERNAL).unwrap());
    if let Some((prefix, len)) = previous {
      // SAFETY: The exclusive destination transferred its old descriptor here.
      // Inline elements are Copy; all other owners are released exactly once.
      unsafe { owner::release_owner::<T>(prefix.owner, prefix.data, len) };
    }
  }

  pub(crate) fn clone_shared(&self) -> Self {
    #[cfg(bytora_atomic)]
    if Layout::<T, S>::TWO_WORD_VEC {
      return Self {
        layout: self.layout.share_two_word(0..self.len()),
      };
    }
    let data = self.layout.data();
    let len = self.layout.external_len();
    // SAFETY: Cloning retains the source's complete external window.
    let owner = unsafe { self.layout.retain_owner(data, len, 0, 0) };
    let layout = Layout::external(
      data.cast(),
      owner,
      len,
      Layout::<T, S>::external_kind(owner),
    );
    Self { layout }
  }
}

#[cfg(bytora_atomic)]
impl<T, S: InlineSize> Drop for Vector<T, S> {
  #[inline(always)]
  fn drop(&mut self) {
    if self.layout.is_two_word_vec() {
      // SAFETY: Dropping this descriptor transfers its sole reference.
      unsafe { self.layout.drop_two_word() };
      return;
    }
    if self.layout.uses_external_owner() {
      let prefix = *self.layout.external_prefix_mut();
      let len = self.layout.external_len();
      // SAFETY: This handle transfers the external owner reference and the
      // window it describes. No concurrent promotion can access this slot.
      unsafe { owner::release_owner::<T>(prefix.owner, prefix.data, len) };
    }
  }
}

#[cfg(all(test, bytora_atomic))]
mod tests;

// The words of split_off_words' suffix while the source narrows. The suffix
// owns the reference retain_window took, so an unwind out of the narrowing
// drops it as the Vector would; the normal path builds the Vector instead.
#[cfg(bytora_atomic)]
struct PendingSuffix<T, S: InlineSize> {
  data: *const T,
  owner: *const (),
  len: usize,
  kind: NonZeroU8,
  _budget: core::marker::PhantomData<S>,
}

#[cfg(bytora_atomic)]
impl<T, S: InlineSize> PendingSuffix<T, S> {
  // Builds the suffix on the normal path, without running the guard.
  #[inline(always)]
  fn into_vector(self) -> Vector<T, S> {
    let this = ManuallyDrop::new(self);
    Vector {
      layout: Layout::external(this.data, this.owner, this.len, this.kind),
    }
  }
}

#[cfg(bytora_atomic)]
impl<T, S: InlineSize> Drop for PendingSuffix<T, S> {
  // Reached only by an unwind out of the narrowing: building the suffix and
  // dropping it releases its reference, as dropping the Vector does.
  fn drop(&mut self) {
    drop(Vector::<T, S> {
      layout: Layout::external(self.data, self.owner, self.len, self.kind),
    });
  }
}
