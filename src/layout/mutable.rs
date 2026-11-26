use super::{InlineSize, Layout};
use crate::{Slice, SliceMut};
use core::{
  marker::PhantomData,
  mem::{self, MaybeUninit},
  num::NonZeroU8,
  ops::{Range, RangeBounds},
  ptr::{self, NonNull},
  slice,
};
#[cfg(any(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec::Vec};

#[cfg(any(feature = "std", feature = "alloc"))]
mod window;
#[cfg(any(feature = "std", feature = "alloc"))]
pub(crate) use window::Window;

// The pointer occupies a whole, aligned word in the payload. Its provenance
// never overlaps the tag. The one-word handle therefore stays inline-only.
#[cfg(any(feature = "std", feature = "alloc"))]
const HEAP: NonZeroU8 = NonZeroU8::MAX;
// An empty immutable handle can exist for arbitrary T, including types that
// require destruction. Its mutable form begins without writable inline slots.
const EMPTY_WITHOUT_COPY: NonZeroU8 = NonZeroU8::new(0xfe).unwrap();

#[repr(transparent)]
pub struct Mutable<T, S, B> {
  storage: B,
  marker: PhantomData<(T, S)>,
}

pub trait Storage<T, S: InlineSize>: Sized {
  const EMPTY: Self;
  fn wrap(layout: Layout<T, S>) -> Self;
  fn layout(&self) -> &Layout<T, S>;
  fn layout_mut(&mut self) -> &mut Layout<T, S>;
}

impl<T, S: InlineSize> Storage<T, S> for Layout<T, S> {
  const EMPTY: Self = Self::empty_inline();

  fn wrap(layout: Self) -> Self {
    layout
  }
  fn layout(&self) -> &Self {
    self
  }
  fn layout_mut(&mut self) -> &mut Self {
    self
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[repr(transparent)]
pub struct Spill<T, S: InlineSize>(Layout<T, S>);

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, S: InlineSize> Storage<T, S> for Spill<T, S> {
  const EMPTY: Self = Self(Layout::empty_inline());

  fn wrap(layout: Layout<T, S>) -> Self {
    Self(layout)
  }
  fn layout(&self) -> &Layout<T, S> {
    &self.0
  }
  fn layout_mut(&mut self) -> &mut Layout<T, S> {
    &mut self.0
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T, S: InlineSize> Drop for Spill<T, S> {
  fn drop(&mut self) {
    drop(self.0.take_window());
  }
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
pub type Spill<T, S> = Layout<T, S>;
#[cfg(target_pointer_width = "64")]
pub type EightBytes<T, S> = Layout<T, S>;
#[cfg(target_pointer_width = "32")]
pub type EightBytes<T, S> = Spill<T, S>;

// Seal the element parameter too: Rust's orphan rules would otherwise allow
// downstream impls of WritableSize<LocalNonCopy> for a foreign positive size.
mod writable {
  pub trait Sealed<T> {}
  impl<T> Sealed<T> for typenum::U0 {}
  impl<T: Copy, U, B> Sealed<T> for typenum::UInt<U, B> {}
}

/// A budget that can construct a fresh mutable buffer for `T`.
///
/// Every positive budget implements this trait for `T: Copy`. The zero budget
/// accepts every element type. Use this bound on code generic over the budget
/// that calls `SliceMut::new`, `SliceMut::with_capacity`, or `SliceMut::freeze`.
///
/// ```compile_fail
/// use paguro::{WritableSize, typenum::U24};
/// struct Element(String);
/// impl WritableSize<Element> for U24 {}
/// ```
///
/// A concrete mutable handle can be coerced to a wider element type. Freezing
/// into inline storage checks that the element type is still `Copy` after
/// that coercion.
///
/// ```compile_fail
/// use core::marker::PhantomData;
/// use paguro::SliceMut;
/// struct Element<F>(u8, PhantomData<F>);
/// impl Copy for Element<fn(&())> {}
/// impl Clone for Element<fn(&())> { fn clone(&self) -> Self { *self } }
/// let value = Element::<fn(&())>(1, PhantomData);
/// let original = SliceMut::<_>::try_from_slice(&[value]).unwrap();
/// let widened: SliceMut<Element<fn(&'static ())>> = original;
/// let _ = widened.freeze();
/// ```
pub trait WritableSize<T>: InlineSize + writable::Sealed<T> {}
impl<T, S: InlineSize + writable::Sealed<T>> WritableSize<T> for S {}

impl<T, S: InlineSize> Layout<T, S> {
  #[cfg(any(feature = "std", feature = "alloc"))]
  const FITS_WINDOW: bool = Self::PAYLOAD_BYTES >= size_of::<NonNull<Window<T>>>();

  #[cfg(any(feature = "std", feature = "alloc"))]
  fn window_ptr(&self) -> Option<NonNull<Window<T>>> {
    if Self::FITS_WINDOW && self.tag == HEAP {
      // SAFETY: The heap tag is written only with a full Box pointer. Layout's
      // payload starts at the word-aligned handle address.
      Some(unsafe { self.payload.as_ptr().cast::<NonNull<Window<T>>>().read() })
    } else {
      None
    }
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  fn take_window(&mut self) -> Option<Box<Window<T>>> {
    let ptr = self.window_ptr()?;
    *self = Self::empty_inline();
    // SAFETY: The cleared descriptor was the sole owner of this Box.
    Some(unsafe { Box::from_raw(ptr.as_ptr()) })
  }

  fn writable_range(&self) -> Range<usize> {
    if self.tag == EMPTY_WITHOUT_COPY {
      0..0
    } else {
      self.inline_range()
    }
  }

  fn inline_limit(&self) -> usize {
    if self.tag == EMPTY_WITHOUT_COPY {
      0
    } else {
      Self::INLINE_CAPACITY
    }
  }

  fn inline_ptr(&mut self, index: usize) -> *mut T {
    if Self::INLINE_CAPACITY == 0 || size_of::<T>() == 0 {
      NonNull::dangling().as_ptr()
    } else {
      self.payload.as_mut_ptr().cast::<T>().wrapping_add(index)
    }
  }

  fn set_range(&mut self, range: Range<usize>) {
    (self.cursor, self.tag) = Self::inline_cursor(range);
  }

  fn move_inline(&mut self, src: usize, dst: usize, len: usize) {
    if src != dst && len != 0 {
      let base = self.inline_ptr(0);
      // SAFETY: Both ranges are inside the payload. Transfer the initialized
      // elements, including across overlapping ranges; only dst stays visible.
      unsafe { ptr::copy(base.add(src), base.add(dst), len) };
    }
    self.set_range(dst..dst + len);
  }
}

impl<T, S: InlineSize> SliceMut<T, S> {
  /// The maximum number of inline elements, as in [`Slice::INLINE_CAPACITY`].
  pub const INLINE_CAPACITY: usize = Slice::<T, S>::INLINE_CAPACITY;

  const fn blank() -> Self {
    Self {
      storage: S::Mutable::EMPTY,
      marker: PhantomData,
    }
  }

  fn wrap(layout: Layout<T, S>) -> Self {
    Self {
      storage: S::Mutable::wrap(layout),
      marker: PhantomData,
    }
  }

  fn unproven_empty() -> Self {
    let mut layout = Layout::empty_inline();
    layout.tag = EMPTY_WITHOUT_COPY;
    Self::wrap(layout)
  }

  pub(crate) fn from_inline_layout(layout: Layout<T, S>) -> Self {
    if layout.len() == 0 {
      return Self::unproven_empty();
    }
    // This is an ownership transfer. Construction-time Copy may no longer
    // hold for the current T after covariance; refreezing checks it anew.
    Self::wrap(layout)
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  pub(crate) fn from_window(window: Window<T>) -> Self {
    Self::from_window_with(|| window)
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  fn from_window_with(make_window: impl FnOnce() -> Window<T>) -> Self {
    assert!(Layout::<T, S>::FITS_WINDOW);
    let mut layout = Layout::<T, S>::empty_inline();
    // Allocate the descriptor before the closure partitions a borrowed source
    // or transfers its live elements. An allocation failure leaves it intact.
    let mut owner = Box::<Window<T>>::new_uninit();
    owner.write(make_window());
    // SAFETY: write initialized the entire window. The Box owns it until its
    // pointer is transferred into the descriptor below.
    let owner = NonNull::from(Box::leak(unsafe { owner.assume_init() }));
    // SAFETY: The complete, word-aligned owner pointer fits in the payload.
    unsafe {
      layout
        .payload
        .as_mut_ptr()
        .cast::<NonNull<Window<T>>>()
        .write(owner)
    };
    layout.tag = HEAP;
    Self::wrap(layout)
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  fn window(&self) -> Option<&Window<T>> {
    let ptr = self.storage.layout().window_ptr()?;
    // SAFETY: The Box stays alive for this shared borrow.
    Some(unsafe { ptr.as_ref() })
  }

  #[cfg(any(feature = "std", feature = "alloc"))]
  fn window_mut(&mut self) -> Option<&mut Window<T>> {
    let mut ptr = self.storage.layout().window_ptr()?;
    // SAFETY: This descriptor exclusively owns its boxed window.
    Some(unsafe { ptr.as_mut() })
  }

  // The pointer covers the whole writable view, even when its initialized
  // length is zero. In particular it is never derived from a shorter slice.
  fn writable_parts(&mut self) -> (*mut T, usize, usize) {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return (window.data_ptr(), window.len(), window.capacity());
    }
    let layout = self.storage.layout_mut();
    let range = layout.writable_range();
    let capacity = layout.inline_limit() - range.start;
    (layout.inline_ptr(range.start), range.len(), capacity)
  }

  /// Creates an allocation-free, empty buffer. Positive budgets start with
  /// inline capacity and require `Copy` elements. A zero budget accepts any T.
  pub const fn new() -> Self
  where
    S: WritableSize<T>,
  {
    Self::blank()
  }

  /// Creates an empty buffer with space for at least `capacity` elements.
  ///
  /// # Panics
  ///
  /// Panics if the requested capacity cannot be represented. Without an
  /// allocator, or with a one-word budget, only the inline capacity is available.
  #[track_caller]
  pub fn with_capacity(capacity: usize) -> Self
  where
    S: WritableSize<T>,
  {
    let mut result = Self::new();
    result.reserve(capacity);
    result
  }

  /// Copies `values` inline, returning `None` when they exceed the inline capacity.
  pub fn try_from_slice(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Layout::try_inline(values).map(Self::wrap)
  }

  /// Borrows the remaining elements.
  pub fn as_slice(&self) -> &[T] {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window() {
      return window.as_slice();
    }
    let layout = self.storage.layout();
    if layout.tag == EMPTY_WITHOUT_COPY {
      return &[];
    }
    layout.as_slice()
  }

  /// Mutably borrows the remaining elements.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    let (data, len, _) = self.writable_parts();
    // SAFETY: The first len slots are initialized and exclusive to this handle.
    unsafe { slice::from_raw_parts_mut(data, len) }
  }

  /// Returns the number of initialized elements in this view.
  pub fn len(&self) -> usize {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window() {
      return window.len();
    }
    self.storage.layout().writable_range().len()
  }

  /// Returns whether the current view is empty.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Returns the number of elements this view can hold without reclaiming or growing.
  pub fn capacity(&self) -> usize {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window() {
      return window.capacity();
    }
    let layout = self.storage.layout();
    layout.inline_limit() - layout.writable_range().start
  }

  /// Skips the first `count` elements in constant time. Heap storage keeps
  /// skipped elements until their prefix is reclaimed or the view drops.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds the current length.
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return window.advance(count);
    }
    if count != 0 {
      assert!(count <= self.len(), "advance past end");
      self.storage.layout_mut().advance(count);
    }
  }

  /// Shortens the initialized view to at most `len` elements and drops its tail.
  pub fn truncate(&mut self, len: usize) {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return window.truncate(len);
    }
    let layout = self.storage.layout_mut();
    let range = layout.writable_range();
    if len < range.len() {
      layout.set_range(range.start..range.start + len);
    }
  }

  /// Drops the current elements while preserving the view's capacity.
  pub fn clear(&mut self) {
    self.truncate(0);
  }

  /// Returns the uninitialized tail exclusively owned by this handle.
  pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
    let (data, len, capacity) = self.writable_parts();
    // SAFETY: The remaining slots are writable and outside our initialized range.
    unsafe { slice::from_raw_parts_mut(data.add(len).cast(), capacity - len) }
  }

  /// Sets the length without initializing or dropping elements.
  ///
  /// # Safety
  ///
  /// `len` must be at most [`Self::capacity`]. Every newly exposed slot must
  /// already contain a valid T. Shortening the view leaks the removed elements
  /// unless the caller has moved or dropped them.
  pub unsafe fn set_len(&mut self, len: usize) {
    assert!(len <= self.capacity(), "length exceeds capacity");
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      // SAFETY: The length and initialization preconditions are as above.
      return unsafe { window.set_len(len) };
    }
    let layout = self.storage.layout_mut();
    if layout.tag != EMPTY_WITHOUT_COPY {
      let start = layout.inline_range().start;
      layout.set_range(start..start + len);
    }
  }

  /// Makes room for `additional` elements using existing storage, returning
  /// false if an allocation would be required. May move the current elements.
  #[must_use]
  pub fn try_reclaim(&mut self, additional: usize) -> bool {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return window.try_reclaim(additional);
    }
    let layout = self.storage.layout_mut();
    let range = layout.writable_range();
    let limit = layout.inline_limit();
    if additional <= limit - range.end {
      return true;
    }
    if additional > limit - range.len() {
      return false;
    }
    layout.move_inline(range.start, 0, range.len());
    true
  }

  /// Ensures space for `additional` elements after the current length.
  ///
  /// # Panics
  ///
  /// Panics on capacity overflow, or when a bounded buffer cannot fit the request.
  #[track_caller]
  pub fn reserve(&mut self, additional: usize) {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return window.reserve(additional);
    }
    if self.try_reclaim(additional) {
      return;
    }
    #[cfg(any(feature = "std", feature = "alloc"))]
    if Layout::<T, S>::FITS_WINDOW {
      let len = self.len();
      let required = len.checked_add(additional).expect("capacity overflow");
      let mut values = Vec::with_capacity(required);
      *self = Self::from_window_with(|| {
        // SAFETY: The new allocation is large enough and distinct. Move the
        // initialized elements; the old inline range is retired immediately.
        unsafe {
          ptr::copy_nonoverlapping(self.as_slice().as_ptr(), values.as_mut_ptr(), len);
          values.set_len(len);
          self.set_len(0);
        }
        Window::from_vec(values)
      });
      return;
    }
    panic!("requested capacity exceeds bounded inline storage");
  }

  /// Copies `elements` onto the end of the buffer.
  #[track_caller]
  pub fn extend_from_slice(&mut self, elements: &[T])
  where
    T: Copy,
  {
    self.reserve(elements.len());
    let len = self.len();
    // SAFETY: reserve provided room; the source is borrowed independently of
    // this exclusive handle, and the destination is outside its active elements.
    unsafe {
      ptr::copy_nonoverlapping(
        elements.as_ptr(),
        self.spare_capacity_mut().as_mut_ptr().cast(),
        elements.len(),
      );
      self.set_len(len + elements.len());
    }
  }

  /// Copies the elements in `range` onto the end of the buffer.
  ///
  /// # Panics
  ///
  /// Panics if the range falls outside the current view or growth exceeds capacity limits.
  #[track_caller]
  pub fn extend_from_within(&mut self, range: impl RangeBounds<usize>)
  where
    T: Copy,
  {
    let len = self.len();
    let range = crate::checked_range(range, len);
    self.reserve(range.len());
    let (base, _, _) = self.writable_parts();
    // SAFETY: Both pointers derive from one base covering the entire writable
    // window. The initialized source and reserved destination are disjoint.
    unsafe {
      ptr::copy_nonoverlapping(base.add(range.start), base.add(len), range.len());
      self.set_len(len + range.len());
    }
  }

  /// Changes the length, cloning `value` into newly initialized slots.
  /// Shrinking drops the removed elements.
  #[track_caller]
  pub fn resize(&mut self, new_len: usize, value: T)
  where
    T: Clone,
  {
    let len = self.len();
    if new_len <= len {
      return self.truncate(new_len);
    }
    self.reserve(new_len - len);
    for initialized in len + 1..new_len {
      self.spare_capacity_mut()[0].write(value.clone());
      // SAFETY: One new element was written. Record it before the next clone can panic.
      unsafe { self.set_len(initialized) };
    }
    self.spare_capacity_mut()[0].write(value);
    // SAFETY: The final slot now contains the original value.
    unsafe { self.set_len(new_len) };
  }

  /// Splits at an index relative to the current start, including in spare capacity.
  /// The left view gets capacity `at`. The returned view gets the remaining space.
  /// Inline splits move elements between handles. Heap splits share disjoint ranges when the
  /// target supports shared allocation ownership, and move the suffix otherwise.
  ///
  /// # Panics
  ///
  /// Panics if `at` exceeds the current capacity.
  #[must_use]
  #[track_caller]
  pub fn split_off(&mut self, at: usize) -> Self {
    assert!(at <= self.capacity(), "split_off past capacity");
    #[cfg(any(feature = "std", feature = "alloc"))]
    if let Some(window) = self.window_mut() {
      return Self::from_window_with(|| window.split_off(at));
    }
    let layout = self.storage.layout_mut();
    if layout.tag == EMPTY_WITHOUT_COPY {
      return Self::unproven_empty();
    }
    let range = layout.inline_range();
    let boundary = range.start + at;
    let mut right = Layout::empty_inline();
    let suffix = range.end.saturating_sub(boundary);
    // SAFETY: The suffix, if present, is initialized. The fresh handle has
    // disjoint storage; we retire these slots from the left before returning.
    unsafe {
      ptr::copy_nonoverlapping(
        layout.inline_ptr(boundary),
        right.inline_ptr(boundary),
        suffix,
      )
    };
    right.set_range(boundary..boundary + suffix);
    // Each inline handle has its own payload. Right-align the left capacity
    // so its implicit end-of-capacity is exactly the split boundary.
    layout.move_inline(range.start, Self::INLINE_CAPACITY - at, range.len().min(at));
    Self::wrap(right)
  }

  /// Removes and returns the first `at` initialized elements.
  ///
  /// # Panics
  ///
  /// Panics if `at` exceeds the current length.
  #[must_use]
  #[track_caller]
  pub fn split_to(&mut self, at: usize) -> Self {
    assert!(at <= self.len(), "split_to past end");
    let right = self.split_off(at);
    mem::replace(self, right)
  }

  /// Removes and returns the initialized elements, leaving the spare capacity here.
  #[must_use]
  pub fn split(&mut self) -> Self {
    self.split_to(self.len())
  }

  /// Joins adjacent views of the same heap allocation without moving elements.
  /// Returns the original `other` when the views cannot be joined in place.
  pub fn try_unsplit(&mut self, other: Self) -> Result<(), Self> {
    if other.capacity() == 0 {
      return Ok(());
    }
    #[cfg(paguro_atomic)]
    {
      let mut other = other;
      if let (Some(left), Some(right)) = (self.window_mut(), other.window_mut()) {
        if left.try_unsplit(right) {
          return Ok(());
        }
      }
      Err(other)
    }
    #[cfg(not(paguro_atomic))]
    {
      Err(other)
    }
  }

  /// Appends `other`, joining adjacent heap views in place when possible.
  #[track_caller]
  pub fn unsplit(&mut self, other: Self) {
    if self.is_empty() {
      *self = other;
      return;
    }
    if let Err(mut other) = self.try_unsplit(other) {
      let count = other.len();
      self.reserve(count);
      let len = self.len();
      // SAFETY: Each view owns a disjoint range (or a distinct inline handle).
      // Transfer the initialized elements exactly once without invoking Clone.
      unsafe {
        ptr::copy_nonoverlapping(
          other.as_slice().as_ptr(),
          self.spare_capacity_mut().as_mut_ptr().cast(),
          count,
        );
        other.set_len(0);
        self.set_len(len + count);
      }
    }
  }

  /// Takes a vector without moving its elements, including its spare capacity.
  /// A one-word budget can represent only an empty, unallocated vector here.
  /// An unrepresentable vector is returned intact.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn try_from_vec(values: Vec<T>) -> Result<Self, Vec<T>> {
    if values.capacity() == 0 {
      return Ok(Self::unproven_empty());
    }
    if !Layout::<T, S>::FITS_WINDOW {
      return Err(values);
    }
    Ok(Self::from_window(Window::from_vec(values)))
  }

  /// Returns the visible elements as a vector. A uniquely held heap buffer
  /// reuses its original allocation, moving a consumed view back to its base.
  /// A still-shared window transfers its elements into a separate vector.
  #[cfg(any(feature = "std", feature = "alloc"))]
  pub fn into_vec(mut self) -> Vec<T> {
    if let Some(window) = self.storage.layout_mut().take_window() {
      return window.into_vec();
    }
    let len = self.len();
    let mut values = Vec::with_capacity(len);
    // SAFETY: The fresh destination has enough space. Transfer the initialized
    // slots and retire them from the source, including after a type coercion.
    unsafe {
      ptr::copy_nonoverlapping(self.as_slice().as_ptr(), values.as_mut_ptr(), len);
      values.set_len(len);
      self.set_len(0);
    }
    values
  }

  fn freeze_inline(mut self) -> Slice<T, S>
  where
    S: WritableSize<T>,
  {
    // U0 has no inline elements. For positive budgets the sealed bound proves
    // the *current* T is Copy, including values written after covariance.
    let mut layout = mem::replace(self.storage.layout_mut(), Layout::empty_inline());
    if layout.tag == EMPTY_WITHOUT_COPY {
      layout = Layout::empty_inline();
    }
    Slice { layout }
  }

  /// Freezes this view and preserves the addresses of heap elements. Handles
  /// of at least three words can freeze disjoint ranges while mutable siblings
  /// are live. A two-word heap window must have a unique allocation and own
  /// the initialized prefix from its base through the visible end. A window
  /// that already relinquished its prefix is returned intact, as is any
  /// window outside the immutable descriptor's limits. Positive budgets
  /// require the current element type to be `Copy`.
  #[cfg(paguro_atomic)]
  pub fn try_freeze(self) -> Result<Slice<T, S>, Self>
  where
    S: WritableSize<T>,
    T: Send + Sync,
  {
    if self.window().is_none() {
      return Ok(self.freeze_inline());
    }
    self.try_freeze_shared()
  }

  /// Freezes into a shared allocation for any `Send + Sync` element type.
  /// Heap elements stay at their current address. With at least three words,
  /// inline elements can move into a new allocation. Returns the original
  /// handle when its byte budget or length cannot represent an external window.
  /// A two-word handle can freeze a unique heap allocation when its initialized
  /// elements begin at the allocation base and their count fits the two-word
  /// vector descriptor. The visible window can start within that prefix.
  #[cfg(paguro_atomic)]
  pub fn try_freeze_shared(mut self) -> Result<Slice<T, S>, Self>
  where
    T: Send + Sync,
  {
    if Layout::<T, S>::TWO_WORD_VEC {
      let Some((values, start)) = self
        .window_mut()
        .and_then(|window| window.take_vec_prefix(Layout::<T, S>::fits_two_word_vec))
      else {
        return Err(self);
      };
      let mut frozen = Slice::try_from_shared_vec(values)
        .unwrap_or_else(|_| unreachable!("the original vector fits in the two-word handle"));
      frozen.advance(start);
      return Ok(frozen);
    }
    if !Layout::<T, S>::FITS_EXTERNAL || self.len() > isize::MAX as usize {
      return Err(self);
    }
    let window = match self.storage.layout_mut().take_window() {
      Some(window) => *window,
      None => Window::from_vec(self.into_vec()),
    };
    Slice::try_from_window(window).map_err(Self::from_window)
  }

  /// Freezes an inline view. Without a shared allocation owner, heap windows
  /// are returned intact.
  #[cfg(not(paguro_atomic))]
  pub fn try_freeze(self) -> Result<Slice<T, S>, Self>
  where
    S: WritableSize<T>,
  {
    #[cfg(any(feature = "std", feature = "alloc"))]
    if self.window().is_some() {
      return Err(self);
    }
    Ok(self.freeze_inline())
  }

  /// Freezes this view, preserving the allocation for heap elements.
  ///
  /// # Panics
  ///
  /// Panics when [`Self::try_freeze`] cannot represent the view. This includes
  /// a two-word heap window with live mutable siblings or a relinquished prefix.
  #[cfg(paguro_atomic)]
  #[track_caller]
  pub fn freeze(self) -> Slice<T, S>
  where
    S: WritableSize<T>,
    T: Send + Sync,
  {
    self
      .try_freeze()
      .ok()
      .expect("heap window cannot fit immutable handle")
  }

  /// Freezes an inline view.
  ///
  /// # Panics
  ///
  /// Panics if the view uses a heap allocation on a target without shared ownership.
  #[cfg(not(paguro_atomic))]
  #[track_caller]
  pub fn freeze(self) -> Slice<T, S>
  where
    S: WritableSize<T>,
  {
    self
      .try_freeze()
      .ok()
      .expect("heap freezing requires shared allocation support")
  }
}

impl<S: InlineSize> SliceMut<u8, S> {
  /// Creates a buffer of `len` initialized zero bytes.
  #[track_caller]
  pub fn zeroed(len: usize) -> Self {
    let mut result = Self::blank();
    result.reserve(len);
    // SAFETY: reserve provided len u8 slots, and all-zero is a valid byte.
    unsafe {
      result.spare_capacity_mut().as_mut_ptr().write_bytes(0, len);
      result.set_len(len);
    }
    result
  }
}

impl<T> SliceMut<T, typenum::U0> {
  /// Creates an allocation-free four-word buffer for any element type.
  pub const fn empty() -> Self {
    Self::blank()
  }
}

/// Takes a vector of any element type with its original allocation.
///
/// ```compile_fail
/// use paguro::SliceMutN;
/// use std::rc::Rc;
/// fn send<T: Send>(_: T) {}
/// send(SliceMutN::<_, 0>::from(vec![Rc::new(1)]));
/// ```
///
/// ```compile_fail
/// use core::cell::Cell;
/// use paguro::SliceMutN;
/// fn sync<T: Sync>() {}
/// sync::<SliceMutN<Cell<u8>, 0>>();
/// ```
///
/// ```compile_fail
/// use core::cell::Cell;
/// use paguro::SliceMutN;
/// struct OnDrop<'a>(&'a Cell<bool>);
/// impl Drop for OnDrop<'_> {
///     fn drop(&mut self) { self.0.set(true); }
/// }
/// let handle;
/// {
///     let flag = Cell::new(false);
///     handle = SliceMutN::<_, 0>::from(vec![OnDrop(&flag)]);
/// }
/// drop(handle);
/// ```
#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> From<Vec<T>> for SliceMut<T, typenum::U0> {
  fn from(values: Vec<T>) -> Self {
    Self::try_from_vec(values)
      .ok()
      .expect("zero-budget mutable handle fits a vector")
  }
}
