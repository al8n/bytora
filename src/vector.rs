use crate::{
  InlineSize, Layout, Vector,
  error::{OutOfBounds, TryPutError},
};
use core::{mem, ops::Range, ptr};

impl<T, S: InlineSize> Vector<T, S> {
  /// Creates an empty vector without allocating, for any element type or size.
  pub const fn new() -> Self {
    Self {
      layout: Layout::empty_inline(),
    }
  }

  /// Returns the part of this vector corresponding to `subset`.
  ///
  /// An empty subset is accepted regardless of its pointer and returns a new
  /// empty vector. For nonempty zero-sized elements, the pointer must match
  /// this vector's pointer and the length must fit. Such pointers do not encode
  /// an element offset or reliably distinguish allocations, so the returned
  /// window uses the corresponding prefix.
  ///
  /// # Panics
  ///
  /// Panics if a nonempty subset is outside this vector's current window.
  /// [`Self::try_slice_ref`] returns `None` instead.
  #[track_caller]
  pub fn slice_ref(&self, subset: &[T]) -> Self {
    if subset.is_empty() {
      return Self::new();
    }
    match self.subset_range(subset) {
      Some(range) => self.slice(range),
      None => subset_outside_vector(),
    }
  }

  /// Returns the part of this vector corresponding to `subset`, as
  /// [`Self::slice_ref`] does, or `None` where that method panics: for a
  /// nonempty subset outside this vector's current window.
  #[inline]
  pub fn try_slice_ref(&self, subset: &[T]) -> Option<Self> {
    if subset.is_empty() {
      return Some(Self::new());
    }
    let range = self.subset_range(subset)?;
    Some(self.slice(range))
  }

  // The range of a nonempty `subset` within the current window.
  #[inline]
  fn subset_range(&self, subset: &[T]) -> Option<Range<usize>> {
    let source = self.as_slice().as_ptr();
    let len = self.len();
    if mem::size_of::<T>() == 0 {
      let fits = ptr::eq(source, subset.as_ptr()) && subset.len() <= len;
      return if fits { Some(0..subset.len()) } else { None };
    }

    let byte_offset = subset.as_ptr().addr().checked_sub(source.addr())?;
    let element_size = mem::size_of::<T>();
    if byte_offset % element_size != 0 {
      return None;
    }
    let start = byte_offset / element_size;
    if start <= len && subset.len() <= len - start {
      Some(start..start + subset.len())
    } else {
      None
    }
  }

  /// Splits at `at`, returning the suffix and retaining the prefix in `self`.
  ///
  /// Static and shared nonempty windows keep their original storage. Inline
  /// windows are copied. Pointers of empty windows are unspecified.
  ///
  /// # Panics
  ///
  /// Panics if `at` exceeds [`Self::len`]. [`Self::try_split_off`] returns an
  /// error instead.
  #[must_use = "consider truncate if the suffix is not needed"]
  #[track_caller]
  pub fn split_off(&mut self, at: usize) -> Self {
    let len = self.len();
    if at > len {
      split_index_out_of_bounds();
    }
    // The arm that ends in words stays here, for the class WORD_SPLIT names,
    // and every other arm is one call that returns the result: a helper with
    // several arms, wrapped once, would take this arm's words apart.
    #[cfg(bytora_atomic)]
    if Self::WORD_SPLIT && at != 0 && at != len && !self.layout.is_inline() {
      // SAFETY: WORD_SPLIT holds, the source is external, `len` is its
      // length, and `0 < at < len`.
      return unsafe { self.split_off_words(at, len) };
    }
    self.split_off_other(at, len)
  }

  /// Splits at `at` as [`Self::split_off`] does, returning the suffix and
  /// retaining the prefix in `self`.
  ///
  /// # Errors
  ///
  /// Returns [`OutOfBounds`] if `at` exceeds [`Self::len`], and leaves `self`
  /// unchanged.
  pub fn try_split_off(&mut self, at: usize) -> Result<Self, OutOfBounds> {
    let len = self.len();
    if at > len {
      return Err(OutOfBounds::new(at, len));
    }
    #[cfg(bytora_atomic)]
    if Self::WORD_SPLIT && at != 0 && at != len && !self.layout.is_inline() {
      // SAFETY: WORD_SPLIT holds, the source is external, `len` is its
      // length, and `0 < at < len`.
      return Ok(unsafe { self.split_off_words(at, len) });
    }
    self.try_split_off_other(at, len)
  }

  // split_off's other arms, one wrapper per form, each returning its form's
  // own result.
  #[inline(never)]
  fn split_off_other(&mut self, at: usize, len: usize) -> Self {
    self.split_off_within(at, len)
  }

  #[inline(never)]
  fn try_split_off_other(&mut self, at: usize, len: usize) -> Result<Self, OutOfBounds> {
    Ok(self.split_off_within(at, len))
  }

  // `len` is the current length, and `at` is at most `len`.
  #[inline]
  fn split_off_within(&mut self, at: usize, len: usize) -> Self {
    if at == len {
      return Self::new();
    }
    if at == 0 {
      return mem::take(self);
    }

    let suffix = self.slice(at..);
    self.truncate(at);
    suffix
  }

  /// Splits at `at`, returning the prefix and retaining the suffix in `self`.
  ///
  /// Static and shared nonempty windows keep their original storage. Inline
  /// windows are copied. Pointers of empty windows are unspecified.
  ///
  /// # Panics
  ///
  /// Panics if `at` exceeds [`Self::len`]. [`Self::try_split_to`] returns an
  /// error instead.
  #[must_use = "consider advance if the prefix is not needed"]
  #[track_caller]
  pub fn split_to(&mut self, at: usize) -> Self {
    let len = self.len();
    if at > len {
      split_index_out_of_bounds();
    }
    self.split_to_within(at, len)
  }

  /// Splits at `at` as [`Self::split_to`] does, returning the prefix and
  /// retaining the suffix in `self`.
  ///
  /// # Errors
  ///
  /// Returns [`OutOfBounds`] if `at` exceeds [`Self::len`], and leaves `self`
  /// unchanged.
  pub fn try_split_to(&mut self, at: usize) -> Result<Self, OutOfBounds> {
    let len = self.len();
    if at > len {
      return Err(OutOfBounds::new(at, len));
    }
    Ok(self.split_to_within(at, len))
  }

  // `len` is the current length, and `at` is at most `len`.
  #[inline]
  fn split_to_within(&mut self, at: usize, len: usize) -> Self {
    if at == len {
      return mem::take(self);
    }
    if at == 0 {
      return Self::new();
    }

    let prefix = self.slice(..at);
    self.advance(at);
    prefix
  }

  /// Shortens the vector to `len` elements, keeping its prefix.
  ///
  /// A length greater than the current length has no effect. Truncating to
  /// zero releases this handle's ownership of the original storage.
  pub fn truncate(&mut self, len: usize) {
    if self.layout.is_inline() {
      self.layout.truncate_inline(len);
    } else if len == 0 {
      *self = Self::new();
    } else if len < self.len() {
      self.truncate_shared(len);
    }
  }

  /// Removes every element, releasing this handle's original storage.
  pub fn clear(&mut self) {
    self.truncate(0);
  }

  /// Copies elements into a new vector.
  ///
  /// Elements are stored inline if they fit. When shared ownership is
  /// available through native atomics or `extra-platforms`, larger inputs use
  /// shared storage when the handle is large enough to hold it.
  ///
  /// # Panics
  ///
  /// Panics when the input does not fit inline and shared storage is
  /// unavailable, the handle is too small, or the length exceeds
  /// `isize::MAX`. [`Self::try_copy_from_slice`] returns an error instead.
  #[cfg(bytora_atomic)]
  #[track_caller]
  pub fn copy_from_slice(values: &[T]) -> Self
  where
    T: Copy + Send + Sync,
  {
    match Self::copy_slice(values) {
      Some(copy) => copy,
      None => slice_does_not_fit(),
    }
  }

  /// Copies elements into a new inline vector.
  ///
  /// # Panics
  ///
  /// Panics when the input does not fit inline. [`Self::try_copy_from_slice`]
  /// returns an error instead.
  #[cfg(not(bytora_atomic))]
  #[track_caller]
  pub fn copy_from_slice(values: &[T]) -> Self
  where
    T: Copy,
  {
    match Self::try_from_slice(values) {
      Some(copy) => copy,
      None => slice_does_not_fit(),
    }
  }

  /// Copies elements inline when possible, or into shared storage when needed.
  ///
  /// Shared copies use the vector limits of [`Self::try_from_vec`].
  ///
  /// # Errors
  ///
  /// Returns [`TryPutError`] where [`Self::copy_from_slice`] panics: when
  /// neither the inline payload nor a vector can represent the input. Its
  /// `available` is the longest input that this handle accepts.
  #[cfg(bytora_atomic)]
  pub fn try_copy_from_slice(values: &[T]) -> Result<Self, TryPutError>
  where
    T: Copy + Send + Sync,
  {
    Self::copy_slice(values).ok_or_else(|| Self::copy_error(values.len()))
  }

  /// Copies elements inline.
  ///
  /// # Errors
  ///
  /// Returns [`TryPutError`] where [`Self::copy_from_slice`] panics: when the
  /// input does not fit inline. Its `available` is the inline capacity.
  #[cfg(not(bytora_atomic))]
  pub fn try_copy_from_slice(values: &[T]) -> Result<Self, TryPutError>
  where
    T: Copy,
  {
    Self::try_from_slice(values).ok_or_else(|| Self::copy_error(values.len()))
  }

  // The copy that both forms share. `None` where neither the inline payload
  // nor a vector can represent the input.
  #[cfg(bytora_atomic)]
  #[inline]
  fn copy_slice(values: &[T]) -> Option<Self>
  where
    T: Copy + Send + Sync,
  {
    if let Some(inline) = Self::try_from_slice(values) {
      return Some(inline);
    }
    if values.len() > isize::MAX as usize
      || (S::HANDLE_BYTES < 3 * mem::size_of::<usize>()
        && !Layout::<T, S>::fits_two_word_vec(values.len()))
    {
      return None;
    }
    Self::try_from_vec(values.to_vec()).ok()
  }

  // The error for a copy of `requested` elements that does not fit.
  #[cold]
  fn copy_error(requested: usize) -> TryPutError {
    TryPutError {
      requested,
      available: Self::copy_limit(),
    }
  }

  // The longest input that `copy_from_slice` accepts for this handle.
  fn copy_limit() -> usize {
    #[cfg(bytora_atomic)]
    {
      if S::HANDLE_BYTES >= 3 * mem::size_of::<usize>() {
        return isize::MAX as usize;
      }
      if Layout::<T, S>::fits_two_word_vec(0) {
        return Layout::<T, S>::TWO_WORD_MAX_LEN;
      }
    }
    Self::INLINE_CAPACITY
  }

  /// Copies elements into a new vector, as [`Self::copy_from_slice`] does.
  /// [`Self::try_from_slice`] copies inline only.
  ///
  /// # Panics
  ///
  /// Panics where [`Self::copy_from_slice`] panics.
  /// [`Self::try_copy_from_slice`] returns an error instead.
  #[cfg(bytora_atomic)]
  #[inline]
  #[track_caller]
  pub fn from_slice(values: &[T]) -> Self
  where
    T: Copy + Send + Sync,
  {
    Self::copy_from_slice(values)
  }

  /// Copies elements into a new inline vector, as [`Self::copy_from_slice`]
  /// does.
  ///
  /// # Panics
  ///
  /// Panics where [`Self::copy_from_slice`] panics.
  /// [`Self::try_copy_from_slice`] returns an error instead.
  #[cfg(not(bytora_atomic))]
  #[inline]
  #[track_caller]
  pub fn from_slice(values: &[T]) -> Self
  where
    T: Copy,
  {
    Self::copy_from_slice(values)
  }
}

impl<T, S: InlineSize> Default for Vector<T, S> {
  fn default() -> Self {
    Self::new()
  }
}

#[cold]
#[inline(never)]
#[track_caller]
fn subset_outside_vector() -> ! {
  panic!("subset is outside this vector")
}

#[cold]
#[inline(never)]
#[track_caller]
fn split_index_out_of_bounds() -> ! {
  panic!("split index out of bounds")
}

#[cold]
#[inline(never)]
#[track_caller]
fn slice_does_not_fit() -> ! {
  panic!("slice does not fit this handle")
}
