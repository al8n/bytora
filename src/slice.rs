use crate::{InlineSize, Layout, Slice};
use core::{mem, ptr};

impl<T, S: InlineSize> Slice<T, S> {
  /// Creates an empty slice without allocating, for any element type or size.
  pub const fn new() -> Self {
    Self {
      layout: Layout::empty_inline(),
    }
  }

  /// Returns the part of this slice corresponding to `subset`.
  ///
  /// An empty subset is accepted regardless of its pointer and returns a new
  /// empty slice. For nonempty zero-sized elements, the pointer must match
  /// this slice's pointer and the length must fit. Such pointers do not encode
  /// an element offset or reliably distinguish allocations, so the returned
  /// window uses the corresponding prefix.
  ///
  /// # Panics
  ///
  /// Panics if a nonempty subset is outside this slice's current window.
  #[track_caller]
  pub fn slice_ref(&self, subset: &[T]) -> Self {
    if subset.is_empty() {
      return Self::new();
    }

    let source = self.as_slice().as_ptr();
    if mem::size_of::<T>() == 0 {
      assert!(
        ptr::eq(source, subset.as_ptr()) && subset.len() <= self.len(),
        "subset is outside this slice"
      );
      return self.slice(..subset.len());
    }

    let byte_offset = subset
      .as_ptr()
      .addr()
      .checked_sub(source.addr())
      .expect("subset is outside this slice");
    let element_size = mem::size_of::<T>();
    assert!(
      byte_offset % element_size == 0,
      "subset is outside this slice"
    );
    let start = byte_offset / element_size;
    assert!(
      start <= self.len() && subset.len() <= self.len() - start,
      "subset is outside this slice"
    );
    self.slice(start..start + subset.len())
  }

  /// Splits at `at`, returning the suffix and retaining the prefix in `self`.
  ///
  /// Static and shared nonempty windows keep their original storage. Inline
  /// windows are copied. Pointers of empty windows are unspecified.
  ///
  /// # Panics
  ///
  /// Panics if `at` exceeds [`Self::len`].
  #[must_use = "consider truncate if the suffix is not needed"]
  #[track_caller]
  pub fn split_off(&mut self, at: usize) -> Self {
    let len = self.len();
    assert!(at <= len, "split index out of bounds");
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
  /// Panics if `at` exceeds [`Self::len`].
  #[must_use = "consider advance if the prefix is not needed"]
  #[track_caller]
  pub fn split_to(&mut self, at: usize) -> Self {
    let len = self.len();
    assert!(at <= len, "split index out of bounds");
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

  /// Shortens the slice to `len` elements, keeping its prefix.
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

  /// Copies elements into a new slice.
  ///
  /// Elements are stored inline if they fit. When shared ownership is
  /// available through native atomics or `extra-platforms`, larger inputs use
  /// shared storage when the handle is large enough to hold it.
  ///
  /// # Panics
  ///
  /// Panics when the input does not fit inline and shared storage is
  /// unavailable, the handle is too small, or the length exceeds
  /// `isize::MAX`. Use [`Self::try_copy_from_slice`] to check these limits.
  #[cfg(paguro_atomic)]
  #[track_caller]
  pub fn copy_from_slice(values: &[T]) -> Self
  where
    T: Copy + Send + Sync,
  {
    Self::try_copy_from_slice(values).expect("slice does not fit this handle")
  }

  /// Copies elements into a new inline slice.
  ///
  /// # Panics
  ///
  /// Panics when the input does not fit inline. Use
  /// [`Self::try_copy_from_slice`] to check this limit.
  #[cfg(not(paguro_atomic))]
  #[track_caller]
  pub fn copy_from_slice(values: &[T]) -> Self
  where
    T: Copy,
  {
    Self::try_copy_from_slice(values).expect("slice does not fit this handle")
  }

  /// Copies elements inline when possible, or into shared storage when needed.
  ///
  /// Shared copies use the vector limits of [`Self::try_from_vec`]. Returns
  /// `None` if neither the inline payload nor a vector can represent the input.
  #[cfg(paguro_atomic)]
  pub fn try_copy_from_slice(values: &[T]) -> Option<Self>
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

  /// Copies elements inline, or returns `None` if they do not fit.
  #[cfg(not(paguro_atomic))]
  pub fn try_copy_from_slice(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Self::try_from_slice(values)
  }
}

impl<T, S: InlineSize> Default for Slice<T, S> {
  fn default() -> Self {
    Self::new()
  }
}
