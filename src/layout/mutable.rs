use super::{InlineSize, Layout};
use crate::{Slice, SliceMut};
use core::marker::PhantomData;
#[cfg(any(feature = "std", feature = "alloc"))]
use core::num::NonZeroUsize;
#[cfg(any(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec::Vec};

// Keeping the concrete storage as a separate type parameter preserves
// covariance in T for each supported size, including owned vectors.
#[repr(transparent)]
pub struct Mutable<T, S, B> {
  storage: B,
  marker: PhantomData<(T, S)>,
}

pub trait Storage<T>: Sized {
  fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy;
  fn as_slice(&self) -> &[T];
  fn as_mut_slice(&mut self) -> &mut [T];
  fn len(&self) -> usize;
  #[track_caller]
  fn advance(&mut self, count: usize);
}

impl<T, S: InlineSize> Storage<T> for Layout<T, S> {
  fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Layout::try_inline(values)
  }

  fn as_slice(&self) -> &[T] {
    Layout::as_slice(self)
  }

  fn as_mut_slice(&mut self) -> &mut [T] {
    Layout::as_mut_slice(self)
  }

  fn len(&self) -> usize {
    Layout::len(self)
  }

  #[track_caller]
  fn advance(&mut self, count: usize) {
    Layout::advance(self, count);
  }
}

impl<T, S: InlineSize> SliceMut<T, S> {
  /// The maximum number of inline elements, as in [`Slice::INLINE_CAPACITY`].
  pub const INLINE_CAPACITY: usize = Slice::<T, S>::INLINE_CAPACITY;

  /// Copies `values` inline. Returns `None` if their length exceeds [`Self::INLINE_CAPACITY`].
  pub fn try_from_slice(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    Some(Self {
      storage: S::Mutable::try_inline(values)?,
      marker: PhantomData,
    })
  }

  /// Borrows the remaining elements.
  pub fn as_slice(&self) -> &[T] {
    self.storage.as_slice()
  }

  /// Mutably borrows the remaining elements.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.storage.as_mut_slice()
  }

  /// Returns the number of remaining elements.
  pub fn len(&self) -> usize {
    self.storage.len()
  }

  /// Returns whether the remaining slice is empty.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Skips the first `count` remaining elements in constant time.
  /// A vector's skipped elements stay owned until the handle drops.
  ///
  /// # Panics
  ///
  /// Panics if `count` exceeds [`Self::len`].
  #[track_caller]
  pub fn advance(&mut self, count: usize) {
    self.storage.advance(count);
  }
}

#[cfg(not(any(feature = "std", feature = "alloc")))]
pub type ZeroStorage<T> = Layout<T, typenum::U0>;

// Only nonempty inputs get an owner. Their original length gives Option a
// niche, and the remaining length lets advance leave the vector untouched.
#[cfg(any(feature = "std", feature = "alloc"))]
pub struct ZeroStorage<T> {
  #[allow(
    clippy::box_collection,
    reason = "boxing the vector header preserves its buffer and leaves room for the cursor"
  )]
  owner: Option<Box<Vec<T>>>,
  original_len: NonZeroUsize,
  len: usize,
}

impl<T> SliceMut<T, typenum::U0> {
  /// Creates an empty three-word mutable handle for any element type, using only `core`.
  pub const fn empty() -> Self {
    #[cfg(any(feature = "std", feature = "alloc"))]
    let storage = ZeroStorage {
      owner: None,
      original_len: NonZeroUsize::MIN,
      len: 0,
    };
    #[cfg(not(any(feature = "std", feature = "alloc")))]
    let storage = Layout {
      _align: [],
      payload: core::mem::MaybeUninit::uninit(),
      cursor: [0],
      tag: core::num::NonZeroU8::MIN,
      _element: PhantomData,
    };
    Self {
      storage,
      marker: PhantomData,
    }
  }
}

/// Takes exclusive ownership of a vector of any element type. A nonempty
/// input keeps its buffer and allocates one owner. An empty input yields
/// [`SliceMut::empty`] without allocating an owner. Available with `std` or `alloc`.
///
/// The handle can be sent across threads when `T: Send` and shared when `T: Sync`.
///
/// ```compile_fail
/// use smol_slice::SliceMutN;
/// use std::rc::Rc;
/// fn send<T: Send>(_: T) {}
/// send(SliceMutN::<_, 0>::from(vec![Rc::new(1)]));
/// ```
///
/// ```compile_fail
/// use core::cell::Cell;
/// use smol_slice::SliceMutN;
/// fn sync<T: Sync>() {}
/// sync::<SliceMutN<Cell<u8>, 0>>();
/// ```
///
/// Borrowed elements must stay valid until their destructors run.
///
/// ```compile_fail
/// use core::cell::Cell;
/// use smol_slice::SliceMutN;
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
    let Some(original_len) = NonZeroUsize::new(values.len()) else {
      return Self::empty();
    };
    Self {
      storage: ZeroStorage {
        owner: Some(Box::new(values)),
        original_len,
        len: original_len.get(),
      },
      marker: PhantomData,
    }
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<T> Storage<T> for ZeroStorage<T> {
  fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    values
      .is_empty()
      .then(|| SliceMut::<T, typenum::U0>::empty().storage)
  }

  fn as_slice(&self) -> &[T] {
    match &self.owner {
      Some(values) => &values[self.original_len.get() - self.len..],
      None => &[],
    }
  }

  fn as_mut_slice(&mut self) -> &mut [T] {
    match &mut self.owner {
      Some(values) => &mut values[self.original_len.get() - self.len..],
      None => &mut [],
    }
  }

  fn len(&self) -> usize {
    self.len
  }

  #[track_caller]
  fn advance(&mut self, count: usize) {
    assert!(count <= self.len, "advance past end");
    self.len -= count;
  }
}
