use core::ops::RangeBounds;
use std::vec::Vec;

#[cfg(feature = "extra-platforms")]
use portable_atomic_util::Arc;
#[cfg(not(feature = "extra-platforms"))]
use std::sync::Arc;

use crate::checked_range;

#[cfg(test)]
mod tests;

enum Descriptor<T> {
  Root(Vec<T>),
  Window {
    root: Arc<Descriptor<T>>,
    start: usize,
    end: usize,
  },
}

/// An experimental pointer-sized shared view of a vector.
///
/// Clones share a descriptor. Slicing a proper subrange allocates a descriptor
/// that retains the original vector; slicing the entire view only shares the
/// existing descriptor. Hidden elements remain alive until the vector is
/// recovered or the last view is dropped.
#[repr(transparent)]
pub struct ThinSlice<T>(Arc<Descriptor<T>>);

impl<T> ThinSlice<T> {
  /// Takes ownership of a vector without moving its elements.
  pub fn from_vec(values: Vec<T>) -> Self {
    Self(Arc::new(Descriptor::Root(values)))
  }

  /// Returns the number of elements in this view.
  pub fn len(&self) -> usize {
    match self.0.as_ref() {
      Descriptor::Root(values) => values.len(),
      Descriptor::Window { start, end, .. } => end - start,
    }
  }

  /// Returns whether this view contains no elements.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Borrows the elements in this view.
  pub fn as_slice(&self) -> &[T] {
    match self.0.as_ref() {
      Descriptor::Root(values) => values,
      Descriptor::Window { root, start, end } => match root.as_ref() {
        Descriptor::Root(values) => &values[*start..*end],
        Descriptor::Window { .. } => unreachable!("windows always point to the root"),
      },
    }
  }

  /// Reports whether this view can recover the vector without cloning elements.
  /// Concurrent clones can change the result immediately.
  pub fn is_unique(&self) -> bool {
    Arc::strong_count(&self.0) == 1
      && match self.0.as_ref() {
        Descriptor::Root(_) => true,
        Descriptor::Window { root, .. } => Arc::strong_count(root) == 1,
      }
  }

  /// Returns a shared view of a range relative to this view.
  ///
  /// # Panics
  ///
  /// Panics if the range starts after its end, exceeds [`Self::len`], or
  /// cannot be represented as a half-open range.
  #[track_caller]
  pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
    let len = self.len();
    let range = checked_range(range, len);
    if range.start == 0 && range.end == len {
      return self.clone();
    }
    let (root, offset) = match self.0.as_ref() {
      Descriptor::Root(_) => (&self.0, 0),
      Descriptor::Window { root, start, .. } => (root, *start),
    };
    Self(Arc::new(Descriptor::Window {
      root: Arc::clone(root),
      start: offset + range.start,
      end: offset + range.end,
    }))
  }

  /// Moves this view into its original vector allocation without cloning.
  ///
  /// The vector retains its original capacity. Elements outside this view are
  /// dropped, and visible elements may move to the start of the allocation.
  /// Returns this view unchanged if another view shares the allocation.
  pub fn try_into_vec(mut self) -> Result<Vec<T>, Self> {
    if matches!(self.0.as_ref(), Descriptor::Root(_)) {
      return match Arc::try_unwrap(self.0) {
        Ok(Descriptor::Root(values)) => Ok(values),
        Ok(Descriptor::Window { .. }) => unreachable!("the descriptor is a root"),
        Err(root) => Err(Self(root)),
      };
    }
    let taken = match Arc::get_mut(&mut self.0) {
      Some(Descriptor::Root(_)) => unreachable!("the descriptor is a window"),
      Some(Descriptor::Window { root, start, end }) => match Arc::get_mut(root) {
        Some(Descriptor::Root(values)) => Some((core::mem::take(values), *start, *end)),
        Some(Descriptor::Window { .. }) => unreachable!("windows always point to the root"),
        None => None,
      },
      None => None,
    };
    let Some((mut values, start, end)) = taken else {
      return Err(self);
    };
    if start != 0 || end != values.len() {
      values.truncate(end);
      drop(values.drain(..start));
    }
    Ok(values)
  }
}

impl<T> Clone for ThinSlice<T> {
  fn clone(&self) -> Self {
    Self(Arc::clone(&self.0))
  }
}
