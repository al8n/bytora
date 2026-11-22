use core::{
  mem::{self, ManuallyDrop, MaybeUninit},
  ops::Range,
  ptr::{self, NonNull},
  slice,
};
use std::vec::Vec;

#[cfg(smol_slice_atomic)]
use super::super::shared::Arc;

// Only the allocation, never its elements, belongs to this object. In particular,
// sibling windows may independently shorten lifetimes and write new T values.
// Each window drops its own initialized elements using its *current* T.
struct Allocation<T> {
  base: NonNull<T>,
  capacity: usize,
}

impl<T> Allocation<T> {
  fn empty() -> Self {
    Self {
      base: NonNull::dangling(),
      capacity: if size_of::<T>() == 0 { usize::MAX } else { 0 },
    }
  }

  fn take(values: Vec<T>) -> Self {
    let mut values = ManuallyDrop::new(values);
    Self {
      // SAFETY: Vec always provides a non-null, aligned pointer.
      base: unsafe { NonNull::new_unchecked(values.as_mut_ptr()) },
      capacity: values.capacity(),
    }
  }

  unsafe fn into_vec(self, len: usize) -> Vec<T> {
    let allocation = ManuallyDrop::new(self);
    // SAFETY: The caller owns this entire allocation and the first len elements.
    unsafe { Vec::from_raw_parts(allocation.base.as_ptr(), len, allocation.capacity) }
  }

  fn grow(&mut self, len: usize, additional: usize) {
    // SAFETY: The window has exclusive use of the whole allocation. MaybeUninit
    // has T's layout and permits moving all the slots, including their padding.
    // On capacity overflow reserve panics with the original Vec still intact;
    // ManuallyDrop leaves ownership with self on that path too.
    let mut buffer = ManuallyDrop::new(unsafe {
      Vec::from_raw_parts(
        self.base.as_ptr().cast::<MaybeUninit<T>>(),
        len,
        self.capacity,
      )
    });
    buffer.reserve(additional);
    self.base = NonNull::new(buffer.as_mut_ptr().cast()).unwrap();
    self.capacity = buffer.capacity();
  }
}

impl<T> Drop for Allocation<T> {
  fn drop(&mut self) {
    // SAFETY: Allocation owns the original Vec allocation. Element destruction
    // belongs to its windows; releasing a zero-length Vec only frees memory.
    unsafe { drop(Vec::from_raw_parts(self.base.as_ptr(), 0, self.capacity)) };
  }
}

// SAFETY: These fields describe an allocation and its destruction only frees
// memory. Sharing this metadata never grants access to any element. Window's
// disjoint ranges and its separate Send/Sync bounds govern all element access.
unsafe impl<T> Send for Allocation<T> {}
// SAFETY: See the Send implementation above.
unsafe impl<T> Sync for Allocation<T> {}

enum Buffer<T> {
  Exclusive(Allocation<T>),
  #[cfg(smol_slice_atomic)]
  Shared(Arc<Allocation<T>>),
}

impl<T> Buffer<T> {
  fn allocation(&self) -> &Allocation<T> {
    match self {
      Self::Exclusive(allocation) => allocation,
      #[cfg(smol_slice_atomic)]
      Self::Shared(allocation) => allocation,
    }
  }

  fn exclusive(&mut self) -> Option<&mut Allocation<T>> {
    match self {
      Self::Exclusive(allocation) => Some(allocation),
      #[cfg(smol_slice_atomic)]
      Self::Shared(allocation) => Arc::get_mut(allocation),
    }
  }

  #[cfg(smol_slice_atomic)]
  fn share(&mut self) -> Self {
    if let Self::Exclusive(allocation) = self {
      // Allocation and the uniqueness check must finish while the original
      // window still owns its live buffer. The swap and install cannot unwind.
      let mut shared = Arc::new(Allocation::empty());
      let slot = Arc::get_mut(&mut shared).expect("fresh shared allocation is unique");
      mem::swap(allocation, slot);
      *self = Self::Shared(shared);
    }
    match self {
      Self::Shared(allocation) => Self::Shared(Arc::clone(allocation)),
      Self::Exclusive(_) => unreachable!(),
    }
  }
}

/// Owns initialized elements in `owned_start..end` and exclusive access to
/// `owned_start..limit`. Advancing only changes `start`. Every live window of
/// the same allocation has a disjoint range, including its writable spare space.
pub(crate) struct Window<T> {
  buffer: Buffer<T>,
  owned_start: usize,
  start: usize,
  end: usize,
  limit: usize,
  // Allocation's thread-safety is independent of T. This window owns actual T.
  element: core::marker::PhantomData<T>,
}

impl<T> Window<T> {
  pub(crate) fn from_vec(values: Vec<T>) -> Self {
    let end = values.len();
    let limit = values.capacity();
    Self {
      buffer: Buffer::Exclusive(Allocation::take(values)),
      owned_start: 0,
      start: 0,
      end,
      limit,
      element: core::marker::PhantomData,
    }
  }

  fn ptr(&self, index: usize) -> *mut T {
    self.buffer.allocation().base.as_ptr().wrapping_add(index)
  }

  pub(super) fn data_ptr(&self) -> *mut T {
    self.ptr(self.start)
  }

  pub(super) fn len(&self) -> usize {
    self.end - self.start
  }

  pub(super) fn capacity(&self) -> usize {
    self.limit - self.start
  }

  pub(crate) fn as_slice(&self) -> &[T] {
    // SAFETY: The active range is initialized, and its owner is borrowed.
    unsafe { slice::from_raw_parts(self.ptr(self.start), self.len()) }
  }

  pub(super) unsafe fn set_len(&mut self, len: usize) {
    // The caller supplies initialized elements within the current capacity.
    self.end = self.start + len;
  }

  pub(super) fn advance(&mut self, count: usize) {
    assert!(count <= self.len(), "advance past end");
    self.start += count;
  }

  // SAFETY: The caller transfers ownership of this initialized, in-bounds
  // range and updates the window before any user destructor can run.
  unsafe fn drop_elements(&self, range: Range<usize>) {
    if mem::needs_drop::<T>() && !range.is_empty() {
      // SAFETY: See the caller's range invariant.
      unsafe {
        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
          self.ptr(range.start),
          range.len(),
        ))
      };
    }
  }

  pub(super) fn truncate(&mut self, len: usize) {
    if len < self.len() {
      let end = self.start + len;
      let previous = mem::replace(&mut self.end, end);
      // SAFETY: This tail was initialized and is no longer tracked.
      unsafe { self.drop_elements(end..previous) };
    }
  }

  #[cfg(smol_slice_atomic)]
  pub(crate) fn keep_range(mut self, range: Range<usize>) -> Self {
    assert!(range.start <= range.end && range.end <= self.len());
    self.truncate(range.end);
    self.advance(range.start);
    self
  }

  fn compact(&mut self, floor: usize, limit: usize) {
    let old = mem::replace(&mut self.owned_start, self.start);
    // SAFETY: The consumed prefix belongs only to us; marking it released
    // first keeps the surviving window valid if an element destructor unwinds.
    unsafe { self.drop_elements(old..self.start) };
    let len = self.len();
    if floor != self.start {
      // SAFETY: Both regions belong to us and may overlap. Moving the active
      // elements into their new slots transfers their ownership exactly once.
      unsafe { ptr::copy(self.ptr(self.start), self.ptr(floor), len) };
    }
    self.owned_start = floor;
    self.start = floor;
    self.end = floor + len;
    self.limit = limit;
  }

  pub(super) fn try_reclaim(&mut self, additional: usize) -> bool {
    self.reclaim(additional, false)
  }

  fn reclaim(&mut self, additional: usize, amortized: bool) -> bool {
    if additional <= self.limit - self.end {
      return true;
    }
    let (floor, limit) = match self.buffer.exclusive() {
      Some(allocation) => (0, allocation.capacity),
      None => (self.owned_start, self.limit),
    };
    // A former sibling may have released the suffix. Reopen it in place.
    if additional <= limit - self.end {
      self.limit = limit;
      return true;
    }
    let len = self.len();
    if additional > limit - floor - len || (amortized && self.start - floor < len) {
      return false;
    }
    self.compact(floor, limit);
    true
  }

  pub(super) fn reserve(&mut self, additional: usize) {
    // Sliding a full buffer forward one element at a time must not move the
    // whole active range at each append. Rebase only once the freed prefix
    // pays for the move. Explicit try_reclaim can always take that tradeoff.
    if self.reclaim(additional, true) {
      return;
    }
    let required = self
      .len()
      .checked_add(additional)
      .expect("capacity overflow");
    let max = if size_of::<T>() == 0 {
      usize::MAX
    } else {
      isize::MAX as usize / size_of::<T>()
    };
    assert!(required <= max, "capacity overflow");
    if let Some(allocation) = self.buffer.exclusive() {
      // Preserve the cursor so Vec's geometric growth leaves room after it.
      // If the offset itself would exceed the allocation limit (notably for
      // a near-maximum ZST length), compact once to make the request fit.
      if additional > max - self.end {
        let capacity = allocation.capacity;
        self.compact(0, capacity);
      }
      let allocation = self.buffer.exclusive().unwrap();
      allocation.grow(self.end, additional);
      self.limit = allocation.capacity;
    } else {
      let capacity = required.max(self.capacity().saturating_mul(2).min(max));
      let mut values = Vec::with_capacity(capacity);
      self.move_visible(&mut values);
      *self = Self::from_vec(values);
    }
  }

  fn move_visible(&mut self, dst: &mut Vec<T>) {
    let len = self.len();
    assert!(dst.is_empty() && dst.capacity() >= len);
    // SAFETY: The fresh allocation cannot overlap our data. Transfer the
    // active elements and remove them from our drop range before returning.
    unsafe {
      ptr::copy_nonoverlapping(self.ptr(self.start), dst.as_mut_ptr(), len);
      dst.set_len(len);
    }
    self.end = self.start;
  }

  pub(crate) fn into_vec(mut self) -> Vec<T> {
    if self.buffer.exclusive().is_some() {
      let capacity = self.buffer.allocation().capacity;
      self.compact(0, capacity);
      let len = mem::replace(&mut self.end, 0);
      let allocation = mem::replace(self.buffer.exclusive().unwrap(), Allocation::empty());
      // SAFETY: compact put the visible elements at the start of this exclusive
      // allocation. Removing them from our drop range transfers them to Vec.
      unsafe { allocation.into_vec(len) }
    } else {
      let mut values = Vec::with_capacity(self.len());
      self.move_visible(&mut values);
      values
    }
  }

  // A complete initialized prefix can become a Vec without moving anything.
  // A former sibling may leave uninitialized slots before owned_start.
  #[cfg(smol_slice_atomic)]
  pub(in crate::layout) fn take_vec_prefix(
    &mut self,
    fits: impl FnOnce(usize) -> bool,
  ) -> Option<(Vec<T>, usize)> {
    if self.owned_start != 0 || !fits(self.end) {
      return None;
    }
    let allocation = mem::replace(self.buffer.exclusive()?, Allocation::empty());
    let start = mem::take(&mut self.start);
    let end = mem::take(&mut self.end);
    self.limit = 0;
    // SAFETY: No other window can access the allocation. This window owned
    // every initialized element in 0..end and now has an empty drop range.
    Some((unsafe { allocation.into_vec(end) }, start))
  }

  pub(super) fn split_off(&mut self, at: usize) -> Self {
    assert!(at <= self.capacity(), "split_off past capacity");
    let boundary = self.start + at;

    #[cfg(smol_slice_atomic)]
    let right = Self {
      buffer: self.buffer.share(),
      owned_start: boundary,
      start: boundary,
      end: self.end.max(boundary),
      limit: self.limit,
      element: core::marker::PhantomData,
    };
    #[cfg(not(smol_slice_atomic))]
    let right = {
      // Without a thread-safe shared allocation owner, move just the suffix.
      let len = self.end.saturating_sub(boundary);
      let capacity = self.limit - boundary;
      let mut values = Vec::with_capacity(capacity);
      // SAFETY: The fresh buffer cannot overlap. The two assignments below
      // remove ownership of these same elements from the old window.
      unsafe {
        ptr::copy_nonoverlapping(self.ptr(boundary), values.as_mut_ptr(), len);
        values.set_len(len);
      }
      let mut window = Self::from_vec(values);
      window.limit = capacity;
      window
    };
    self.end = self.end.min(boundary);
    self.limit = boundary;
    right
  }

  #[cfg(smol_slice_atomic)]
  pub(super) fn try_unsplit(&mut self, other: &mut Self) -> bool {
    let (Buffer::Shared(left), Buffer::Shared(right)) = (&self.buffer, &other.buffer) else {
      return false;
    };
    // Compare slot indices even for ZSTs, whose pointers are all equal.
    if !Arc::ptr_eq(left, right)
      || self.end != other.start
      || self.limit != other.owned_start
      || other.start != other.owned_start
    {
      return false;
    }
    self.end = other.end;
    self.limit = other.limit;
    other.owned_start = other.end;
    other.start = other.end;
    other.limit = other.end;
    true
  }
}

impl<T> Drop for Window<T> {
  fn drop(&mut self) {
    // SAFETY: This window owns exactly this initialized range, including
    // consumed elements. Its buffer field stays alive through element drops.
    unsafe { self.drop_elements(self.owned_start..self.end) };
  }
}
