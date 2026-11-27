use core::{
  mem::{self, ManuallyDrop, MaybeUninit},
  ops::Range,
  ptr::{self, NonNull},
  slice,
};
use std::vec::Vec;

use crate::error::TryReserveError;

#[cfg(bytora_atomic)]
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

  // The capacity grow asks for: Vec's amortized growth, twice the capacity
  // and at least the required length and Vec's smallest non-zero capacity,
  // but capped at the length limit, as grown_capacity is. The caller keeps
  // `len + additional` within the limit.
  fn grown_capacity(&self, len: usize, additional: usize) -> usize {
    grown_capacity::<T>(self.capacity, len + additional)
      .max(min_non_zero_capacity::<T>().min(max_len::<T>()))
  }

  // Makes room for `additional` slots after the first `len`, which it keeps.
  // The caller keeps `len + additional` within the length limit.
  fn grow(&mut self, len: usize, additional: usize) {
    if additional <= self.capacity - len {
      return;
    }
    let capacity = self.grown_capacity(len, additional);
    self.grow_to(len, capacity);
  }

  // grow's twin: on failure the allocation and its contents are untouched.
  fn try_grow(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
    if additional <= self.capacity - len {
      return Ok(());
    }
    let capacity = self.grown_capacity(len, additional);
    self.try_grow_to(len, capacity)
  }

  // Grows the allocation to exactly `capacity` slots, keeping the first `len`
  // in place. The caller keeps `len <= capacity <= max_len`.
  fn grow_to(&mut self, len: usize, capacity: usize) {
    // SAFETY: The window has exclusive use of the whole allocation. MaybeUninit
    // has T's layout and permits moving all the slots, including their padding.
    // On capacity overflow reserve_exact panics with the original Vec still
    // intact; ManuallyDrop leaves ownership with self on that path too.
    let mut buffer = ManuallyDrop::new(unsafe {
      Vec::from_raw_parts(
        self.base.as_ptr().cast::<MaybeUninit<T>>(),
        len,
        self.capacity,
      )
    });
    buffer.reserve_exact(capacity - len);
    self.base = NonNull::new(buffer.as_mut_ptr().cast()).unwrap();
    self.capacity = buffer.capacity();
  }

  // grow_to's twin: on failure the allocation and its contents are untouched.
  fn try_grow_to(&mut self, len: usize, capacity: usize) -> Result<(), TryReserveError> {
    // SAFETY: As in grow_to. On failure try_reserve_exact leaves the Vec intact,
    // and ManuallyDrop leaves ownership with self.
    let mut buffer = ManuallyDrop::new(unsafe {
      Vec::from_raw_parts(
        self.base.as_ptr().cast::<MaybeUninit<T>>(),
        len,
        self.capacity,
      )
    });
    buffer
      .try_reserve_exact(capacity - len)
      .map_err(|_| exact_error::<T>(capacity))?;
    self.base = NonNull::new(buffer.as_mut_ptr().cast()).unwrap();
    self.capacity = buffer.capacity();
    Ok(())
  }
}

// The most elements an allocation can hold: isize::MAX bytes, or any number of
// zero-sized elements.
pub(super) const fn max_len<T>() -> usize {
  if size_of::<T>() == 0 {
    usize::MAX
  } else {
    isize::MAX as usize / size_of::<T>()
  }
}

// What an exact request's refusal stands for. Past isize::MAX bytes it is an
// overflow; otherwise the allocator failed. alloc's TryReserveError::kind is
// unstable, so the request is classified the way RawVec classifies it.
pub(super) fn exact_error<T>(required: usize) -> TryReserveError {
  if required > max_len::<T>() {
    TryReserveError::CapacityOverflow
  } else {
    TryReserveError::AllocError
  }
}

// The capacity an amortized growth asks for: twice the old capacity, and at
// least the required length, but never past the length limit. Vec doubles
// past it and then refuses, which made reserve panic, and its twin fail, on a
// length that fits. `required` must not exceed the limit.
fn grown_capacity<T>(capacity: usize, required: usize) -> usize {
  required.max(capacity.saturating_mul(2).min(max_len::<T>()))
}

// Vec's smallest non-zero capacity, the floor of its amortized growth.
const fn min_non_zero_capacity<T>() -> usize {
  if size_of::<T>() == 1 {
    8
  } else if size_of::<T>() <= 1024 {
    4
  } else {
    1
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
  #[cfg(bytora_atomic)]
  Shared(Arc<Allocation<T>>),
}

impl<T> Buffer<T> {
  fn allocation(&self) -> &Allocation<T> {
    match self {
      Self::Exclusive(allocation) => allocation,
      #[cfg(bytora_atomic)]
      Self::Shared(allocation) => allocation,
    }
  }

  fn exclusive(&mut self) -> Option<&mut Allocation<T>> {
    match self {
      Self::Exclusive(allocation) => Some(allocation),
      #[cfg(bytora_atomic)]
      Self::Shared(allocation) => Arc::get_mut(allocation),
    }
  }

  #[cfg(bytora_atomic)]
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

  #[cfg(bytora_atomic)]
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

  // Makes room in place where it can. An amortized reserve compacts only once
  // the consumed prefix is at least as long as the elements, which pays for
  // the move: sliding a full buffer forward one element at a time must not
  // move the whole active range at each append. An explicit try_reclaim can
  // always take that tradeoff.
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

  #[track_caller]
  pub(super) fn reserve(&mut self, additional: usize) {
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
    let (len, end) = (self.len(), self.end);
    if let Some(allocation) = self.buffer.exclusive() {
      if additional <= max - end {
        // Preserve the cursor so the geometric growth leaves room after it.
        allocation.grow(end, additional);
        self.limit = allocation.capacity;
        return;
      }
      // The offset would put a growth in place past the length limit
      // (notably for a near-maximum ZST length). Grow, where the allocation
      // cannot already hold the request, with the slots kept in place, and
      // then compact. try_reserve takes the same steps, so that its refused
      // growth leaves the window as it was.
      if additional > allocation.capacity - len {
        let capacity = allocation.grown_capacity(len, additional);
        allocation.grow_to(end, capacity);
      }
      let capacity = allocation.capacity;
      self.compact(0, capacity);
    } else {
      let capacity = grown_capacity::<T>(self.capacity(), required);
      let mut values = Vec::with_capacity(capacity);
      self.move_visible(&mut values);
      *self = Self::from_vec(values);
    }
  }

  // reserve's twin: an error where reserve panics or the allocation fails, and
  // on an error the window is as it was.
  pub(super) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
    if self.reclaim(additional, true) {
      return Ok(());
    }
    let max = max_len::<T>();
    let (len, end, visible) = (self.len(), self.end, self.capacity());
    let required = match len.checked_add(additional) {
      Some(required) if required <= max => required,
      _ => return Err(TryReserveError::CapacityOverflow),
    };
    if let Some(allocation) = self.buffer.exclusive() {
      if additional <= max - end {
        allocation.try_grow(end, additional)?;
        self.limit = allocation.capacity;
        return Ok(());
      }
      // As reserve: grow where the allocation cannot already hold the
      // request, then compact. A refused growth leaves the window as it was.
      if additional > allocation.capacity - len {
        let capacity = allocation.grown_capacity(len, additional);
        allocation.try_grow_to(end, capacity)?;
      }
      let capacity = allocation.capacity;
      self.compact(0, capacity);
      return Ok(());
    }
    // A shared allocation: the elements move to a new allocation of the
    // capacity reserve makes. It is made first, so a refusal leaves this
    // window as it was.
    let capacity = grown_capacity::<T>(visible, required);
    let mut values = Vec::new();
    values
      .try_reserve_exact(capacity)
      .map_err(|_| exact_error::<T>(capacity))?;
    self.move_visible(&mut values);
    *self = Self::from_vec(values);
    Ok(())
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
  #[cfg(bytora_atomic)]
  pub(super) fn take_vec_prefix(
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

  #[track_caller]
  pub(super) fn split_off(&mut self, at: usize) -> Self {
    assert!(at <= self.capacity(), "split_off past capacity");
    let boundary = self.start + at;

    #[cfg(bytora_atomic)]
    let right = Self {
      buffer: self.buffer.share(),
      owned_start: boundary,
      start: boundary,
      end: self.end.max(boundary),
      limit: self.limit,
      element: core::marker::PhantomData,
    };
    #[cfg(not(bytora_atomic))]
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

  #[cfg(bytora_atomic)]
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

#[cfg(test)]
mod tests;
