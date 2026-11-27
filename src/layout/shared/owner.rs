use super::vector::{self, Vector};
use super::{Arc, STATIC_OWNER};
use crate::{InlineSize, Slice, layout::mutable::Window};
use core::{mem::ManuallyDrop, ops::Range};
use std::vec::Vec;

// Each callback receives exactly the tagged or untagged strong reference used to
// select this table. A take callback consumes the reference on success or panic.
// On failure it leaves that same strong reference in the caller's ownership.
// The window pointer retains its provenance but carries no input type: the
// stored table must be covariant in T, just like the Slice exposing its window.
pub(super) type Take<R> = unsafe fn(*const (), *const (), usize) -> Option<R>;
pub(super) type ZstBase = unsafe fn(*const ()) -> *const ();

pub(super) struct StorageVtable<T> {
  pub(super) clone: unsafe fn(*const ()),
  pub(super) into_vec: Take<Vec<T>>,
  pub(super) into_mut: Take<Window<T>>,
  pub(super) is_unique: unsafe fn(*const ()) -> bool,
  pub(super) drop: unsafe fn(*const ()),
  // A recoverable ZST window stores a logical slot offset in the data word.
  // This callback supplies its real, aligned data pointer for a public borrow.
  pub(super) zst_base: Option<ZstBase>,
}

impl<T> StorageVtable<T> {
  const STATIC: Self = Self {
    clone: |_| {},
    into_vec: |_, _, _| None,
    into_mut: |_, _, _| None,
    is_unique: |_| false,
    drop: |_| {},
    zst_base: None,
  };

  const VECTOR: Self = Self {
    clone: retain::<Vector<T>>,
    into_vec: vector_into_vec,
    into_mut: vector_into_window,
    is_unique: |owner| {
      // SAFETY: A direct vector supplies a strong reference to its header.
      vector::raw(owner) || unsafe { unique::<Vector<T>>(owner) }
    },
    drop: release::<Vector<T>>,
    zst_base: Some(|owner| {
      // SAFETY: ZST vectors always use a live direct Arc<Vector<T>>.
      unsafe { (*owner.cast::<Vector<T>>()).base.cast() }
    }),
  };
}

// Promoted Vec owners keep their direct Arc<Vector<T>> pointer. Other managed owners put
// a pointer to their typed operations first and tag their Arc pointer. Keeping
// the table in constant storage lets scoped T (such as &'a str) use it too.
#[repr(C)]
struct Managed<T, O> {
  ops: *const StorageVtable<T>,
  value: O,
}

impl<T, O> Managed<T, O> {
  const OPAQUE_OPS: *const StorageVtable<T> = &StorageVtable {
    clone: retain_tagged::<T, O>,
    drop: release_tagged::<T, O>,
    ..StorageVtable::<T>::STATIC
  };

  fn into_owner(this: Arc<Self>) -> *const () {
    // repr(C)'s pointer first field gives the allocation at least two-byte alignment.
    Arc::into_raw(this).wrapping_byte_add(1).cast()
  }
}

impl<T> Managed<T, Window<T>> {
  const WINDOW_OPS: *const StorageVtable<T> = &StorageVtable {
    clone: retain_tagged::<T, Window<T>>,
    into_vec: window_into_vec,
    into_mut: take_window,
    is_unique: |owner| {
      // SAFETY: This table belongs only to tagged Arc<Managed<T, Window<T>>>.
      unsafe { unique::<Self>(untag(owner)) }
    },
    drop: release_tagged::<T, Window<T>>,
    zst_base: Some(|owner| {
      // SAFETY: This table receives a live tagged Arc<Managed<T, Window<T>>>.
      unsafe {
        (&*untag(owner).cast::<Self>())
          .value
          .as_slice()
          .as_ptr()
          .cast()
      }
    }),
  };
}

fn untag(owner: *const ()) -> *const () {
  owner.wrapping_byte_sub(1)
}

// The caller borrows a live external descriptor for T. Its table is in
// constant storage, while the tagged owner's header lives as long as the handle.
pub(super) unsafe fn operations<'a, T>(owner: *const ()) -> &'a StorageVtable<T> {
  if owner.addr() & 1 == 0 || vector::raw(owner) {
    &StorageVtable::<T>::VECTOR
  } else if owner == STATIC_OWNER {
    &StorageVtable::<T>::STATIC
  } else {
    // SAFETY: This tagged pointer identifies a managed allocation.
    unsafe { managed_operations(owner) }
  }
}

pub(super) unsafe fn managed_operations<'a, T>(owner: *const ()) -> &'a StorageVtable<T> {
  // SAFETY: The caller excludes the static sentinel and direct Vec pointers.
  // Untagging recovers the live Managed<T, O> whose first field holds its table.
  unsafe { &**untag(owner).cast::<*const StorageVtable<T>>() }
}

// Retain a direct Vec reference without a dynamic call. Static references need
// no retain. A false result leaves a tagged managed reference for its table.
#[inline]
pub(super) unsafe fn retain_direct<T>(owner: *const ()) -> bool {
  if owner.addr() & 1 != 0 {
    return owner == STATIC_OWNER;
  }
  // SAFETY: The caller holds a live external descriptor. Its untagged owner
  // came from Arc<Vector<T>>::into_raw. Raw descriptors are promoted before retain.
  unsafe { (StorageVtable::<T>::VECTOR.clone)(owner) };
  true
}

#[inline]
pub(super) unsafe fn release_owner<T>(owner: *const ()) {
  if owner.addr() & 1 == 0 {
    // SAFETY: The untagged owner transfers a direct Arc<Vector<T>> reference.
    unsafe { (StorageVtable::<T>::VECTOR.drop)(owner) };
  } else if owner != STATIC_OWNER {
    // SAFETY: The tagged managed owner transfers a reference to its own table.
    unsafe { (managed_operations::<T>(owner).drop)(owner) };
  }
}

unsafe fn retain<A>(raw: *const ()) {
  // SAFETY: The selected table identifies A, and the source owns a live Arc.
  unsafe { Arc::<A>::increment_strong_count(raw.cast()) };
}

unsafe fn release<A>(raw: *const ()) {
  // SAFETY: The caller transfers one strong reference to this concrete Arc.
  unsafe { Arc::<A>::decrement_strong_count(raw.cast()) };
}

unsafe fn retain_tagged<T, O>(owner: *const ()) {
  // SAFETY: This callback was installed by Managed<T, O>::into_owner.
  unsafe { retain::<Managed<T, O>>(untag(owner)) };
}

unsafe fn release_tagged<T, O>(owner: *const ()) {
  // SAFETY: This callback was installed by Managed<T, O>::into_owner.
  unsafe { release::<Managed<T, O>>(untag(owner)) };
}

unsafe fn unique<A>(raw: *const ()) -> bool {
  // SAFETY: Borrow the caller's strong reference, without decrementing it.
  let arc = ManuallyDrop::new(unsafe { Arc::<A>::from_raw(raw.cast()) });
  Arc::strong_count(&arc) == 1
}

unsafe fn take_arc<A>(raw: *const ()) -> Option<A> {
  // SAFETY: The caller transfers one strong reference and suppresses its Drop.
  match Arc::try_unwrap(unsafe { Arc::<A>::from_raw(raw.cast()) }) {
    Ok(value) => Some(value),
    Err(arc) => {
      // The caller will resume ownership of exactly this reference.
      let _ = Arc::into_raw(arc);
      None
    }
  }
}

pub(super) fn initial_data<T>(data: *const T) -> *const T {
  if size_of::<T>() == 0 {
    core::ptr::null()
  } else {
    data
  }
}

fn active_range<T>(original: &[T], data: *const T, len: usize) -> Range<usize> {
  // A ZST descriptor for these two recoverable backings stores an element
  // index relative to the original window. It is never used as a reference.
  let start = if size_of::<T>() == 0 {
    data.addr()
  } else {
    (data.addr() - original.as_ptr().addr()) / size_of::<T>()
  };
  debug_assert!(start <= original.len() && len <= original.len() - start);
  start..start + len
}

unsafe fn vector_into_vec<T>(owner: *const (), data: *const (), len: usize) -> Option<Vec<T>> {
  // SAFETY: This callback consumes the source's raw vector or its strong reference.
  let mut original = unsafe { take_vector(owner, data, len) }?;
  let range = active_range(&original, data.cast::<T>(), len);
  original.truncate(range.end);
  drop(original.drain(..range.start));
  Some(original)
}

unsafe fn take_vector<T>(owner: *const (), data: *const (), len: usize) -> Option<Vec<T>> {
  let buffer = if vector::raw(owner) {
    // SAFETY: A consuming take transfers the raw descriptor's original vector.
    unsafe { Vector::<T>::take_raw(owner, data, len) }
  } else {
    // SAFETY: A direct descriptor transfers its strong reference, or restores
    // it when other immutable views still exist.
    unsafe { take_arc::<Vector<T>>(owner) }?
  };
  Some(buffer.into_vec())
}

unsafe fn vector_into_window<T>(
  owner: *const (),
  data: *const (),
  len: usize,
) -> Option<Window<T>> {
  // SAFETY: This callback consumes the source's raw vector or its strong reference.
  let original = unsafe { take_vector(owner, data, len) }?;
  let range = active_range(&original, data.cast::<T>(), len);
  Some(Window::from_vec(original).keep_range(range))
}

unsafe fn take_window<T>(owner: *const (), data: *const (), len: usize) -> Option<Window<T>> {
  // SAFETY: The window table receives an Arc<Managed<T, Window<T>>> reference.
  let window = unsafe { take_arc::<Managed<T, Window<T>>>(untag(owner)) }?.value;
  let range = active_range(window.as_slice(), data.cast::<T>(), len);
  Some(window.keep_range(range))
}

unsafe fn window_into_vec<T>(owner: *const (), data: *const (), len: usize) -> Option<Vec<T>> {
  // SAFETY: The window table supplies the same owner and range as take_window.
  unsafe { take_window(owner, data, len) }.map(Window::into_vec)
}

impl<T, S: InlineSize> Slice<T, S> {
  pub(crate) fn try_from_window(window: Window<T>) -> Result<Self, Window<T>>
  where
    T: Send + Sync,
  {
    let values = window.as_slice();
    let Some(mut layout) = super::Layout::try_external(
      initial_data(values.as_ptr()),
      core::ptr::null(),
      values.len(),
    ) else {
      return Err(window);
    };
    let managed = Arc::new(Managed {
      ops: Managed::<T, Window<T>>::WINDOW_OPS,
      value: window,
    });
    layout.external_prefix_mut().owner = Managed::into_owner(managed);
    Ok(Self { layout })
  }

  /// Keeps an owner alive until the last handle or subrange is dropped. The
  /// owner's slice is borrowed once at its final address, even for a short or
  /// empty slice. Returns the owner if the handle cannot hold an external
  /// descriptor or its length exceeds `isize::MAX`.
  ///
  /// A custom owner grants shared access. For `T: Clone`, [`Self::into_vec`]
  /// and [`Self::into_mut`] copy the current window.
  pub fn try_from_owner<O>(owner: O) -> Result<Self, O>
  where
    T: Sync,
    O: AsRef<[T]> + Send + 'static,
  {
    if !super::Layout::<T, S>::FITS_EXTERNAL {
      return Err(owner);
    }
    let managed = Arc::new(Managed {
      ops: Managed::<T, O>::OPAQUE_OPS,
      value: owner,
    });
    // Project only after the owner reaches its stable address. Once projected
    // its value is never accessed again until final drop. Even a Send, !Sync
    // owner may therefore back concurrent readers of Sync elements.
    let values = managed.value.as_ref();
    let Some(mut layout) =
      super::Layout::try_external(values.as_ptr(), core::ptr::null(), values.len())
    else {
      return Err(
        Arc::try_unwrap(managed)
          .ok()
          .expect("owner has not been shared")
          .value,
      );
    };
    layout.external_prefix_mut().owner = Managed::into_owner(managed);
    Ok(Self { layout })
  }
}

#[cfg(test)]
mod covariance {
  use super::*;

  type AnyArg = for<'a> fn(&'a str) -> &'a str;
  type StaticArg = fn(&'static str) -> &'static str;
  type Frozen<T> = Managed<T, Window<T>>;

  // A handle can undergo this subtype coercion with its original table and
  // allocation still inside. Both stored types must admit the same coercion.
  const _: fn(StorageVtable<AnyArg>) -> StorageVtable<StaticArg> = |ops| ops;
  const _: fn(Frozen<AnyArg>) -> Frozen<StaticArg> = |managed| managed;
  const _: fn(Vector<AnyArg>) -> Vector<StaticArg> = |buffer| buffer;
}
