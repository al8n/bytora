use super::{
  InlineSize, Layout,
  mutable::{Storage, Window},
};
use crate::{Slice, SliceMut};
use core::{mem::ManuallyDrop, ops::Range, ptr, slice};
use std::{boxed::Box, vec::Vec};

mod pool;
use pool::{INDEX_BITS, Key, Reservation};

const INDEX_MASK: u64 = (1 << INDEX_BITS) - 1;
const FIELD_BITS: u32 = 20;
const FIELD_MASK: u64 = (1 << FIELD_BITS) - 1;
const LEN_SHIFT: u32 = INDEX_BITS + FIELD_BITS;
const EXTERNAL: u64 = 1 << 63;
const MUTABLE: u64 = 0xff << 56;

const _: () = assert!(INDEX_BITS + 2 * FIELD_BITS == 63);

impl<T, S: InlineSize> Layout<T, S> {
  const fn table_word(&self) -> u64 {
    assert!(Self::TABLE_VEC && !Self::inline_tag(self.tag));
    // SAFETY: both table constructors initialize all eight bytes as an
    // integer. This read never touches inline T or a provenance-bearing word.
    u64::from_le(unsafe { ptr::from_ref(self).cast::<u64>().read() })
  }

  pub(super) const fn table_len(&self) -> usize {
    ((self.table_word() >> LEN_SHIFT) & FIELD_MASK) as usize
  }

  fn from_table_word(word: u64) -> Self {
    assert!(Self::TABLE_VEC && word & EXTERNAL != 0);
    let mut layout = Self::empty_inline();
    // SAFETY: this exact eight-byte layout is word-aligned. The final
    // address byte is a nonzero tag on either endian, and the cursor is empty.
    unsafe { ptr::from_mut(&mut layout).cast::<u64>().write(word.to_le()) };
    layout
  }

  fn table_key(&self) -> Key {
    // SAFETY: a live immutable or mutable table descriptor holds this slot.
    unsafe { Key::from_bits((self.table_word() & INDEX_MASK) as u32) }
  }

  fn table_start(&self) -> usize {
    ((self.table_word() >> INDEX_BITS) & FIELD_MASK) as usize
  }

  fn table_window(key: Key, start: usize, len: usize) -> Self {
    debug_assert!(Self::fits_table_vec(start + len));
    Self::from_table_word(
      EXTERNAL
        | ((len as u64) << LEN_SHIFT)
        | ((start as u64) << INDEX_BITS)
        | u64::from(key.bits()),
    )
  }

  pub(super) fn table_slice(&self) -> &[T] {
    // SAFETY: the typed handle pins its allocation; the represented range
    // stays within the original initialized elements, including empty and ZST
    // windows. Only the pointer read from that slot supplies provenance.
    unsafe {
      let base = self.table_key().base::<T>();
      slice::from_raw_parts(base.add(self.table_start()), self.table_len())
    }
  }

  pub(super) fn table_unique(&self) -> bool {
    self.table_key().unique()
  }

  pub(super) fn share_table(&self, range: Range<usize>) -> Self {
    let key = self.table_key();
    let layout = Self::table_window(key, self.table_start() + range.start, range.len());
    // SAFETY: the source remains borrowed until the peer's ownership is acquired.
    unsafe { key.retain() };
    layout
  }

  pub(super) fn keep_table(&mut self, range: Range<usize>) {
    *self = Self::table_window(
      self.table_key(),
      self.table_start() + range.start,
      range.len(),
    );
  }

  pub(super) unsafe fn drop_table(&mut self) {
    // SAFETY: the caller transfers this immutable descriptor's strong reference.
    unsafe { self.table_key().release::<T>() };
  }

  pub(super) fn table_box_ptr(&self) -> ptr::NonNull<Window<T>> {
    // SAFETY: the mutable HEAP tag selects its own unique Box<Window<T>>
    // slot. The bound of the returned borrow is enforced by Window's caller.
    unsafe { ptr::NonNull::new_unchecked(self.table_key().base()) }
  }

  fn keyed_box(&mut self) -> (Reservation, Box<Window<T>>) {
    let key = self.table_key();
    *self = Self::empty_inline();
    // SAFETY: the mutable descriptor was the unique owner and is now cleared.
    unsafe { key.take_box() }
  }

  pub(super) fn take_table_box(&mut self) -> Box<Window<T>> {
    self.keyed_box().1
  }
}

impl<T, S: InlineSize> Slice<T, S> {
  pub(super) fn from_table_vec<const INLINE_EMPTY: bool>(values: Vec<T>) -> Result<Self, Vec<T>>
  where
    T: Send + Sync,
  {
    let len = values.len();
    if !Layout::<T, S>::fits_table_vec(len) {
      return Err(values);
    }
    if INLINE_EMPTY && values.capacity() == 0 {
      return Ok(Self::new());
    }
    let Some(reservation) = Reservation::claim() else {
      return Err(values);
    };
    let key = reservation.install_vec(values);
    Ok(Self {
      layout: Layout::table_window(key, 0, len),
    })
  }

  fn take_table(self) -> Result<(Reservation, Vec<T>, Range<usize>), Self> {
    let this = ManuallyDrop::new(self);
    let start = this.layout.table_start();
    let range = start..start + this.layout.table_len();
    // SAFETY: the current typed descriptor owns this reference. Successful
    // extraction transfers the full Vec and vacant slot, and suppresses Drop.
    match unsafe { this.layout.table_key().try_take_vec::<T>() } {
      Some((reservation, values)) => Ok((reservation, values, range)),
      None => Err(ManuallyDrop::into_inner(this)),
    }
  }

  pub(super) fn take_table_vec(self) -> Result<Vec<T>, Self> {
    let (reservation, mut values, range) = self.take_table()?;
    drop(reservation);
    if values.len() != range.len() {
      values.truncate(range.end);
      drop(values.drain(..range.start));
    }
    Ok(values)
  }

  pub(super) fn take_table_mut(self) -> Result<SliceMut<T, S>, Self> {
    let (reservation, values, range) = self.take_table()?;
    Ok(SliceMut::install_table_window(reservation, || {
      Window::from_vec(values).keep_range(range)
    }))
  }
}

impl<T, S: InlineSize> SliceMut<T, S> {
  fn install_table_window(
    reservation: Reservation,
    make_window: impl FnOnce() -> Window<T>,
  ) -> Self {
    // Reserve both allocations before partitioning a source window. If the
    // closure unwinds, its vacant slot and uninitialized Box are reclaimed.
    let mut owner = Box::<Window<T>>::new_uninit();
    owner.write(make_window());
    // SAFETY: write initialized the entire boxed window.
    let key = reservation.install_box(unsafe { owner.assume_init() });
    Self::wrap(Layout::from_table_word(MUTABLE | u64::from(key.bits())))
  }

  pub(super) fn from_table_window(make_window: impl FnOnce() -> Window<T>) -> Self {
    Self::install_table_window(
      Reservation::claim().expect("owner table exhausted"),
      make_window,
    )
  }

  pub(super) fn try_from_table_vec(values: Vec<T>) -> Result<Self, Vec<T>> {
    let Some(reservation) = Reservation::claim() else {
      return Err(values);
    };
    Ok(Self::install_table_window(reservation, || {
      Window::from_vec(values)
    }))
  }

  pub(super) fn freeze_table(mut self) -> Result<Slice<T, S>, Self>
  where
    T: Send + Sync,
  {
    let Some((values, start)) = self
      .window_mut()
      .and_then(|window| window.take_vec_prefix(Layout::<T, S>::fits_table_vec))
    else {
      return Err(self);
    };
    let (reservation, window) = self.storage.layout_mut().keyed_box();
    drop(window);
    let len = values.len() - start;
    let key = reservation.install_vec(values);
    Ok(Slice {
      layout: Layout::table_window(key, start, len),
    })
  }
}
