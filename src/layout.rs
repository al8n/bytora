use core::{
  fmt::Debug,
  marker::PhantomData,
  mem::MaybeUninit,
  num::NonZeroU8,
  ops::Range,
  ptr::{self, NonNull},
};
use typenum::Unsigned;

#[cfg(any(feature = "std", feature = "alloc"))]
pub(super) mod local;
pub(super) mod mutable;
mod shared;

/// The byte arrays used by the inline cursor and raw element storage.
///
/// # Safety
///
/// Both types must be arrays of `u8`. Together with the one-byte tag they
/// must fill the physical handle. The cursor must have the width
/// required to encode every index in the payload.
#[doc(hidden)]
unsafe trait Repr {
  type Payload;
  type Cursor: Copy + AsRef<[u8]> + AsMut<[u8]>;
  const ZERO_CURSOR: Self::Cursor;
}

/// A type-level byte size for a [`Slice`](crate::Slice) or [`SliceMut`](crate::SliceMut).
///
/// A positive size is a multiple of the target's word size through [`MAX_HANDLE_BYTES`]
/// and counts all handle bytes, including the cursor. [`typenum::U0`] disables
/// inline elements and uses a three-word handle. The trait is sealed.
/// [`INLINE_CAPACITY`](crate::Slice::INLINE_CAPACITY) also depends on the element type.
#[allow(
  private_bounds,
  reason = "the backing arrays are private to the layout"
)]
pub trait InlineSize: Unsigned + Debug + Repr {
  /// The size in bytes of the handle and its `Option`.
  const HANDLE_BYTES: usize = handle_bytes(Self::USIZE);

  #[doc(hidden)]
  type Mutable<T>: mutable::Storage<T, Self>;
}

const fn handle_bytes(budget: usize) -> usize {
  if budget == 0 {
    3 * size_of::<usize>()
  } else {
    budget
  }
}

/// Maps an array of `N` bytes to the type-level handle size used by
/// [`SliceN`](crate::SliceN) and [`SliceMutN`](crate::SliceMutN).
///
/// Use `where [u8; N]: AssocInlineSize` in code generic over `const N: usize`.
pub trait AssocInlineSize {
  /// The [`InlineSize`] for this byte budget.
  type Size: InlineSize;
}

// Stable Rust cannot subtract metadata from a generic array length. Select the
// concrete arrays generated for the target and verify every supported size.
macro_rules! size {
  (4, $size:ty, $cursor:literal) => {
    size!(4, $size, $cursor, Layout<T, $size>);
  };
  (8, $size:ty, $cursor:literal) => {
    size!(8, $size, $cursor, mutable::EightBytes<T, $size>);
  };
  ($bytes:literal, $size:ty, $cursor:literal) => {
    size!($bytes, $size, $cursor, mutable::Spill<T, $size>);
  };
  ($bytes:literal, $size:ty, $cursor:literal, $mutable:ty) => {
    // SAFETY: Both fields are byte arrays. verify_size checks their lengths
    // against the cursor encoding and the typenum value for every mapping.
    unsafe impl Repr for $size {
      type Payload = [u8; handle_bytes($bytes) - $cursor - 1];
      type Cursor = [u8; $cursor];
      const ZERO_CURSOR: Self::Cursor = [0; $cursor];
    }
    impl InlineSize for $size {
      type Mutable<T> = $mutable;
    }
    impl AssocInlineSize for [u8; $bytes] {
      type Size = $size;
    }
    const _: () = verify_size::<<[u8; $bytes] as AssocInlineSize>::Size>($bytes);
  };
}
size!(0, typenum::U0, 1, mutable::Spill<T, typenum::U0>);
include!(concat!(env!("OUT_DIR"), "/sizes.rs"));

#[allow(
  dead_code,
  reason = "Rust 1.85 does not count calls from anonymous constants as use"
)]
const fn verify_size<S: InlineSize>(budget: usize) {
  use crate::{Slice, SliceMut};

  let bytes = S::HANDLE_BYTES;
  let payload = size_of::<S::Payload>();
  let cursor = size_of::<S::Cursor>();
  let expected_cursor = if bytes <= 8 {
    0
  } else if bytes <= 128 {
    1
  } else {
    3
  };
  let cursor_limit = match cursor {
    0 => 7,
    1 => 126,
    3 => 32511,
    _ => 0,
  };
  assert!(bytes > 0 && bytes % size_of::<usize>() == 0 && bytes <= MAX_HANDLE_BYTES);
  assert!(S::USIZE == budget);
  assert!(cursor == expected_cursor);
  assert!(payload + cursor + 1 == bytes);
  assert!(payload <= cursor_limit);
  if Layout::<u8, S>::COMPACT_EXTERNAL {
    assert!(cursor == 1 && payload + 1 < COMPACT_STATIC.get() as usize);
  }
  if Layout::<u8, S>::FITS_EXTERNAL {
    assert!(size_of::<ExternalPrefix>() == 2 * size_of::<usize>());
    if bytes == 3 * size_of::<usize>() {
      assert!(cursor == 1 && payload + 2 == bytes);
    } else {
      assert!(payload >= 3 * size_of::<usize>());
    }
  }
  assert!(Layout::<u8, S>::INLINE_CAPACITY == if budget == 0 { 0 } else { payload });
  assert!(core::mem::offset_of!(Layout<u8, S>, payload) == 0);
  assert!(core::mem::offset_of!(Layout<u8, S>, cursor) == payload);
  assert!(core::mem::offset_of!(Layout<u8, S>, tag) == bytes - 1);
  assert!(align_of::<Slice<u8, S>>() == align_of::<usize>());
  assert!(align_of::<SliceMut<u8, S>>() == align_of::<usize>());
  assert!(size_of::<Slice<u8, S>>() == bytes);
  assert!(size_of::<Option<Slice<u8, S>>>() == bytes);
  assert!(size_of::<SliceMut<u8, S>>() == bytes);
  assert!(size_of::<Option<SliceMut<u8, S>>>() == bytes);
  if !cfg!(any(feature = "std", feature = "alloc")) || budget == size_of::<usize>() {
    assert!(!core::mem::needs_drop::<SliceMut<u8, S>>());
  }
}

const TINY_INLINE: u8 = 0x40;
// An external-capable handle of at most 32 bytes uses tags 1..=31 for inline
// endpoints. 0x40 identifies the real static owner. In a three-word handle
// the high length byte shares the tag, so long windows use 0x80..=0xff
// regardless of owner. Tiny inline-only handles have a separate tag encoding.
const COMPACT_STATIC: NonZeroU8 = NonZeroU8::new(0x40).unwrap();
const EXTERNAL: u8 = 0x80;
const TAG_SHIFT: u32 = usize::BITS - 8;

// Both pointers stay typed when written to or read from the payload.
// The next word stores the length. In three-word handles its last byte
// doubles as the tag; the word is little-endian there, even on big-endian targets.
// Recoverable ZST owners use the data word for a provenance-free logical index
// and supply the original aligned pointer themselves.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ExternalPrefix {
  data: *const (),
  owner: *const (),
}

#[repr(transparent)]
struct Payload<P> {
  #[cfg(smol_slice_atomic)]
  bytes: core::cell::UnsafeCell<MaybeUninit<P>>,
  #[cfg(not(smol_slice_atomic))]
  bytes: MaybeUninit<P>,
}

impl<P> Payload<P> {
  const fn new(bytes: MaybeUninit<P>) -> Self {
    Self {
      #[cfg(smol_slice_atomic)]
      bytes: core::cell::UnsafeCell::new(bytes),
      #[cfg(not(smol_slice_atomic))]
      bytes,
    }
  }

  const fn as_ptr(&self) -> *const P {
    #[cfg(smol_slice_atomic)]
    return self.bytes.get().cast();
    #[cfg(not(smol_slice_atomic))]
    self.bytes.as_ptr()
  }

  const fn as_mut_ptr(&mut self) -> *mut P {
    ptr::from_mut(self).cast()
  }
}

// SAFETY: Shared writes are confined to the aligned atomic owner word of a
// promotable external descriptor. A handle's tag, window and inline elements
// are immutable through shared references. P is a sealed byte array.
#[cfg(smol_slice_atomic)]
unsafe impl<P: Sync> Sync for Payload<P> {}
#[cfg(smol_slice_atomic)]
impl<P: core::panic::RefUnwindSafe> core::panic::RefUnwindSafe for Payload<P> {}

#[repr(C)]
pub struct Layout<T, S: InlineSize> {
  // This zero-sized field aligns the start of the payload. Both byte arrays
  // have alignment one and no padding. Moving MaybeUninit preserves pointer
  // provenance and any uninitialized bytes in a stored T's padding.
  _align: [usize; 0],
  payload: Payload<S::Payload>,
  cursor: S::Cursor,
  tag: NonZeroU8,
  _element: PhantomData<T>,
}

impl<T, S: InlineSize> Layout<T, S> {
  const PAYLOAD_BYTES: usize = size_of::<S::Payload>();
  const TWO_WORD_VEC: bool = cfg!(smol_slice_atomic) && S::HANDLE_BYTES == 2 * size_of::<usize>();
  const FITS_EXTERNAL: bool = S::HANDLE_BYTES >= 3 * size_of::<usize>();
  const PACKED_LEN: bool = S::HANDLE_BYTES == 3 * size_of::<usize>();
  pub(super) const COMPACT_EXTERNAL: bool = Self::FITS_EXTERNAL && S::HANDLE_BYTES <= 32;
  pub(super) const INLINE_CAPACITY: usize = if S::USIZE == 0 {
    0
  } else if size_of::<T>() == 0 {
    Self::PAYLOAD_BYTES
  } else if align_of::<T>() > align_of::<Self>() {
    0
  } else {
    Self::PAYLOAD_BYTES / size_of::<T>()
  };

  pub(crate) const fn empty_inline() -> Self {
    Self {
      _align: [],
      payload: Payload::new(MaybeUninit::uninit()),
      cursor: S::ZERO_CURSOR,
      tag: if size_of::<S::Cursor>() == 0 {
        NonZeroU8::new(TINY_INLINE).unwrap()
      } else {
        NonZeroU8::MIN
      },
      _element: PhantomData,
    }
  }

  pub(super) fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    let len = values.len();
    if len > Self::INLINE_CAPACITY {
      return None;
    }
    let (cursor, tag) = Self::inline_cursor(0..len);
    let mut layout = Self {
      _align: [],
      payload: Payload::new(MaybeUninit::uninit()),
      cursor,
      tag,
      _element: PhantomData,
    };
    if len != 0 && size_of::<T>() != 0 {
      // SAFETY: The capacity check guarantees alignment and space for len elements.
      // The fresh payload cannot overlap values. T: Copy permits the duplicates.
      unsafe { ptr::copy_nonoverlapping(values.as_ptr(), layout.payload.as_mut_ptr().cast(), len) };
    }
    Some(layout)
  }

  pub(super) fn is_inline(&self) -> bool {
    Self::inline_tag(self.tag)
  }

  const fn inline_tag(tag: NonZeroU8) -> bool {
    if Self::TWO_WORD_VEC {
      // Tiny inline endpoints set bit 6. A sixteen-byte inline handle
      // has a one-byte endpoint in the range 1 through 15.
      return if size_of::<S::Cursor>() == 0 {
        tag.get() & 0xc0 == TINY_INLINE
      } else {
        tag.get() <= Self::PAYLOAD_BYTES as u8 + 1
      };
    }
    let first_external = if Self::COMPACT_EXTERNAL {
      COMPACT_STATIC.get()
    } else {
      EXTERNAL
    };
    !Self::FITS_EXTERNAL || tag.get() < first_external
  }

  // Inline and compact static descriptors can be copied without an owner.
  #[cfg(smol_slice_atomic)]
  fn uses_external_owner(&self) -> bool {
    Self::FITS_EXTERNAL && self.tag.get() >= EXTERNAL
  }

  pub(super) fn clone_inline(&self) -> Self {
    assert!(self.is_inline(), "expected inline storage");
    // SAFETY: self is a valid, aligned layout. The inline tag excludes an external
    // owner. Nonempty immutable inline elements entered with a Copy proof;
    // they cannot be replaced while immutable. Refreezing checks Copy again.
    // Reading the MaybeUninit payload as part of Self preserves both
    // pointer provenance and uninitialized bytes, including element padding.
    unsafe { ptr::read(self) }
  }

  #[track_caller]
  pub(super) fn slice_inline(&self, range: Range<usize>) -> Self {
    let current = self.inline_range();
    assert!(
      range.start <= range.end && range.end <= current.len(),
      "slice range out of bounds"
    );
    // The new absolute endpoints stay inside the initialized window. Copy the
    // inline elements with their padding and provenance before changing the cursor.
    let mut layout = self.clone_inline();
    (layout.cursor, layout.tag) =
      Self::inline_cursor(current.start + range.start..current.start + range.end);
    layout
  }

  pub(super) fn truncate_inline(&mut self, len: usize) {
    let current = self.inline_range();
    if len < current.len() {
      // Only the end moves, and it stays within the initialized window.
      (self.cursor, self.tag) = Self::inline_cursor(current.start..current.start + len);
    }
  }

  const fn external_ptr(&self) -> *const ExternalPrefix {
    assert!(!Self::inline_tag(self.tag), "expected external storage");
    self.payload.as_ptr().cast()
  }

  const fn data(&self) -> *const () {
    // SAFETY: This field is initialized in every external descriptor and only
    // changes through &mut. Do not borrow the whole prefix across a promotion.
    unsafe { (*self.external_ptr()).data }
  }

  pub(super) const fn len(&self) -> usize {
    if Self::inline_tag(self.tag) {
      let range = self.inline_range();
      return range.end - range.start;
    }
    #[cfg(smol_slice_atomic)]
    if Self::TWO_WORD_VEC {
      return self.two_word_len();
    }
    self.external_len()
  }

  #[inline]
  const fn external_len(&self) -> usize {
    assert!(!Self::inline_tag(self.tag), "expected external storage");
    // SAFETY: Every external constructor initializes this entire word. Its
    // address derives from the enclosing handle, as the packed form spans the
    // payload, cursor and tag. It is disjoint from the mutable owner slot.
    let word = unsafe { ptr::from_ref(self).cast::<usize>().add(2).read() };
    if !Self::PACKED_LEN {
      return word;
    }
    let len = usize::from_le(word);
    if self.tag.get() < EXTERNAL {
      len & ((1 << TAG_SHIFT) - 1)
    } else {
      len & (isize::MAX as usize)
    }
  }

  const fn cursor_bytes(&self) -> &[u8] {
    // SAFETY: Sealed Repr guarantees that Cursor is an initialized [u8; N].
    // Borrow exactly its N bytes, including the zero-byte cursor of tiny layouts.
    unsafe {
      core::slice::from_raw_parts(ptr::from_ref(&self.cursor).cast(), size_of::<S::Cursor>())
    }
  }

  const fn inline_range(&self) -> Range<usize> {
    let tag = self.tag.get();
    assert!(Self::inline_tag(self.tag), "expected inline storage");
    // Tiny tags contain both 3-bit cursors. Larger tags encode the high part
    // of end plus one, leaving zero for Option and the top bit for external.
    match self.cursor_bytes() {
      [] => ((tag >> 3) & 7) as usize..(tag & 7) as usize,
      [start] => *start as usize..(tag - 1) as usize,
      [lo, hi, end_low] => {
        u16::from_le_bytes([*lo, *hi]) as usize..u16::from_le_bytes([*end_low, tag - 1]) as usize
      }
      _ => unreachable!(),
    }
  }

  // Callers supply a window within the inline capacity for this size.
  fn inline_cursor(range: Range<usize>) -> (S::Cursor, NonZeroU8) {
    let mut cursor = S::ZERO_CURSOR;
    let tag = match cursor.as_mut() {
      [] => TINY_INLINE | ((range.start as u8) << 3) | range.end as u8,
      [start] => {
        *start = range.start as u8;
        range.end as u8 + 1
      }
      [lo, hi, end_low] => {
        [*lo, *hi] = (range.start as u16).to_le_bytes();
        let [low, high] = (range.end as u16).to_le_bytes();
        *end_low = low;
        high + 1
      }
      _ => unreachable!(),
    };
    (cursor, NonZeroU8::new(tag).unwrap())
  }

  // Empty and zero-sized slices need only a non-null, T-aligned pointer.
  // Nonempty, non-ZST windows lie wholly in the aligned payload initialized by try_inline.
  fn window(payload: NonNull<T>, range: Range<usize>) -> *mut [T] {
    let data = if range.is_empty() || size_of::<T>() == 0 {
      NonNull::dangling().as_ptr()
    } else {
      payload.as_ptr().wrapping_add(range.start)
    };
    ptr::slice_from_raw_parts_mut(data, range.len())
  }

  pub(super) fn as_slice(&self) -> &[T] {
    // SAFETY: window selects initialized elements (or an aligned empty slice) and
    // the shared payload borrow covers the lifetime of the returned slice.
    unsafe { &*Self::window(NonNull::from(&self.payload).cast(), self.inline_range()) }
  }

  #[track_caller]
  pub(super) fn advance(&mut self, count: usize) {
    let range = self.inline_range();
    assert!(count <= range.len(), "advance past end");
    let start = range.start + count;
    match self.cursor.as_mut() {
      [] => {
        self.tag = NonZeroU8::new(TINY_INLINE | ((start as u8) << 3) | range.end as u8).unwrap()
      }
      [cursor] => *cursor = start as u8,
      [lo, hi, _] => [*lo, *hi] = (start as u16).to_le_bytes(),
      _ => unreachable!(),
    }
  }
}

// A bare Layout describes a window. Its enclosing Slice manages the external owner.
impl<T, S: InlineSize> Layout<T, S> {
  const fn set_external_len(&mut self, len: usize, kind: NonZeroU8) {
    assert!(Self::FITS_EXTERNAL);
    let word = if Self::PACKED_LEN {
      // The owner-specific tag is available only when the high byte is free.
      let tag = if len >> TAG_SHIFT == 0 {
        kind.get()
      } else {
        EXTERNAL
      };
      (len | ((tag as usize) << TAG_SHIFT)).to_le()
    } else {
      if Self::COMPACT_EXTERNAL {
        // Initialize the whole final word, including the byte cursor. Compact
        // handles are routinely moved as words, even when their spare bytes
        // have no semantic meaning. The separate length precedes this word.
        let last = S::HANDLE_BYTES / size_of::<usize>() - 1;
        let tag_word = ((kind.get() as usize) << TAG_SHIFT).to_le();
        // SAFETY: The fourth or later word is aligned and disjoint from the
        // other three. Its final address byte receives the nonzero tag.
        unsafe {
          ptr::from_mut(self)
            .cast::<usize>()
            .add(last)
            .write(tag_word)
        };
      } else {
        self.tag = kind;
      }
      len
    };
    // SAFETY: verify_size checks that the third word fits, and the handle
    // is usize-aligned. The packed write initializes the cursor and writes
    // a nonzero high byte into the tag. The wide write is wholly in the payload.
    unsafe { ptr::from_mut(self).cast::<usize>().add(2).write(word) };
  }

  #[cfg(any(test, smol_slice_atomic))]
  const fn try_external(data: *const T, owner: *const (), len: usize) -> Option<Self> {
    if !Self::FITS_EXTERNAL || len > isize::MAX as usize {
      return None;
    }
    Some(Self::external(
      data,
      owner,
      len,
      NonZeroU8::new(EXTERNAL).unwrap(),
    ))
  }

  #[inline]
  const fn external(data: *const T, owner: *const (), len: usize, kind: NonZeroU8) -> Self {
    assert!(Self::FITS_EXTERNAL);
    let mut layout = Self::empty_inline();
    let prefix = ExternalPrefix {
      data: data.cast(),
      owner,
    };
    // SAFETY: The two-pointer prefix fits the checked external payload.
    unsafe {
      layout
        .payload
        .as_mut_ptr()
        .cast::<ExternalPrefix>()
        .write(prefix)
    };
    layout.set_external_len(len, kind);
    layout
  }
}

#[cfg(test)]
mod tests;
