use core::{
  fmt::Debug,
  marker::PhantomData,
  mem::MaybeUninit,
  num::NonZeroU8,
  ops::Range,
  ptr::{self, NonNull},
};
use typenum::Unsigned;

/// The byte arrays used by the inline cursor and raw element storage.
///
/// # Safety
///
/// Both types must be arrays of `u8`. Together with the one-byte tag they
/// must fill exactly the size's byte budget. The cursor must have the width
/// required to encode every index in the payload.
#[doc(hidden)]
unsafe trait Repr {
  type Payload;
  type Cursor: Default + AsRef<[u8]> + AsMut<[u8]>;
}

/// A type-level byte size for a [`Slice`](crate::Slice) or [`SliceMut`](crate::SliceMut).
///
/// Implemented for the [`typenum`] unsigned integers that are positive multiples
/// of the target's word size through [`MAX_HANDLE_BYTES`]. The trait is sealed.
/// [`S::USIZE`](Unsigned::USIZE) counts all bytes, including the cursor. The
/// [`INLINE_CAPACITY`](crate::Slice::INLINE_CAPACITY) also depends on the element type.
#[allow(
  private_bounds,
  reason = "the backing arrays are private to the layout"
)]
pub trait InlineSize: Unsigned + Debug + Repr {}
impl<S: Unsigned + Debug + Repr> InlineSize for S {}

/// Maps an array of `N` bytes to the type-level handle size used by
/// [`SliceN`](crate::SliceN) and [`SliceMutN`](crate::SliceMutN).
///
/// Use `where [u8; N]: AssocInlineSize` in code generic over `const N: usize`.
pub trait AssocInlineSize {
  /// The [`InlineSize`] for this total byte count.
  type Size: InlineSize;
}

// Stable Rust cannot subtract metadata from a generic array length. Select the
// concrete arrays generated for the target and verify every supported size.
macro_rules! size {
  ($bytes:literal, $size:ty, $cursor:literal) => {
    // SAFETY: Both fields are byte arrays. verify_size checks their lengths
    // against the cursor encoding and the typenum value for every mapping.
    unsafe impl Repr for $size {
      type Payload = [u8; $bytes - $cursor - 1];
      type Cursor = [u8; $cursor];
    }
    impl AssocInlineSize for [u8; $bytes] {
      type Size = $size;
    }
    const _: () = verify_size::<<[u8; $bytes] as AssocInlineSize>::Size>($bytes);
  };
}
include!(concat!(env!("OUT_DIR"), "/sizes.rs"));

#[allow(
  dead_code,
  reason = "Rust 1.85 does not count calls from anonymous constants as use"
)]
const fn verify_size<S: InlineSize>(bytes: usize) {
  use crate::{Slice, SliceMut};

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
  assert!(S::USIZE == bytes);
  assert!(cursor == expected_cursor);
  assert!(payload + cursor + 1 == bytes);
  assert!(payload <= cursor_limit);
  assert!(Layout::<u8, S>::INLINE_CAPACITY == payload);
  assert!(core::mem::offset_of!(Layout<u8, S>, payload) == 0);
  assert!(core::mem::offset_of!(Layout<u8, S>, cursor) == payload);
  assert!(core::mem::offset_of!(Layout<u8, S>, tag) == bytes - 1);
  assert!(align_of::<Slice<u8, S>>() == align_of::<usize>());
  assert!(align_of::<SliceMut<u8, S>>() == align_of::<usize>());
  assert!(size_of::<Slice<u8, S>>() == bytes);
  assert!(size_of::<Option<Slice<u8, S>>>() == bytes);
  assert!(size_of::<SliceMut<u8, S>>() == bytes);
  assert!(size_of::<Option<SliceMut<u8, S>>>() == bytes);
}

const TINY_INLINE: u8 = 0x40;
const EXTERNAL: u8 = 0x80;
const LEN_LOW_BYTES: usize = size_of::<usize>() - 2;

// The two high length bytes live in the cursor and tag. Both pointers stay
// typed when written to or read from the payload, including on big-endian targets.
#[repr(C, packed)]
struct ExternalPrefix {
  data: *const (),
  owner: *const (),
  len_low: [u8; LEN_LOW_BYTES],
}

#[repr(C)]
pub(super) struct Layout<T, S: InlineSize> {
  // This zero-sized field aligns the start of the payload. Both byte arrays
  // have alignment one and no padding. Moving MaybeUninit preserves pointer
  // provenance and any uninitialized bytes in a stored T's padding.
  _align: [usize; 0],
  payload: MaybeUninit<S::Payload>,
  cursor: S::Cursor,
  tag: NonZeroU8,
  _element: PhantomData<T>,
}

impl<T, S: InlineSize> Layout<T, S> {
  const PAYLOAD_BYTES: usize = size_of::<S::Payload>();
  pub(super) const INLINE_CAPACITY: usize = if size_of::<T>() == 0 {
    Self::PAYLOAD_BYTES
  } else if align_of::<T>() > align_of::<Self>() {
    0
  } else {
    Self::PAYLOAD_BYTES / size_of::<T>()
  };

  pub(super) fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    let len = values.len();
    if len > Self::INLINE_CAPACITY {
      return None;
    }
    let mut cursor = S::Cursor::default();
    let tag = match cursor.as_mut() {
      [] => TINY_INLINE | len as u8,
      [_] => len as u8 + 1,
      [_, _, end_low] => {
        let [low, high] = (len as u16).to_le_bytes();
        *end_low = low;
        high + 1
      }
      _ => unreachable!(),
    };
    let mut layout = Self {
      _align: [],
      payload: MaybeUninit::uninit(),
      cursor,
      tag: NonZeroU8::new(tag).unwrap(),
      _element: PhantomData,
    };
    if len != 0 && size_of::<T>() != 0 {
      // SAFETY: The capacity check guarantees alignment and space for len elements.
      // The fresh payload cannot overlap values. T: Copy permits the duplicates.
      unsafe { ptr::copy_nonoverlapping(values.as_ptr(), layout.payload.as_mut_ptr().cast(), len) };
    }
    Some(layout)
  }

  // An inert descriptor. Only inline layouts can be put into a public handle.
  #[cfg_attr(not(test), expect(dead_code, reason = "reserved for external owners"))]
  fn try_external(data: *const T, owner: *const (), len: usize) -> Option<Self> {
    if S::USIZE <= 8
      || Self::PAYLOAD_BYTES < size_of::<ExternalPrefix>()
      || len > isize::MAX as usize
    {
      return None;
    }
    let bytes = len.to_le_bytes();
    let mut len_low = [0; LEN_LOW_BYTES];
    len_low.copy_from_slice(&bytes[..LEN_LOW_BYTES]);
    let mut cursor = S::Cursor::default();
    cursor.as_mut()[0] = bytes[LEN_LOW_BYTES];
    let mut payload = MaybeUninit::<S::Payload>::uninit();
    // SAFETY: The packed prefix has alignment one and fits the checked payload.
    unsafe {
      payload
        .as_mut_ptr()
        .cast::<ExternalPrefix>()
        .write(ExternalPrefix {
          data: data.cast(),
          owner,
          len_low,
        });
    }
    Some(Self {
      _align: [],
      payload,
      cursor,
      tag: NonZeroU8::new(EXTERNAL | bytes[LEN_LOW_BYTES + 1]).unwrap(),
      _element: PhantomData,
    })
  }

  pub(super) fn len(&self) -> usize {
    if self.tag.get() < EXTERNAL {
      return self.inline_range().len();
    }
    // SAFETY: Only try_external sets the external tag, after writing the prefix.
    let prefix = unsafe { &*self.payload.as_ptr().cast::<ExternalPrefix>() };
    let mut bytes = [0; size_of::<usize>()];
    bytes[..LEN_LOW_BYTES].copy_from_slice(&prefix.len_low);
    bytes[LEN_LOW_BYTES] = self.cursor.as_ref()[0];
    bytes[LEN_LOW_BYTES + 1] = self.tag.get() & !EXTERNAL;
    usize::from_le_bytes(bytes)
  }

  fn inline_range(&self) -> Range<usize> {
    let tag = self.tag.get();
    assert!(tag < EXTERNAL, "expected inline storage");
    // Tiny tags contain both 3-bit cursors. Larger tags encode the high part
    // of end plus one, leaving zero for Option and the top bit for external.
    match self.cursor.as_ref() {
      [] => usize::from((tag >> 3) & 7)..usize::from(tag & 7),
      [start] => usize::from(*start)..usize::from(tag - 1),
      [lo, hi, end_low] => {
        usize::from(u16::from_le_bytes([*lo, *hi]))
          ..usize::from(u16::from_le_bytes([*end_low, tag - 1]))
      }
      _ => unreachable!(),
    }
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

  pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
    let range = self.inline_range();
    // SAFETY: The same window invariant applies, with exclusive payload access.
    unsafe { &mut *Self::window(NonNull::from(&mut self.payload).cast(), range) }
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

#[cfg(test)]
mod tests;
