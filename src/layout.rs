use core::{
  marker::PhantomData,
  mem::MaybeUninit,
  num::NonZeroU8,
  ops::Range,
  ptr::{self, NonNull},
};

mod sealed {
  pub trait Repr {
    type Align;
    type Payload;
    type Cursor: Default + AsRef<[u8]> + AsMut<[u8]>;
  }
}
use sealed::Repr;

/// A supported total byte size for a [`Slice`](crate::Slice) or [`SliceMut`](crate::SliceMut).
///
/// On 32- and 64-bit targets, implemented for `[(); BYTES]` when `BYTES` is a
/// nonzero multiple of the target's word size, up to 2048 bytes. Generic callers
/// can use `where [(); BYTES]: SupportedBytes`.
pub trait SupportedBytes: Repr {}
impl<const BYTES: usize> SupportedBytes for [(); BYTES] where [(); BYTES]: Repr {}

// Stable Rust cannot subtract metadata from a generic array length. Select the
// concrete arrays here and verify the whole public handle for every supported size.
macro_rules! sizes {
  ($align:ty, $cursor:literal; $($bytes:expr),+ $(,)?) => {$(
    impl Repr for [(); $bytes] {
      type Align = $align;
      type Payload = [u8; $bytes - $cursor - 1];
      type Cursor = [u8; $cursor];
    }
    const _: () = {
      use crate::{Slice, SliceMut};
      assert!(core::mem::offset_of!(Layout<u8, { $bytes }>, payload) == 0);
      assert!(core::mem::offset_of!(Layout<u8, { $bytes }>, tag) == $bytes - 1);
      assert!(align_of::<Slice<u8, { $bytes }>>() == align_of::<$align>());
      assert!(size_of::<Slice<u8, { $bytes }>>() == $bytes);
      assert!(size_of::<Option<Slice<u8, { $bytes }>>>() == $bytes);
      assert!(size_of::<SliceMut<u8, { $bytes }>>() == $bytes);
      assert!(size_of::<Option<SliceMut<u8, { $bytes }>>>() == $bytes);
    };
  )+};
}
#[cfg(target_pointer_width = "32")]
sizes!(u32, 0; 4);
sizes!(usize, 0; 8);

// Each larger 64-bit size also anchors the preceding 32-bit-only size in the
// same cursor tier.
macro_rules! word_sizes {
  ($cursor:literal; $($bytes:expr),+ $(,)?) => {
    sizes!(usize, $cursor; $($bytes),+);
    #[cfg(target_pointer_width = "32")]
    sizes!(usize, $cursor; $($bytes - 4),+);
  };
}
word_sizes!(1; 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128);

macro_rules! wide_sizes {
  ($($base:literal),+) => {$(
    word_sizes!(3;
      $base + 8, $base + 16, $base + 24, $base + 32, $base + 40, $base + 48, $base + 56,
      $base + 64, $base + 72, $base + 80, $base + 88, $base + 96, $base + 104, $base + 112,
      $base + 120, $base + 128,
    );
  )+};
}
wide_sizes!(
  128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920
);

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
pub(super) struct Layout<T, const BYTES: usize>
where
  [(); BYTES]: SupportedBytes,
{
  // This zero-sized field aligns the start of the payload. Both byte arrays
  // have alignment one and no padding. Moving MaybeUninit preserves pointer
  // provenance and any uninitialized bytes in a stored T's padding.
  _align: [<[(); BYTES] as Repr>::Align; 0],
  payload: MaybeUninit<<[(); BYTES] as Repr>::Payload>,
  cursor: <[(); BYTES] as Repr>::Cursor,
  tag: NonZeroU8,
  _element: PhantomData<T>,
}

impl<T, const BYTES: usize> Layout<T, BYTES>
where
  [(); BYTES]: SupportedBytes,
{
  const PAYLOAD_BYTES: usize = size_of::<<[(); BYTES] as Repr>::Payload>();
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
    let mut cursor = <[(); BYTES] as Repr>::Cursor::default();
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
    if BYTES <= 8 || Self::PAYLOAD_BYTES < size_of::<ExternalPrefix>() || len > isize::MAX as usize
    {
      return None;
    }
    let bytes = len.to_le_bytes();
    let mut len_low = [0; LEN_LOW_BYTES];
    len_low.copy_from_slice(&bytes[..LEN_LOW_BYTES]);
    let mut cursor = <[(); BYTES] as Repr>::Cursor::default();
    cursor.as_mut()[0] = bytes[LEN_LOW_BYTES];
    let mut payload = MaybeUninit::<<[(); BYTES] as Repr>::Payload>::uninit();
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
