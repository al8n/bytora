use core::{
  mem::{ManuallyDrop, MaybeUninit},
  num::NonZeroU8,
  ops::Range,
  slice,
};

const EXTERNAL: u8 = 0x80;
const MAX_INLINE: usize = (EXTERNAL - 2) as usize;
const LEN_LOW_BYTES: usize = size_of::<usize>() - 2;

// The final two length bytes live outside the union, at the same offset as
// the inline cursor for each choice of T and N. Keep pointers as pointers.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ExternalPrefix {
  data: *const (),
  owner: *const (),
  len_low: [u8; LEN_LOW_BYTES],
}

#[repr(C)]
union Body<T, const N: usize> {
  inline: ManuallyDrop<[MaybeUninit<T>; N]>,
  external: ExternalPrefix,
}

#[repr(C)]
pub(super) struct Layout<T, const N: usize> {
  // A zero-sized field aligns the handle to a word without rounding up the packed prefix.
  // The typed union also aligns the body to T, including over-aligned and zero-sized T.
  _word_align: [usize; 0],
  body: Body<T, N>,
  start_or_len: u8,
  // 0 is the Option niche. 1..=127 encode inline end + 1. 128..=255 select external
  // storage and carry the upper seven bits of the external length.
  tag: NonZeroU8,
}

impl<T, const N: usize> Layout<T, N> {
  pub(super) const INLINE_CAPACITY: usize = if N < MAX_INLINE { N } else { MAX_INLINE };

  pub(super) fn try_inline(values: &[T]) -> Option<Self>
  where
    T: Copy,
  {
    if values.len() > Self::INLINE_CAPACITY {
      return None;
    }
    let mut slots = [MaybeUninit::uninit(); N];
    for (slot, &value) in slots.iter_mut().zip(values) {
      slot.write(value);
    }
    Some(Self {
      _word_align: [],
      body: Body {
        inline: ManuallyDrop::new(slots),
      },
      start_or_len: 0,
      tag: NonZeroU8::new(values.len() as u8 + 1).unwrap(),
    })
  }

  // An inert descriptor. Only inline layouts can be put into a public handle.
  #[cfg_attr(not(test), expect(dead_code, reason = "reserved for external owners"))]
  fn try_external(data: *const T, owner: *const (), len: usize) -> Option<Self> {
    if len > isize::MAX as usize {
      return None;
    }
    let bytes = len.to_le_bytes();
    let mut len_low = [0; LEN_LOW_BYTES];
    len_low.copy_from_slice(&bytes[..LEN_LOW_BYTES]);
    Some(Self {
      _word_align: [],
      body: Body {
        external: ExternalPrefix {
          data: data.cast(),
          owner,
          len_low,
        },
      },
      start_or_len: bytes[LEN_LOW_BYTES],
      tag: NonZeroU8::new(EXTERNAL | bytes[LEN_LOW_BYTES + 1]).unwrap(),
    })
  }

  pub(super) fn len(&self) -> usize {
    if self.tag.get() < EXTERNAL {
      return self.inline_range().len();
    }
    let mut bytes = [0; size_of::<usize>()];
    // SAFETY: The external tag selects the initialized external prefix. The byte
    // array has alignment one even when borrowed inside a packed struct.
    bytes[..LEN_LOW_BYTES].copy_from_slice(unsafe { &self.body.external.len_low });
    bytes[LEN_LOW_BYTES] = self.start_or_len;
    bytes[LEN_LOW_BYTES + 1] = self.tag.get() & !EXTERNAL;
    usize::from_le_bytes(bytes)
  }

  fn inline_range(&self) -> Range<usize> {
    let tag = self.tag.get();
    assert!(tag < EXTERNAL, "expected inline storage");
    usize::from(self.start_or_len)..usize::from(tag - 1)
  }

  pub(super) fn as_slice(&self) -> &[T] {
    let range = self.inline_range();
    // SAFETY: The tag selects inline storage. try_inline initialized every slot
    // up to the encoded end, and advance only moves the start within that prefix.
    // MaybeUninit<T> has T's size and alignment, and the borrow is tied to self.
    unsafe {
      let slots = &self.body.inline[range];
      slice::from_raw_parts(slots.as_ptr().cast(), slots.len())
    }
  }

  pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
    let range = self.inline_range();
    // SAFETY: The same initialized-range and alignment invariants as as_slice
    // apply. The borrow of self gives exclusive access to every returned slot.
    unsafe {
      let inline: &mut [_; N] = &mut self.body.inline;
      let slots = &mut inline[range];
      slice::from_raw_parts_mut(slots.as_mut_ptr().cast(), slots.len())
    }
  }

  #[track_caller]
  pub(super) fn advance(&mut self, count: usize) {
    assert!(count <= self.inline_range().len(), "advance past end");
    // start + count <= end <= MAX_INLINE.
    self.start_or_len += count as u8;
  }
}

#[cfg(test)]
mod tests;
