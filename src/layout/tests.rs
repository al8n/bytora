use super::*;
#[cfg(target_pointer_width = "32")]
use typenum::{Diff, U4, U12, U20, U132};
use typenum::{U8, U16, U24, U32, U64, U128, U136, U2048};

#[test]
fn external_lengths_round_trip_on_every_descriptor_tier() {
  fn check<S: InlineSize>(fits: bool) {
    let make = |len| Layout::<u8, S>::try_external(ptr::null(), ptr::null(), len);
    if !fits {
      assert!(make(0).is_none());
      assert!(make(1).is_none());
      return;
    }
    for bit in 0..usize::BITS - 1 {
      let boundary = 1usize << bit;
      for len in [boundary - 1, boundary, boundary + 1] {
        assert_eq!(make(len).unwrap().len(), len);
      }
    }
    let max = isize::MAX as usize;
    assert_eq!(make(max).unwrap().len(), max);
    assert!(make(max + 1).is_none());
    assert!(make(usize::MAX).is_none());
  }
  #[cfg(target_pointer_width = "32")]
  {
    check::<U4>(false);
    check::<U12>(true);
    check::<U20>(true);
    check::<U132>(true);
    check::<Diff<U2048, U4>>(true);
  }
  check::<typenum::U0>(true);
  check::<U8>(false);
  check::<U16>(size_of::<usize>() == 4);
  check::<U24>(true);
  check::<U32>(true);
  check::<U128>(true);
  check::<U136>(true);
  check::<U2048>(true);
}

#[test]
fn compact_static_windows_cross_the_fast_tag_length_boundary() {
  const BOUNDARY: usize = 1 << (usize::BITS - 8);
  static DATA: [(); BOUNDARY + 1] = [(); BOUNDARY + 1];
  fn check<S: InlineSize>() {
    let mut source = crate::Vector::<(), S>::try_from_static(&DATA).unwrap();
    assert_eq!(
      source.layout.tag == COMPACT_STATIC,
      !Layout::<(), S>::PACKED_LEN
    );
    assert_eq!(source.clone().len(), BOUNDARY + 1);
    source.advance(1);
    assert_eq!(
      source.layout.tag == COMPACT_STATIC,
      !Layout::<(), S>::PACKED_LEN
    );
    assert_eq!(source.len(), BOUNDARY);
    let subrange = source.slice(1..);
    assert_eq!(subrange.layout.tag, COMPACT_STATIC);
    assert!(!subrange.layout.is_inline());
    assert_eq!(subrange.clone().len(), BOUNDARY - 1);
    source.advance(1);
    assert_eq!(source.layout.tag, COMPACT_STATIC);
    assert_eq!(source.as_slice().len(), BOUNDARY - 1);
  }
  check::<typenum::U0>();
  check::<crate::DefaultSize>();
  #[cfg(target_pointer_width = "32")]
  check::<U12>();
  #[cfg(target_pointer_width = "64")]
  check::<U24>();
  check::<U32>();
}

#[test]
fn packed_static_tags_stay_external_across_short_windows() {
  struct Field(u8);
  static DATA: [Field; 64] = [const { Field(42) }; 64];
  for len in [0, 1, 7, 22, 23, 31, 32, 63, 64] {
    let mut source =
      crate::VectorN::<Field, { 3 * size_of::<usize>() }>::try_from_static(&DATA[..len]).unwrap();
    assert_eq!(source.layout.tag, COMPACT_STATIC);
    assert!(!source.layout.is_inline());
    assert_eq!(source.as_slice().as_ptr(), DATA.as_ptr());
    let start = len / 2;
    let window = source.slice(start..);
    assert_eq!(window.layout.tag, COMPACT_STATIC);
    assert_eq!(
      window.as_slice().as_ptr(),
      DATA.as_ptr().wrapping_add(start)
    );
    assert_eq!(window.len(), len - start);
    if let Some(field) = window.as_slice().first() {
      assert_eq!(field.0, 42);
    }
    source.advance(start);
    let clone = source.clone();
    assert_eq!(clone.layout.tag, COMPACT_STATIC);
    assert_eq!(clone.as_slice().as_ptr(), DATA.as_ptr().wrapping_add(start));
    assert_eq!(clone.len(), len - start);
  }
}

#[test]
fn moved_external_prefix_preserves_pointer_provenance() {
  fn check<S: InlineSize>() {
    let values = [11u16, 13];
    let owner = 17u32;
    let data = ptr::from_ref(&values[1]);
    let owner_ptr = ptr::from_ref(&owner);
    let mut bytes = core::array::from_fn(|i| i as u8 + 1);
    bytes[size_of::<usize>() - 1] = 0x65;
    let len = usize::from_le_bytes(bytes);
    let repr = Layout::<u16, S>::try_external(data, owner_ptr.cast(), len).unwrap();
    let moved = core::hint::black_box(Some(repr)).unwrap();
    assert_eq!(moved.len(), len);
    if Layout::<u16, S>::PACKED_LEN {
      assert_eq!(moved.cursor.as_ref()[0], bytes[size_of::<usize>() - 2]);
      assert_eq!(moved.tag.get(), 0xe5);
    } else {
      assert_eq!(moved.tag.get(), EXTERNAL);
    }
    let stored_data = moved.data().cast::<u16>();
    // SAFETY: This private descriptor stores the unchanged pointer to the local.
    let stored_owner = unsafe { (*moved.external_ptr()).owner.cast::<u32>() };
    assert!(ptr::eq(data, stored_data));
    assert!(ptr::eq(owner_ptr, stored_owner));
    // SAFETY: These are the original pointers to the live locals above.
    unsafe {
      assert_eq!(*stored_data, 13);
      assert_eq!(*stored_owner, 17);
    }
  }
  #[cfg(target_pointer_width = "32")]
  {
    check::<U12>();
    check::<U16>();
    check::<U20>();
    check::<U132>();
    check::<Diff<U2048, U4>>();
  }
  check::<typenum::U0>();
  check::<U24>();
  check::<U64>();
  check::<U136>();
  check::<U2048>();
}

#[test]
#[should_panic(expected = "expected inline storage")]
fn external_descriptors_cannot_be_borrowed_as_inline_elements() {
  let repr = Layout::<u8, U24>::try_external(ptr::null(), ptr::null(), 1).unwrap();
  let _ = repr.as_slice();
}

#[test]
#[should_panic(expected = "expected inline storage")]
fn external_descriptors_cannot_be_cloned_as_inline_elements() {
  let repr = Layout::<u8, U24>::try_external(ptr::null(), ptr::null(), 1).unwrap();
  let _ = repr.clone_inline();
}
