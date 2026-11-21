use super::*;

#[test]
fn external_lengths_round_trip_on_every_descriptor_tier() {
  fn check<const BYTES: usize>(fits: bool)
  where
    [(); BYTES]: SupportedBytes,
  {
    let make = |len| Layout::<u8, Size<BYTES>>::try_external(ptr::null(), ptr::null(), len);
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
    check::<4>(false);
    check::<12>(true);
    check::<20>(true);
    check::<132>(true);
    check::<2044>(true);
  }
  check::<8>(false);
  check::<16>(size_of::<usize>() == 4);
  check::<24>(true);
  check::<32>(true);
  check::<128>(true);
  check::<136>(true);
  check::<2048>(true);
}

#[test]
fn moved_external_prefix_preserves_pointer_provenance() {
  fn check<const BYTES: usize>()
  where
    [(); BYTES]: SupportedBytes,
  {
    let values = [11u16, 13];
    let owner = 17u32;
    let data = ptr::from_ref(&values[1]);
    let owner_ptr = ptr::from_ref(&owner);
    let mut bytes = core::array::from_fn(|i| i as u8 + 1);
    bytes[LEN_LOW_BYTES + 1] = 0x65;
    let len = usize::from_le_bytes(bytes);
    let repr = Layout::<u16, Size<BYTES>>::try_external(data, owner_ptr.cast(), len).unwrap();
    let moved = core::hint::black_box(Some(repr)).unwrap();
    assert_eq!(moved.len(), len);
    assert_eq!(moved.cursor.as_ref()[0], bytes[LEN_LOW_BYTES]);
    assert_eq!(moved.tag.get(), 0xe5);
    // SAFETY: try_external wrote the packed prefix. Copy its fields by value
    // to avoid taking unaligned references to the pointers.
    let prefix = unsafe { &*moved.payload.as_ptr().cast::<ExternalPrefix>() };
    assert_eq!(prefix.len_low, bytes[..LEN_LOW_BYTES]);
    let stored_data = prefix.data.cast::<u16>();
    let stored_owner = prefix.owner.cast::<u32>();
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
    check::<12>();
    check::<16>();
    check::<20>();
    check::<132>();
    check::<2044>();
  }
  check::<24>();
  check::<64>();
  check::<136>();
  check::<2048>();
}

#[test]
#[should_panic(expected = "expected inline storage")]
fn external_descriptors_cannot_be_borrowed_as_inline_elements() {
  let repr = Layout::<u8, Size<24>>::try_external(ptr::null(), ptr::null(), 1).unwrap();
  let _ = repr.as_slice();
}
