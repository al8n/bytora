use super::*;
use core::{mem::offset_of, ptr};

#[test]
fn cursor_follows_the_larger_of_the_byte_payload_and_external_prefix() {
  fn check<const N: usize>() {
    let body_bytes = N.max(size_of::<ExternalPrefix>());
    assert_eq!(offset_of!(Layout<u8, N>, body), 0);
    assert_eq!(offset_of!(Layout<u8, N>, start_or_len), body_bytes);
    assert_eq!(offset_of!(Layout<u8, N>, tag), body_bytes + 1);
  }
  assert_eq!(align_of::<ExternalPrefix>(), 1);
  assert_eq!(
    size_of::<ExternalPrefix>(),
    2 * size_of::<*const ()>() + LEN_LOW_BYTES
  );
  check::<0>();
  check::<22>();
  check::<30>();
  check::<62>();
}

#[test]
fn external_lengths_round_trip_through_every_bit() {
  let make = |len| Layout::<u8, 22>::try_external(ptr::null(), ptr::null(), len);
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

#[test]
fn external_parts_keep_their_bytes_and_pointer_provenance() {
  let values = [11u16, 13];
  let owner = 17u32;
  let data = ptr::from_ref(&values[1]);
  let owner_ptr = ptr::from_ref(&owner);
  let mut bytes: [u8; size_of::<usize>()] = core::array::from_fn(|i| i as u8 + 1);
  bytes[LEN_LOW_BYTES + 1] = 0x65;
  let len = usize::from_le_bytes(bytes);
  let repr = Layout::<u16, 62>::try_external(data, owner_ptr.cast(), len).unwrap();
  let moved = core::hint::black_box(repr);
  assert_eq!(moved.len(), len);
  assert_eq!(moved.start_or_len, bytes[LEN_LOW_BYTES]);
  assert_eq!(moved.tag.get(), 0xe5);
  // SAFETY: try_external selected and initialized this union member. Read its packed
  // fields by value so that no unaligned pointer reference is formed.
  let ExternalPrefix {
    data: stored_data,
    owner: stored_owner,
    len_low,
  } = unsafe { moved.body.external };
  assert_eq!(len_low, bytes[..LEN_LOW_BYTES]);
  let stored_data = stored_data.cast::<u16>();
  let stored_owner = stored_owner.cast::<u32>();
  assert!(ptr::eq(data, stored_data));
  assert!(ptr::eq(owner_ptr, stored_owner));
  // SAFETY: These are the pointers to the live locals supplied to try_external.
  unsafe {
    assert_eq!(*stored_data, 13);
    assert_eq!(*stored_owner, 17);
  }
}
