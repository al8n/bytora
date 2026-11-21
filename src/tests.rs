use super::{AssocInlineSize, MAX_HANDLE_BYTES, SliceMutN, SliceN};
use core::{hint::black_box, ptr};

#[test]
fn full_byte_windows_move_and_advance() {
  fn check<const BYTES: usize>(capacity: usize)
  where
    [u8; BYTES]: AssocInlineSize,
  {
    let source = core::array::from_fn::<_, BYTES, _>(|i| i as u8);
    assert_eq!(SliceN::<u8, BYTES>::INLINE_CAPACITY, capacity);
    assert!(SliceN::<u8, BYTES>::try_from_slice(&source[..=capacity]).is_none());
    let mut bytes = SliceN::<u8, BYTES>::try_from_slice(&source[..capacity]).unwrap();
    let base = bytes.as_slice().as_ptr();
    let start = capacity / 2;
    bytes.advance(start);
    assert_eq!(bytes.as_slice().as_ptr(), base.wrapping_add(start));
    let mut moved = black_box(Some(bytes)).unwrap();
    assert_eq!(moved.len(), capacity - start);
    assert_eq!(moved.as_slice(), &source[start..capacity]);
    moved.advance(capacity - start);
    moved.advance(0);
    assert!(moved.is_empty());
    assert!(moved.as_slice().is_empty());
  }
  #[cfg(target_pointer_width = "32")]
  {
    check::<4>(3);
    check::<12>(10);
    check::<20>(18);
    check::<132>(128);
    check::<2044>(2040);
    check::<4100>(4096);
  }
  check::<8>(7);
  check::<16>(14);
  check::<24>(22);
  check::<32>(30);
  check::<40>(38);
  check::<48>(46);
  check::<56>(54);
  check::<64>(62);
  check::<80>(78);
  check::<128>(126);
  check::<136>(132);
  check::<256>(252);
  check::<264>(260);
  check::<2048>(2044);
  check::<2056>(2052);
  check::<4096>(4092);
  check::<4104>(4100);
  check::<MAX_HANDLE_BYTES>(8188);
}

#[test]
fn large_payloads_are_inline_within_the_handle_budget() {
  let source = core::array::from_fn::<_, 8189, _>(|i| i as u8);
  let mut four_kib = SliceMutN::<u8, 4104>::try_from_slice(&source[..4096]).unwrap();
  assert_eq!(
    four_kib.as_slice().as_ptr(),
    ptr::from_ref(&four_kib).cast::<u8>()
  );
  four_kib.advance(4095);
  assert_eq!(four_kib.as_mut_slice(), &source[4095..4096]);

  let mut eight_kib = SliceMutN::<u8, MAX_HANDLE_BYTES>::try_from_slice(&source[..8188]).unwrap();
  assert_eq!(
    eight_kib.as_slice().as_ptr(),
    ptr::from_ref(&eight_kib).cast::<u8>()
  );
  eight_kib.advance(4096);
  let mut moved = black_box(Some(eight_kib)).unwrap();
  assert_eq!(
    moved.as_slice().as_ptr(),
    ptr::from_ref(&moved).cast::<u8>().wrapping_add(4096)
  );
  assert_eq!(moved.as_slice(), &source[4096..8188]);
  moved.as_mut_slice()[4091] = 17;
  assert_eq!(moved.as_slice()[4091], 17);
  moved.advance(4092);
  assert!(moved.is_empty());

  assert!(SliceN::<u8, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  assert!(SliceMutN::<u8, MAX_HANDLE_BYTES>::try_from_slice(&source).is_none());
  assert_eq!(SliceN::<u32, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 2047);
  assert_eq!(SliceN::<(), MAX_HANDLE_BYTES>::INLINE_CAPACITY, 8188);
  assert!(SliceN::<(), MAX_HANDLE_BYTES>::try_from_slice(&[(); 8189]).is_none());
  let mut zsts = SliceMutN::<(), MAX_HANDLE_BYTES>::try_from_slice(&[(); 8188]).unwrap();
  zsts.advance(8188);
  assert!(zsts.is_empty());
}

#[test]
fn tiny_cursors_represent_every_window() {
  fn check<const BYTES: usize>()
  where
    [u8; BYTES]: AssocInlineSize,
  {
    let source = *b"abcdefg";
    for end in 0..=SliceMutN::<u8, BYTES>::INLINE_CAPACITY {
      for start in 0..=end {
        let mut values = SliceMutN::<u8, BYTES>::try_from_slice(&source[..end]).unwrap();
        values.advance(start);
        assert_eq!(values.as_mut_slice(), &source[start..end]);
        assert_eq!(values.len(), end - start);
      }
    }
  }
  #[cfg(target_pointer_width = "32")]
  check::<4>();
  check::<8>();
}

#[test]
fn u32_capacity_and_alignment_follow_the_byte_budget() {
  #[cfg(target_pointer_width = "32")]
  assert_eq!(SliceN::<u32, 4>::INLINE_CAPACITY, 0);
  assert_eq!(SliceN::<u32, 8>::INLINE_CAPACITY, 1);
  assert_eq!(SliceN::<u32, 24>::INLINE_CAPACITY, 5);
  assert_eq!(SliceMutN::<u32, 32>::INLINE_CAPACITY, 7);
  assert_eq!(SliceMutN::<u32, 64>::INLINE_CAPACITY, 15);
  assert!(SliceN::<u32, 24>::try_from_slice(&[0; 6]).is_none());
  let mut values = SliceMutN::<u32, 24>::try_from_slice(&[2, 3, 5, 7, 11]).unwrap();
  values.advance(2);
  values.as_mut_slice()[1] = 13;
  let moved = black_box(values);
  assert!(moved.as_slice().as_ptr().is_aligned());
  assert_eq!(moved.as_slice(), &[5, 13, 11]);
}

#[test]
fn over_aligned_elements_support_aligned_empty_windows() {
  #[repr(align(64))]
  #[derive(Copy, Clone)]
  struct Aligned {
    _byte: u8,
  }

  assert_eq!(size_of::<Option<SliceMutN<Aligned, 8>>>(), 8);
  assert_eq!(size_of::<Option<SliceN<Aligned, 256>>>(), 256);
  assert_eq!(SliceN::<Aligned, 256>::INLINE_CAPACITY, 0);
  let value = Aligned { _byte: 3 };
  assert!(SliceN::<Aligned, 256>::try_from_slice(&[value]).is_none());
  let mut empty = SliceMutN::<Aligned, 8>::try_from_slice(&[]).unwrap();
  empty.advance(0);
  assert!(empty.is_empty());
  assert!(empty.as_slice().as_ptr().is_aligned());
  assert!(empty.as_mut_slice().as_mut_ptr().is_aligned());
}

#[test]
fn aligned_zero_sized_elements_use_the_payload_byte_count() {
  #[repr(align(64))]
  #[derive(Clone, Copy)]
  struct AlignedZst;

  assert_eq!(SliceN::<AlignedZst, 8>::INLINE_CAPACITY, 7);
  assert!(SliceN::<AlignedZst, 8>::try_from_slice(&[AlignedZst; 8]).is_none());
  let mut zsts = SliceMutN::<AlignedZst, 8>::try_from_slice(&[AlignedZst; 7]).unwrap();
  assert!(zsts.as_slice().as_ptr().is_aligned());
  zsts.advance(2);
  assert_eq!(zsts.as_mut_slice().len(), 5);
  assert!(zsts.as_mut_slice().as_mut_ptr().is_aligned());
  zsts.advance(5);
  assert!(zsts.as_mut_slice().is_empty());
}

#[test]
fn pointer_elements_survive_moves_advances_and_writes() {
  let numbers = [2, 3];
  let mut refs = SliceMutN::<&i32, 24>::try_from_slice(&[&numbers[0], &numbers[1]]).unwrap();
  refs.advance(1);
  assert_eq!(*refs.as_slice()[0], 3);
  refs.as_mut_slice()[0] = &numbers[0];
  let moved = black_box(Some(refs)).unwrap();
  assert!(ptr::eq(moved.as_slice()[0], &numbers[0]));
  assert_eq!(*moved.as_slice()[0], 2);
}

#[test]
fn copies_preserve_pointer_fields_and_uninitialized_padding() {
  #[repr(C)]
  #[derive(Clone, Copy)]
  struct Padded<'a> {
    byte: u8,
    number: &'a u32,
  }

  let numbers = [13, 17];
  let first = Padded {
    byte: 3,
    number: &numbers[0],
  };
  let second = Padded {
    byte: 5,
    number: &numbers[1],
  };
  let mut values = SliceMutN::<Padded<'_>, 48>::try_from_slice(&[first, second]).unwrap();
  values.advance(1);
  let mut moved = black_box(Some(values)).unwrap();
  assert_eq!(moved.as_slice()[0].byte, 5);
  assert_eq!(*moved.as_slice()[0].number, 17);
  moved.as_mut_slice()[0] = first;
  let again = black_box(moved);
  assert!(ptr::eq(again.as_slice()[0].number, &numbers[0]));
  assert_eq!(*again.as_slice()[0].number, 13);
}
