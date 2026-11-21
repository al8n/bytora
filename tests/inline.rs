use smol_slice::{Slice, SliceMut, SupportedBytes};

#[test]
fn handles_own_their_copied_elements() {
  let mut source = [2, 3, 5, 7];
  let mut read = Slice::<u32, 24>::try_from_slice(&source).unwrap();
  let mut write = SliceMut::<u32, 24>::try_from_slice(&source).unwrap();
  source.fill(0);
  read.advance(2);
  write.advance(1);
  write.as_mut_slice()[1] = 11;
  assert_eq!(read.as_slice(), &[5, 7]);
  assert_eq!(write.as_slice(), &[3, 11, 7]);
}

#[test]
fn failed_advances_leave_the_window_intact() {
  fn check<const BYTES: usize>(source: &[u8], start: usize)
  where
    [(); BYTES]: SupportedBytes,
  {
    let mut values = SliceMut::<u8, BYTES>::try_from_slice(source).unwrap();
    values.advance(start);
    for count in [values.len() + 1, usize::MAX] {
      assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.advance(count))).is_err()
      );
      assert_eq!(values.as_mut_slice(), &source[start..]);
    }
  }
  check::<8>(b"abcdefg", 4);
  check::<24>(b"abcdefg", 4);
  let wide: [u8; 521] = core::array::from_fn(|i| i as u8);
  for start in [254, 255, 256, 257, 511, 512, 521] {
    check::<536>(&wide, start);
  }
}
