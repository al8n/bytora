use smol_slice::{Slice, SliceMut};

#[test]
fn handles_own_their_copied_elements() {
  let mut source = [2, 3, 5, 7];
  let mut read = Slice::<u32, 4>::try_from_slice(&source).unwrap();
  let mut write = SliceMut::<u32, 4>::try_from_slice(&source).unwrap();
  source.fill(0);
  read.advance(2);
  write.advance(1);
  write.as_mut_slice()[1] = 11;
  assert_eq!(read.as_slice(), &[5, 7]);
  assert_eq!(write.as_slice(), &[3, 11, 7]);
}
