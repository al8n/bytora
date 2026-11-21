use smol_slice::SliceMut;

fn main() {
  let mut bytes = SliceMut::<u8, 22>::try_from_slice(b"head:body").unwrap();
  bytes.advance(5);
  bytes.as_mut_slice().make_ascii_uppercase();
  assert_eq!(bytes.as_slice(), b"BODY");
}
