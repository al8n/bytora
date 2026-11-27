use bytora::VectorMutN;

fn main() {
  let mut bytes = VectorMutN::<u8, 24>::try_from_slice(b"head:body").unwrap();
  bytes.advance(5);
  bytes.as_mut_slice().make_ascii_uppercase();
  assert_eq!(bytes.as_slice(), b"BODY");
}
