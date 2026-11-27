//! The byte aliases, and the traits that only byte handles have.

use bytora::{
  Bytes, BytesMut, BytesMutN, BytesN, DefaultSize, InlineSize, Vector, VectorMut, VectorMutN,
  VectorN,
};
use core::cmp::Ordering::{self, Equal, Greater, Less};
use std::panic::catch_unwind;

// Checks `==` and `partial_cmp` in both directions between a handle and another value.
macro_rules! assert_compares {
  ($handle:expr, $other:expr, $order:expr) => {{
    let (handle, other, order): (_, _, Ordering) = (&$handle, &$other, $order);
    assert_eq!(*handle == *other, order.is_eq());
    assert_eq!(*other == *handle, order.is_eq());
    assert_eq!(handle.partial_cmp(other), Some(order));
    assert_eq!(other.partial_cmp(handle), Some(order.reverse()));
  }};
}

fn byte_count(bytes: &[u8]) -> usize {
  bytes.len()
}

#[test]
fn the_default_aliases_need_no_type_annotation() {
  // Each value flows only into uses that name no budget.
  let empty = Bytes::new();
  assert_eq!(byte_count(&empty), 0);
  let fallback = Bytes::default();
  assert_eq!(byte_count(&fallback), 0);
  let copied = Bytes::copy_from_slice(b"abc");
  assert_eq!(byte_count(&copied), 3);
  let mut buffer = BytesMut::new();
  buffer.extend_from_slice(b"abc");
  assert_eq!(byte_count(&buffer), 3);
  let reserved = BytesMut::with_capacity(4);
  assert_eq!(byte_count(&reserved), 0);
  let zeroed = BytesMut::zeroed(3);
  assert_eq!(byte_count(&zeroed), 3);
}

#[test]
fn the_aliases_name_the_byte_handles() {
  let read: Vector<u8, DefaultSize> = Bytes::new();
  let write: VectorMut<u8, DefaultSize> = BytesMut::new();
  let wide: VectorN<u8, 24> = BytesN::<24>::new();
  let wide_write: VectorMutN<u8, 24> = BytesMutN::<24>::new();
  assert!(read.is_empty() && write.is_empty() && wide.is_empty() && wide_write.is_empty());
  assert_eq!(DefaultSize::HANDLE_BYTES, 4 * size_of::<usize>());
  assert_eq!(size_of::<Bytes>(), size_of::<bytes::Bytes>());
  assert_eq!(size_of::<BytesMut>(), size_of::<bytes::BytesMut>());
}

#[test]
fn byte_handles_compare_with_text() {
  let handle = Bytes::from_static(b"bcd");
  let buffer = BytesMut::from("bcd");
  for (text, order) in [
    ("bcd", Equal),
    ("bce", Less),
    ("bcc", Greater),
    ("bc", Greater),
    ("bcde", Less),
  ] {
    assert_compares!(handle, *text, order);
    assert_compares!(handle, text, order);
    assert_compares!(buffer, *text, order);
    assert_compares!(buffer, text, order);
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn byte_handles_compare_with_strings() {
  let handle = Bytes::from_static(b"bcd");
  let buffer = BytesMut::from(&b"bcd"[..]);
  for (text, order) in [("bcd", Equal), ("bce", Less), ("bc", Greater)] {
    let owned = String::from(text);
    assert_compares!(handle, owned, order);
    assert_compares!(buffer, owned, order);
  }
}

#[test]
fn hex_formatting_matches_bytes() {
  const DATA: &[u8] = b"\x00\x0f\xa0\xff ok";
  let oracle = bytes::Bytes::from_static(DATA);
  let handle = Bytes::from_static(DATA);
  let buffer = BytesMut::from(DATA);
  assert_eq!(format!("{handle:x}"), "000fa0ff206f6b");
  for (ours, theirs) in [
    (format!("{handle:x}"), format!("{oracle:x}")),
    (format!("{handle:X}"), format!("{oracle:X}")),
    (format!("{handle:#x}"), format!("{oracle:#x}")),
    (format!("{buffer:x}"), format!("{oracle:x}")),
    (format!("{buffer:X}"), format!("{oracle:X}")),
  ] {
    assert_eq!(ours, theirs);
  }
}

#[test]
fn static_bytes_and_text_convert_without_copying() {
  static TEXT: &str = "hello";
  let text = Bytes::from(TEXT);
  assert_eq!(text.as_ptr(), TEXT.as_ptr());
  assert_eq!(text, *"hello");
  let raw = Bytes::from(TEXT.as_bytes());
  assert_eq!(raw.as_ptr(), TEXT.as_ptr());
  assert_eq!(raw, text);
}

#[test]
fn buffers_copy_bytes_and_text_and_freeze_into_handles() {
  let buffer = BytesMut::from(&b"abc"[..]);
  assert_eq!(buffer, *b"abc");
  let text = BytesMut::from("abc");
  assert_eq!(text, buffer);
  let frozen = Bytes::from(buffer);
  assert_eq!(frozen, *b"abc");

  // A buffer that cannot grow panics on bytes past its inline capacity.
  type Tiny = BytesMutN<{ size_of::<usize>() }>;
  let long = [7u8; 64];
  assert!(catch_unwind(|| Tiny::from(&long[..])).is_err());
  let copied = catch_unwind(|| BytesMut::from(&long[..]));
  if cfg!(any(feature = "std", feature = "alloc")) {
    assert_eq!(copied.unwrap(), long);
  } else {
    assert!(copied.is_err());
  }
}

#[cfg(bytora_atomic)]
#[test]
fn heap_buffers_and_strings_convert_without_copying() {
  let mut buffer = BytesMut::with_capacity(64);
  buffer.extend_from_slice(&[5; 40]);
  let base = buffer.as_ptr();
  let frozen = Bytes::from(buffer);
  assert_eq!(frozen.as_ptr(), base);

  let text = String::from("a string that is longer than the inline payload");
  let base = text.as_ptr();
  let shared = Bytes::from(text);
  assert_eq!(shared.as_ptr(), base);
  assert_eq!(shared, *"a string that is longer than the inline payload");
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn buffers_convert_into_byte_vectors() {
  let values: Vec<u8> = BytesMut::from("xyz").into();
  assert_eq!(values, b"xyz");
}
