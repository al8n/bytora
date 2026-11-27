//! The standard traits of both handles, checked against the slices they view.

use bytora::{Vector, VectorMut, VectorMutN, VectorN};
use core::cmp::Ordering::{self, Equal, Greater, Less};
use std::panic::{AssertUnwindSafe, catch_unwind};

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

#[test]
fn deref_as_ref_and_borrow_view_the_current_window() {
  use std::borrow::{Borrow, BorrowMut};

  let mut vector = VectorN::<u16, 24>::try_from_slice(&[1, 2, 3, 4]).unwrap();
  vector.advance(1);
  assert_eq!(&*vector, &[2, 3, 4]);
  assert_eq!(vector.iter().sum::<u16>(), 9);
  assert_eq!(vector.first(), Some(&2));
  assert_eq!(AsRef::<[u16]>::as_ref(&vector), &[2, 3, 4]);
  assert_eq!(Borrow::<[u16]>::borrow(&vector), &[2, 3, 4]);

  let mut buffer = VectorMutN::<u16, 24>::try_from_slice(&[1, 2, 3, 4]).unwrap();
  buffer.advance(1);
  buffer[0] = 20;
  AsMut::<[u16]>::as_mut(&mut buffer)[1] = 30;
  BorrowMut::<[u16]>::borrow_mut(&mut buffer)[2] = 40;
  assert_eq!(&*buffer, &[20, 30, 40]);
  buffer.reverse();
  assert_eq!(AsRef::<[u16]>::as_ref(&buffer), &[40, 30, 20]);
  assert_eq!(Borrow::<[u16]>::borrow(&buffer), &[40, 30, 20]);
}

#[test]
fn handles_compare_with_slices_arrays_and_each_other() {
  let vector = VectorN::<u8, 24>::try_from_slice(b"bcd").unwrap();
  let buffer = VectorMutN::<u8, 24>::try_from_slice(b"bcd").unwrap();
  for (other, order) in [
    (&b"bcd"[..], Equal),
    (&b"bce"[..], Less),
    (&b"bcc"[..], Greater),
    (&b"bc"[..], Greater),
    (&b"bcde"[..], Less),
  ] {
    assert_compares!(vector, *other, order);
    assert_compares!(vector, other, order);
    assert_compares!(buffer, *other, order);
    assert_compares!(buffer, other, order);
  }
  assert_compares!(vector, *b"bcd", Equal);
  assert_compares!(vector, *b"bca", Greater);
  assert_compares!(buffer, *b"bcd", Equal);
  assert_compares!(buffer, *b"bcz", Less);

  assert_compares!(vector, buffer, Equal);
  let larger = VectorN::<u8, 24>::try_from_static(b"bcz").unwrap();
  assert_compares!(vector, larger, Less);
  let smaller = VectorMutN::<u8, 24>::try_from_slice(b"b").unwrap();
  assert_compares!(buffer, smaller, Greater);
  assert_compares!(larger, buffer, Greater);
  assert_compares!(smaller, vector, Less);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn handles_compare_with_vectors() {
  let vector = VectorN::<u16, 24>::try_from_slice(&[1, 2]).unwrap();
  let buffer = VectorMutN::<u16, 24>::try_from_slice(&[1, 2]).unwrap();
  for (other, order) in [
    ([1u16, 2].to_vec(), Equal),
    ([1u16, 3].to_vec(), Less),
    ([1u16].to_vec(), Greater),
  ] {
    assert_compares!(vector, other, order);
    assert_compares!(buffer, other, order);
  }
}

#[test]
fn partial_orders_follow_partially_ordered_elements() {
  let nan = VectorN::<f32, 24>::try_from_slice(&[f32::NAN]).unwrap();
  let same = nan.clone();
  assert!(nan != same);
  assert_eq!(nan.partial_cmp(&same), None);
  assert_eq!(nan.partial_cmp(&[1.0f32][..]), None);
  let one = VectorMutN::<f32, 24>::try_from_slice(&[1.0]).unwrap();
  assert_eq!(one.partial_cmp(&[2.0f32][..]), Some(Less));
  assert_eq!(one.partial_cmp(&nan), None);
}

#[test]
fn order_hash_and_borrow_follow_the_visible_elements() {
  use std::{
    collections::{BTreeSet, HashSet},
    hash::{BuildHasher, BuildHasherDefault, DefaultHasher},
  };

  let hasher = BuildHasherDefault::<DefaultHasher>::default();
  let mut inline = VectorN::<u8, 24>::try_from_slice(b"xabc").unwrap();
  inline.advance(1);
  let borrowed = VectorN::<u8, 24>::try_from_static(b"abc").unwrap();
  assert_eq!(inline, borrowed);
  assert_eq!(inline.cmp(&borrowed), Equal);
  assert_eq!(hasher.hash_one(&inline), hasher.hash_one(b"abc".as_slice()));
  assert_eq!(hasher.hash_one(&inline), hasher.hash_one(&borrowed));

  let other = VectorN::<u8, 24>::try_from_slice(b"abd").unwrap();
  assert_eq!(inline.cmp(&other), Less);
  let set: HashSet<_> = [inline.clone(), other.clone()].into_iter().collect();
  assert!(set.contains(b"abc".as_slice()));
  assert!(!set.contains(b"ab".as_slice()));
  let ordered: BTreeSet<_> = [other, inline].into_iter().collect();
  assert!(ordered.contains(b"abd".as_slice()));
  assert_eq!(
    ordered.first().map(|first| first.as_slice()),
    Some(&b"abc"[..])
  );

  let mut buffer = VectorMutN::<u8, 24>::try_from_slice(b"xabc").unwrap();
  buffer.advance(1);
  let copy = VectorMutN::<u8, 24>::try_from_slice(b"abc").unwrap();
  assert_eq!(buffer, copy);
  assert_eq!(buffer.cmp(&copy), Equal);
  assert_eq!(hasher.hash_one(&buffer), hasher.hash_one(b"abc".as_slice()));
  let next = VectorMutN::<u8, 24>::try_from_slice(b"abd").unwrap();
  assert_eq!(buffer.cmp(&next), Less);
  let set: HashSet<_> = [buffer, next].into_iter().collect();
  assert!(set.contains(b"abd".as_slice()));
  let ordered: BTreeSet<_> = set.into_iter().collect();
  assert_eq!(
    ordered.last().map(|last| last.as_slice()),
    Some(&b"abd"[..])
  );
}

#[test]
fn debug_prints_the_elements_as_a_slice_does() {
  let vector = VectorN::<u8, 24>::try_from_slice(b"a\0'").unwrap();
  let slice = vector.as_slice();
  assert_eq!(format!("{vector:?}"), format!("{slice:?}"));
  assert_eq!(format!("{vector:?}"), "[97, 0, 39]");
  // The `b"…"` text is one call away, with the escapes of std.
  assert_eq!(vector.escape_ascii().to_string(), "a\\x00\\'");

  let buffer = VectorMutN::<u16, 24>::try_from_slice(&[1, 2]).unwrap();
  assert_eq!(format!("{buffer:?}"), "[1, 2]");
  assert_eq!(format!("{buffer:#?}"), format!("{:#?}", [1u16, 2]));
}

#[test]
fn references_iterate_over_the_visible_elements() {
  let mut vector = VectorN::<u8, 24>::try_from_slice(b"xabc").unwrap();
  vector.advance(1);
  let mut seen = Vec::new();
  for byte in &vector {
    seen.push(*byte);
  }
  assert_eq!(seen, b"abc");

  let mut buffer = VectorMutN::<u8, 24>::try_from_slice(b"xabc").unwrap();
  buffer.advance(1);
  for byte in &mut buffer {
    *byte = byte.to_ascii_uppercase();
  }
  let mut seen = Vec::new();
  for byte in &buffer {
    seen.push(*byte);
  }
  assert_eq!(seen, b"ABC");
}

#[test]
fn extend_appends_elements_references_and_handles() {
  let mut buffer = VectorMutN::<u8, 24>::new();
  buffer.extend([1u8, 2]);
  buffer.extend(&[3u8, 4]);
  buffer.extend([
    Vector::<u8>::try_from_slice(&[5, 6]).unwrap(),
    Vector::<u8>::from_static(&[7]),
  ]);
  assert_eq!(buffer, [1, 2, 3, 4, 5, 6, 7]);
  assert_eq!(buffer.capacity(), VectorMutN::<u8, 24>::INLINE_CAPACITY);
}

#[test]
fn extend_past_a_bounded_buffer_panics_after_the_elements_that_fit() {
  type Tiny = VectorMutN<u8, { size_of::<usize>() }>;
  let capacity = Tiny::INLINE_CAPACITY;
  let mut buffer = Tiny::new();
  // The lower bound of the size hint is reserved first, so nothing is appended.
  let hinted = catch_unwind(AssertUnwindSafe(|| {
    buffer.extend(std::iter::repeat_n(1u8, capacity + 1));
  }));
  assert!(hinted.is_err());
  assert!(buffer.is_empty());

  // Without a lower bound, the elements that fit are appended before the panic.
  let mut next = 0u8;
  let unhinted = std::iter::from_fn(move || {
    next += 1;
    Some(next)
  })
  .take(capacity + 1);
  let partial = catch_unwind(AssertUnwindSafe(|| buffer.extend(unhinted)));
  assert!(partial.is_err());
  assert_eq!(buffer.len(), capacity);
  assert_eq!(buffer[capacity - 1], u8::try_from(capacity).unwrap());
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn extend_moves_elements_into_heap_storage() {
  let mut words = VectorMutN::<String, 0>::new();
  words.extend(["one".to_owned(), "two".to_owned()]);
  words.extend(std::iter::once("three".to_owned()));
  assert_eq!(words.as_slice(), ["one", "two", "three"]);

  let mut bytes = VectorMut::<u8>::new();
  bytes.extend(std::iter::repeat_n(9u8, 40));
  assert_eq!(bytes.as_slice(), [9; 40]);
}

#[test]
fn collect_builds_both_handles() {
  let buffer: VectorMutN<u8, 24> = (1..=3u8).collect();
  assert_eq!(buffer, [1, 2, 3]);
  let copies: VectorMutN<u8, 24> = [4u8, 5].iter().collect();
  assert_eq!(copies, [4, 5]);
  let frozen: VectorN<u8, 24> = (1..=3u8).collect();
  assert_eq!(frozen, buffer);

  // Past the inline capacity a buffer grows on the heap, and only shared storage freezes it.
  let grown = catch_unwind(|| (0..40u8).collect::<VectorMut<u8>>());
  if cfg!(any(feature = "std", feature = "alloc")) {
    assert_eq!(grown.unwrap().len(), 40);
  } else {
    assert!(grown.is_err());
  }
  let shared = catch_unwind(|| (0..40u8).collect::<Vector<u8>>());
  #[cfg(bytora_atomic)]
  assert_eq!(shared.unwrap(), (0..40u8).collect::<Vec<_>>());
  #[cfg(not(bytora_atomic))]
  assert!(shared.is_err());
}

#[test]
fn default_and_clone_give_independent_buffers() {
  let empty = VectorMutN::<u8, 24>::default();
  assert!(empty.is_empty());
  assert_eq!(empty.capacity(), VectorMutN::<u8, 24>::INLINE_CAPACITY);

  let mut source = VectorMutN::<u8, 24>::try_from_slice(b"abcdef").unwrap();
  source.advance(2);
  let mut copy = source.clone();
  copy[0] = b'X';
  assert_eq!(source, *b"cdef");
  assert_eq!(copy, *b"Xdef");
  assert_ne!(copy.as_ptr(), source.as_ptr());

  // A buffer thawed from an empty generic handle has no inline capacity, and nor has its clone.
  let thawed = Vector::<u8>::new().try_into_mut().ok().unwrap();
  assert_eq!(thawed.capacity(), 0);
  assert_eq!(thawed.clone().capacity(), 0);
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn clones_of_heap_buffers_copy_into_a_new_allocation() {
  let words = VectorMutN::<String, 0>::from(vec!["a".to_owned(), "b".to_owned()]);
  let copy = words.clone();
  assert_eq!(copy, words);
  assert_ne!(copy.as_ptr(), words.as_ptr());
  assert_eq!(VectorMutN::<String, 0>::default().capacity(), 0);

  let mut bytes = VectorMut::<u8>::with_capacity(64);
  bytes.extend_from_slice(&[7; 40]);
  let copy = bytes.clone();
  assert_eq!(copy, bytes);
  assert_ne!(copy.as_ptr(), bytes.as_ptr());
}

#[test]
fn static_slices_convert_without_copying() {
  static WORDS: [u16; 3] = [1, 2, 3];
  let vector: Vector<u16> = Vector::from(&WORDS[..]);
  assert_eq!(vector.as_ptr(), WORDS.as_ptr());
  assert_eq!(vector, WORDS);
}

#[cfg(bytora_atomic)]
#[test]
fn vectors_convert_into_both_four_word_handles_without_copying() {
  let words = vec!["one".to_owned(), "two".to_owned()];
  let base = words.as_ptr();
  let shared: Vector<String> = Vector::from(words);
  assert_eq!(shared.as_ptr(), base);
  assert_eq!(shared.clone().as_ptr(), base);

  let bytes = vec![1u8, 2, 3];
  let base = bytes.as_ptr();
  let zero = VectorN::<u8, 0>::from(bytes);
  assert_eq!(zero.as_ptr(), base);
  assert_eq!(zero, [1, 2, 3]);

  // An unallocated vector gives an empty handle that owns nothing.
  let empty: Vector<u8> = Vector::from(Vec::new());
  assert!(empty.is_empty());
  assert!(empty.is_unique());
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn buffers_convert_into_vectors() {
  let buffer = VectorMutN::<u16, 24>::try_from_slice(&[1, 2]).unwrap();
  assert_eq!(Vec::from(buffer), [1, 2]);

  let words = VectorMutN::<String, 0>::from(vec!["w".to_owned()]);
  let base = words.as_ptr();
  let values: Vec<String> = words.into();
  assert_eq!(values.as_ptr(), base);
}
