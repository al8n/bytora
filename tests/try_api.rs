//! The non-panicking twins and their errors, against the panicking forms.

use bytora::{
  ByteEditError, Bytes, InlineSize, OutOfBounds, RangeOutOfBounds, TryPutError, TryReserveError,
  Vector, VectorMut, VectorMutN, VectorN, error,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

// The message of the panic that `operation` raises.
fn panic_message(operation: impl FnOnce()) -> String {
  let payload = catch_unwind(AssertUnwindSafe(operation)).expect_err("the operation panics");
  match payload.downcast::<&str>() {
    Ok(message) => (*message).to_owned(),
    Err(payload) => *payload.downcast::<String>().expect("a text panic payload"),
  }
}

#[test]
fn vector_split_twins_return_what_the_panicking_forms_return() {
  // An external source keeps its storage, so both forms point into it alike.
  fn check<S: InlineSize>(source: Vector<u8, S>, external: bool) {
    let len = source.len();
    for at in 0..=len {
      let mut panicking = source.clone();
      let mut fallible = source.clone();
      let expected = panicking.split_off(at);
      let suffix = fallible.try_split_off(at).unwrap();
      assert_eq!((&suffix, &fallible), (&expected, &panicking));
      if external && !suffix.is_empty() {
        assert_eq!(suffix.as_ptr(), expected.as_ptr());
      }

      let mut panicking = source.clone();
      let mut fallible = source.clone();
      let expected = panicking.split_to(at);
      let prefix = fallible.try_split_to(at).unwrap();
      assert_eq!((&prefix, &fallible), (&expected, &panicking));
      if external && !fallible.is_empty() {
        assert_eq!(fallible.as_ptr(), panicking.as_ptr());
      }
    }
    for at in [len + 1, usize::MAX] {
      let mut fallible = source.clone();
      assert_eq!(
        fallible.try_split_off(at).err(),
        Some(OutOfBounds::new(at, len))
      );
      assert_eq!(
        fallible.try_split_to(at).err(),
        Some(OutOfBounds::new(at, len))
      );
      assert_eq!(fallible, source);
      let message = panic_message(|| {
        let _ = source.clone().split_off(at);
      });
      assert_eq!(message, "split index out of bounds");
      let message = panic_message(|| {
        let _ = source.clone().split_to(at);
      });
      assert_eq!(message, "split index out of bounds");
    }
  }

  static DATA: &[u8] = b"abcdef";
  check(Vector::<u8>::try_from_slice(DATA).unwrap(), false);
  check(
    VectorN::<u8, { size_of::<usize>() }>::try_from_slice(&DATA[..3]).unwrap(),
    false,
  );
  check(Vector::<u8>::from_static(DATA), true);
  #[cfg(bytora_atomic)]
  check(Vector::<u8>::from(DATA.to_vec()), true);
}

#[test]
fn mutable_split_to_twin_returns_what_split_to_returns() {
  fn check<S: InlineSize>(source: &VectorMut<u8, S>) {
    let len = source.len();
    for at in 0..=len {
      let mut panicking = source.clone();
      let mut fallible = source.clone();
      let expected = panicking.split_to(at);
      let prefix = fallible.try_split_to(at).unwrap();
      assert_eq!((&prefix, &fallible), (&expected, &panicking));
      assert_eq!(
        (prefix.capacity(), fallible.capacity()),
        (expected.capacity(), panicking.capacity())
      );
    }
    for at in [len + 1, usize::MAX] {
      let mut fallible = source.clone();
      assert_eq!(
        fallible.try_split_to(at).err(),
        Some(OutOfBounds::new(at, len))
      );
      assert_eq!(fallible, *source);
      let message = panic_message(|| {
        let _ = source.clone().split_to(at);
      });
      assert_eq!(message, "split_to past end");
    }
  }

  let mut inline = VectorMutN::<u8, 24>::try_from_slice(b"abcdef").unwrap();
  inline.advance(1);
  check(&inline);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    let mut heap = VectorMutN::<u8, 0>::from(b"abcdef".to_vec());
    heap.advance(1);
    check(&heap);
  }
}

#[test]
fn try_slice_ref_returns_none_exactly_where_slice_ref_panics() {
  static DATA: [u8; 6] = *b"abcdef";
  let window = Vector::from_static(&DATA[1..5]);
  for start in 0..=DATA.len() {
    for end in start..=DATA.len() {
      let subset = &DATA[start..end];
      let fallible = window.try_slice_ref(subset);
      if subset.is_empty() || ((1..=5).contains(&start) && end <= 5) {
        let expected = window.slice_ref(subset);
        if !subset.is_empty() {
          assert_eq!(expected.as_ptr(), subset.as_ptr());
        }
        assert_eq!(fallible, Some(expected));
      } else {
        assert!(fallible.is_none());
        let message = panic_message(|| {
          let _ = window.slice_ref(subset);
        });
        assert_eq!(message, "subset is outside this vector");
      }
    }
  }

  let inline = Vector::<u8>::try_from_slice(b"bcde").unwrap();
  assert_eq!(
    inline.try_slice_ref(&inline[1..3]),
    Some(inline.slice(1..3))
  );
  assert!(inline.try_slice_ref(&DATA[1..3]).is_none());
  assert_eq!(inline.try_slice_ref(&DATA[..0]), Some(Vector::new()));

  let zero_sized = VectorN::<(), 24>::try_from_slice(&[(); 4]).unwrap();
  let all = zero_sized.as_slice();
  assert_eq!(
    zero_sized.try_slice_ref(&all[..2]).map(|part| part.len()),
    Some(2)
  );
  assert!(zero_sized.slice(..2).try_slice_ref(all).is_none());
}

#[test]
fn try_copy_from_slice_errs_exactly_where_copy_from_slice_panics() {
  let short = b"abc";
  assert_eq!(
    Bytes::try_copy_from_slice(short),
    Ok(Bytes::copy_from_slice(short))
  );
  assert_eq!(Bytes::from_slice(short), Bytes::copy_from_slice(short));

  type Tiny = VectorN<u8, { size_of::<usize>() }>;
  let long = [7u8; 64];
  let overflow = TryPutError {
    requested: 64,
    available: Tiny::INLINE_CAPACITY,
  };
  assert_eq!(Tiny::try_copy_from_slice(&long).err(), Some(overflow));
  let message = panic_message(|| {
    let _ = Tiny::copy_from_slice(&long);
  });
  assert_eq!(message, "slice does not fit this handle");
  let message = panic_message(|| {
    let _ = Tiny::from_slice(&long);
  });
  assert_eq!(message, "slice does not fit this handle");

  // Shared storage takes what the inline payload cannot.
  let spilled = Bytes::try_copy_from_slice(&long);
  #[cfg(bytora_atomic)]
  assert_eq!(spilled, Ok(Bytes::from_slice(&long)));
  #[cfg(not(bytora_atomic))]
  assert_eq!(
    spilled.err(),
    Some(TryPutError {
      requested: 64,
      available: Bytes::INLINE_CAPACITY,
    })
  );
}

#[cfg(all(bytora_atomic, target_pointer_width = "32"))]
#[test]
fn two_word_copies_report_the_two_word_vector_limit() {
  type TwoWords = VectorN<u8, 8>;
  if TwoWords::try_from_vec(vec![0; TwoWords::INLINE_CAPACITY + 1]).is_err() {
    // This target keeps no vector in two words.
    return;
  }
  let limit = (1usize << (usize::BITS / 2 - 1)) - 1;
  let input = vec![1u8; limit + 1];
  assert_eq!(
    TwoWords::try_copy_from_slice(&input).err(),
    Some(TryPutError {
      requested: limit + 1,
      available: limit,
    })
  );
}

#[test]
fn errors_print_their_numbers_and_chain_their_sources() {
  use core::error::Error;

  let index = OutOfBounds::new(5, 3);
  assert_eq!((index.requested, index.available), (5, 3));
  assert_eq!(
    index.to_string(),
    "index out of bounds: requested 5 but only 3 available"
  );
  let range = RangeOutOfBounds::new(2, 9, 4);
  assert_eq!((range.start, range.end, range.available), (2, 9, 4));
  assert_eq!(
    range.to_string(),
    "range out of bounds: requested 2..9 but only 4 available"
  );
  let put = TryPutError {
    requested: 8,
    available: 2,
  };
  assert_eq!(
    put.to_string(),
    "Not enough capacity remaining in buffer to write value (requested 8 but only 2 available)"
  );
  for leaf in [&index as &dyn Error, &range, &put] {
    assert!(leaf.source().is_none());
  }

  let bounded = TryReserveError::from(put);
  assert_eq!(bounded, TryReserveError::Bounded(put));
  assert_eq!(bounded.to_string(), put.to_string());
  assert_eq!(
    bounded.source().map(ToString::to_string),
    Some(put.to_string())
  );
  assert_eq!(
    TryReserveError::CapacityOverflow.to_string(),
    "memory allocation failed because the computed capacity exceeded the collection's maximum"
  );
  assert_eq!(
    TryReserveError::AllocError.to_string(),
    "memory allocation failed because the memory allocator returned an error"
  );
  assert!(TryReserveError::CapacityOverflow.source().is_none());
  assert!(TryReserveError::AllocError.source().is_none());

  let edit = ByteEditError::from(range);
  assert_eq!(edit, ByteEditError::Range(range));
  assert_eq!(edit.to_string(), range.to_string());
  assert_eq!(
    edit.source().map(ToString::to_string),
    Some(range.to_string())
  );
  let edit = ByteEditError::from(bounded);
  assert_eq!(edit, ByteEditError::Capacity(bounded));
  assert_eq!(edit.to_string(), put.to_string());
  assert_eq!(
    edit.source().map(ToString::to_string),
    Some(put.to_string())
  );
}

#[test]
fn errors_are_plain_values_at_the_root_and_in_the_error_module() {
  fn plain<E: core::error::Error + Copy + Eq + core::hash::Hash + Send + Sync + 'static>() {}
  plain::<error::OutOfBounds>();
  plain::<error::RangeOutOfBounds>();
  plain::<error::TryPutError>();
  plain::<error::TryReserveError>();
  plain::<error::ByteEditError>();
  let index: OutOfBounds = error::OutOfBounds::new(1, 0);
  assert_eq!(index, OutOfBounds::new(1, 0));
}

#[cfg(feature = "std")]
#[test]
fn errors_convert_into_io_errors() {
  use std::io::{Error, ErrorKind};

  assert_eq!(
    Error::from(OutOfBounds::new(1, 0)).kind(),
    ErrorKind::InvalidInput
  );
  assert_eq!(
    Error::from(RangeOutOfBounds::new(1, 0, 0)).kind(),
    ErrorKind::InvalidInput
  );
  let put = Error::from(TryPutError {
    requested: 1,
    available: 0,
  });
  assert_eq!(put.kind(), ErrorKind::Other);
  assert!(put.get_ref().is_some_and(|inner| inner.is::<TryPutError>()));
}

#[test]
fn ranged_twins_err_exactly_where_the_ranged_methods_panic() {
  use core::ops::Bound::{Excluded, Unbounded};

  fn check<S: InlineSize>(source: Vector<u8, S>) {
    let len = source.len();
    for start in 0..=len + 1 {
      for end in 0..=len + 1 {
        match source.try_slice(start..end) {
          Ok(range) => assert_eq!(range, source.slice(start..end)),
          Err(error) => {
            assert_eq!(error, RangeOutOfBounds::new(start, end, len));
            let message = panic_message(|| {
              let _ = source.slice(start..end);
            });
            assert_eq!(message, "slice range out of bounds");
          }
        }
      }
    }
    // An endpoint past usize::MAX is reported as usize::MAX, and so is an end that is not read
    // because the start already was.
    assert_eq!(
      source
        .try_slice((Excluded(usize::MAX), Unbounded))
        .unwrap_err(),
      RangeOutOfBounds::new(usize::MAX, usize::MAX, len)
    );
    assert_eq!(
      source.try_slice(..=usize::MAX).unwrap_err(),
      RangeOutOfBounds::new(0, usize::MAX, len)
    );
  }
  check(VectorN::<u8, 32>::try_from_slice(b"abcdef").unwrap());
  check(Vector::<u8>::from_static(b"static window"));

  let mut buffer = VectorMut::<u8>::new();
  buffer.extend_from_slice(b"abc");
  assert_eq!(
    buffer.try_extend_from_within(2..5).unwrap_err(),
    ByteEditError::Range(RangeOutOfBounds::new(2, 5, 3))
  );
  assert_eq!(buffer.as_slice(), b"abc");
  buffer.try_extend_from_within(1..3).unwrap();
  assert_eq!(buffer.as_slice(), b"abcbc");
}

#[test]
fn advance_twins_err_exactly_where_advance_panics() {
  fn check<S: InlineSize>(source: Vector<u8, S>) {
    let len = source.len();
    for count in 0..=len + 1 {
      let mut fallible = source.clone();
      let mut panicking = source.clone();
      if count <= len {
        panicking.advance(count);
        fallible.try_advance(count).unwrap();
        assert_eq!(fallible, panicking);
      } else {
        assert_eq!(
          fallible.try_advance(count).unwrap_err(),
          OutOfBounds::new(count, len)
        );
        assert_eq!(fallible, source);
        assert_eq!(
          panic_message(|| panicking.advance(count)),
          "advance past end"
        );
      }
    }
  }
  check(VectorN::<u8, 32>::try_from_slice(b"abcdef").unwrap());
  check(Vector::<u8>::from_static(b"static window"));

  let mut buffer = VectorMut::<u8>::new();
  buffer.extend_from_slice(b"abc");
  assert_eq!(buffer.try_advance(4).unwrap_err(), OutOfBounds::new(4, 3));
  assert_eq!(buffer.as_slice(), b"abc");
  buffer.try_advance(2).unwrap();
  assert_eq!(buffer.as_slice(), b"c");
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn local_twins_err_exactly_where_the_local_methods_panic() {
  use bytora::LocalVector;
  let source = LocalVector::<u8>::from_rc(std::rc::Rc::from(&b"abcdef"[..]));
  let len = source.len();
  for start in 0..=len + 1 {
    for end in 0..=len + 1 {
      match source.try_slice(start..end) {
        Ok(range) => assert_eq!(range.as_slice(), source.slice(start..end).as_slice()),
        Err(error) => assert_eq!(error, RangeOutOfBounds::new(start, end, len)),
      }
    }
  }
  let mut fallible = source.clone();
  assert_eq!(
    fallible.try_advance(len + 1).unwrap_err(),
    OutOfBounds::new(len + 1, len)
  );
  assert_eq!(fallible.as_slice(), source.as_slice());
  fallible.try_advance(2).unwrap();
  assert_eq!(fallible.as_slice(), b"cdef");
}

#[test]
fn mutable_split_off_twin_errs_exactly_past_capacity() {
  let mut source = VectorMut::<u8>::new();
  source.extend_from_slice(b"abcd");
  let capacity = source.capacity();
  for at in [0, 2, 4, capacity] {
    let mut panicking = source.clone();
    let mut fallible = source.clone();
    let expected = panicking.split_off(at);
    let suffix = fallible.try_split_off(at).unwrap();
    assert_eq!(
      (suffix.as_slice(), fallible.as_slice()),
      (expected.as_slice(), panicking.as_slice())
    );
  }
  let mut fallible = source.clone();
  assert_eq!(
    fallible.try_split_off(capacity + 1).unwrap_err(),
    OutOfBounds::new(capacity + 1, capacity)
  );
  assert_eq!(fallible.as_slice(), b"abcd");
}

// try_reserve errs exactly where reserve panics, and on either outcome the
// elements are unchanged. Requests stay small or overflow, so that no reserve
// reaches an allocation the allocator would refuse, which aborts.
#[test]
fn try_reserve_errs_exactly_where_reserve_panics() {
  fn check<S: InlineSize>(buffer: VectorMut<u8, S>) {
    let len = buffer.len();
    for additional in [0, 1, 7, 64, 4096, usize::MAX - len, usize::MAX] {
      let mut fallible = buffer.clone();
      let result = fallible.try_reserve(additional);
      let mut panicking = buffer.clone();
      let panicked = catch_unwind(AssertUnwindSafe(|| panicking.reserve(additional))).is_err();
      assert_eq!(result.is_err(), panicked, "additional {additional}");
      assert_eq!(fallible.as_slice(), buffer.as_slice());
      match result {
        Ok(()) => assert!(fallible.capacity() - fallible.len() >= additional),
        Err(TryReserveError::Bounded(error)) => assert_eq!(error.requested, additional),
        Err(error) => assert_eq!(error, TryReserveError::CapacityOverflow),
      }
    }
  }
  let mut inline = VectorMut::<u8>::new();
  inline.extend_from_slice(b"abc");
  check(inline.clone());
  let mut narrow = VectorMutN::<u8, 16>::new();
  narrow.extend_from_slice(b"abc");
  check(narrow);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    // A heap buffer reserves through its window.
    let mut heap = VectorMut::<u8>::with_capacity(256);
    heap.extend_from_slice(b"heap bytes");
    heap.advance(5);
    check(heap);
  }
}

#[test]
fn allocating_twins_return_what_the_allocating_methods_return() {
  for len in [0, 3, 40, 1000] {
    match VectorMut::<u8>::try_with_capacity(len) {
      Ok(buffer) => assert!(buffer.capacity() >= len && buffer.is_empty()),
      Err(error) => assert!(matches!(error, TryReserveError::Bounded(_))),
    }
    match VectorMut::<u8>::try_zeroed(len) {
      Ok(zeros) => assert_eq!(zeros.as_slice(), VectorMut::<u8>::zeroed(len).as_slice()),
      Err(error) => assert!(matches!(error, TryReserveError::Bounded(_))),
    }
    let mut fallible = VectorMut::<u8>::new();
    let elements = vec![7u8; len];
    match fallible.try_extend_from_slice(&elements) {
      Ok(()) => assert_eq!(fallible.as_slice(), &elements[..]),
      Err(error) => {
        assert!(matches!(error, TryReserveError::Bounded(_)));
        assert!(fallible.is_empty());
      }
    }
  }
  assert!(VectorMut::<u8>::try_with_capacity(usize::MAX).is_err());
  assert!(VectorMut::<u8>::try_zeroed(usize::MAX).is_err());
}

// try_resize errs exactly where resize panics, with the elements unchanged, and otherwise resizes
// as resize does. Lengths stay small or overflow, as in the try_reserve test above.
#[test]
fn try_resize_errs_exactly_where_resize_panics() {
  fn check<S: InlineSize>(buffer: VectorMut<u8, S>) {
    let len = buffer.len();
    let cap = buffer.capacity();
    for new_len in [0, 1, len, len + 1, cap, cap + 1, 64, 4096, usize::MAX] {
      let mut fallible = buffer.clone();
      let result = fallible.try_resize(new_len, 7);
      let mut panicking = buffer.clone();
      let panicked = catch_unwind(AssertUnwindSafe(|| panicking.resize(new_len, 7))).is_err();
      assert_eq!(result.is_err(), panicked, "new_len {new_len}");
      match result {
        Ok(()) => assert_eq!(fallible, panicking),
        Err(TryReserveError::Bounded(error)) => {
          assert_eq!(error.requested, new_len - len);
          assert_eq!(fallible, buffer);
        }
        Err(error) => {
          assert_eq!(error, TryReserveError::CapacityOverflow);
          assert_eq!(fallible, buffer);
        }
      }
    }
  }
  // The one-word budget never grows. The 16-byte budget grows only with an allocator.
  let mut tiny = VectorMutN::<u8, { size_of::<usize>() }>::new();
  tiny.extend_from_slice(b"abc");
  check(tiny);
  let mut narrow = VectorMutN::<u8, 16>::new();
  narrow.extend_from_slice(b"abc");
  check(narrow);
  let mut inline = VectorMut::<u8>::new();
  inline.extend_from_slice(b"abc");
  check(inline);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    // A heap buffer whose view starts past the base of its allocation.
    let mut heap = VectorMut::<u8>::with_capacity(256);
    heap.extend_from_slice(b"heap bytes");
    heap.advance(5);
    check(heap);
  }
}

// try_slice returns the error at a start past usize::MAX without reading the end bound, as slice
// panics there without reading it.
#[test]
fn a_start_past_usize_max_is_refused_before_the_end_is_read() {
  use core::ops::{Bound, RangeBounds};

  struct StartOverflows;
  impl RangeBounds<usize> for StartOverflows {
    fn start_bound(&self) -> Bound<&usize> {
      Bound::Excluded(&usize::MAX)
    }
    fn end_bound(&self) -> Bound<&usize> {
      panic!("the end bound was read")
    }
  }
  let vector = Vector::<u8>::from_static(b"bytes");
  let error = vector.try_slice(StartOverflows).unwrap_err();
  assert_eq!((error.start, error.end), (usize::MAX, usize::MAX));
}

// try_extend errs exactly where extend panics. Its elements are then unchanged, where extend keeps
// what it appended before the panic, and otherwise it appends what extend appends. It reserves the
// lower size hint first, then each further element as it comes. Counts stay small or overflow.
#[test]
fn try_extend_errs_exactly_where_extend_panics() {
  use core::iter::{from_fn, repeat_n};

  fn compare<S: InlineSize, I: Iterator<Item = u8>>(
    buffer: &VectorMut<u8, S>,
    items: impl Fn() -> I,
  ) -> Result<(), TryReserveError> {
    let mut fallible = buffer.clone();
    let result = fallible.try_extend(items());
    let mut panicking = buffer.clone();
    let panicked = catch_unwind(AssertUnwindSafe(|| panicking.extend(items()))).is_err();
    assert_eq!(result.is_err(), panicked);
    if result.is_ok() {
      assert_eq!(fallible, panicking);
    } else {
      assert_eq!(fallible, *buffer);
    }
    result
  }

  fn check<S: InlineSize>(buffer: VectorMut<u8, S>) {
    for count in [0, 1, 3, 7, 16, 64, 4096] {
      // An exact size hint reserves every element at once.
      if let Err(error) = compare(&buffer, || repeat_n(7, count)) {
        assert!(matches!(error, TryReserveError::Bounded(put) if put.requested == count));
      }
      // Without one, each element reserves its own slot, and the first that finds none fails.
      let unhinted = || {
        let mut items = repeat_n(7, count);
        from_fn(move || items.next())
      };
      if let Err(error) = compare(&buffer, unhinted) {
        let full = TryPutError {
          requested: 1,
          available: 0,
        };
        assert_eq!(error, TryReserveError::Bounded(full));
      }
    }
    // A size hint that no buffer can hold is refused before any element is taken.
    assert!(compare(&buffer, || repeat_n(7, usize::MAX)).is_err());
  }
  // The one-word budget never grows. The 16-byte budget grows only with an allocator.
  let mut tiny = VectorMutN::<u8, { size_of::<usize>() }>::new();
  tiny.extend_from_slice(b"abc");
  check(tiny);
  let mut narrow = VectorMutN::<u8, 16>::new();
  narrow.extend_from_slice(b"abc");
  check(narrow);
  let mut inline = VectorMut::<u8>::new();
  inline.extend_from_slice(b"abc");
  check(inline);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    // A heap buffer whose view starts past the base of its allocation.
    let mut heap = VectorMut::<u8>::with_capacity(256);
    heap.extend_from_slice(b"heap bytes");
    heap.advance(5);
    check(heap);
  }
}

// write_str appends where extend_from_slice does and errs exactly where it panics, and a failed
// call appends nothing. Lengths stay small, as in the try_reserve test above.
#[test]
fn write_str_errs_exactly_where_extend_from_slice_panics() {
  use core::fmt::Write;

  fn check<S: InlineSize>(buffer: VectorMut<u8, S>) {
    for len in [0, 1, 3, 7, 16, 64, 4096] {
      let text = "x".repeat(len);
      let mut writer = buffer.clone();
      let result = writer.write_str(&text);
      let mut panicking = buffer.clone();
      let panicked = catch_unwind(AssertUnwindSafe(|| {
        panicking.extend_from_slice(text.as_bytes())
      }))
      .is_err();
      assert_eq!(result.is_err(), panicked, "len {len}");
      if result.is_ok() {
        assert_eq!(writer, panicking);
      } else {
        assert_eq!(writer, buffer);
      }
    }
  }
  // The one-word budget never grows. The 16-byte budget grows only with an allocator.
  let mut tiny = VectorMutN::<u8, { size_of::<usize>() }>::new();
  tiny.extend_from_slice(b"abc");
  check(tiny);
  let mut narrow = VectorMutN::<u8, 16>::new();
  narrow.extend_from_slice(b"abc");
  check(narrow);
  let mut inline = VectorMut::<u8>::new();
  inline.extend_from_slice(b"abc");
  check(inline);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    // A heap buffer whose view starts past the base of its allocation.
    let mut heap = VectorMut::<u8>::with_capacity(256);
    heap.extend_from_slice(b"heap bytes");
    heap.advance(5);
    check(heap);
  }
}

// write! makes one write_str call per piece. A piece past the room of a buffer that cannot grow is
// refused whole and the pieces before it stay. A buffer with room holds what format! prints.
#[test]
fn write_keeps_the_pieces_before_the_one_that_does_not_fit() {
  use core::fmt::Write;

  // Variables rather than literal arguments, which `format_args!` may fold into a single piece.
  let (fits, too_long) = ("ab", "cdefghijk");
  let mut tiny = VectorMutN::<u8, { size_of::<usize>() }>::new();
  assert_eq!(write!(tiny, "{fits}{too_long}"), Err(core::fmt::Error));
  assert_eq!(tiny.as_slice(), b"ab");

  let (number, byte) = (1, 255u8);
  let mut buffer = VectorMut::<u8>::new();
  write!(buffer, "{number}-{byte:02x}-{fits}").unwrap();
  assert_eq!(
    buffer.as_slice(),
    format!("{number}-{byte:02x}-{fits}").as_bytes()
  );
}

// A failed try_extend leaves the elements as they were. A bounded inline buffer reclaims its
// consumed prefix before it fails, as extend does, so the capacity may have grown.
#[test]
fn a_failed_try_extend_leaves_the_elements_as_they_were() {
  let mut buffer = VectorMutN::<u8, { size_of::<usize>() }>::zeroed(size_of::<usize>() - 1);
  buffer.advance(1);
  let capacity = buffer.capacity();
  let elements = buffer.as_slice().to_vec();
  // The filter hides the length, so the lower size hint is zero.
  let more = (0..u8::MAX).filter(|_| true);
  assert!(buffer.try_extend(more).is_err());
  assert_eq!(buffer.as_slice(), elements.as_slice());
  assert!(buffer.capacity() >= capacity);
}

// try_extend reserves where extend does, so it moves the elements to the front of the allocation
// rather than grow it where extend does. Here the first reserve, for a lower size hint past the
// spare capacity, finds a consumed prefix as long as the elements.
#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn try_extend_grows_only_where_extend_would() {
  let build = || {
    let mut buffer = VectorMut::<u8>::with_capacity(64);
    buffer.resize(62, 7);
    buffer.advance(31);
    buffer
  };
  let mut panicking = build();
  panicking.extend([1, 2, 3]);
  let mut fallible = build();
  fallible.try_extend([1, 2, 3]).unwrap();
  assert_eq!(fallible, panicking);
  assert_eq!(fallible.capacity(), panicking.capacity());
}

// try_extend drops skipped elements where extend does, before it polls the iterator again: here
// its first reserve reclaims the prefix that holds the last strong reference, so the iterator
// finds the value gone in both.
#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn try_extend_drops_skipped_elements_where_extend_does() {
  use std::rc::{Rc, Weak};

  // A zero budget, as elements that are not Copy need.
  fn build() -> (VectorMutN<Option<Rc<u8>>, 0>, Weak<u8>) {
    let value = Rc::new(7);
    let weak = Rc::downgrade(&value);
    let mut buffer = VectorMutN::<Option<Rc<u8>>, 0>::with_capacity(64);
    buffer.resize(1, Some(value));
    buffer.resize(64, None);
    // A full buffer whose skipped prefix is as long as its elements.
    buffer.advance(32);
    assert_eq!(buffer.capacity(), buffer.len());
    (buffer, weak)
  }
  let (mut panicking, weak) = build();
  panicking.extend(core::iter::once_with(|| weak.upgrade()));
  let (mut fallible, weak) = build();
  fallible
    .try_extend(core::iter::once_with(|| weak.upgrade()))
    .unwrap();
  assert_eq!(fallible, panicking);
  assert!(fallible.as_slice().iter().all(Option::is_none));
}

// try_extend polls an iterator no further once it has ended, as extend does, even one that
// promises to stay ended.
#[test]
fn try_extend_polls_an_ended_iterator_no_further() {
  struct Ended<'a>(&'a core::cell::Cell<usize>);
  impl Iterator for Ended<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
      self.0.set(self.0.get() + 1);
      None
    }
  }
  impl core::iter::FusedIterator for Ended<'_> {}

  fn check<S: InlineSize>(mut buffer: VectorMut<u8, S>) {
    let polls = core::cell::Cell::new(0);
    buffer.try_extend(Ended(&polls)).unwrap();
    assert_eq!(polls.get(), 1);
  }
  check(VectorMut::<u8>::new());
  check(VectorMutN::<u8, { size_of::<usize>() }>::new());
  #[cfg(any(feature = "std", feature = "alloc"))]
  check(VectorMut::<u8>::with_capacity(64));
}
