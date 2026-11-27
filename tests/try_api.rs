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
