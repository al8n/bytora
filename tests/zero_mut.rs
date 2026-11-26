use core::{cell::Cell, mem::needs_drop};
use paguro::{SliceMut, SliceMutN, typenum::U24};

#[test]
fn empty_mutable_zero_budget_matches_bytes_and_accepts_any_alignment() {
  #[repr(align(256))]
  struct Aligned;
  assert_eq!(
    size_of::<SliceMutN<Aligned, 0>>(),
    size_of::<bytes::Bytes>()
  );
  assert_eq!(align_of::<SliceMutN<Aligned, 0>>(), align_of::<usize>());
  assert_eq!(SliceMutN::<Aligned, 0>::INLINE_CAPACITY, 0);
  assert!(SliceMutN::<(), 0>::try_from_slice(&[()]).is_none());
  let mut empty = const { SliceMutN::<Aligned, 0>::empty() };
  empty.advance(0);
  assert!(empty.is_empty());
  assert!(empty.as_mut_slice().as_mut_ptr().is_aligned());
  assert!(empty.as_slice().as_ptr().is_aligned());
  assert_eq!(
    needs_drop::<SliceMut<String, U24>>(),
    cfg!(any(feature = "std", feature = "alloc"))
  );

  fn send<T: Send>() {}
  fn sync<T: Sync>() {}
  send::<SliceMutN<Cell<u8>, 0>>();
  sync::<SliceMutN<String, 0>>();
}

#[test]
fn mutable_reference_elements_are_covariant_for_each_budget() {
  fn shorten_inline<'a>(v: SliceMut<&'static str, U24>, s: &'a str) -> SliceMut<&'a str, U24> {
    let mut shorter: SliceMut<&'a str, U24> = v;
    shorter.as_mut_slice()[0] = s;
    shorter
  }
  fn shorten_zero<'a>(v: SliceMutN<&'static str, 0>, _: &'a str) -> SliceMutN<&'a str, 0> {
    v
  }

  let local = String::from("local");
  let inline = SliceMut::<_, U24>::try_from_slice(&["static"]).unwrap();
  assert_eq!(shorten_inline(inline, &local).as_slice(), &["local"]);
  #[cfg(any(feature = "std", feature = "alloc"))]
  {
    let mut owned = shorten_zero(SliceMut::from(vec!["static"]), &local);
    owned.as_mut_slice()[0] = &local;
    assert_eq!(owned.as_slice(), &["local"]);
  }
  assert!(shorten_zero(SliceMutN::<_, 0>::empty(), &local).is_empty());
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn exclusive_vector_keeps_its_buffer_and_drops_every_element_once() {
  use std::rc::Rc;

  struct CountDrop<'a>(&'a Cell<usize>, Rc<Cell<u8>>);
  impl Drop for CountDrop<'_> {
    fn drop(&mut self) {
      self.0.set(self.0.get() + 1);
    }
  }

  let drops = [Cell::new(0), Cell::new(0), Cell::new(0)];
  let values = Rc::new(Cell::new(0));
  let mut input = Vec::with_capacity(16);
  input.extend(drops.iter().map(|d| CountDrop(d, Rc::clone(&values))));
  let base = input.as_ptr();
  let mut handle = SliceMutN::<_, 0>::from(input);
  assert_eq!(handle.as_slice().as_ptr(), base);
  handle.advance(1);
  assert_eq!(
    handle.as_mut_slice().as_mut_ptr().cast_const(),
    base.wrapping_add(1)
  );
  handle.as_mut_slice()[0].1.set(42);
  assert_eq!(values.get(), 42);
  let before = handle.len();
  assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.advance(3))).is_err());
  assert_eq!(handle.len(), before);
  handle.advance(2);
  assert!(handle.is_empty());
  assert_eq!(handle.as_slice().as_ptr(), base.wrapping_add(3));
  assert!(drops.iter().all(|d| d.get() == 0));
  drop(handle);
  assert!(drops.iter().all(|d| d.get() == 1));
}

#[cfg(any(feature = "std", feature = "alloc"))]
#[test]
fn exclusive_vector_supports_the_entire_zero_sized_length_range() {
  let mut all = SliceMutN::<(), 0>::from(vec![(); usize::MAX]);
  assert_eq!(all.as_mut_slice().len(), usize::MAX);
  all.advance(usize::MAX - 1);
  assert_eq!(all.as_slice(), &[()]);
  all.advance(1);
  assert!(all.is_empty());

  let empty = SliceMutN::<String, 0>::from(Vec::with_capacity(5));
  assert!(empty.is_empty());
}
