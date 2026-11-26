//! Report allocation counts separately from the uninstrumented timing benchmarks.
//!
//! Run with `cargo run --release --example lazy_vec_allocations`. The input Vec
//! and its contents are created before each measured stage. The output shows
//! observations rather than asserting counts, so the same runner can compare
//! different implementations of the owner.

#[cfg(bytora_atomic)]
mod measurement {
  use bytes::Bytes;
  use bytora::{Slice, SliceN};
  use std::hint::black_box;

  const LEN: usize = 64;
  const CONTENTS: [u8; LEN] = [0x5a; LEN];

  #[derive(Clone, Copy)]
  enum Capacity {
    Exact,
    Spare,
  }

  impl Capacity {
    fn name(self) -> &'static str {
      match self {
        Self::Exact => "exact",
        Self::Spare => "spare",
      }
    }
  }

  fn input(kind: Capacity) -> Vec<u8> {
    let requested = match kind {
      Capacity::Exact => LEN,
      Capacity::Spare => LEN * 2,
    };
    let mut values = Vec::with_capacity(requested);
    values.extend_from_slice(&CONTENTS);
    match kind {
      Capacity::Exact => assert_eq!(values.capacity(), LEN),
      Capacity::Spare => assert!(values.capacity() > LEN),
    }
    values
  }

  fn report<T: Clone>(
    name: &str,
    kind: Capacity,
    from_vec: impl Fn(Vec<u8>) -> T,
    into_vec: impl Fn(T) -> Vec<u8>,
    as_slice: impl Fn(&T) -> &[u8],
  ) {
    let values = input(kind);
    let pointer = values.as_ptr();
    let mut owner = None;
    let construction = allocation_counter::measure(|| {
      owner = Some(from_vec(black_box(values)));
    });
    let owner = owner.expect("constructed owner");
    assert_eq!(as_slice(&owner), CONTENTS);
    assert_eq!(as_slice(&owner).as_ptr(), pointer);

    let mut cloned = None;
    let first_clone = allocation_counter::measure(|| {
      cloned = Some(black_box(&owner).clone());
    });
    let cloned = cloned.expect("first clone");
    assert_eq!(as_slice(&cloned), CONTENTS);
    assert_eq!(as_slice(&cloned).as_ptr(), pointer);

    let mut steady = None;
    let steady_clone = allocation_counter::measure(|| {
      steady = Some(black_box(&owner).clone());
    });
    let steady = steady.expect("steady clone");
    assert_eq!(as_slice(&steady), CONTENTS);
    assert_eq!(as_slice(&steady).as_ptr(), pointer);
    drop((owner, cloned, steady));

    // A separate owner has never been cloned before its unique consume.
    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let unique_owner = from_vec(values);
    let mut returned = None;
    let consume = allocation_counter::measure(|| {
      returned = Some(into_vec(black_box(unique_owner)));
    });
    let returned = returned.expect("consumed owner");
    assert_eq!(returned.as_ptr(), pointer);
    assert_eq!(returned.capacity(), capacity);
    assert_eq!(returned, CONTENTS);
    drop(returned);

    let values = input(kind);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let mut returned = None;
    let round_trip = allocation_counter::measure(|| {
      returned = Some(into_vec(from_vec(black_box(values))));
    });
    let returned = returned.expect("round trip");
    assert_eq!(returned.as_ptr(), pointer);
    assert_eq!(returned.capacity(), capacity);
    assert_eq!(returned, CONTENTS);
    drop(returned);

    for (stage, info) in [
      ("construct", construction),
      ("consume_unique", consume),
      ("round_trip", round_trip),
      ("first_clone", first_clone),
      ("steady_clone", steady_clone),
    ] {
      println!(
        "{name:<14} {:<5} {stage:<14} allocations={:<2} bytes={}",
        kind.name(),
        info.count_total,
        info.bytes_total,
      );
    }
  }

  pub fn run() {
    // Warm up thread-local accounting and prove the counter sees an allocation.
    let _ = allocation_counter::measure(|| {});
    let mut probe = None;
    let measured = allocation_counter::measure(|| {
      probe = Some(black_box(Box::new(1u64)));
    });
    assert_eq!(measured.count_total, 1);
    drop(probe);

    type DefaultSlice = Slice<u8>;
    type WideSlice = SliceN<u8, 32>;
    for kind in [Capacity::Exact, Capacity::Spare] {
      report(
        "slice_default",
        kind,
        |values| DefaultSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
        DefaultSlice::as_slice,
      );
      report(
        "slice_32",
        kind,
        |values| WideSlice::try_from_vec(values).unwrap(),
        |slice| slice.try_into_vec().ok().expect("unique slice"),
        WideSlice::as_slice,
      );
      report("bytes", kind, Bytes::from, Vec::from, Bytes::as_ref);
    }
  }
}

fn main() {
  #[cfg(bytora_atomic)]
  measurement::run();
  #[cfg(not(bytora_atomic))]
  eprintln!("allocation measurements require shared ownership support");
}
