//! Check ThinSlice allocation phases separately from the timing benchmark.
//!
//! Run with `cargo run --release --example thin_allocations --features experimental-thin`.

#[cfg(smol_slice_atomic)]
mod measurement {
  use smol_slice::ThinSlice;
  use std::hint::black_box;

  const CONTENTS: [u8; 64] = {
    let mut values = [0; 64];
    let mut i = 0;
    while i < values.len() {
      values[i] = i as u8;
      i += 1;
    }
    values
  };

  fn input(len: usize, capacity: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(capacity);
    values.extend_from_slice(&CONTENTS[..len]);
    assert_eq!((values.len(), values.capacity()), (len, capacity));
    values
  }

  fn phase(name: &str, allocations: u64, deallocations: i64, action: impl FnOnce()) {
    let info = allocation_counter::measure(action);
    assert_eq!(info.count_total, allocations, "{name} allocations");
    assert_eq!(
      info.count_total as i64 - info.count_current,
      deallocations,
      "{name} deallocations"
    );
    println!(
      "  {name:<20} allocations={allocations} deallocations={deallocations} bytes_allocated={}",
      info.bytes_total
    );
  }

  fn shared_lifetime(len: usize, capacity: usize) {
    let mut values = None;
    phase("vec", 1, 0, || {
      values = Some(black_box(input(len, capacity)));
    });
    let values = values.expect("input vector");
    let pointer = values.as_ptr();
    let mut owner = None;
    phase("from_vec", 1, 0, || {
      owner = Some(ThinSlice::from_vec(black_box(values)));
    });
    let owner = owner.expect("owner");
    assert_eq!(owner.as_slice(), &CONTENTS[..len]);
    assert_eq!(owner.as_slice().as_ptr(), pointer);
    assert!(owner.is_unique());

    let mut cloned = None;
    phase("clone_full", 0, 0, || {
      cloned = Some(black_box(&owner).clone());
    });
    let cloned = cloned.expect("full clone");
    let mut full = None;
    phase("slice_full", 0, 0, || {
      full = Some(black_box(&owner).slice(..));
    });
    let full = full.expect("full range");
    assert_eq!(cloned.as_slice(), &CONTENTS[..len]);
    assert_eq!(full.as_slice(), &CONTENTS[..len]);
    assert_eq!(cloned.as_slice().as_ptr(), pointer);
    assert_eq!(full.as_slice().as_ptr(), pointer);

    let mut window = None;
    phase("slice_window", 1, 0, || {
      window = Some(black_box(&owner).slice(1..len - 1));
    });
    let window = window.expect("window");
    let mut nested = None;
    phase("slice_nested", 1, 0, || {
      nested = Some(black_box(&window).slice(1..len - 3));
    });
    let nested = nested.expect("nested window");
    assert_eq!(owner.as_slice(), &CONTENTS[..len]);
    assert_eq!(window.as_slice(), &CONTENTS[1..len - 1]);
    assert_eq!(nested.as_slice(), &CONTENTS[2..len - 2]);
    assert_eq!(window.as_slice().as_ptr(), pointer.wrapping_add(1));
    assert_eq!(nested.as_slice().as_ptr(), pointer.wrapping_add(2));

    phase("drop_full_clones", 0, 0, || drop((cloned, full)));
    phase("drop_window", 0, 1, || drop(window));
    phase("drop_owner", 0, 0, || drop(owner));
    assert_eq!(nested.as_slice(), &CONTENTS[2..len - 2]);
    phase("drop_last_window", 0, 3, || drop(nested));
  }

  fn consume_full(len: usize, capacity: usize) {
    let values = input(len, capacity);
    let pointer = values.as_ptr();
    let owner = ThinSlice::from_vec(values);
    assert!(owner.is_unique());
    let mut slot = Some(owner);
    let mut returned = None;
    phase("consume_full", 0, 1, || {
      returned = Some(
        black_box(slot.take().expect("unique owner"))
          .try_into_vec()
          .ok()
          .expect("unique owner"),
      );
    });
    let returned = returned.expect("returned vector");
    assert_eq!(returned, &CONTENTS[..len]);
    assert_eq!(returned.as_ptr(), pointer);
    assert_eq!((returned.len(), returned.capacity()), (len, capacity));
    phase("drop_full_vec", 0, 1, || drop(returned));
  }

  fn consume_window(len: usize, capacity: usize) {
    let values = input(len, capacity);
    let pointer = values.as_ptr();
    let owner = ThinSlice::from_vec(values);
    let window = owner.slice(1..len - 1);
    drop(owner);
    assert!(window.is_unique());
    let mut slot = Some(window);
    let mut returned = None;
    phase("consume_window", 0, 2, || {
      returned = Some(
        black_box(slot.take().expect("unique window"))
          .try_into_vec()
          .ok()
          .expect("unique window"),
      );
    });
    let returned = returned.expect("returned vector");
    assert_eq!(returned, &CONTENTS[1..len - 1]);
    assert_eq!(returned.as_ptr(), pointer);
    assert_eq!((returned.len(), returned.capacity()), (len - 2, capacity));
    phase("drop_window_vec", 0, 1, || drop(returned));
  }

  pub fn run() {
    let _ = allocation_counter::measure(|| {});
    let mut probe = None;
    let info = allocation_counter::measure(|| {
      probe = Some(black_box(Box::new(1u64)));
    });
    assert_eq!((info.count_total, info.count_current), (1, 1));
    drop(probe);
    for (len, capacity) in [(7, 7), (64, 64), (64, 65)] {
      println!("len={len} capacity={capacity}");
      shared_lifetime(len, capacity);
      consume_full(len, capacity);
      consume_window(len, capacity);
    }
  }
}

fn main() {
  #[cfg(smol_slice_atomic)]
  measurement::run();
  #[cfg(not(smol_slice_atomic))]
  eprintln!("allocation measurements require shared ownership support");
}
