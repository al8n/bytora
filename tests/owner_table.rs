#![cfg(smol_slice_owner_table)]

use smol_slice::{SliceMutN, SliceN};
use std::{
  panic::{AssertUnwindSafe, catch_unwind},
  sync::{
    Barrier,
    atomic::{AtomicUsize, Ordering::Relaxed},
  },
  thread,
};

type Tiny<T> = SliceN<T, 8>;
type TinyMut<T> = SliceMutN<T, 8>;

#[test]
fn seven_borrowed_bytes_stay_inline_and_owned_views_keep_the_original_allocation() {
  assert_eq!(size_of::<Tiny<u8>>(), 8);
  assert_eq!(size_of::<Option<Tiny<u8>>>(), 8);
  assert_eq!(size_of::<Option<TinyMut<u8>>>(), 8);
  assert_eq!(Tiny::<u8>::INLINE_CAPACITY, 7);
  let bytes = *b"1234567";
  let inline = Tiny::try_from_slice(&bytes).unwrap();
  assert_eq!(inline.as_slice(), &bytes);
  assert_ne!(inline.as_slice().as_ptr(), bytes.as_ptr());
  assert!(inline.clone().is_unique());

  const fn length<T>(s: &Tiny<T>) -> usize {
    s.len()
  }
  for (len, capacity) in [(0, 7), (7, 7), (64, 64), (64, 65)] {
    let mut values = Vec::with_capacity(capacity);
    values.extend((0..len).map(|v| v as u8));
    let (base, cap) = (values.as_ptr(), values.capacity());
    let mut source = Tiny::try_from_vec(values).unwrap();
    assert_eq!(source.as_slice().as_ptr(), base);
    assert_eq!(length(&source), len);
    let mut last = inline.clone();
    last.clone_from(&source);
    let empty = source.slice(len..);
    assert_eq!(empty.as_slice().as_ptr(), base.wrapping_add(len));
    assert!(source.slice_ref(&[]).is_unique());
    assert!(!source.is_unique());
    source = source.try_into_mut().err().unwrap();
    last = last.try_into_vec().err().unwrap();
    drop(source);
    assert!(!last.is_unique());
    drop(empty);
    last.advance(len / 2);
    last.truncate(3);
    let expected = if len == 0 { 0 } else { 3 };
    assert_eq!(length(&last), expected);
    let result = last.try_into_vec().ok().unwrap();
    assert_eq!((result.as_ptr(), result.capacity()), (base, cap));
    assert_eq!(
      result,
      (len / 2..len / 2 + expected)
        .map(|v| v as u8)
        .collect::<Vec<_>>()
    );
  }
}

#[test]
fn typed_slots_preserve_scoped_elements_and_mutable_windows() {
  struct Element<'a>(usize, &'a AtomicUsize);
  impl Drop for Element<'_> {
    fn drop(&mut self) {
      self.1.fetch_add(1, Relaxed);
    }
  }
  let drops = AtomicUsize::new(0);
  let mut values = Vec::with_capacity(17);
  values.extend((0..8).map(|i| Element(i, &drops)));
  let (base, cap) = (values.as_ptr(), values.capacity());
  let source = Tiny::try_from_shared_vec(values).ok().unwrap();
  let shared = source.slice(1..7);
  thread::scope(|scope| {
    scope
      .spawn(move || {
        let piece = shared.slice(1..4);
        assert_eq!(piece.as_slice()[0].0, 2);
        drop(shared);
        assert!(piece.try_into_mut().is_err());
      })
      .join()
      .unwrap();
  });
  let mut source = source;
  source.advance(1);
  source.truncate(5);
  let mut mutable = source.try_into_mut().ok().unwrap();
  assert_eq!(mutable.as_mut_slice().as_ptr(), base.wrapping_add(1));
  assert_eq!(mutable.capacity(), cap - 1);
  assert_eq!(drops.load(Relaxed), 2);
  let right = mutable.split_off(2);
  let mut left = mutable.try_freeze_shared().err().unwrap();
  assert!(left.try_unsplit(right).is_ok());
  left.as_mut_slice()[1].0 = 19;
  let frozen = left.try_freeze_shared().ok().unwrap();
  assert_eq!(frozen.as_slice().as_ptr(), base.wrapping_add(1));
  let values = frozen.try_into_vec().ok().unwrap();
  assert_eq!(
    values.iter().map(|v| v.0).collect::<Vec<_>>(),
    [1, 19, 3, 4, 5]
  );
  assert_eq!((values.as_ptr(), values.capacity()), (base, cap));
  assert_eq!(drops.load(Relaxed), 3);
  drop(values);
  assert_eq!(drops.load(Relaxed), 8);
}

#[test]
fn real_page_growth_and_reuse_survive_cross_thread_last_releases() {
  let owners = if cfg!(miri) { 72 } else { 8_193 };
  let workers = 4;
  let barrier = Barrier::new(workers);
  for generation in 0..3 {
    let mut handles = Vec::with_capacity(owners);
    for i in 0..owners {
      let v = vec![(generation, i)];
      let key = Tiny::try_from_shared_vec(v).unwrap();
      handles.push(key);
    }
    thread::scope(|scope| {
      for worker in 0..workers {
        let source = &handles;
        let barrier = &barrier;
        scope.spawn(move || {
          barrier.wait();
          for index in (worker..owners).step_by(workers) {
            let peer = source[index].clone();
            assert_eq!(peer.as_slice(), &[(generation, index)]);
            assert!(peer.try_into_vec().is_err());
          }
        });
      }
    });
    thread::scope(|scope| {
      for chunk in handles.chunks_mut(owners.div_ceil(workers)) {
        scope.spawn(move || {
          for live in chunk {
            let owned = std::mem::take(live);
            assert!(owned.is_unique());
            assert_eq!(owned.try_into_vec().ok().unwrap()[0].0, generation);
          }
        });
      }
    });
  }
}

#[test]
fn mixed_origin_last_releases_cross_the_cache_limit_and_thread_exit() {
  struct Token<'a>(&'a AtomicUsize);
  impl Drop for Token<'_> {
    fn drop(&mut self) {
      assert_eq!(self.0.fetch_add(1, Relaxed), 0);
    }
  }
  let drops: Vec<_> = (0..160).map(|_| AtomicUsize::new(0)).collect();
  thread::scope(|scope| {
    let producers: Vec<_> = drops
      .chunks(40)
      .map(|part| {
        scope.spawn(move || {
          part
            .iter()
            .map(|counter| {
              Tiny::try_from_shared_vec(vec![Token(counter)])
                .ok()
                .unwrap()
            })
            .collect::<Vec<_>>()
        })
      })
      .collect();
    let mut batches: Vec<_> = producers.into_iter().map(|p| p.join().unwrap()).collect();
    scope
      .spawn(move || {
        for _ in 0..40 {
          for batch in &mut batches {
            drop(batch.pop().unwrap());
          }
        }
      })
      .join()
      .unwrap();
  });
  assert!(drops.iter().all(|counter| counter.load(Relaxed) == 1));
}

#[test]
fn panicking_reentrant_element_drops_release_each_slot_and_all_hidden_values() {
  struct Element<'a> {
    index: usize,
    explode: usize,
    drops: &'a [AtomicUsize; 5],
  }
  impl Drop for Element<'_> {
    fn drop(&mut self) {
      self.drops[self.index].fetch_add(1, Relaxed);
      let another = Tiny::try_from_vec(vec![37; 7]).unwrap();
      assert_eq!(another.try_into_vec().ok().unwrap(), [37; 7]);
      assert_ne!(
        self.index, self.explode,
        "unwind while releasing an element"
      );
    }
  }
  for (mode, explode) in [(0, 0), (1, 0), (1, 4), (2, 4), (3, 2)] {
    let drops = core::array::from_fn(|_| AtomicUsize::new(0));
    let values = (0..5)
      .map(|index| Element {
        index,
        explode,
        drops: &drops,
      })
      .collect();
    let mut source = Tiny::try_from_shared_vec(values).ok().unwrap();
    source.advance(1);
    source.truncate(3);
    let result = catch_unwind(AssertUnwindSafe(|| match mode {
      0 => drop(source),
      1 => drop(source.try_into_vec().ok().unwrap()),
      2 => drop(source.try_into_mut().ok().unwrap()),
      _ => source.clear(),
    }));
    assert!(result.is_err());
    assert!(drops.iter().all(|d| d.load(Relaxed) == 1));
  }
}

#[test]
fn largest_immutable_window_and_first_rejected_length_keep_their_exact_geometry() {
  const MAX: usize = (1 << 20) - 1;
  #[repr(align(256))]
  #[derive(Clone, Copy)]
  struct Zst;
  let boxed: Box<[Zst]> = Box::new([Zst; MAX]);
  let values = boxed.into_vec();
  let base = values.as_ptr();
  let mut source = Tiny::try_from_shared_vec(values).ok().unwrap();
  assert_eq!(source.len(), MAX);
  let last = source.slice(MAX - 1..);
  let empty = last.slice(1..);
  assert_eq!(
    (last.as_slice().as_ptr(), empty.as_slice().as_ptr()),
    (base, base)
  );
  source.advance(MAX);
  drop(last);
  drop(empty);
  let recovered = source.try_into_vec().ok().unwrap();
  assert_eq!(
    (recovered.as_ptr(), recovered.len(), recovered.capacity()),
    (base, 0, usize::MAX)
  );
  let boxed: Box<[Zst]> = Box::new([Zst; MAX + 1]);
  let rejected = boxed.into_vec();
  let base = rejected.as_ptr();
  let rejected = Tiny::try_from_vec(rejected).err().unwrap();
  assert_eq!(
    (rejected.as_ptr(), rejected.len(), rejected.capacity()),
    (base, MAX + 1, usize::MAX)
  );
}

#[test]
fn subtype_coercion_and_exclusive_non_send_values_use_the_current_typed_handle() {
  use std::{marker::PhantomData, rc::Rc};
  struct Mark<F>(u8, PhantomData<F>);
  impl Copy for Mark<fn(&())> {}
  impl Clone for Mark<fn(&())> {
    fn clone(&self) -> Self {
      *self
    }
  }
  let original = Tiny::try_from_vec(vec![Mark::<fn(&())>(13, PhantomData)])
    .ok()
    .unwrap();
  let widened: Tiny<Mark<fn(&'static ())>> = original;
  let mut mutable = widened.try_into_mut().ok().unwrap();
  mutable.as_mut_slice()[0] = Mark(17, PhantomData);
  let frozen = mutable.try_freeze_shared().ok().unwrap();
  assert_eq!(frozen.clone().as_slice()[0].0, 17);

  let local = Rc::new(7);
  let values = vec![Rc::clone(&local)];
  let base = values.as_ptr();
  let mutable = TinyMut::try_from_vec(values).ok().unwrap();
  assert_eq!(mutable.as_slice().as_ptr(), base);
  drop(mutable);
  assert_eq!(Rc::strong_count(&local), 1);
}

// Run in its own test process. It intentionally fills every slot and retains
// the resulting 256 MiB of pages until the process exits.
#[test]
#[ignore = "allocates the entire 8,388,608-slot directory"]
fn full_table_returns_inputs_and_reuses_slots_in_place() {
  let slots = 1 << 23;
  let per_worker = 96;
  let workers = 4;
  let mut handles = Vec::with_capacity(slots - 1 - workers * per_worker);
  let original = vec![41u8; 64];
  let base = original.as_ptr();
  let root = Tiny::try_from_vec(original).unwrap();
  let mut remote = thread::scope(|scope| {
    let tasks: Vec<_> = (0..workers)
      .map(|_| {
        scope.spawn(|| {
          (0..per_worker)
            .map(|_| Tiny::try_from_shared_vec(vec![()]).unwrap())
            .collect::<Vec<_>>()
        })
      })
      .collect();
    tasks
      .into_iter()
      .map(|task| task.join().unwrap())
      .collect::<Vec<_>>()
  });
  for _ in (1 + workers * per_worker)..slots {
    handles.push(Tiny::try_from_shared_vec(vec![()]).unwrap());
  }
  let short = Tiny::try_from_vec(vec![23u8; 7]).unwrap();
  assert!(short.clone().is_unique());
  assert_eq!(short.as_slice(), &[23; 7]);
  let values = vec![61u8; 64];
  let raw = (values.as_ptr(), values.capacity());
  let values = Tiny::try_from_vec(values).err().unwrap();
  let values = Tiny::try_from_shared_vec(values).err().unwrap();
  let values = TinyMut::try_from_vec(values).err().unwrap();
  assert_eq!((values.as_ptr(), values.capacity()), raw);
  let mut mutable = root.try_into_mut().ok().unwrap();
  assert!(catch_unwind(AssertUnwindSafe(|| drop(mutable.split_off(1)))).is_err());
  assert_eq!((mutable.as_slice().as_ptr(), mutable.len()), (base, 64));
  let root = mutable.try_freeze_shared().ok().unwrap();
  assert_eq!(root.as_slice().as_ptr(), base);
  // A single consumer interleaves all four origin shards. Filling these
  // exact vacancies again detects lost, duplicated, or stranded cache keys,
  // including the partially full batch flushed by its TLS destructor.
  thread::spawn(move || {
    for _ in 0..per_worker {
      for group in &mut remote {
        drop(group.pop().unwrap());
      }
    }
  })
  .join()
  .unwrap();
  for _ in 0..workers * per_worker {
    handles.push(Tiny::try_from_shared_vec(vec![()]).unwrap());
  }
  assert!(Tiny::try_from_shared_vec(vec![()]).is_err());
  drop(root);
  let recovered = Tiny::try_from_vec(values)
    .unwrap()
    .try_into_vec()
    .ok()
    .unwrap();
  assert_eq!((recovered.as_ptr(), recovered.capacity()), raw);
}
