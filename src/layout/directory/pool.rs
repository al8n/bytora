#[cfg(all(feature = "std", unix))]
use core::cell::Cell;
use core::{
  cell::UnsafeCell,
  mem::{self, ManuallyDrop},
  ptr,
  sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering, fence},
};
use std::{boxed::Box, vec::Vec};

const SHARD_BITS: u32 = 5;
const SHARDS: usize = 1 << SHARD_BITS;
pub(super) const INDEX_BITS: u32 = 23;
const PER_SHARD: u32 = 1 << (INDEX_BITS - SHARD_BITS);
const FIRST_PAGE_BITS: u32 = 5;
const PAGES: usize = (INDEX_BITS - SHARD_BITS - FIRST_PAGE_BITS + 1) as usize;
const NONE: u32 = u32::MAX;

// The pointer is always stored and moved as a pointer. Its target's type and
// lifetime are supplied by the owning typed handle. No integer here designates
// a pointee, except as an index into permanently allocated slot storage.
struct Parts {
  base: *mut (),
  len: usize,
  capacity: usize,
}

impl Parts {
  const EMPTY: Self = Self {
    base: ptr::null_mut(),
    len: 0,
    capacity: 0,
  };
}

struct Slot {
  strong: AtomicUsize,
  parts: UnsafeCell<Parts>,
}

impl Slot {
  const fn empty() -> Self {
    Self {
      strong: AtomicUsize::new(0),
      parts: UnsafeCell::new(Parts::EMPTY),
    }
  }
}

// SAFETY: parts is written only by its exclusive reservation or the last
// owner. Live immutable handles only read it. The public handle's T governs
// access, transfer, destruction, and the lifetime of the pointed-to elements.
unsafe impl Sync for Slot {}

struct Free {
  head: u32,
  next: u32,
}

// Each page is independently allocated and stays at the published address for
// the process lifetime. The shard lock covers just the two allocation cursors
// and the intrusive free list. Allocation and element drops happen outside it.
#[repr(align(64))]
struct Shard {
  pages: [AtomicPtr<Slot>; PAGES],
  free: UnsafeCell<Free>,
  locked: AtomicBool,
}

// SAFETY: only with_free accesses Free. Published page pointers never change.
unsafe impl Sync for Shard {}

impl Shard {
  const fn new() -> Self {
    Self {
      pages: [const { AtomicPtr::new(ptr::null_mut()) }; PAGES],
      free: UnsafeCell::new(Free {
        head: NONE,
        next: 0,
      }),
      locked: AtomicBool::new(false),
    }
  }

  fn with_free<R>(&self, operation: impl FnOnce(&mut Free) -> R) -> R {
    struct Unlock<'a>(&'a AtomicBool);
    impl Drop for Unlock<'_> {
      fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
      }
    }
    while self
      .locked
      .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
      .is_err()
    {
      while self.locked.load(Ordering::Relaxed) {
        core::hint::spin_loop();
      }
    }
    let _guard = Unlock(&self.locked);
    // SAFETY: this thread owns the lock until the closure returns or unwinds.
    operation(unsafe { &mut *self.free.get() })
  }

  fn claim(&self, shard: usize) -> Option<Key> {
    loop {
      let result = self.with_free(|free| {
        if free.head != NONE {
          let key = Key(free.head);
          free.head = key.slot().strong.load(Ordering::Relaxed) as u32;
          return Ok(Some(key));
        }
        if free.next == PER_SHARD {
          return Ok(None);
        }
        let key = Key((free.next << SHARD_BITS) | shard as u32);
        let (page, _) = key.location();
        if self.pages[page].load(Ordering::Acquire).is_null() {
          return Err(page);
        }
        free.next += 1;
        Ok(Some(key))
      });
      match result {
        Ok(key) => return key,
        // Allocate outside the lock. No index is consumed until publication
        // succeeds, even if an allocation failure unwinds.
        Err(missing_page) => self.grow(missing_page),
      }
    }
  }

  #[cold]
  fn grow(&self, page: usize) {
    let count = 1 << (FIRST_PAGE_BITS + page.saturating_sub(1) as u32);
    let mut slots = Vec::with_capacity(count);
    slots.resize_with(count, Slot::empty);
    let raw = Box::into_raw(slots.into_boxed_slice());
    if self.pages[page]
      .compare_exchange(
        ptr::null_mut(),
        raw.cast(),
        Ordering::AcqRel,
        Ordering::Acquire,
      )
      .is_err()
    {
      // SAFETY: this candidate was never published. Its full Box pointer
      // still carries both the original allocation provenance and its length.
      unsafe {
        drop(Box::from_raw(raw));
      }
    }
  }
}

static DIRECTORY: [Shard; SHARDS] = [const { Shard::new() }; SHARDS];
static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
pub(super) struct Key(u32);

impl Key {
  pub(super) fn bits(self) -> u32 {
    self.0
  }

  /// The integer must come from a live table descriptor for its current owner.
  pub(super) unsafe fn from_bits(bits: u32) -> Self {
    Self(bits)
  }

  fn shard(self) -> &'static Shard {
    &DIRECTORY[self.0 as usize & (SHARDS - 1)]
  }

  fn location(self) -> (usize, usize) {
    let local = self.0 >> SHARD_BITS;
    if local < (1 << FIRST_PAGE_BITS) {
      return (0, local as usize);
    }
    let high = local.ilog2();
    (
      (high - FIRST_PAGE_BITS + 1) as usize,
      (local ^ (1 << high)) as usize,
    )
  }

  fn slot(self) -> &'static Slot {
    let (page, offset) = self.location();
    let base = self.shard().pages[page].load(Ordering::Acquire);
    // SAFETY: claim publishes this page before exposing a key. Slots never
    // move, its allocation is permanent, and location indexes this size class.
    unsafe { &*base.add(offset) }
  }

  fn return_to_shard(self) {
    self.shard().with_free(|free| {
      self
        .slot()
        .strong
        .store(free.head as usize, Ordering::Relaxed);
      free.head = self.0;
    });
  }

  pub(super) fn unique(self) -> bool {
    self.slot().strong.load(Ordering::Acquire) == 1
  }

  /// The caller pins a correctly typed Vec or boxed-window slot throughout the read.
  pub(super) unsafe fn base<T>(self) -> *mut T {
    // SAFETY: the source pins these parts throughout this read.
    unsafe { (*self.slot().parts.get()).base.cast() }
  }

  /// The caller's shared borrow must pin this immutable owner until return.
  pub(super) unsafe fn retain(self) {
    let old = self.slot().strong.fetch_add(1, Ordering::Relaxed);
    if old > isize::MAX as usize {
      super::super::shared::vector::refcount_overflow();
    }
  }

  /// The caller transfers the final reference and the stored pointer was
  /// obtained from a Vec whose elements can be owned at the caller's T.
  unsafe fn take_vec<T>(self, slot: &Slot) -> (Reservation, Vec<T>) {
    // SAFETY: no peer can access the parts. Clear the global pointer before
    // running any destructor or making the slot available for reuse.
    let parts = unsafe { slot.parts.get().replace(Parts::EMPTY) };
    // SAFETY: the pointer and capacity are the original allocator parts. Its
    // entire original prefix, including any hidden ends, remains initialized.
    (Reservation(self), unsafe {
      Vec::from_raw_parts(parts.base.cast(), parts.len, parts.capacity)
    })
  }

  /// The caller owns the typed handle and suppresses its Drop on success.
  pub(super) unsafe fn try_take_vec<T>(self) -> Option<(Reservation, Vec<T>)> {
    // No Weak handles exist. While consuming the sole handle, nobody can
    // create another strong reference. Acquire observes preceding releases.
    let slot = self.slot();
    (slot.strong.load(Ordering::Acquire) == 1).then(|| unsafe { self.take_vec(slot) })
  }

  /// The caller transfers one live immutable owner of this T exactly once.
  pub(super) unsafe fn release<T>(self) {
    let slot = self.slot();
    if slot.strong.fetch_sub(1, Ordering::Release) == 1 {
      fence(Ordering::Acquire);
      // SAFETY: the final release owns every original element and the allocation.
      let (reservation, values) = unsafe { self.take_vec::<T>(slot) };
      drop(reservation);
      drop(values);
    }
  }

  /// The caller transfers the only mutable handle, whose slot contains a Box<T>.
  pub(super) unsafe fn take_box<T>(self) -> (Reservation, Box<T>) {
    // SAFETY: a mutable window slot never has an immutable peer.
    let parts = unsafe { self.slot().parts.get().replace(Parts::EMPTY) };
    // SAFETY: install_box transferred this exact allocation to its handle.
    (Reservation(self), unsafe {
      Box::from_raw(parts.base.cast())
    })
  }
}

// Bound idle reservations per live thread and return all of them on thread
// exit. try_with falls back to the shard during or after TLS destruction.
#[cfg(all(feature = "std", unix))]
struct Cache {
  keys: [Cell<u32>; 64],
  len: Cell<usize>,
  shard: Cell<u32>,
}

#[cfg(all(feature = "std", unix))]
impl Cache {
  fn take(&self) -> Result<Key, usize> {
    let len = self.len.get();
    if len != 0 {
      self.len.set(len - 1);
      return Ok(Key(self.keys[len - 1].get()));
    }
    let mut shard = self.shard.get();
    if shard == NONE {
      shard = (NEXT_SHARD.fetch_add(1, Ordering::Relaxed) & (SHARDS - 1)) as u32;
      self.shard.set(shard);
    }
    Err(shard as usize)
  }

  fn put(&self, key: Key) {
    let mut len = self.len.get();
    if len == self.keys.len() {
      len /= 2;
      self.publish_above(len);
    }
    self.keys[len].set(key.0);
    self.len.set(len + 1);
  }

  #[cold]
  fn publish_above(&self, keep: usize) {
    let len = self.len.replace(keep);
    if len == keep {
      return;
    }
    let mut chains = [(NONE, NONE); SHARDS];
    let mut touched = 0u32;
    for entry in &self.keys[keep..len] {
      let key = Key(entry.get());
      let shard = key.0 as usize & (SHARDS - 1);
      let (head, tail) = &mut chains[shard];
      if *head == NONE {
        *tail = key.0;
        touched |= 1 << shard;
      } else {
        // Every detached entry is an exclusive empty reservation. No handle,
        // cache or shard can read its link until the chain is published.
        key.slot().strong.store(*head as usize, Ordering::Relaxed);
      }
      *head = key.0;
    }
    while touched != 0 {
      let shard = touched.trailing_zeros() as usize;
      touched &= touched - 1;
      let (head, tail) = chains[shard];
      let tail = &Key(tail).slot().strong;
      DIRECTORY[shard].with_free(|free| {
        tail.store(free.head as usize, Ordering::Relaxed);
        free.head = head;
      });
      // The unlock publishes every initialized link to the next lock holder.
    }
  }
}

#[cfg(all(feature = "std", unix))]
impl Drop for Cache {
  fn drop(&mut self) {
    self.publish_above(0);
  }
}

#[cfg(all(feature = "std", unix))]
std::thread_local! {
  static CACHE: Cache = const { Cache {
    keys: [const { Cell::new(0) }; 64], len: Cell::new(0), shard: Cell::new(NONE),
  } };
}

pub(super) struct Reservation(Key);

impl Reservation {
  pub(super) fn claim() -> Option<Self> {
    #[cfg(all(feature = "std", unix))]
    let hint = match CACHE.try_with(Cache::take) {
      Ok(Ok(key)) => return Some(Self(key)),
      Ok(Err(shard)) => shard,
      Err(_) => NEXT_SHARD.fetch_add(1, Ordering::Relaxed),
    };
    #[cfg(not(all(feature = "std", unix)))]
    let hint = NEXT_SHARD.fetch_add(1, Ordering::Relaxed);
    for delta in 0..SHARDS {
      let index = hint.wrapping_add(delta) & (SHARDS - 1);
      if let Some(key) = DIRECTORY[index].claim(index) {
        return Some(Self(key));
      }
    }
    None
  }

  fn install(self, parts: Parts) -> Key {
    let key = self.0;
    let slot = key.slot();
    // SAFETY: a reservation has unique access. It came from this same thread
    // or a lock-acquired free list, or nobody has used the slot previously.
    unsafe { slot.parts.get().write(parts) };
    slot.strong.store(1, Ordering::Relaxed);
    mem::forget(self);
    key
  }

  pub(super) fn install_vec<T>(self, values: Vec<T>) -> Key {
    let mut values = ManuallyDrop::new(values);
    self.install(Parts {
      base: values.as_mut_ptr().cast(),
      len: values.len(),
      capacity: values.capacity(),
    })
  }

  pub(super) fn install_box<T>(self, value: Box<T>) -> Key {
    self.install(Parts {
      base: Box::into_raw(value).cast(),
      len: 0,
      capacity: 0,
    })
  }
}

impl Drop for Reservation {
  fn drop(&mut self) {
    #[cfg(all(feature = "std", unix))]
    if CACHE.try_with(|cache| cache.put(self.0)).is_ok() {
      return;
    }
    self.0.return_to_shard();
  }
}

#[cfg(test)]
mod tests;
