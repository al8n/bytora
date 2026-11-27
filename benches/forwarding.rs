//! A store-to-load forwarding probe. It holds no product code.
//!
//! The N = 0 `consume` and the 32 B `warmed_clone` reload words from a stack temporary that was
//! written in other widths, and the gap to `Bytes` follows the CPU. This bench writes a 64-byte
//! stack slot and reads it back in those shapes, one round trip per iteration, so that the price of
//! each shape on each CPU is a row of its own:
//!
//! | shape | stores | reload |
//! |---|---|---|
//! | `u8_over_u64` | one 8-byte word, then a 1-byte store into its last byte | the 8-byte word, across both |
//! | `u64x2_to_u128` | two 8-byte words | one 16-byte load, across both |
//! | `u128_to_u64` | one 16-byte store | an 8-byte load of its upper half |
//! | `u64x2_to_ldp` | two 8-byte words | an AArch64 `ldp` whose halves come from two stores |
//! | `no_forward` | an 8-byte word | an 8-byte load of another cache line: the control |
//! | `alias_4k` | an 8-byte word | an 8-byte load 4096 bytes away, the 4 KiB aliasing control |
//!
//! Each shape runs in six variants, `<lock>_<sink>`:
//!
//! - `plain`: nothing else. `locked`: a relaxed `fetch_add` (the clone's `lock incq`, or the
//!   AArch64 `ldadd`) between the stores and the reload. `lock_first`: the same before the stores,
//!   as the clone orders it.
//! - `store`: the reload feeds the next round's stores, so a failed forward is on the critical
//!   path. `branch`: the reload is only compared, and the next round does not depend on it.
//!
//! The same stack slot is used on every iteration. Every access is a volatile load or store of the
//! stated width (a vector access for 16 bytes on x86-64 and AArch64, a `u128` elsewhere), so the
//! compiler can neither fold a store into its reload nor merge two stores; `ldp` is inline `asm!`.
//! `cargo bench --bench forwarding` is the only caller; ids are `forwarding/<shape>/<variant>`, which
//! the summarizer lists flat.

use criterion::{
  BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use std::{
  hint::black_box,
  ptr::{read_volatile, write_volatile},
  sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(target_arch = "x86_64")]
mod wide {
  use core::arch::x86_64::{__m128i, _mm_cvtsi128_si64, _mm_set_epi64x};
  use std::ptr::{read_volatile, write_volatile};

  /// A 16-byte volatile vector store of `lo` then `hi` to `p`, which is 16-byte aligned.
  #[inline(always)]
  pub(super) unsafe fn store(p: *mut u8, lo: u64, hi: u64) {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe { write_volatile(p.cast::<__m128i>(), _mm_set_epi64x(hi as i64, lo as i64)) }
  }

  /// A 16-byte volatile vector load from `p`, which is 16-byte aligned. Returns its low half.
  #[inline(always)]
  pub(super) unsafe fn load(p: *const u8) -> u64 {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe { _mm_cvtsi128_si64(read_volatile(p.cast::<__m128i>())) as u64 }
  }
}

#[cfg(target_arch = "aarch64")]
mod wide {
  use core::arch::aarch64::{uint64x2_t, vdupq_n_u64, vgetq_lane_u64, vsetq_lane_u64};
  use std::ptr::{read_volatile, write_volatile};

  /// A 16-byte volatile vector store of `lo` then `hi` to `p`, which is 16-byte aligned.
  #[inline(always)]
  pub(super) unsafe fn store(p: *mut u8, lo: u64, hi: u64) {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe {
      write_volatile(
        p.cast::<uint64x2_t>(),
        vsetq_lane_u64::<1>(hi, vdupq_n_u64(lo)),
      )
    }
  }

  /// A 16-byte volatile vector load from `p`, which is 16-byte aligned. Returns its low half.
  #[inline(always)]
  pub(super) unsafe fn load(p: *const u8) -> u64 {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe { vgetq_lane_u64::<0>(read_volatile(p.cast::<uint64x2_t>())) }
  }

  /// One `ldp` of the two words at `p`, which is 8-byte aligned. Returns their xor.
  #[inline(always)]
  pub(super) unsafe fn load_pair(p: *const u8) -> u64 {
    let (low, high): (u64, u64);
    // SAFETY: the caller passes a valid `p` with 16 readable bytes; `ldp` writes only its outputs.
    unsafe {
      core::arch::asm!(
        "ldp {low}, {high}, [{p}]",
        low = out(reg) low,
        high = out(reg) high,
        p = in(reg) p,
        options(nostack, readonly, preserves_flags),
      );
    }
    low ^ high
  }
}

// Neither vector ISA: a `u128` access stands in for the 16-byte one.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod wide {
  use std::ptr::{read_volatile, write_volatile};

  #[inline(always)]
  pub(super) unsafe fn store(p: *mut u8, lo: u64, hi: u64) {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe { write_volatile(p.cast::<u128>(), (u128::from(hi) << 64) | u128::from(lo)) }
  }

  #[inline(always)]
  pub(super) unsafe fn load(p: *const u8) -> u64 {
    // SAFETY: the caller passes a valid, 16-byte aligned `p`.
    unsafe { read_volatile(p.cast::<u128>()) as u64 }
  }
}

const SLOT_BYTES: usize = 8192;

/// The stack slot: 64-byte aligned, and long enough for a load 4096 bytes from a store.
#[repr(C, align(64))]
struct Slot([u8; SLOT_BYTES]);

static COUNTER: AtomicUsize = AtomicUsize::new(0);

const BYTE_OVER_WORD: u8 = 0;
const WORDS_TO_WIDE: u8 = 1;
const WIDE_TO_WORD: u8 = 2;
#[cfg(target_arch = "aarch64")]
const WORDS_TO_PAIR: u8 = 3;
const NO_FORWARD: u8 = 4;
const ALIAS_4K: u8 = 5;

const PLAIN: u8 = 0;
const LOCKED: u8 = 1;
const LOCK_FIRST: u8 = 2;

const STORE: u8 = 0;
const BRANCH: u8 = 1;

/// The locked increment between the stores and the reload.
#[inline(always)]
fn locked<const LOCK: u8>() {
  if LOCK == LOCKED {
    let _ = COUNTER.fetch_add(1, Ordering::Relaxed);
  }
}

/// One round trip: store `data` in the shape's widths, reload, and return what was reloaded.
///
/// # Safety
///
/// `p` points at a `Slot`.
#[inline(always)]
unsafe fn round<const SHAPE: u8, const LOCK: u8>(p: *mut u8, data: u64) -> u64 {
  if LOCK == LOCK_FIRST {
    let _ = COUNTER.fetch_add(1, Ordering::Relaxed);
  }
  // SAFETY: every offset is inside the slot and aligned for its access.
  unsafe {
    match SHAPE {
      BYTE_OVER_WORD => {
        write_volatile(p.cast::<u64>(), data);
        write_volatile(p.add(7), (data >> 3) as u8);
        locked::<LOCK>();
        read_volatile(p.cast::<u64>())
      }
      WORDS_TO_WIDE => {
        write_volatile(p.cast::<u64>(), data);
        write_volatile(p.add(8).cast::<u64>(), data.wrapping_add(1));
        locked::<LOCK>();
        wide::load(p)
      }
      WIDE_TO_WORD => {
        wide::store(p, data, data.wrapping_add(1));
        locked::<LOCK>();
        read_volatile(p.add(8).cast::<u64>())
      }
      #[cfg(target_arch = "aarch64")]
      WORDS_TO_PAIR => {
        write_volatile(p.cast::<u64>(), data);
        write_volatile(p.add(8).cast::<u64>(), data.wrapping_add(1));
        locked::<LOCK>();
        wide::load_pair(p)
      }
      NO_FORWARD => {
        write_volatile(p.cast::<u64>(), data);
        locked::<LOCK>();
        read_volatile(p.add(256).cast::<u64>())
      }
      ALIAS_4K => {
        write_volatile(p.cast::<u64>(), data);
        locked::<LOCK>();
        read_volatile(p.add(4096).cast::<u64>())
      }
      _ => unreachable!("unknown shape"),
    }
  }
}

fn register<const SHAPE: u8, const LOCK: u8, const SINK: u8>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  id: &str,
) {
  group.bench_function(id, |b| {
    let mut slot = Slot([0; SLOT_BYTES]);
    let p = slot.0.as_mut_ptr();
    let mut acc = 1u64;
    let mut hits = 0u64;
    b.iter(|| {
      // SAFETY: `p` points at `slot`, which outlives the loop.
      let loaded = unsafe { round::<SHAPE, LOCK>(p, acc) };
      if SINK == STORE {
        acc = loaded.wrapping_add(1);
      } else {
        acc = acc.wrapping_add(1);
        if loaded == u64::MAX {
          hits = hits.wrapping_add(1);
        }
      }
      acc ^ hits
    });
    black_box(&slot);
  });
}

macro_rules! shape {
  ($group:ident, $name:literal, $shape:expr) => {
    register::<{ $shape }, PLAIN, STORE>(&mut $group, concat!($name, "/plain_store"));
    register::<{ $shape }, PLAIN, BRANCH>(&mut $group, concat!($name, "/plain_branch"));
    register::<{ $shape }, LOCKED, STORE>(&mut $group, concat!($name, "/locked_store"));
    register::<{ $shape }, LOCKED, BRANCH>(&mut $group, concat!($name, "/locked_branch"));
    register::<{ $shape }, LOCK_FIRST, STORE>(&mut $group, concat!($name, "/lock_first_store"));
    register::<{ $shape }, LOCK_FIRST, BRANCH>(&mut $group, concat!($name, "/lock_first_branch"));
  };
}

fn benchmarks(c: &mut Criterion) {
  let mut group = c.benchmark_group("forwarding");
  shape!(group, "u8_over_u64", BYTE_OVER_WORD);
  shape!(group, "u64x2_to_u128", WORDS_TO_WIDE);
  shape!(group, "u128_to_u64", WIDE_TO_WORD);
  #[cfg(target_arch = "aarch64")]
  shape!(group, "u64x2_to_ldp", WORDS_TO_PAIR);
  shape!(group, "no_forward", NO_FORWARD);
  shape!(group, "alias_4k", ALIAS_4K);
  group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
