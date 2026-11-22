# smol-slice

`Slice<T, S>` and `SliceMut<T, S>` own small sequences of `Copy` elements
inline. An immutable `Slice` can also borrow static data or share owned data.
With `std` or `alloc`, a `SliceMut` can hold a growable vector-backed window.
For a positive typenum integer `S`, the handle occupies exactly `S` bytes,
including its cursor. `U0` disables inline elements and uses three machine words.
Wrapping either handle in `Option` uses the same number
of bytes. `Slice<T>` and `SliceMut<T>` default to the three-word size of a
`Vec<T>`, or 12 bytes on 32-bit targets and 24 bytes on 64-bit targets.

`SliceN<T, N>` and `SliceMutN<T, N>` name the same types using a numeric byte count.
The inherent APIs draw on [`bytes::Bytes`](https://docs.rs/bytes/1.12.2/bytes/struct.Bytes.html)
and [`bytes::BytesMut`](https://docs.rs/bytes/1.12.2/bytes/struct.BytesMut.html),
while their generic element types and inline storage introduce the limits described below.

```rust
use smol_slice::{SliceMut, SliceMutN, typenum::U24};

let mut bytes: SliceMut<u8, U24> = SliceMutN::<u8, 24>::try_from_slice(b"key=value").unwrap();
bytes.advance(4);
bytes.as_mut_slice().make_ascii_uppercase();
assert_eq!(bytes.as_slice(), b"VALUE");
```

Positive budgets are multiples of 4 through 8192 on 32-bit targets,
and multiples of 8 through 8192 on 64-bit targets.
Advancing changes the cursor in constant time. Moving an inline handle moves
its elements. The inline constructors and operations are available with only `core`.

| Requested budget | Inline cursor bytes | Inline capacity for `u8` |
| --- | --- | --- |
| 0 (three-word handle) | — | 0 |
| 4 (32-bit only), 8 | 1 | 3, 7 |
| 12 (32-bit only) or 16 through 128 | 2 | 10 or 14 through 126 |
| 132 (32-bit only) or 136 through 8192 | 4 | 128 or 132 through 8188 |

For example, `SliceN<u8, 24>` holds 22 bytes and `SliceN<u32, 24>` holds five
elements. `INLINE_CAPACITY` gives the limit for a particular type and budget.
`try_from_slice` returns `None` if the input exceeds it. All handles have word
alignment. A nonzero-sized element that needs greater alignment has zero
inline capacity. Each zero-sized element counts as one inline payload byte,
including when it has greater alignment than the handle.
The same size and alignment rules apply to downstream types, including
`#[repr(transparent)]` wrappers.

Use `MAX_HANDLE_BYTES` for the largest supported total handle size.

```rust
use smol_slice::{SliceN, MAX_HANDLE_BYTES};

#[repr(transparent)]
struct UserId(u16);

assert_eq!(MAX_HANDLE_BYTES, 8192);
assert_eq!(SliceN::<u8, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 8188);
assert_eq!(SliceN::<UserId, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
```

The layout is available for non-`Copy` types too. `try_from_slice` needs
`T: Copy` to copy elements from the borrowed slice.

Generic code can use `S: InlineSize` for either handle. `S::USIZE` gives
the requested budget. `S::HANDLE_BYTES` gives the actual handle size,
including for `U0`. `INLINE_CAPACITY` counts the elements that fit.
Generic mutable constructors can use `S: WritableSize<T>`: positive sizes require
`T: Copy`, while `U0` accepts any element type.

```rust
use smol_slice::{InlineSize, Slice, typenum::U24};

fn copy_inline<T: Copy, S: InlineSize>(values: &[T]) -> Option<Slice<T, S>> {
    assert_eq!(size_of::<Slice<T, S>>(), S::HANDLE_BYTES);
    Slice::try_from_slice(values)
}

let read = copy_inline::<_, U24>(b"small").unwrap();
assert_eq!(read.as_slice(), b"small");
```

Functions that take a `const N: usize` can use `AssocInlineSize` to look up its
corresponding type-level size. The numeric aliases cover every supported size,
including values without a named typenum constant.

```rust
use smol_slice::{AssocInlineSize, SliceN};

fn copy_inline<T: Copy, const N: usize>(values: &[T]) -> Option<SliceN<T, N>>
where
    [u8; N]: AssocInlineSize,
{
    SliceN::<T, N>::try_from_slice(values)
}

assert_eq!(copy_inline::<_, 4104>(b"small").unwrap().as_slice(), b"small");
```

An unsupported size is a compile-time error.

```compile_fail
use smol_slice::{InlineSize, typenum::U1};
fn supported<S: InlineSize>() {}
supported::<U1>();
```

```compile_fail
use core::mem::size_of;
use smol_slice::SliceN;
let _ = SliceN::<u8, { size_of::<usize>() / 2 }>::try_from_slice(b"a");
```

```compile_fail
use core::mem::size_of;
use smol_slice::SliceN;
let _ = SliceN::<u8, { 2 * size_of::<usize>() + size_of::<usize>() / 2 }>::try_from_slice(b"abc");
```

```compile_fail
use core::mem::size_of;
use smol_slice::SliceMutN;
let _ = SliceMutN::<u8, { 8192 + size_of::<usize>() }>::try_from_slice(&[]);
```

The crate seals `InlineSize` so each supported size has a checked layout.

```compile_fail
use smol_slice::InlineSize;
struct CustomSize;
impl InlineSize for CustomSize {}
```

## Cloning, slicing, and shared storage

Cloning an inline `Slice` copies the entire handle, including its cursor.
`Slice::slice(range)` returns an independent window relative to the current
view. It copies an inline handle once. A static or heap-backed window keeps
pointing into the original storage, even if the range is short or empty.
`slice_ref` accepts a borrowed subslice of the current view; any empty slice is
accepted. For zero-sized elements, addresses cannot encode the element offset
or reliably distinguish allocations, so a matching pointer and fitting length
select a prefix. `split_off` and `split_to` partition the view; an empty side
uses a new empty handle. `truncate(0)` and `clear()` release this handle's
ownership, even if an earlier empty `slice` retained it.
`Slice::from_static` borrows a `&'static [T]` directly, even for short inputs.
Constructing, slicing, cloning, and dropping static handles use no allocations or reference counts.
Static storage works with just `core` on targets with or without atomics.
Elements can be any `Sync` type, including types with no `Clone` implementation.

```rust
use smol_slice::{Slice, SliceN};

static HEADER: Slice<u8> = Slice::from_static(b"content-type");
let field = HEADER.slice(8..);
assert_eq!(field.as_slice(), b"type");
assert_eq!(field.as_slice().as_ptr(), HEADER.as_slice()[8..].as_ptr());
assert!(SliceN::<u8, 8>::try_from_static(b"id").is_none());
```

Cloning a static or heap-backed value constructs a descriptor without copying
the elements. For a wide handle, reusing an existing destination with
`clone_from` can let the compiler replace just that descriptor. An ordinary
`clone` returns the whole handle, and subsequent moves can copy its full width.
The generated code depends on the compiler and the caller.

```rust
use smol_slice::SliceN;

let source = SliceN::<u8, 4096>::try_from_static(b"shared").unwrap();
let mut destination = SliceN::<u8, 4096>::try_from_slice(&[]).unwrap();
destination.clone_from(&source);
assert_eq!(destination.as_slice().as_ptr(), source.as_slice().as_ptr());
```

`try_from_static` returns `None` when the handle occupies fewer than three
words or the slice contains more than `isize::MAX` elements. The default
`from_static` constructor panics on an oversized slice of zero-sized elements.
Static constructors, `len` and `is_empty` are const functions. With shared heap
ownership enabled, the handle contains an interior-mutable owner slot. Rust
allows a `static` handle to be initialized at compile time and read at runtime.
For a compile-time length, use the original static slice's length. In core-only
builds, a constant can also read the length from the static handle itself.

With `std` or `alloc` and shared ownership available,
`Slice::try_from_vec` takes a `Vec<T>` of `Copy + Send + Sync` elements. A
handle of at least three machine words keeps the vector's allocation, even for
a short vector or an empty vector with spare capacity. A smaller handle copies
the elements inline when they fit. An empty vector without an allocation and
short vectors of zero-sized elements can also use inline storage.

On targets with suitable pointer alignment, an exact-capacity vector of
non-zero-sized elements needs no additional allocation on adoption or drop.
The descriptor records skipped prefixes and truncated suffixes of up to
`(1 << ((usize::BITS - 2) / 2)) - 1` elements each. The first clone, shared
range, or larger window adjustment allocates a reference-counted owner.
Concurrent first clones can allocate competing owners and publish only one.

For a vector with spare capacity, adoption initializes the owner immediately.
It uses aligned space beyond the initialized elements when the four-word owner
fits, and allocates a separate owner otherwise. Clones and ranges then need no
additional allocation. Shared zero-sized vectors and exact-capacity vectors on
targets with insufficient pointer alignment also use a separate owner.
Every shared range keeps an independent cursor. The final handle owns the
entire original vector allocation, including its spare capacity.

`try_from_shared_vec` always uses external storage, even for short inputs,
and accepts any `T: Send + Sync`. `Slice::from_owner` and `try_from_owner`
keep an `AsRef<[T]> + Send + 'static` owner alive without copying its slice;
their elements must be `Sync`. Shared ownership and its target requirements
are described under [Features and targets](#features-and-targets).

`Slice::try_from_boxed_slice` and `Slice::try_from_arc` adopt a `Box<[T]>`
or an already shared `Arc<[T]>` without copying the elements. Both accept
borrowed and non-`Clone` elements with `T: Send + Sync`. A box follows the
exact-capacity vector path, including allocation recovery through
`try_into_vec`. An Arc-backed window stays read-only. Adoption and ordinary
ranges reuse its existing reference count. A range whose hidden ends exceed
the descriptor's counts allocates a small owner that retains the original Arc.
`Slice` also implements `TryFrom` for both input types. A failed conversion
returns the original owner.

`LocalSlice<T>` adopts an already shared `Rc<[T]>` and stays on the same
thread. It requires only `alloc` or `std`, including on targets without
pointer-width atomics. Its default handle and its `Option` occupy three
machine words. `LocalSliceN<T, N>` selects a byte budget. Adoption, clones,
and normally sized ranges reuse the original Rc without allocating. A range
whose hidden prefix or suffix exceeds `(1 << (usize::BITS / 2 - 1)) - 1`
elements allocates a small owner for that view and its descendants.
`into_original_rc` returns the complete original allocation from any window.

```rust
# #[cfg(any(feature = "std", feature = "alloc"))]
# {
extern crate alloc;
use alloc::rc::Rc;
use core::cell::Cell;
use smol_slice::LocalSlice;

let values: Rc<[Cell<u8>]> = Rc::from([Cell::new(3), Cell::new(9)]);
let whole = LocalSlice::<_>::from_rc(values.clone());
let tail = whole.slice(1..);
tail.as_slice()[0].set(12);
assert_eq!(values[1].get(), 12);
assert!(Rc::ptr_eq(&values, &tail.into_original_rc()));
# }
```

Shared storage requires a handle of at least three machine words and a length
of at most `isize::MAX` elements. If the elements need shared storage and the
handle or length cannot accommodate them, the constructor returns the original
vector in `Err`.
The default `Slice<u8>` on a 64-bit target keeps up to 22 bytes inline.
`try_copy_from_slice` copies inline when possible and otherwise uses shared
storage when available, returning `None` if the input cannot fit;
`copy_from_slice` panics in that case. The sharing-enabled form requires
`T: Copy + Send + Sync`; without shared ownership it requires `T: Copy`.

```rust
# #[cfg(all(any(feature = "std", feature = "alloc"), any(target_has_atomic = "ptr", feature = "extra-platforms")))]
# {
use smol_slice::Slice;

let whole = Slice::<u8>::try_from_vec((0..32).collect()).unwrap();
let mut tail = whole.clone();
tail.advance(30);
assert_eq!(tail.as_slice(), &[30, 31]);
assert_eq!(whole.len(), 32);
# }
```

`Slice<T, U0>` and `SliceN<T, 0>` have zero inline capacity for every `T`,
including zero-sized types. `SliceN::<T, 0>::empty()` uses only `core` and
accepts any element type. `try_from_static` borrows a static slice of `Sync`
elements. With shared ownership available, `TryFrom<Vec<T>>`
takes ownership of `Send + Sync` elements, including types that implement
neither `Copy` nor `Clone`. It retains an empty vector's allocation when one
exists. An empty vector without an allocation yields `Slice::empty()`. Clones
and ranges keep the original vector's elements alive until the last handle
drops, even when its window is empty.

```rust
# #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
# {
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use smol_slice::SliceN;

let counters = SliceN::<AtomicUsize, 0>::try_from(vec![AtomicUsize::new(0)]).unwrap();
let peer = counters.clone();
peer.as_slice()[0].fetch_add(1, Relaxed);
assert_eq!(counters.as_slice()[0].load(Relaxed), 1);
# }
```

`is_unique` reports whether the current immutable window can become mutable
without copying. Concurrent clones can change that answer. `try_into_mut`
returns the original handle when a vector-backed window is shared or when
storage is static or supplied by a custom owner. Thawing an empty inline
slice, including `Slice::new()`, gives the mutable handle zero capacity.
Its first growth needs heap storage. With `alloc` or `std`, `into_mut` clones
the visible elements when needed. `try_into_vec` transfers movable elements
without cloning and returns the handle otherwise. `into_vec` can clone when
`T: Clone`. A uniquely owned vector allocation can be reused, while an
inline or disjoint window may need a new vector.

## Mutable storage

`SliceMut<T, U0>` and `SliceMutN<T, 0>` have zero inline capacity. Their
`empty()` constructor works with only `core`. With `alloc` or `std`,
`From<Vec<T>>` accepts any element type, including thread-local types and
atomics, and preserves the vector's allocation and spare capacity. A vector
with no capacity needs no heap window; an empty vector with spare capacity keeps
that capacity. This storage also works on targets without pointer-width atomics.
For other sizes, `try_from_vec` can take any element type and returns the vector
unchanged if the handle cannot hold a heap window.

`as_mut_slice()` borrows the current window exclusively. `advance` skips a
prefix in constant time. Each window owns its elements, including any skipped
prefix, until they are reclaimed or that window is dropped.

`SliceMut::new` starts empty with inline capacity for positive budgets and
`Copy` elements; `U0` accepts any element type and starts without inline space.
`reserve` and `with_capacity` may spill a handle of at least two machine words
to a vector-backed window with `std` or `alloc`. A one-word handle, or any
handle built without an allocator, remains bounded by its inline capacity.
`try_from_slice` only attempts inline storage; it does not spill.

Mutable `split_off(at)` divides both the current elements and capacity, and
accepts `at <= capacity`, including an index in spare capacity. `split_to(at)`
accepts `at <= len`, while `split()` takes the initialized part. Inline splits
move the suffix into a separate inline handle. Heap splits use disjoint shared
windows when shared ownership is available; otherwise the suffix moves to a new
allocation.
`try_unsplit` joins eligible adjacent shared heap windows without moving
elements, or returns the other window on failure. `unsplit` can instead move
elements and grow storage when an in-place join is unavailable.

`try_freeze` converts a mutable window into a `Slice` without copying elements
and returns the original on failure. `freeze` panics on the same failures.
Both methods take `S: WritableSize<T>`. Positive budgets require `T: Copy`,
and `U0` accepts any element type. When shared ownership is enabled, both
also require `T: Send + Sync`. Without shared ownership, only inline windows
can freeze.

With shared ownership, `try_freeze_shared` accepts any `Send + Sync` element
type for any budget. It preserves the allocation of a heap window and moves
the elements of a nonempty inline window into shared storage. A shared
descriptor needs three machine words and a length of at most `isize::MAX`.
A two-word mutable handle can spill, but a request to freeze that spill returns
the original handle through either fallible method. `SliceMut::into_vec` reuses
a unique heap allocation where possible and otherwise moves its elements to a
new vector without requiring `Clone`.

```rust
# #[cfg(any(feature = "std", feature = "alloc"))]
# {
use smol_slice::SliceMutN;

let mut words = SliceMutN::<String, 0>::from(vec!["one".into(), "two".into()]);
words.advance(1);
words.as_mut_slice()[0].push('!');
assert_eq!(words.as_slice(), &["two!"]);
# }
```

```rust
# #[cfg(all(any(feature = "std", feature = "alloc"), any(target_has_atomic = "ptr", feature = "extra-platforms")))]
# {
use smol_slice::SliceMut;

let mut first = SliceMut::<u8>::with_capacity(32);
first.extend_from_slice(b"hello");
let second = first.split_off(3);
assert!(first.try_unsplit(second).is_ok());
let frozen = first.freeze();
let mut writable = frozen.try_into_mut().ok().unwrap();
writable.as_mut_slice()[0] = b'H';
assert_eq!(writable.into_vec().as_slice(), b"Hello");
# }
```

For a concrete budget, a `SliceMut` containing references can be moved into a
type with shorter reference lifetimes. When the budget is still a generic
`S: InlineSize`, Rust can shorten the lifetime of a borrowed view returned by
`as_slice()`, but cannot prove the same conversion for the whole mutable handle.

## Features and targets

The default `std` feature and the optional `alloc` feature support vector-backed
mutable storage and Rc-backed `LocalSlice`. Without `extra-platforms`, targets
with native pointer-width
compare-and-swap also enable shared ownership. Vectors use a strong reference
count, and custom owners and frozen mutable windows use the allocation library's
`Arc`. With neither feature, inline and static operations use only `core`.

The `extra-platforms` feature enables `alloc` and selects
`portable-atomic` and [`portable-atomic-util`](https://docs.rs/portable-atomic-util/)
for internal thread-safe sharing, including on targets with native atomics.
`Slice::try_from_arc` accepts `alloc::sync::Arc` wherever native pointer-width
atomics are available and `portable_atomic_util::Arc` on portable-only
targets. On a target without native
compare-and-swap, the application must configure a working portable-atomic
backend. For example, a single-core Cortex-M application can provide a real
critical section with these downstream dependencies:

```toml
[dependencies]
smol-slice = { path = "../smol-slice", default-features = false, features = ["extra-platforms"] }
portable-atomic = { version = "1", default-features = false, features = ["critical-section"] }
cortex-m = { version = "0.7", features = ["critical-section-single-core"] }
```

This particular provider is suitable only when the application's target and
execution model satisfy its single-core requirements. Other applications should
choose a [critical-section implementation](https://docs.rs/critical-section/)
and [portable-atomic configuration](https://docs.rs/portable-atomic/#optional-featurescfgs)
appropriate to their platform. An allocator is also required for heap storage.

Host checks can exercise the feature combinations with `cargo test`,
`cargo test --no-default-features`, `cargo test --no-default-features --features alloc`,
and `cargo test --no-default-features --features extra-platforms`.

Licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

Copyright (c) 2021 Al Liu.
