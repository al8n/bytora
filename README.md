# smol-slice

`Slice<T, S>` and `SliceMut<T, S>` own small sequences of `Copy` elements
inline. An immutable `Slice` can also borrow static data or share a vector's allocation.
The typenum integer `S` gives the exact size in bytes of the handle,
including its cursor. Wrapping either handle in `Option` uses the same number
of bytes. `Slice<T>` and `SliceMut<T>` default to the three-word size of a
`Vec<T>`, or 12 bytes on 32-bit targets and 24 bytes on 64-bit targets.

`SliceN<T, N>` and `SliceMutN<T, N>` name the same types using a numeric byte count.

```rust
use smol_slice::{SliceMut, SliceMutN, typenum::U24};

let mut bytes: SliceMut<u8, U24> = SliceMutN::<u8, 24>::try_from_slice(b"key=value").unwrap();
bytes.advance(4);
bytes.as_mut_slice().make_ascii_uppercase();
assert_eq!(bytes.as_slice(), b"VALUE");
```

On 32-bit targets, supported sizes are multiples of 4 from 4 through 8192 bytes.
On 64-bit targets, supported sizes are multiples of 8 from 8 through 8192 bytes.
Advancing changes the cursor in constant time. Moving an inline handle moves
its elements. The inline constructors and operations are available with only `core`.

| Handle size | Cursor bytes | Inline capacity for `u8` |
| --- | --- | --- |
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
the total byte budget. `INLINE_CAPACITY` counts the elements that fit.

```rust
use smol_slice::{InlineSize, Slice, typenum::U24};

fn copy_inline<T: Copy, S: InlineSize>(values: &[T]) -> Option<Slice<T, S>> {
    assert_eq!(size_of::<Slice<T, S>>(), S::USIZE);
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
use smol_slice::{InlineSize, typenum::U0};
fn supported<S: InlineSize>() {}
supported::<U0>();
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

For an explicit handle size, `try_from_static` returns `None` if its budget is
below three words or its length exceeds `isize::MAX` elements. The default
`from_static` constructor panics on an oversized slice of zero-sized elements.

With the `std` or `alloc` feature on targets with pointer-width atomics,
`Slice::try_from_vec` takes a `Vec<T>` of `Copy + Send + Sync` elements. It
copies short inputs inline. For larger inputs it retains the vector's buffer
and allocates a shared owner. Each clone or range increments the owner's atomic
reference count and keeps its own cursor. Dropping the last handle frees the
owner and the vector.

Shared storage requires a handle of at least three machine words and a length
of at most `isize::MAX` elements. If the elements need shared storage and the
handle or length cannot accommodate them, the constructor returns the original
vector in `Err`.
The default `Slice<u8>` on a 64-bit target keeps up to 22 bytes inline.

```rust
# #[cfg(all(any(feature = "std", feature = "alloc"), target_has_atomic = "ptr"))]
# {
use smol_slice::Slice;

let whole = Slice::<u8>::try_from_vec((0..32).collect()).unwrap();
let mut tail = whole.clone();
tail.advance(30);
assert_eq!(tail.as_slice(), &[30, 31]);
assert_eq!(whole.len(), 32);
# }
```

Licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

Copyright (c) 2021 Al Liu.
