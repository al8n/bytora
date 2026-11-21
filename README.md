# smol-slice

`Slice<T, BYTES>` and `SliceMut<T, BYTES>` own small sequences of `Copy`
elements inline. `BYTES` is the exact size of the handle, including its cursor.
Wrapping either handle in `Option` uses the same number of bytes. The default
matches the three-word size of a `Vec<T>`, or 12 bytes on 32-bit targets and
24 bytes on 64-bit targets.

```rust
use smol_slice::SliceMut;

let mut bytes = SliceMut::<u8, 24>::try_from_slice(b"key=value").unwrap();
bytes.advance(4);
bytes.as_mut_slice().make_ascii_uppercase();
assert_eq!(bytes.as_slice(), b"VALUE");
```

On 32-bit targets, supported sizes are multiples of 4 from 4 through 8192 bytes.
On 64-bit targets, supported sizes are multiples of 8 from 8 through 8192 bytes.
Advancing changes the cursor in constant time. The elements stay in place until
the handle moves. Both handles use only `core`.

| Handle size | Cursor bytes | Inline capacity for `u8` |
| --- | --- | --- |
| 4 (32-bit only), 8 | 1 | 3, 7 |
| 12 (32-bit only) or 16 through 128 | 2 | 10 or 14 through 126 |
| 132 (32-bit only) or 136 through 8192 | 4 | 128 or 132 through 8188 |

For example, `Slice<u8, 24>` holds 22 bytes and `Slice<u32, 24>` holds five
elements. `INLINE_CAPACITY` gives the limit for a particular type and budget.
Construction returns `None` if the input exceeds it. All handles have word
alignment. A nonzero-sized element that needs greater alignment supports only
the empty slice. Each zero-sized element counts as one payload byte, including
when it has greater alignment than the handle.
The same size and alignment rules apply to downstream types, including
`#[repr(transparent)]` wrappers; no registration of element types is needed.

Use `MAX_HANDLE_BYTES` for the largest supported total handle size:

```rust
use smol_slice::{Slice, MAX_HANDLE_BYTES};

#[repr(transparent)]
struct UserId(u16);

assert_eq!(MAX_HANDLE_BYTES, 8192);
assert_eq!(Slice::<u8, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 8188);
assert_eq!(Slice::<UserId, MAX_HANDLE_BYTES>::INLINE_CAPACITY, 4094);
```

The layout is available for non-`Copy` types too; `try_from_slice` needs
`T: Copy` to copy elements from the borrowed slice.

Generic functions can express the size constraint with `SupportedBytes`.

```rust
use smol_slice::{Slice, SupportedBytes};

fn copy_inline<T: Copy, const BYTES: usize>(values: &[T]) -> Option<Slice<T, BYTES>>
where
    [(); BYTES]: SupportedBytes,
{
    Slice::try_from_slice(values)
}

assert_eq!(copy_inline::<_, 8>(b"small").unwrap().as_slice(), b"small");
```

An unsupported size is a compile-time error.

```compile_fail
use core::mem::size_of;
use smol_slice::Slice;
let _ = Slice::<u8, { size_of::<usize>() / 2 }>::try_from_slice(b"a");
```

```compile_fail
use core::mem::size_of;
use smol_slice::Slice;
let _ = Slice::<u8, { 2 * size_of::<usize>() + size_of::<usize>() / 2 }>::try_from_slice(b"abc");
```

```compile_fail
use core::mem::size_of;
use smol_slice::SliceMut;
let _ = SliceMut::<u8, { 8192 + size_of::<usize>() }>::try_from_slice(&[]);
```

All public constructors store elements inline. Handles of at least three
machine words also have room for a private external descriptor with two
pointers and a length.

Licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

Copyright (c) 2021 Al Liu.
