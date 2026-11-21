# smol-slice

`Slice<T, N>` and `SliceMut<T, N>` own small sequences of `Copy` elements inline.
Both handles expose the remaining slice and can advance past a prefix. `SliceMut`
also supports in-place mutation.

```rust
use smol_slice::SliceMut;

let mut bytes = SliceMut::<u8, 22>::try_from_slice(b"key=value").unwrap();
bytes.advance(4);
bytes.as_mut_slice().make_ascii_uppercase();
assert_eq!(bytes.as_slice(), b"VALUE");
```

The inline cursor addresses up to `min(N, 126)` elements. Construction returns
`None` when the input exceeds that capacity. Both handles use only `core` and
can be used with default features disabled.

On 64-bit targets with 8-byte word alignment, the byte handles with capacities
22, 30 and 62 occupy 24, 32 and 64 bytes respectively, including inside
`Option`. Typed inline storage preserves element alignment, so other element
types can require more space.

The layout reserves two pointers and a length for an external owner. All current
public constructors store elements inline. Each tier uses the compact external
descriptor. Direct wide spill metadata and external ownership are deferred.

Licensed under either [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

Copyright (c) 2021 Al Liu.
