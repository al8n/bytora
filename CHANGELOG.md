# Changelog

All notable changes to this crate are documented here. The project follows semantic versioning;
before 1.0, a minor bump (0.x to 0.y) signals a breaking change.

New entries go under `## Unreleased`. A release renames that section to `## X.Y.Z (YYYY-MM-DD)` and
opens a new, empty `## Unreleased` above it. `ci/changelog.sh` checks this structure on every pull
request, and `release.yml` publishes the released section as the GitHub Release notes.

## Unreleased

- Default and zero-budget `Vector`, `VectorMut`, and `LocalVector` handles and
  their `Option`s use four machine words, matching `bytes::Bytes` (16 bytes on
  32-bit targets and 32 bytes on 64-bit targets). Both default `Vector<u8>` and
  `VectorMut<u8>` hold 30 inline bytes on 64-bit targets and 14 on 32-bit targets.
  The zero-budget inline capacity remains zero.
- `Bytes`, `BytesMut`, `BytesN` and `BytesMutN` name the byte handles, and
  `DefaultSize` names the default budget. `Bytes` and `BytesMut` have no budget
  parameter, so their constructors need no type annotation.
- `Vector` and `VectorMut` implement `Deref` to a slice, `AsRef`, `Borrow`,
  `PartialEq`, `Eq`, `PartialOrd`, `Ord` and `Hash` by their elements, the same
  comparisons with slices, arrays and vectors, and for bytes with `str` and
  `String`, on either side, `Debug` as a slice prints, `LowerHex` and `UpperHex`
  for bytes, and `IntoIterator` by reference. `VectorMut` adds `DerefMut`,
  `AsMut`, `BorrowMut`, `Default`, `Clone`, `Extend` and `FromIterator`, and
  `Vector` adds `FromIterator`. The conversions of `bytes` are added: from static
  slices and text without copying, from `String`, from `VectorMut<u8>` into
  `Vector<u8>`, from `VectorMut` into `Vec`, and from `&[u8]` and `&str` into
  `VectorMut<u8>`.
- `From<Vec<T>>` replaces `TryFrom<Vec<T>>` for `Vector<T, U0>`, and the default
  budget has it too. It adopts the vector as `try_from_vec` does, so an
  unallocated empty vector now gives an inline empty handle. A vector of
  zero-sized elements longer than `isize::MAX` panics; `try_from_shared_vec`
  returns it.
- The `error` module holds the errors of the `try_*` operations:
  `OutOfBounds`, `RangeOutOfBounds`, `TryPutError`, `TryReserveError` and
  `ByteEditError`. `try_split_to`, `try_split_off` and `try_slice_ref` return an
  error or `None` where `split_to`, `split_off` and `slice_ref` panic.
  `try_copy_from_slice` returns `Result<_, TryPutError>` instead of `Option`, and
  `from_slice` copies as `copy_from_slice` does.
- The twins of the operations that guard storage: `try_slice` on `Vector` and
  `LocalVector`, and `try_advance` on those and on `VectorMut`;
  `VectorMut::try_split_off`,
  `try_reserve`, `try_with_capacity`, `try_extend_from_slice`,
  `try_extend_from_within` and `try_zeroed`. Each returns an error where its
  panicking form panics, and leaves the elements unchanged when it does. The
  allocating twins also return `TryReserveError::AllocError` when the
  allocator refuses, where the panicking forms abort. `VectorMut::try_resize`
  and `try_extend` are the twins of `resize` and `Extend`, and `VectorMut<u8>`
  implements `fmt::Write`, which returns `fmt::Error` and appends nothing where
  `try_extend_from_slice` returns an error.
