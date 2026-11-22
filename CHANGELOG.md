# Changelog

All notable changes to this crate are documented here. The project follows semantic versioning;
before 1.0, a minor bump (0.x to 0.y) signals a breaking change.

New entries go under `## Unreleased`. A release renames that section to `## X.Y.Z (YYYY-MM-DD)` and
opens a new, empty `## Unreleased` above it. `ci/changelog.sh` checks this structure on every pull
request, and `release.yml` publishes the released section as the GitHub Release notes.

## Unreleased

- Default and zero-budget `Slice`, `SliceMut`, and `LocalSlice` handles and
  their `Option`s use four machine words, matching `bytes::Bytes` (16 bytes on
  32-bit targets and 32 bytes on 64-bit targets). Both default `Slice<u8>` and
  `SliceMut<u8>` hold 30 inline bytes on 64-bit targets and 14 on 32-bit targets.
  The zero-budget inline capacity remains zero.
