#!/bin/bash
set -euo pipefail

# Miri under Tree Borrows, on one target.
#
# This script and `ci/miri_sb.sh` are deliberate near-duplicates: they differ in `MIRIFLAGS` and in
# nothing else, and are two files so that each workflow runs exactly one aliasing model.
#
# Usage: ci/miri_tb.sh <TARGET>

if [ -z "${1:-}" ]; then
  echo "Error: TARGET is not provided" >&2
  exit 1
fi

TARGET="$1"

# A C compiler and libc headers for the target. `cargo miri test` builds the dev-dependencies for
# TARGET, and a build script among them (criterion's `alloca` compiles C) needs both. Miri itself
# links nothing. The headers are named explicitly: `gcc-<arch>-linux-gnu` only Recommends its
# `libc6-dev-<arch>-cross`, so `--no-install-recommends` leaves out `bits/libc-header-start.h` and
# the build script fails with it. `cc` finds the compiler by its `<arch>-linux-gnu-gcc` name.
if [ "$(uname)" = "Linux" ]; then
  packages=()
  case "$TARGET" in
    aarch64-unknown-linux-gnu) packages=(gcc-aarch64-linux-gnu libc6-dev-arm64-cross) ;;
    i686-unknown-linux-gnu) packages=(gcc-multilib) ;;
    powerpc64-unknown-linux-gnu) packages=(gcc-powerpc64-linux-gnu libc6-dev-ppc64-cross) ;;
    s390x-unknown-linux-gnu) packages=(gcc-s390x-linux-gnu libc6-dev-s390x-cross) ;;
    riscv64gc-unknown-linux-gnu) packages=(gcc-riscv64-linux-gnu libc6-dev-riscv64-cross) ;;
  esac
  if [ "${#packages[@]}" -gt 0 ]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${packages[@]}"
  fi
fi

rustup toolchain install nightly --profile minimal --component miri --component rust-src
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check -Zmiri-tree-borrows"

# Two passes, the two ends of the feature range: everything on, then nothing on. The second is the
# core tier, where code gated on `not(feature = "std")` lives and `--all-features` never compiles
# it. Add a pass for any feature that gates `unsafe` code these two do not cover together.
#
# No `--all-targets`: it would interpret the criterion benches, which are measurement code and not
# code under test. The default selection runs the unit tests, the integration tests and the
# doctests, and only builds the examples.
cargo miri test --target "$TARGET" --all-features
cargo miri test --target "$TARGET" --no-default-features
