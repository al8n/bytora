# Source this from a bash step (`source ci/bench/cargo_artifacts.sh`); it defines `cargo_json`.
#
#   cargo_json FILE exe NAME          executable path of the bench target NAME
#   cargo_json FILE pid NAME          package_id of the package that owns bench target NAME
#   cargo_json FILE cfgs NAME         `msg` once per build-script-executed message of the package that
#                                     owns bench target NAME, then `cfg:<cfg>` for each cfg it set
#
# Also here: sha256_of, timed_collect and llvm_tools_dir, below.
#
# FILE is the stdout of `cargo ... --message-format=json-render-diagnostics`. Uses jq, or else
# `ci/bench/cargo_artifacts.py` (python3 or python); fails when neither works. A path comes out as
# cargo wrote it, so on Windows it is a Windows path: convert it with `cygpath -u` before bash uses
# it. Lines that are not JSON are skipped.

_cj_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

_cj_jq_exe='fromjson? | select(.reason == "compiler-artifact" and .target.name == $name and any(.target.kind[]; . == "bench") and .executable != null) | .executable'
_cj_jq_pid='fromjson? | select(.reason == "compiler-artifact" and .target.name == $name and any(.target.kind[]; . == "bench")) | .package_id'
# One jq run reads every message, so that no package_id (a `path+file:///...` string, which an MSYS
# shell may rewrite on its way to a native jq.exe) has to be passed on the command line.
_cj_jq_cfgs='[split("\n")[] | fromjson?] as $m | ($m | map(select(.reason == "compiler-artifact" and .target.name == $name and any(.target.kind[]; . == "bench"))) | (first | .package_id)) as $pid | $m[] | select(.reason == "build-script-executed" and .package_id == $pid) | "msg", (.cfgs[]? | "cfg:" + .)'

cargo_json() {
  local file=$1 mode=$2 arg=$3 out="" py found=0
  if command -v jq >/dev/null 2>&1; then
    case "$mode" in
      exe) out=$(jq -rR --arg name "$arg" "$_cj_jq_exe" "$file") || return $? ;;
      pid) out=$(jq -rR --arg name "$arg" "$_cj_jq_pid" "$file") || return $? ;;
      cfgs) out=$(jq -rRs --arg name "$arg" "$_cj_jq_cfgs" "$file") || return $? ;;
      *) echo "cargo_json: unknown mode $mode" >&2; return 2 ;;
    esac
  else
    for py in python3 python; do
      if command -v "$py" >/dev/null 2>&1 && "$py" -I -c 'import json' >/dev/null 2>&1; then
        out=$("$py" -I "$_cj_dir/cargo_artifacts.py" "$file" "$mode" "$arg") || return $?
        found=1
        break
      fi
    done
    if [ "$found" = 0 ]; then
      echo "cargo_json: neither jq nor python is usable" >&2
      return 1
    fi
  fi
  # jq.exe may end its lines with CR.
  out=${out//$'\r'/}
  if [ -n "$out" ]; then printf '%s\n' "$out"; fi
}

# sha256_of FILE: the file's sha256, as bare hex.
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# timed_collect JSONL ROOT NAME OUT: copies the executable of bench NAME, as the messages in JSONL
# name it (a relative path is taken from ROOT), into OUT, with the `.pdb` and `.map` linked beside
# it when there are any, and prints the executable's original path. Returns 3, printing nothing, when
# the messages name no executable for NAME. Needs RUNNER_OS.
timed_collect() {
  local jsonl=$1 root=$2 name=$3 out=$4 bin stem dir ext="" f
  bin=$(cargo_json "$jsonl" exe "$name" | tail -n 1) || return $?
  if [ -z "$bin" ]; then return 3; fi
  if [ "$RUNNER_OS" = Windows ]; then
    bin=$(cygpath -u "$bin")
    ext=".exe"
  fi
  case "$bin" in
    /*) ;;
    *) bin="$root/$bin" ;;
  esac
  if [ ! -f "$bin" ]; then
    echo "timed_collect: cargo named $bin for $name, which is not a file" >&2
    return 1
  fi
  mkdir -p "$out"
  cp "$bin" "$out/$name$ext"
  stem=${bin%.exe}
  dir=$(dirname "$bin")
  for f in "$stem.pdb" "$dir/$name"*.pdb; do
    if [ -f "$f" ]; then
      cp "$f" "$out/$name.pdb"
      break
    fi
  done
  # link.exe writes the map file named after its output, beside it or in its own directory.
  for f in "$stem.map" "$dir/$name"*.map "./$(basename "$stem").map"; do
    if [ -f "$f" ]; then
      cp "$f" "$out/$name.map"
      break
    fi
  done
  printf '%s\n' "$bin"
}

# llvm_tools_dir: the directory of rustup's `llvm-objdump` and `llvm-nm`. Needs RUNNER_OS.
llvm_tools_dir() {
  local sysroot host tools ext="" found
  sysroot=$(rustc --print sysroot)
  if [ "$RUNNER_OS" = Windows ]; then
    sysroot=$(cygpath -u "$sysroot")
    ext=".exe"
  fi
  host=$(rustc -vV | sed -n 's/^host: //p')
  tools="$sysroot/lib/rustlib/$host/bin"
  if [ ! -f "$tools/llvm-objdump$ext" ]; then
    found=$(find "$sysroot/lib/rustlib" -name "llvm-objdump$ext" -type f 2>/dev/null | tail -n 1 || true)
    if [ -n "$found" ]; then tools=$(dirname "$found"); fi
  fi
  printf '%s\n' "$tools"
}
