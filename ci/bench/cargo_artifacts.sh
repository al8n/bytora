# Source this from a bash step (`source ci/bench/cargo_artifacts.sh`); it defines `cargo_json`.
#
#   cargo_json FILE exe NAME          executable path of the bench target NAME
#   cargo_json FILE pid NAME          package_id of the package that owns bench target NAME
#   cargo_json FILE cfgs NAME         `msg` once per build-script-executed message of the package that
#                                     owns bench target NAME, then `cfg:<cfg>` for each cfg it set
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
