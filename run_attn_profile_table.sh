#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/attn_test.cu"
BUILD_DIR="${SCRIPT_DIR}/.bench_build"
OUT_MD="${1:-${SCRIPT_DIR}/attn_profile_table.md}"
CUDA_ARCH="${CUDA_ARCH:-sm_90a}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found in PATH." >&2
  exit 1
fi
if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: source not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

declare -a PAIRS=(
  "128:32"
  "128:64"
  "128:128"
  "128:256"
  "256:32"
  "256:64"
  "256:128"
)

if [[ -n "${ATTN_BM_BN_PAIRS:-}" ]]; then
  IFS=',' read -r -a PAIRS <<< "${ATTN_BM_BN_PAIRS}"
fi

{
  echo "| BM | BN | LoadQ(avg) | LoadK(avg) | LoadV(avg) | GEMM-QK(avg) | SOFTMAX(avg) | GEMM-PV(avg) |"
  echo "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
} > "${OUT_MD}"

for pair in "${PAIRS[@]}"; do
  pair="$(echo "${pair}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  BM="${pair%%:*}"
  BN="${pair##*:}"
  BM="$(echo "${BM}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  BN="$(echo "${BN}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

  if [[ -z "${BM}" || -z "${BN}" || "${pair}" != *:* || ! "${BM}" =~ ^[0-9]+$ || ! "${BN}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid pair '${pair}', expected BM:BN." >&2
    exit 1
  fi

  TMP_CU="${BUILD_DIR}/attn_test_bm${BM}_bn${BN}.cu"
  BIN="${BUILD_DIR}/attn_test_bm${BM}_bn${BN}"

  sed -E \
    -e "s/^([[:space:]]*constexpr int BM = ).*;/\1${BM};/" \
    -e "s/^([[:space:]]*constexpr int BN = ).*;/\1${BN};/" \
    "${SRC}" > "${TMP_CU}"

  echo "[build] BM=${BM} BN=${BN}"
  nvcc -std=c++17 -arch="${CUDA_ARCH}" -O3 -I"${SCRIPT_DIR}" "${TMP_CU}" -o "${BIN}" -lcuda

  echo "[run]   BM=${BM} BN=${BN}"
  if ! RUN_OUT="$("${BIN}" 2>&1)"; then
    echo "ERROR: run failed for BM=${BM} BN=${BN}" >&2
    printf '%s\n' "${RUN_OUT}" >&2
    exit 1
  fi
  LINE="$(printf '%s\n' "${RUN_OUT}" | grep 'LoadQ(avg):' || true)"
  if [[ -z "${LINE}" ]]; then
    echo "ERROR: failed to parse output for BM=${BM} BN=${BN}" >&2
    echo "Kernel output:" >&2
    printf '%s\n' "${RUN_OUT}" >&2
    exit 1
  fi

  PARSED="$(printf '%s\n' "${LINE}" | sed -E \
    's/.*LoadQ\(avg\): ([0-9.]+), LoadK\(avg\): ([0-9.]+), LoadV\(avg\): ([0-9.]+), GEMM-QK\(avg\): ([0-9.]+), SOFTMAX\(avg\): ([0-9.]+), GEMM-PV\(avg\): ([0-9.]+) cycle/\1 \2 \3 \4 \5 \6/')"
  if [[ "${PARSED}" == "${LINE}" ]]; then
    echo "ERROR: parse regex mismatch for BM=${BM} BN=${BN}" >&2
    echo "Raw line: ${LINE}" >&2
    exit 1
  fi

  read -r LOADQ LOADK LOADV GEMMQK SOFTMAX GEMMPV <<< "${PARSED}"
  printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "${BM}" "${BN}" "${LOADQ}" "${LOADK}" "${LOADV}" "${GEMMQK}" "${SOFTMAX}" "${GEMMPV}" >> "${OUT_MD}"
done

echo "Done. Table written to: ${OUT_MD}"
