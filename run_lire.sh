#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BINARY="${ROOT_DIR}/build/faiss_ivfflat_lire_run_eval"
CONFIG="${ROOT_DIR}/configs/ivfflat_lire_sift10m_drift.json"
DATASET="${HOME}/data/sift10M/drift_fast"

LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/result"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/ivfflat_lire_sift10m_drift_${TIMESTAMP}.log"
RESULT_FILE="${RESULT_DIR}/ivfflat_lire_sift10m_drift_${TIMESTAMP}.json"

if [[ ! -x "${BINARY}" ]]; then
    echo "Error: executable not found or not executable: ${BINARY}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -d "${DATASET}" ]]; then
    echo "Error: dataset directory not found: ${DATASET}" >&2
    exit 1
fi

if ! command -v numactl >/dev/null 2>&1; then
    echo "Error: numactl is required but was not found." >&2
    exit 1
fi

echo "Binary : ${BINARY}"
echo "Config : ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Log    : ${LOG_FILE}"
echo "Result : ${RESULT_FILE}"
echo "NUMA   : CPU node 0, memory node 0"
echo

numactl --cpunodebind=0 --membind=0 \
    "${BINARY}" \
    "${CONFIG}" \
    "${DATASET}" \
    "${RESULT_FILE}" \
    2>&1 | tee "${LOG_FILE}"
