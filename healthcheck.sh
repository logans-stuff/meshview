#!/usr/bin/env bash
#
# meshview health check & auto-restart script
#
# Runs alongside meshview (outside the container). Periodically queries the
# meshview API to verify the service is healthy and receiving new data.
# If it detects a problem, it restarts the Docker container.
#
# Usage:
#   ./healthcheck.sh                  # run once (good for cron)
#   ./healthcheck.sh --loop           # run continuously
#   ./healthcheck.sh --loop --dry-run # continuous, log-only (no restart)
#
# Cron example (every 5 minutes):
#   */5 * * * * /path/to/healthcheck.sh >> /var/log/meshview-healthcheck.log 2>&1
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration – override via environment variables if needed
# ---------------------------------------------------------------------------

# Base URL of the meshview instance
MESHVIEW_URL="${MESHVIEW_URL:-http://localhost:8081}"

# Docker container name to restart
CONTAINER_NAME="${CONTAINER_NAME:-meshview}"

# How many seconds of no new data before we consider data "stale"
# Default: 600 seconds (10 minutes)
STALE_THRESHOLD_SECS="${STALE_THRESHOLD_SECS:-600}"

# Seconds between checks when running in --loop mode
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

# How many consecutive failures before triggering a restart
FAIL_THRESHOLD="${FAIL_THRESHOLD:-3}"

# Seconds to wait after a restart before resuming checks (let the app boot)
RESTART_COOLDOWN="${RESTART_COOLDOWN:-120}"

# Curl timeout for each request (seconds)
CURL_TIMEOUT="${CURL_TIMEOUT:-10}"

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
LOOP_MODE=false
DRY_RUN=false
consecutive_failures=0

for arg in "$@"; do
    case "$arg" in
        --loop)   LOOP_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Returns 0 if healthy, 1 otherwise. Sets $health_detail.
check_health_endpoint() {
    local http_code body
    http_code=$(curl -s -o /tmp/meshview_health.json -w "%{http_code}" \
        --max-time "$CURL_TIMEOUT" "${MESHVIEW_URL}/health" 2>/dev/null) || {
        health_detail="connection refused or timed out"
        return 1
    }

    if [[ "$http_code" -ne 200 ]]; then
        health_detail="HTTP $http_code"
        body=$(cat /tmp/meshview_health.json 2>/dev/null || true)
        if [[ -n "$body" ]]; then
            local status
            status=$(echo "$body" | jq -r '.status // empty' 2>/dev/null || true)
            local db
            db=$(echo "$body" | jq -r '.database // empty' 2>/dev/null || true)
            [[ -n "$status" ]] && health_detail="$health_detail (status=$status, db=$db)"
        fi
        return 1
    fi

    health_detail="ok"
    return 0
}

# Returns 0 if fresh data exists, 1 if stale. Sets $data_detail.
check_data_freshness() {
    local http_code body
    http_code=$(curl -s -o /tmp/meshview_packets.json -w "%{http_code}" \
        --max-time "$CURL_TIMEOUT" "${MESHVIEW_URL}/api/packets?limit=1" 2>/dev/null) || {
        data_detail="connection refused or timed out"
        return 1
    }

    if [[ "$http_code" -ne 200 ]]; then
        data_detail="HTTP $http_code fetching packets"
        return 1
    fi

    body=$(cat /tmp/meshview_packets.json 2>/dev/null || true)

    # latest_import_time is in microseconds
    local latest_us
    latest_us=$(echo "$body" | jq -r '.latest_import_time // empty' 2>/dev/null || true)

    if [[ -z "$latest_us" || "$latest_us" == "null" ]]; then
        data_detail="no latest_import_time in response"
        return 1
    fi

    # Convert to seconds and compare against wall clock
    local latest_secs now_secs age_secs
    latest_secs=$(( latest_us / 1000000 ))
    now_secs=$(date +%s)
    age_secs=$(( now_secs - latest_secs ))

    if [[ "$age_secs" -gt "$STALE_THRESHOLD_SECS" ]]; then
        data_detail="last data ${age_secs}s ago (threshold: ${STALE_THRESHOLD_SECS}s)"
        return 1
    fi

    data_detail="last data ${age_secs}s ago"
    return 0
}

do_restart() {
    if $DRY_RUN; then
        log "DRY-RUN: would restart container '$CONTAINER_NAME'"
        return 0
    fi

    log "Restarting container '$CONTAINER_NAME' ..."
    if docker container restart "$CONTAINER_NAME"; then
        log "Container restarted successfully."
    else
        log "ERROR: docker restart failed (exit code $?)."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main check logic
# ---------------------------------------------------------------------------

run_check() {
    local failed=false reason=""

    # 1. Health endpoint
    if ! check_health_endpoint; then
        failed=true
        reason="health: $health_detail"
        log "FAIL  /health – $health_detail"
    else
        log "OK    /health – $health_detail"
    fi

    # 2. Data freshness (only if health endpoint was reachable)
    if [[ "$health_detail" != "connection refused or timed out" ]]; then
        if ! check_data_freshness; then
            failed=true
            reason="${reason:+$reason; }data: $data_detail"
            log "FAIL  data freshness – $data_detail"
        else
            log "OK    data freshness – $data_detail"
        fi
    fi

    # 3. Evaluate
    if $failed; then
        consecutive_failures=$(( consecutive_failures + 1 ))
        log "Consecutive failures: $consecutive_failures / $FAIL_THRESHOLD"

        if [[ "$consecutive_failures" -ge "$FAIL_THRESHOLD" ]]; then
            log "Failure threshold reached. Reason: $reason"
            do_restart
            consecutive_failures=0

            if $LOOP_MODE; then
                log "Cooling down for ${RESTART_COOLDOWN}s before next check ..."
                sleep "$RESTART_COOLDOWN"
            fi
        fi
    else
        if [[ "$consecutive_failures" -gt 0 ]]; then
            log "Recovered after $consecutive_failures failure(s)."
        fi
        consecutive_failures=0
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

log "meshview health check starting (url=$MESHVIEW_URL container=$CONTAINER_NAME stale=${STALE_THRESHOLD_SECS}s)"
log "fail_threshold=$FAIL_THRESHOLD interval=${CHECK_INTERVAL}s cooldown=${RESTART_COOLDOWN}s"
[[ "$DRY_RUN" == "true" ]] && log "DRY-RUN mode enabled – no restarts will be performed"

if $LOOP_MODE; then
    log "Running in loop mode (Ctrl+C to stop) ..."
    while true; do
        run_check
        sleep "$CHECK_INTERVAL"
    done
else
    run_check
fi
