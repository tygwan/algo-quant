#!/bin/bash
#
# Analytics Visualizer - CLI charts for cc-initializer metrics
#
# Usage:
#   ./analytics-visualizer.sh [command] [options]
#
# Commands:
#   summary     - Quick overview (default)
#   tools       - Tool usage bar chart
#   errors      - Error distribution
#   activity    - Time-based activity
#   agents      - Agent usage stats
#   full        - Full report
#

set -e

# Configuration
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
METRICS_FILE="$PROJECT_ROOT/.claude/analytics/metrics.jsonl"
SESSION_LOG="$PROJECT_ROOT/.claude/session.log"

# Colors
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Chart characters
BAR_FULL="█"
BAR_HALF="▓"
BAR_LIGHT="▒"
BAR_EMPTY="░"

# Sparkline characters
SPARK_CHARS="▁▂▃▄▅▆▇█"

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

check_dependencies() {
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Warning: jq not found, using fallback parsing${NC}"
        USE_JQ=false
    else
        USE_JQ=true
    fi
}

check_metrics_file() {
    if [[ ! -f "$METRICS_FILE" ]]; then
        echo -e "${YELLOW}No metrics file found at $METRICS_FILE${NC}"
        echo "Metrics will be collected as you use tools."
        exit 0
    fi
}

get_line_count() {
    wc -l < "$METRICS_FILE" | tr -d ' '
}

# ─────────────────────────────────────────────────────────────
# Chart Drawing Functions
# ─────────────────────────────────────────────────────────────

# Draw horizontal bar
# Args: $1=value, $2=max_value, $3=bar_width (default 30)
draw_bar() {
    local value=$1
    local max_value=$2
    local bar_width=${3:-30}

    if [[ $max_value -eq 0 ]]; then
        printf "%${bar_width}s" ""
        return
    fi

    local filled=$(( (value * bar_width) / max_value ))
    local empty=$(( bar_width - filled ))

    printf "${GREEN}"
    for ((i=0; i<filled; i++)); do printf "$BAR_FULL"; done
    printf "${DIM}"
    for ((i=0; i<empty; i++)); do printf "$BAR_EMPTY"; done
    printf "${NC}"
}

# Draw percentage bar with color coding
# Args: $1=percentage (0-100), $2=bar_width
draw_percentage_bar() {
    local pct=$1
    local bar_width=${2:-20}
    local filled=$(( (pct * bar_width) / 100 ))
    local empty=$(( bar_width - filled ))

    # Color based on percentage
    local color=$GREEN
    if [[ $pct -lt 50 ]]; then color=$RED
    elif [[ $pct -lt 80 ]]; then color=$YELLOW
    fi

    printf "${color}"
    for ((i=0; i<filled; i++)); do printf "$BAR_FULL"; done
    printf "${DIM}"
    for ((i=0; i<empty; i++)); do printf "$BAR_EMPTY"; done
    printf "${NC}"
}

# Draw sparkline from array of values
# Args: values as arguments
draw_sparkline() {
    local values=("$@")
    local max=0

    # Find max
    for v in "${values[@]}"; do
        [[ $v -gt $max ]] && max=$v
    done

    [[ $max -eq 0 ]] && max=1

    local result=""
    for v in "${values[@]}"; do
        local idx=$(( (v * 7) / max ))
        result+="${SPARK_CHARS:$idx:1}"
    done

    echo "$result"
}

# Print section header
print_header() {
    local title="$1"
    echo -e "\n${BOLD}${CYAN}$title${NC}"
    echo -e "${DIM}$(printf '─%.0s' {1..50})${NC}"
}

# ─────────────────────────────────────────────────────────────
# Data Extraction Functions (jq-based)
# ─────────────────────────────────────────────────────────────

get_tool_counts() {
    if $USE_JQ; then
        jq -s 'group_by(.name) | map({name: .[0].name, count: length}) | sort_by(-.count)' "$METRICS_FILE" 2>/dev/null
    else
        # Fallback: grep + awk
        grep -o '"name":"[^"]*"' "$METRICS_FILE" | cut -d'"' -f4 | sort | uniq -c | sort -rn | \
            awk '{print "{\"name\":\"" $2 "\",\"count\":" $1 "}"}'
    fi
}

get_category_counts() {
    if $USE_JQ; then
        jq -s 'group_by(.category) | map({category: .[0].category, count: length}) | sort_by(-.count)' "$METRICS_FILE" 2>/dev/null
    else
        grep -o '"category":"[^"]*"' "$METRICS_FILE" | cut -d'"' -f4 | sort | uniq -c | sort -rn | \
            awk '{print "{\"category\":\"" $2 "\",\"count\":" $1 "}"}'
    fi
}

get_success_rate() {
    if $USE_JQ; then
        jq -s '{total: length, success: [.[] | select(.success == true)] | length}' "$METRICS_FILE" 2>/dev/null
    else
        local total=$(wc -l < "$METRICS_FILE")
        local success=$(grep -c '"success":true' "$METRICS_FILE" || echo 0)
        echo "{\"total\":$total,\"success\":$success}"
    fi
}

get_agent_counts() {
    if $USE_JQ; then
        jq -s '[.[] | select(.category == "agent")] | group_by(.context) | map({agent: .[0].context, count: length}) | sort_by(-.count)' "$METRICS_FILE" 2>/dev/null
    else
        grep '"category":"agent"' "$METRICS_FILE" | grep -o '"context":"[^"]*"' | cut -d'"' -f4 | sort | uniq -c | sort -rn | \
            awk '{print "{\"agent\":\"" $2 "\",\"count\":" $1 "}"}'
    fi
}

get_hourly_distribution() {
    if $USE_JQ; then
        jq -s 'group_by(.ts[11:13]) | map({hour: .[0].ts[11:13], count: length}) | sort_by(.hour)' "$METRICS_FILE" 2>/dev/null
    else
        grep -o '"ts":"[^"]*"' "$METRICS_FILE" | cut -d'T' -f2 | cut -c1-2 | sort | uniq -c | \
            awk '{print "{\"hour\":\"" $2 "\",\"count\":" $1 "}"}'
    fi
}

get_error_types() {
    if $USE_JQ; then
        jq -s '[.[] | select(.success == false)] | group_by(.name) | map({tool: .[0].name, count: length}) | sort_by(-.count)' "$METRICS_FILE" 2>/dev/null
    else
        grep '"success":false' "$METRICS_FILE" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sort | uniq -c | sort -rn | \
            awk '{print "{\"tool\":\"" $2 "\",\"count\":" $1 "}"}'
    fi
}

# ─────────────────────────────────────────────────────────────
# Report Sections
# ─────────────────────────────────────────────────────────────

show_tool_usage() {
    print_header "Tool Usage"

    local data=$(get_tool_counts)
    local max_count=0

    # Parse and find max
    if $USE_JQ; then
        max_count=$(echo "$data" | jq -r '.[0].count // 0')

        echo "$data" | jq -r '.[] | "\(.name) \(.count)"' | while read -r name count; do
            printf "%-10s │" "$name"
            draw_bar "$count" "$max_count" 25
            printf "│ %3d\n" "$count"
        done
    else
        # Fallback parsing
        local first=true
        echo "$data" | while read -r line; do
            local name=$(echo "$line" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
            local count=$(echo "$line" | grep -o '"count":[0-9]*' | cut -d':' -f2)

            if $first; then
                max_count=$count
                first=false
            fi

            printf "%-10s │" "$name"
            draw_bar "$count" "${max_count:-1}" 25
            printf "│ %3d\n" "$count"
        done
    fi
}

show_category_distribution() {
    print_header "Category Distribution"

    local data=$(get_category_counts)
    local total=$(get_line_count)

    if $USE_JQ; then
        echo "$data" | jq -r '.[] | "\(.category) \(.count)"' | while read -r category count; do
            local pct=$(( (count * 100) / total ))
            printf "%-12s " "$category"
            draw_percentage_bar "$pct" 20
            printf " %3d%% (%d)\n" "$pct" "$count"
        done
    else
        # Fallback parsing
        echo "$data" | while read -r line; do
            local category=$(echo "$line" | grep -o '"category":"[^"]*"' | cut -d'"' -f4)
            local count=$(echo "$line" | grep -o '"count":[0-9]*' | cut -d':' -f2)
            [[ -z "$category" ]] && continue
            local pct=$(( (count * 100) / total ))
            printf "%-12s " "$category"
            draw_percentage_bar "$pct" 20
            printf " %3d%% (%d)\n" "$pct" "$count"
        done
    fi
}

show_success_rate() {
    print_header "Success Rate"

    local data=$(get_success_rate)
    local total success pct

    if $USE_JQ; then
        total=$(echo "$data" | jq -r '.total')
        success=$(echo "$data" | jq -r '.success')
    else
        total=$(echo "$data" | grep -o '"total":[0-9]*' | cut -d':' -f2)
        success=$(echo "$data" | grep -o '"success":[0-9]*' | cut -d':' -f2)
    fi

    [[ $total -eq 0 ]] && total=1
    pct=$(( (success * 100) / total ))

    printf "Overall    "
    draw_percentage_bar "$pct" 25
    printf " %3d%% (%d/%d)\n" "$pct" "$success" "$total"
}

show_agent_usage() {
    print_header "Agent Usage"

    local data=$(get_agent_counts)

    if [[ -z "$data" ]] || [[ "$data" == "[]" ]]; then
        echo -e "${DIM}No agent calls recorded${NC}"
        return
    fi

    local max_count=0

    if $USE_JQ; then
        max_count=$(echo "$data" | jq -r '.[0].count // 0')

        echo "$data" | jq -r '.[] | "\(.agent) \(.count)"' | head -10 | while read -r agent count; do
            [[ -z "$agent" ]] && continue
            printf "%-18s │" "${agent:0:18}"
            draw_bar "$count" "$max_count" 20
            printf "│ %3d\n" "$count"
        done
    else
        # Fallback parsing
        local first=true
        echo "$data" | head -10 | while read -r line; do
            local agent=$(echo "$line" | grep -o '"agent":"[^"]*"' | cut -d'"' -f4)
            local count=$(echo "$line" | grep -o '"count":[0-9]*' | cut -d':' -f2)
            [[ -z "$agent" ]] && continue

            if $first; then
                max_count=$count
                first=false
            fi

            printf "%-18s │" "${agent:0:18}"
            draw_bar "$count" "${max_count:-1}" 20
            printf "│ %3d\n" "$count"
        done
    fi
}

show_activity_sparkline() {
    print_header "Hourly Activity"

    local data=$(get_hourly_distribution)

    # Get counts for each hour (0-23)
    local -a hourly_counts
    for i in {0..23}; do
        hourly_counts[$i]=0
    done

    if $USE_JQ; then
        while read -r hour count; do
            local h=$((10#$hour))  # Remove leading zero
            hourly_counts[$h]=$count
        done < <(echo "$data" | jq -r '.[] | "\(.hour) \(.count)"')
    else
        # Fallback parsing - use process substitution to avoid subshell
        while read -r line; do
            local hour=$(echo "$line" | grep -o '"hour":"[^"]*"' | cut -d'"' -f4)
            local count=$(echo "$line" | grep -o '"count":[0-9]*' | cut -d':' -f2)
            [[ -z "$hour" ]] && continue
            local h=$((10#$hour))
            hourly_counts[$h]=$count
        done < <(echo "$data")
    fi

    # Draw sparkline
    local sparkline=$(draw_sparkline "${hourly_counts[@]}")
    echo -e "Activity: ${CYAN}$sparkline${NC}"
    echo -e "${DIM}          00    06    12    18    23${NC}"
}

show_error_distribution() {
    print_header "Error Distribution"

    local data=$(get_error_types)

    if [[ -z "$data" ]] || [[ "$data" == "[]" ]]; then
        echo -e "${GREEN}No errors recorded!${NC}"
        return
    fi

    local max_count=0

    if $USE_JQ; then
        max_count=$(echo "$data" | jq -r '.[0].count // 0')

        echo "$data" | jq -r '.[] | "\(.tool) \(.count)"' | while read -r tool count; do
            printf "%-12s │" "$tool"
            printf "${RED}"
            draw_bar "$count" "$max_count" 20
            printf "${NC}│ %3d\n" "$count"
        done
    else
        # Fallback parsing
        local first=true
        echo "$data" | while read -r line; do
            local tool=$(echo "$line" | grep -o '"tool":"[^"]*"' | cut -d'"' -f4)
            local count=$(echo "$line" | grep -o '"count":[0-9]*' | cut -d':' -f2)
            [[ -z "$tool" ]] && continue

            if $first; then
                max_count=$count
                first=false
            fi

            printf "%-12s │" "$tool"
            printf "${RED}"
            draw_bar "$count" "${max_count:-1}" 20
            printf "${NC}│ %3d\n" "$count"
        done
    fi
}

# ─────────────────────────────────────────────────────────────
# Main Report Functions
# ─────────────────────────────────────────────────────────────

show_summary() {
    local total=$(get_line_count)
    local today=$(date '+%Y-%m-%d')

    echo ""
    echo -e "${BOLD}${MAGENTA}📊 Analytics Summary${NC} ${DIM}($today)${NC}"
    echo -e "${DIM}$(printf '═%.0s' {1..50})${NC}"

    # Quick stats
    local data=$(get_success_rate)
    local success total_calls pct

    if $USE_JQ; then
        total_calls=$(echo "$data" | jq -r '.total')
        success=$(echo "$data" | jq -r '.success')
    else
        total_calls=$(echo "$data" | grep -o '"total":[0-9]*' | cut -d':' -f2)
        success=$(echo "$data" | grep -o '"success":[0-9]*' | cut -d':' -f2)
    fi

    [[ $total_calls -eq 0 ]] && total_calls=1
    pct=$(( (success * 100) / total_calls ))

    echo -e "${BOLD}Total Calls:${NC} $total_calls  ${BOLD}Success Rate:${NC} ${GREEN}$pct%${NC}"
    echo ""

    show_tool_usage
    show_success_rate
    show_activity_sparkline
}

show_full_report() {
    local today=$(date '+%Y-%m-%d')

    echo ""
    echo -e "${BOLD}${MAGENTA}📊 Full Analytics Report${NC} ${DIM}($today)${NC}"
    echo -e "${DIM}$(printf '═%.0s' {1..50})${NC}"

    show_tool_usage
    echo ""
    show_category_distribution
    echo ""
    show_success_rate
    echo ""
    show_agent_usage
    echo ""
    show_error_distribution
    echo ""
    show_activity_sparkline

    echo ""
    echo -e "${DIM}Data source: $METRICS_FILE${NC}"
}

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

main() {
    check_dependencies
    check_metrics_file

    local command="${1:-summary}"

    case "$command" in
        summary|s)
            show_summary
            ;;
        tools|t)
            show_tool_usage
            ;;
        errors|e)
            show_error_distribution
            ;;
        activity|a)
            show_activity_sparkline
            ;;
        agents|ag)
            show_agent_usage
            ;;
        categories|c)
            show_category_distribution
            ;;
        full|f)
            show_full_report
            ;;
        *)
            echo "Usage: $0 [summary|tools|errors|activity|agents|categories|full]"
            exit 1
            ;;
    esac
}

main "$@"
