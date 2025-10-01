#!/bin/bash

# =======DECLARATIONS========

#===INPUTS===

# Timeout to be applied to all compilation commands called from the script
TIMEOUT_DURATION="20m"

# Dynamatic executable location
DYNAMATIC="./bin/dynamatic"

# Directory containing the test subdirectories
TEST_DIR="./integration-test"

# Input frontend compilation script template (.dyn)
COMPILE_SCRIPT_TEMPLATE="./skip/templates/compile-with-dynamatic-compile-template.dyn"

# Input frontend simulation script template (.dyn)
SIMULATE_SCRIPT_TEMPLATE="./skip/templates/compile-with-dynamatic-simulate-template.dyn"

# Tcl template for vivado
TCL_TEMPLATE="./skip/templates/call-vivado-impl-for-dynamatic.tcl"

# Template file to feed constraints to vivado
CONSTRAINTS_TEMPLATE="./skip/templates/call-vivado-impl-for-dynamatic-contraints.xdc"

SIM_RESULT="out/sim/report.txt"

PLACE_AND_ROUTE_TIMING="/out/vivado/timing_post_pr.rpt"



#===OUTPUTS===

# Output file containing the compilation time of Dynamatic
COMPILATION_TIME_FILE="out/vivado/compilation-time.txt"

CP_FILE="out/vivado/cp.txt"

FIND_CP_OUTPUT="out/vivado/find_cp.rpt"

# Temporary frontend script
SCRIPT_TEMP="temp_frontend-script-compile-with-dynamatic.dyn"

# Temporary file to hold the list of C files
FILE_LIST="temp-file-list-compile-with-dynamatic.txt"

# Temporary tcl file
TCL_TEMPORARY="temp-call-vivado-impl-for-dynamatic.tcl"

# File containing a list of tests that have timeout
OUT_TIMEOUTS="./skip/timeouts-compile-with-dynamatic.txt"

COMPILATION_ERRORS="./skip/benchmark-compilation-errors.txt"


DUMP_FILE="vivado_dump.txt"

REPORT_LOCATION="out/vivado"

BEST_CP_GLOB=100

MAX_JOBS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute path to run log
RUN_LOG="./run-robust.log"
echo "Run started at $(date --rfc-3339=seconds)" > "$RUN_LOG"
# =======END DECLARATIONS========


# =======HELPERS========

wait_for_free_slot() {
  while true; do
    running=$(jobs -pr | wc -l)
    if [ "$running" -lt "$MAX_JOBS" ]; then break; fi
    sleep 2
  done
}

safe_run() {
  local timeout_d=$1; shift
  local out_file=$1; shift
  local err_file=$1; shift
  if [ "$1" = "--" ]; then shift; fi
  timeout -k 30s "$timeout_d" bash -c "$*" >"$out_file" 2>"$err_file"
  return $?
}


place_and_route() {
  local dir="$1"; shift
  local basename="$1"; shift
  local tcl_template="$1"; shift
  local constraints_template="$1"; shift

  local jobtmp="$dir/$REPORT_LOCATION"


  local tcl_temp="$jobtmp/temp-call-vivado-impl-for-dynamatic.tcl"
  cp "$tcl_template" "$tcl_temp" || { echo "ERROR: copy tcl template failed" >>"$RUN_LOG"; return 1; }

  sed -i -e "s|PATH_TO_TEST_FOLDER|$dir|g" \
         -e "s|PATH_TO_CONSTRAINTS_FILE|$constraints_template|g" \
         -e "s|TOP_MODULE|$basename|g" "$tcl_temp" || true

  local vivado_stdout="$jobtmp/vivado.stdout"
  local vivado_stderr="$jobtmp/vivado.stderr"

  pwd >> "$RUN_LOG"
  cd "$SCRIPT_DIR/.." && \
  pwd >> "$RUN_LOG"
  echo "Calling vivado for $basename at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
  (
    cd "$SCRIPT_DIR/.." || exit 1
    TMPDIR="$jobtmp" timeout -k 30s "$TIMEOUT_DURATION" vivado \
      -notrace -nolog -nojournal -mode tcl -script "$tcl_temp" \
      > "$vivado_stdout" 2> "$vivado_stderr"
  )

  rc=$?
  if [ $rc -ne 0 ]; then
      echo "Vivado failed for $basename with exit code $rc" >> "$RUN_LOG"
      cat "$RUN_LOG/vivado.stderr" >> "$RUN_LOG"
  fi

  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "$basename : place and routing timeout" >> "$OUT_TIMEOUTS"
    echo "P&R timeout for $basename" >> "$jobtmp/timing_error.log"
    return 124
  fi
  return 0
}


compile() {
  local dir="$1"; shift
  local basename="$1"; shift
  local clock="$1"; shift
  local n="$1"; shift
  local script_template="$1"; shift

  local script_temp="$dir/temp_frontend-script-compile-with-dynamatic.dyn"
  cp "$script_template" "$script_temp" || { echo "ERROR: copy compile template failed" >>"$RUN_LOG"; return 1; }
  sed -i "s|PATH_TO_C_FILE|$cfile|g; s|CLOCK_PERIOD|$clock|g; s|SKIPPABLE_SEQ_N|$n|g" "$script_temp" || true

  mkdir -p "$dir/$REPORT_LOCATION/compilation"
  local comp_out="$dir/$REPORT_LOCATION/compilation/dynamatic.stdout"
  local comp_err="$dir/$REPORT_LOCATION/compilation/dynamatic.stderr"

  echo "Calling dynamatic for $cfile (clock=$clock n=$n) at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
  ( cd "$jobtmp" && timeout -k 30s "$TIMEOUT_DURATION" bash -c "{ time \"$DYNAMATIC\" --run \"$script_temp\" 2>> \"$COMPILATION_ERRORS\" ; }" ) >"$comp_out" 2>"$comp_err"
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "$basename : compilation timeout" >> $OUT_TIMEOUTS
    echo "Compilation timeout for $cfile (clock=$clock n=$n)" >> "$dir/$basename/$REPORT_LOCATION/compilation/error.log"
    return 124
  fi
  return 0
}


simulate() {
  local cfile="$1"; shift
  local simulate_template="$1"; shift

  local jobtmp="$dir/$basename/$REPORT_LOCATION"

  local script_temp="$jobtmp/temp_frontend-script-simulate.dyn"
  cp "$simulate_template" "$script_temp" || { echo "ERROR: copy simulate template failed" >>"$RUN_LOG"; return 1; }
  sed -i -e "s|PATH_TO_C_FILE|$cfile|g" "$script_temp" || true

  local sim_out="$jobtmp/sim.stdout"
  local sim_err="$jobtmp/sim.stderr"

  echo "Simulating $cfile at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
  ( cd "$jobtmp" && timeout -k 30s "$TIMEOUT_DURATION" bash -c "\"$DYNAMATIC\" --run \"$script_temp\"" ) >"$sim_out" 2>"$sim_err"
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "$basename : simulation timeout" >> $OUT_TIMEOUTS
    echo "Simulation timeout for $cfile" >> "$jobtmp/sim/error.log"
    return 124
  fi
  return 0
}


get_slack() {
  local report=$1

  # You can call the `check_slack` function or directly parse the slack from the report
  slack_value=$(grep "Slack\s*(MET)" "$report" | sed 's/.*Slack\s*(MET)\s*:\s*\([0-9.]*\)ns.*/\1/')
  if [ -z "$slack_value" ]; then
      slack_value=$(grep "Slack\s*(VIOLATED)" "$report" | sed 's/.*Slack\s*(VIOLATED)\s*:\s*\([+-]*[0-9.]*\)ns.*/\1/')
  fi
  echo "$slack_value"
}


find_real_cp() {
  local dir="$1"; shift
  local basename="$1"; shift
  local start_cp="$1"; shift
  local constraints_template="$1"

  local jobtmp="$dir/$REPORT_LOCATION"
  local report_file="$jobtmp/timing_post_pr.rpt"
  local current_cp="$start_cp"
  local step="-0.1"

  echo "find_real_cp start $start_cp for $basename" >> "$RUN_LOG"
  echo "----------------------" >> "$jobtmp/find_cp.rpt"
  echo "Running for CP = $start_cp" >> "$jobtmp/find_cp.rpt"


  printf "Current CP %.6f\n" "$current_cp" >> "$jobtmp/find_cp.rpt"
  half_period=$(echo "$current_cp / 2" | bc -l)

  local constraints_copy="$jobtmp/constraints.xdc"
  cp "$CONSTRAINTS_TEMPLATE" "$constraints_copy" 2>/dev/null || cp "$constraints_template" "$constraints_copy" 2>/dev/null || true
  sed -i "s/create_clock -name clk -period .*/create_clock -name clk -period $current_cp -waveform {0.000 $half_period} [get_ports clk]/" "$constraints_copy" 2>/dev/null || true

  echo "Using constraint line:" >> "$jobtmp/find_cp.rpt"
  grep create_clock "$constraints_copy" >> "$jobtmp/find_cp.rpt"

  place_and_route "$dir" "$basename" "$TCL_TEMPLATE" "$constraints_copy"
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "P&R timed out while searching CP. Stopping CP search." >> "$jobtmp/find_cp.rpt"
    break
  fi

  slack=$(get_slack "$report_file")
  echo "Slack: $slack" >> "$jobtmp/find_cp.rpt"
  echo "$start_cp - $slack" | bc -l
    
}



find_best_timing() {
  local dir="$1"; shift
  local basename="$1"; shift
  local n="$1"; shift

  # Absolute path to out directory
  OUT_DIR="$SCRIPT_DIR/../integration-test/$basename/out/"
  mkdir -p "$OUT_DIR/vivado" "$OUT_DIR/sim"

  ### Rouzbeh
  local cp=4
  local best_timing_cp=0
  local buffer_cp=0

  echo > "$dir/$REPORT_LOCATION/cp.txt"

  rm -rf "$OUT_DIR"

  while true; do
      echo "Compile $basename at CP=$cp (unbuffered) at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
      compile "$dir" "$basename" "$cp" "$n" "$COMPILE_SCRIPT_TEMPLATE" || true

      if [ -s "$OUT_DIR/comp/handshake_buffered.mlir" ]; then
          echo "Found handshake_buffered, compile done at CP=$cp" >> "$RUN_LOG"
          break
      else
          echo "handshake_buffered not found, increasing CP and retrying..." >> "$RUN_LOG"
          cp=$(echo "$cp + 1" | bc -l)
          echo "handshake_buffered not found, increasing CP and retrying..." >> "$RUN_LOG"
      fi
  done
  
  echo "Running buffering with CP=$cp..." >> "$RUN_LOG"
  python3 buffer.py . $basename $n $cp > "$dir/$REPORT_LOCATION/buffer.log" 2>&1 \
    || echo "buffer.py failed for $basename" > "$dir/$REPORT_LOCATION/buffer.err"

  # start_cp=$(echo "$cp - 0.5" | bc -l)
  start_cp=$cp
  echo "Final place & route on buffered circuit..." >> "$RUN_LOG"
  final_cp=$(find_real_cp "$dir" "$basename" "$start_cp" "$CONSTRAINTS_TEMPLATE") || final_cp=999
  echo "Final CP after buffering P&R: $final_cp" >> "$dir/$REPORT_LOCATION/find_cp.rpt"
  echo "Final CP after buffering P&R: $final_cp" >> "$dir/$REPORT_LOCATION/cp.txt"

  cycles=$(grep -A1 "\*\* Note: Simulation done!" "$dir/$SIM_RESULT" | grep "Latency" | awk '{print $8}')

  echo "Clock cycles: $cycles" > "$dir/$CP_FILE"
  echo "Real CP: $final_cp"  >> "$dir/$CP_FILE"

  cd skip
  python3 retrieve-information-dynamatic-n.py $basename $n $cp
  cd ..
}


process_file() {
  local cfile="$1"
  cfile=$(echo "$cfile" | sed 's#^\.\./##')
  local basename=$(basename "$cfile" .c)
  local dir=$(dirname "$cfile")


  echo "Starting job for $cfile at $(date --rfc-3339=seconds)" >> "$RUN_LOG"

  local n_values=(3)
  for n in "${n_values[@]}"; do
    find_best_timing "$dir" "$basename" "$n"
  done

  echo "Job finished for $cfile at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
}



#### main ####

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
cd ..

echo > "$RUN_LOG"

USE_FILE=false
FILE_LIST=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -f|--file)
      FILE_LIST="$SCRIPT_DIR/$2"
      USE_FILE=true
      shift 2
      ;;
    -j|--jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$USE_FILE" == false ]]; then
  FILE_LIST=$(mktemp -t file_list_XXXX)
  find "$TEST_DIR" -type d -name "out" -prune -o -type f -name "*.c" -print > "$FILE_LIST"
fi

while IFS= read -r cfile; do
  [ -z "$cfile" ] && continue
  wait_for_free_slot
  process_file "$cfile" &
  sleep 0.2
done < "$FILE_LIST"

wait

if [[ "$USE_FILE" == false ]]; then
  rm -f "$FILE_LIST"
fi

echo "Run finished at $(date --rfc-3339=seconds)" >> "$RUN_LOG"
echo "Finished. Check $RUN_LOG and per-test reports under each test's $REPORT_LOCATION directory for details."