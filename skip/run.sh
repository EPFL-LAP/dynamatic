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
# TCL_TEMPLATE="./vitis-benchmark/templates/call-vivado-impl-for-dynamatic.tcl"
TCL_TEMPLATE="./skip/templates/call-vivado-impl-for-dynamatic.tcl"


# Template file to feed constraints to vivado
CONSTRAINTS_TEMPLATE="./skip/templates/call-vivado-impl-for-dynamatic-contraints.xdc"

PLACE_AND_ROUTE_TIMING="/out/vivado/timing_post_pr.rpt"

SIM_RESULT="out/sim/report.txt"

FIND_CP_OUTPUT="out/vivado/find_cp.rpt"

#===OUTPUTS===

# Output file containing the compilation time of Dynamatic
COMPILATION_TIME_FILE="out/vivado/compilation-time.txt"

CP_FILE="out/vivado/cp.txt"

# Temporary frontend script
SCRIPT_TEMP="temp_frontend-script-compile-with-dynamatic.dyn"

# Temporary file to hold the list of C files
FILE_LIST="temp-file-list-compile-with-dynamatic.txt"

# Temporary tcl file
TCL_TEMPORARY="temp-call-vivado-impl-for-dynamatic.tcl"


# File containing a list of tests that have timeout
OUT_TIMEOUTS="./skip/timeouts-compile-with-dynamatic.txt"

COMPILATION_ERRORS="./skip/benchmark-compilation-errors.txt"


# DUMP_FILE="/dev/null"
DUMP_FILE="vivado_dump.txt"

REPORT_LOCATION="out/vivado"


# =======END DECLARATIONS========

USE_FILE=false

# Make sure that the relative paths are always at the same level as the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Dynamatic expects the top level folder as starting folder when executing
cd ..

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--file) # If the -f or --file flag is passed
            FILE_LIST="$SCRIPT_DIR/$2"
            USE_FILE=true
            shift 2
            ;;
        *) # For any unknown argument
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$USE_FILE" == false ]]; then
    # Find all .c files in the test directory and its subdirectories, that are not inside an out directory
    find $TEST_DIR -type d -name "out" -prune -o -type f -name "*.c" -print > $FILE_LIST
fi


check_slack() {
    local report_file=$1
    echo $report_file
    if grep -q "Slack\s*(VIOLATED)" "$report_file"; then
        echo "Timing violated"
        return 1  # Timing not met
    elif grep -q "Slack\s*(MET)" "$report_file"; then
        echo "Timing met"
        return 0  # Timing met
    else
        echo "No slack information found"
        return 2  # No slack information
    fi
}


modify_xdc() {
    local period=$1
    local half_period=$(echo "$period / 2" | bc -l)

    echo "Creating clock constraint with period $period ns and half-period $half_period ns"

    # Replace the create_clock line in the XDC file (adjust as needed for your XDC format)
    sed -i "s/create_clock -name clk -period .*/create_clock -name clk -period $period -waveform {0.000 $half_period} [get_ports clk]/" "$CONSTRAINTS_TEMPLATE"
}

binary_search() {
    local min_period=3.00
    local max_period=10.00
    local tolerance=0.1
    local best_period=$max_period

    echo > "$dir/$CP_FILE"

    while (( $(echo "$max_period - $min_period > $tolerance" | bc -l) )); do
        # Calculate the midpoint
        local mid_period=$(echo "scale=1; ($min_period + $max_period) / 2" | bc -l)
        echo "Testing clock period: $mid_period ns"

        compile $mid_period
        simulate
        modify_xdc $mid_period
        place_and_route

        check_slack "$dir/$PLACE_AND_ROUTE_TIMING"
        result=$?
        echo $result
        if [ $result -eq 0 ]; then
            # Timing met, search for a faster clock period
            echo "Timing met with period $mid_period ns. Searching for a faster clock."
            best_period=$mid_period
            max_period=$mid_period  # Narrow the search to the upper half
        else
            # Timing not met, search for a slower clock period
            echo "Timing NOT met with period $mid_period ns. Searching for a slower clock."
            min_period=$mid_period  # Narrow the search to the lower half
        fi

        echo "$mid_period  $result" >> "$dir/$CP_FILE"


        # Optional: Pause for a brief moment before the next iteration
        sleep 1  # 1 second pause
    done

    echo "Best clock period found: $best_period ns"
    echo "BEST $best_period" >> "$dir/$CP_FILE"
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




compile() {
  clock=$1
  echo compiling
  # Create a new frontend file from the template
  cp "$COMPILE_SCRIPT_TEMPLATE" "$SCRIPT_TEMP"

  # Complete the set-src command with the appropriate path
  sed -i "s|PATH_TO_C_FILE|$cfile|g; \
          s|CLOCK_PERIOD|$clock|g" "$SCRIPT_TEMP"

  echo "Calling dynamatic for $cfile..."

  mkdir -p "$(dirname $dir/$COMPILATION_TIME_FILE)"

  # timeout $TIMEOUT_DURATION bash -c "{ time \"$DYNAMATIC\" --run \"$SCRIPT_TEMP\" 2>> $COMPILATION_ERRORS ; } 2> \"$dir/$COMPILATION_TIME_FILE\""
  # if [ $? -eq 124 ]; then
  #   echo "$basename : compilation timeout" >> $OUT_TIMEOUTS
  #   ((timeout_count++))
  # fi


  { time timeout "$TIMEOUT_DURATION" "$DYNAMATIC" --run "$SCRIPT_TEMP" \
      > "$dir/$basename.compile.log" 2>&1 ; } 2> "$dir/$COMPILATION_TIME_FILE"

  rc=$?

  if [ $rc -eq 124 ]; then
    echo "$basename : compilation timeout" >> $OUT_TIMEOUTS
    ((timeout_count++))
    return 124
  elif [ $rc -ne 0 ]; then
    echo "$basename : compilation failed (exit $rc)" >> "$COMPILATION_ERRORS"
    return $rc
  fi
}

simulate(){
  echo simulating
  # Create a new frontend file from the template
  cp "$SIMULATE_SCRIPT_TEMPLATE" "$SCRIPT_TEMP"

  half_clock_period=$(echo "$clock / 2" | bc -l)  
  #Compelte the set-src command with the appropriate path
  sed -i \
  -e "s|PATH_TO_C_FILE|$cfile|g" \
  "$SCRIPT_TEMP"

  timeout $TIMEOUT_DURATION bash -c "\"$DYNAMATIC\" --run \"$SCRIPT_TEMP\""
  if [ $? -eq 124 ]; then
    echo "$basename : simulation timeout" >> $OUT_TIMEOUTS
    ((timeout_count++))
  fi
}


place_and_route(){
  # Create directory for reports
  mkdir -p $dir/$REPORT_LOCATION

  # Create a new config file from the template
  cp "$TCL_TEMPLATE" "$TCL_TEMPORARY"
  

  # Replace necessary information on tcl template
  sed -i -e "s|PATH_TO_TEST_FOLDER|$dir|g" \
        -e "s|PATH_TO_CONSTRAINTS_FILE|$CONSTRAINTS_TEMPLATE|g" \
        -e "s|TOP_MODULE|$basename|g" \
        "$TCL_TEMPORARY"

  echo 
  echo "Calling vivado for $cfile..."


  timeout $TIMEOUT_DURATION bash -c "vivado -notrace -nolog -nojournal -mode tcl -script \"$TCL_TEMPORARY\" >> $DUMP_FILE"
  if [ $? -eq 124 ]; then
    echo "$basename : place and routing timeout" >> $OUT_TIMEOUTS
    ((timeout_count++))
  fi
}


find_best_cp(){

  tolerance=0.1
  initial_cp=4.0
  report_file="$dir/$PLACE_AND_ROUTE_TIMING"
  current_cp=$initial_cp
  previous_cp=0
  best_cp=0

  while true; do
    compile $current_cp
    simulate $current_cp
    modify_xdc $current_cp
    place_and_route

    slack=$(get_slack "$report_file")
      
    
    slack_value=$(echo "$slack" | awk '{print int($1*10)/10}')

    if (( $(echo "$slack < 0" | bc -l) )); then
      change="-$tolerance"
    else
      change=$tolerance
    fi

    slack_value=$(echo "$slack_value + $change" | bc)

    current_cp=$(echo "$current_cp - $slack_value" | bc)
    echo $current_cp

      
    
    # 5. If the difference between the old and new clock periods is less than the tolerance, stop
    period_diff=$(echo "$previous_cp - $current_cp" | bc)
    abs_period_diff=$period_diff
    if (( $(echo "$period_diff < 0" | bc -l) )); then
      abs_period_diff="-$period_diff"
    fi


    if (( $(echo "$abs_period_diff <= $tolerance" | bc -l) )) then
        best_cp=$current_cp
        if (( $(echo "$slack_value < 0" | bc -l) )); then
          best_cp=$previous_cp
        fi
        break
    fi

      # Update previous clock period
    previous_cp=$current_cp
  done

  echo "Best clock period found: $best_cp ns"

}


find_real_cp(){
  step=-0.1
  initial_cp=$1
  report_file="$dir/$PLACE_AND_ROUTE_TIMING"
  current_cp=$initial_cp

  echo find real
  while true; do
    echo "Current CP $current_cp" >> "$dir/$FIND_CP_OUTPUT"
    modify_xdc $current_cp
    place_and_route

    slack=$(get_slack "$report_file")

    echo "Slack $slack"
    echo "Slack $slack" >> "$dir/$FIND_CP_OUTPUT"
    

    if [ -z "$slack" ]; then
      echo "No slack information found, stopping search."
      echo "No slack information found, stopping search." >> "$dir/$FIND_CP_OUTPUT"
      break
    fi
      
    if (( $(echo "$slack > 0" | bc -l) )); then
      
      break
    fi

    slack_value=$(echo "$slack" | awk '{print int($1*10)/10}')
    if (( $(echo "$slack_value == 0" | bc -l) )); then
      slack_value=$step
    fi

    echo "Slack Value: $slack_value"

    current_cp=$(echo "$current_cp - $slack_value" | bc)
  done

  BEST_CP_GLOB=$current_cp
  echo "BEST CP found: $BEST_CP_GLOB"
  echo "BEST CP found: $BEST_CP_GLOB" >> "$dir/$FIND_CP_OUTPUT"

}


find_best_timing(){
  basename=$1
  l=$2
  s=$3

  cp=4.5 # Start value of CP

  best_cycles=10000000
  best_timing=10000000
  best_timing_cp=0
  best_timing_cc=0

  echo "$dir/$CP_FILE"
  > "$dir/$CP_FILE"

  echo > "$dir/$FIND_CP_OUTPUT"


  while true; do
      echo "----------------------"
      echo "Running for CP = $cp"
      echo "----------------------" >> "$dir/$FIND_CP_OUTPUT"
      echo "Running for CP = $cp" >> "$dir/$FIND_CP_OUTPUT"

      # Call the compile function (assuming it's a script or command)
      # compile $cp
      python3 ./buffer.py . $basename $l $s $cp

      # Run the simulation (modify this to match your actual simulation command)
      simulate

      
      cycles=$(grep -A1 "\*\* Note: Simulation done!" "$dir/$SIM_RESULT" | grep "Latency" | awk '{print $8}')

      echo "Clock cycles: $cycles"
      echo "Clock cycles: $cycles" >> "$dir/$FIND_CP_OUTPUT"
      echo "Clock cycles: $cycles" > "$dir/$CP_FILE"

      find_real_cp $cp

      echo $BEST_CP_GLOB
      real_cp=$BEST_CP_GLOB
      echo $real_cp


      timing=$(echo "$cycles * $real_cp" | bc -l)
      echo "timing is $timing"

      echo "Real CP: $real_cp"  >> "$dir/$CP_FILE"
      echo "Timing: $timing" >> "$dir/$CP_FILE"

      cd skip
      python3 ./retrieve-information-dynamatic-lsq.py $basename $l $s $cp
      cd ..


      # if (( $(echo "$cycles < $best_cycles" | bc -l) )); then
      #   best_cycles=$cycles
      # fi

      if (( $(echo "$timing < $best_timing" | bc -l) )); then
        best_timing=$timing
      #   best_timing_cp=$real_cp
      #   best_timing_cc=$cycles
      fi

      if (( $(echo "$timing > 1.05 * $best_timing" | bc -l) )); then
        break
      fi


      # Break condition (optional): stop after a certain CP value
      if (( $(echo "$cp > 3.5" | bc -l) )); then
          break
      fi

      # Increment CP by 0.5
      cp=$(echo "$cp + 0.5" | bc)
  done


}


timeout_count=0

while IFS= read -r cfile; do
    # Remove the prefix "../" using sed. Note that there was a change in directory
    cfile=$(echo "$cfile" | sed 's#^\.\./##')

    # Extract the base name of the file without the extension
    basename=$(basename "$cfile" .c)

    # Extract the directory of the file
    dir=$(dirname "$cfile")

    # Construct the corresponding .h file path
    hfile="$dir/$basename.h"


    load_values=(10)
    store_values=(10)

    for l in "${load_values[@]}"; do
      for s in "${store_values[@]}"; do
        if [ "$l" != "$s" ]; then
            continue
        fi

        # python3 ./skip/change_lsq.py $l $s

        # find_best_cp
        # binary_search
        find_best_timing $basename $l $s

    echo 
      done
    done

    echo 

done < $FILE_LIST

#Remove temporary files
rm -f $SCRIPT_TEMP

# If the -f flag was provided do not erase the file_list file, it was modified by the user
if [[ "$USE_FILE" == false ]]; then
    rm -f $FILE_LIST
fi

# Come back to starting directory
cd $SCRIPT_DIR

# Print the number of timeouts that occurred at the end
if [ $timeout_count -gt 0 ]; then
    echo "Finished with errors. $timeout_count command(s) timed out."
else
    echo "Succesfully finished!"
fi