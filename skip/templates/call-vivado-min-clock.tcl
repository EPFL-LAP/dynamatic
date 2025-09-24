# Parameters
set min_period 3.0    ;# Minimum period in ns (for binary search)
set max_period 10.0  ;# Maximum period in ns
set tolerance 0.1     ;# Acceptable tolerance for slack (in ns)

# Paths to report files
set timing_post_syn_report "PATH_TO_TEST_FOLDER/out/vivado/timing_post_syn.rpt"
set timing_post_pr_report "PATH_TO_TEST_FOLDER/out/vivado/timing_post_pr.rpt"
#set xdc_file "PATH_TO_CONSTRAINTS_FILE"  ;# Your XDC file

# Function to modify the XDC file with a new clock period
proc modify_xdc {period} {
    puts "a\n"
    puts $period
    set half_period [expr $period / 2]
    puts "a\n"
    puts $half_period

    catch {eval create_clock -name clk -period $period -waveform {0.000 $half_period} [get_ports clk]} result
    

}

# Function to check if the slack is met from a report file
proc check_timing_met {report_address} {

    set file_id [open $report_address r]

    set report_content [read $file_id]

    # Find the slack line in the timing report
    if {[regexp {Slack\s*\(VIOLATED\)} $report_content]} {
        puts "Timing violated"
        return 0  ;# Timing is violated
    } elseif {[regexp {Slack\s*\(met\)} $report_content]} {
        # Check if the report contains "Slack (met)"
        puts "Timing met"
        return 1  ;# Timing is met
    }
    return 1
}

# Function to run synthesis, place, and route
proc run_vivado {} {
    # Run Vivado's synthesis, place, and route commands
    catch {
        synth_design -top TOP_MODULE -part xc7k160tfbg484-3 -no_iobuf -mode out_of_context
        report_utilization > "PATH_TO_TEST_FOLDER/out/vivado/utilization_post_syn.rpt"
        report_timing > "PATH_TO_TEST_FOLDER/out/vivado/timing_post_syn.rpt"
        place_design
        route_design
        report_utilization > "PATH_TO_TEST_FOLDER/out/vivado/utilization_post_pr.rpt"
        report_timing > "PATH_TO_TEST_FOLDER/out/vivado/timing_post_pr.rpt"
    } result
    return $result
}

# Perform binary search for the best clock period
proc binary_search {} {
    global min_period max_period tolerance timing_post_syn_report timing_post_pr_report

    set best_period $max_period

    while {[format "%.1f" [expr ($max_period - $min_period)]] > $tolerance} {
        set mid_period [format "%.1f" [expr ($max_period + $min_period) / 2]]
        puts "Testing clock period: $mid_period ns"
        
        # Modify the XDC file to change the clock period
        modify_xdc $mid_period

        # Run Vivado (synthesis, placement, routing)
        puts "Running Vivado with period $mid_period..."
        run_vivado

        # Check the post-route timing report (timing_post_pr.rpt)
        if {[check_timing_met $timing_post_pr_report]} {
            # Timing met, search for a faster clock period
            puts "Timing met with period $mid_period ns. Searching for a faster clock."
            set best_period $mid_period
            set max_period $mid_period  ;# Narrow the search to the upper half
        } else {
            # Timing not met, search for a slower clock period
            puts "Timing NOT met with period $mid_period ns. Searching for a slower clock."
            set min_period $mid_period  ;# Narrow the search to the lower half
        }
        
        # Optional: Pause for a brief moment before the next iteration
        after 1000  ;# 1 second pause
    }

    # Return the best clock period found
    puts "Best clock period found: $best_period ns"
    return $best_period
}

# Run the binary search to find the best clock period
binary_search
