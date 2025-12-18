# The exception catching is there because otherwise the bash script becomes incoherent
# when the tcl console fails and tries to resume execution of commands inside the tcl console, failing all of them
if {[catch {
  read_vhdl -vhdl2008 [glob skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/hdl/*.vhd]
  # This may not be a fatal error in the script
  if {[catch {read_verilog [glob skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/hdl/*.v]} result2]} {
    puts "No verilog files found: $result2"
  }
  read_xdc skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/vivado/Timing_based_target_5.00_cp_4.70/constraints.xdc
  synth_design -top histogram -part xc7k160tfbg484-3 -no_iobuf -mode out_of_context
  report_utilization > skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/vivado/Timing_based_target_5.00_cp_4.70/utilization_post_syn.rpt
  report_timing > skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/vivado/Timing_based_target_5.00_cp_4.70/timing_post_syn.rpt
  place_design
  route_design
  report_utilization > skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/vivado/Timing_based_target_5.00_cp_4.70/utilization_post_pr.rpt
  report_timing > skip2/runs/run_2025-12-16_15-08-39/histogram-7-17/out/vivado/Timing_based_target_5.00_cp_4.70/timing_post_pr.rpt
} result]} {
  puts "Error during tcl script execution: $result"
  exit
}

exit