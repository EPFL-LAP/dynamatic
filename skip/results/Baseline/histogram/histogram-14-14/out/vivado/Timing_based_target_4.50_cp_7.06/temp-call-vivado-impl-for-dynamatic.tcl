# The exception catching is there because otherwise the bash script becomes incoherent
# when the tcl console fails and tries to resume execution of commands inside the tcl console, failing all of them
if {[catch {
  read_vhdl -vhdl2008 [glob skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/hdl/*.vhd]
  # This may not be a fatal error in the script
  if {[catch {read_verilog [glob skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/hdl/*.v]} result2]} {
    puts "No verilog files found: $result2"
  }
  read_xdc skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/vivado/Timing_based_target_4.50_cp_7.06/constraints.xdc
  synth_design -top histogram -part xc7k160tfbg484-3 -no_iobuf -mode out_of_context
  report_utilization > skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/vivado/Timing_based_target_4.50_cp_7.06/utilization_post_syn.rpt
  report_timing > skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/vivado/Timing_based_target_4.50_cp_7.06/timing_post_syn.rpt
  place_design
  route_design
  report_utilization > skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/vivado/Timing_based_target_4.50_cp_7.06/utilization_post_pr.rpt
  report_timing > skip2/runs/run_2025-12-16_16-06-24/histogram-14-14/out/vivado/Timing_based_target_4.50_cp_7.06/timing_post_pr.rpt
} result]} {
  puts "Error during tcl script execution: $result"
  exit
}

exit