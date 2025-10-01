# The exception catching is there because otherwise the bash script becomes incoherent
# when the tcl console fails and tries to resume execution of commands inside the tcl console, failing all of them
if {[catch {
  read_vhdl -vhdl2008 [glob PATH_TO_TEST_FOLDER/out/hdl/*.vhd]
  # This may not be a fatal error in the script
  if {[catch {read_verilog [glob PATH_TO_TEST_FOLDER/out/hdl/*.v]} result2]} {
    puts "No verilog files found: $result2"
  }
  read_xdc PATH_TO_CONSTRAINTS_FILE
  synth_design -top TOP_MODULE -part xc7k160tfbg484-3 -no_iobuf -mode out_of_context
  report_utilization > PATH_TO_TEST_FOLDER/out/vivado/utilization_post_syn.rpt
  report_timing > PATH_TO_TEST_FOLDER/out/vivado/timing_post_syn.rpt
  place_design
  route_design
  report_utilization > PATH_TO_TEST_FOLDER/out/vivado/utilization_post_pr.rpt
  report_timing > PATH_TO_TEST_FOLDER/out/vivado/timing_post_pr.rpt
} result]} {
  puts "Error during tcl script execution: $result"
  exit
}

exit