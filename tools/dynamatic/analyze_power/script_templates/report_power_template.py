
vector_base_report_power_tcl = r"""
# Date: %date
# This tcl script is used to synthesis the design and report the power
set TOP_DESIGN %design
set VHDL_SRC %hdlsrc

# Read all source files
%inputs

# Read the design constraints
read_xdc period.xdc

# Run synthesis
synth_design -top $TOP_DESIGN -part xc7k160tfbg484-1 -no_iobuf -mode out_of_context -flatten_hierarchy none

## Report pre_all power
read_saif -out_file unmatched_pre_all.rpt -file %saif2
report_power -file %{report_folder}/%{date}_pre_all.pwr
reset_switching_activity -all

# Ciao!
exit

"""