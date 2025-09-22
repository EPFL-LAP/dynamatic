set_param general.maxThreads 8
set vivado_ver [version -short]
set fpo_ver 7.1
if {[regexp -nocase {2015\.1.*}  match]} {
    set fpo_ver 7.0
}

read_vhdl -vhdl2008 [glob /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/hdl/*.vhd]


source /home/shundroid/dynamatic/tools/backend/synth-resources/addf_vitis_hls_single_precision_lat_8.tcl
source /home/shundroid/dynamatic/tools/backend/synth-resources/cmpf_vitis_hls_single_precision_lat_0.tcl
source /home/shundroid/dynamatic/tools/backend/synth-resources/divf_vitis_hls_single_precision_lat_28.tcl
source /home/shundroid/dynamatic/tools/backend/synth-resources/mulf_vitis_hls_single_precision_lat_4.tcl
source /home/shundroid/dynamatic/tools/backend/synth-resources/subf_vitis_hls_single_precision_lat_8.tcl
read_xdc /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/period_5.000.xdc
synth_design -top collision_donut -part xc7k160tfbg484-2 -no_iobuf -mode out_of_context
report_utilization > /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/utilization_post_syn.rpt
report_timing > /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/timing_post_syn.rpt
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
report_utilization > /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/utilization_post_pr.rpt
report_timing > /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/timing_post_pr.rpt

write_checkpoint -force /home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/impl.dcp

set outFile [open "/home/shundroid/dynamatic/integration-test/collision_donut/out_5/synth/primitive_counts.txt" w]
proc count_primitives {cells outFile} {
    set primitive_counts {}
    set total_primitive_counts 0
    foreach cell $cells {
        if {[get_property IS_PRIMITIVE $cell]} {
            set group [get_property PRIMITIVE_GROUP $cell]
            if {[dict exists $primitive_counts $group]} {
                dict incr $primitive_counts $group
            } else {
                dict set $primitive_counts $group 1
            }
        } else {
            puts "not primitive cell: $cell"
        }
        incr total_primitive_counts [get_property PRIMITIVE_COUNT $cell]
    }
    puts $outFile "Total Primitive Count: $total_primitive_counts"
    puts $outFile $primitive_counts
    return $primitive_counts
}

# Now call it with different filters
puts $outFile "All Primitives:"
count_primitives [get_cells -leaf] $outFile
puts $outFile "Spec*:"
count_primitives [get_cells spec* -leaf] $outFile
puts $outFile "Buffers:"
count_primitives [get_cells buffer* -leaf] $outFile
# puts $outFile "Fork:"
# count_primitives [get_cells fork* -leaf] $outFile
puts $outFile "Others:"
count_primitives [get_cells -leaf -filter {NAME !~ "buffer*" && NAME !~ "spec*"}] $outFile
close $outFile

exit
