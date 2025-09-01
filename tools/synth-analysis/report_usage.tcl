open_checkpoint SYNTH_DIR/impl.dcp
set outFile [open "primitive_counts.txt" w]
proc count_primitives {cells outFile} {
    set primitive_counts {}
    set total_primitive_counts 0
    foreach cell $cells {
        if {[get_property IS_PRIMITIVE $cell]} {
            set group [get_property PRIMITIVE_GROUP $cell]
            if {[dict exists $primitive_counts $group]} {
                dict incr primitive_counts $group
            } else {
                dict set primitive_counts $group 1
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
