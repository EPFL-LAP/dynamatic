KERNEL_NAME="elastic_miter_c_lhs_c_rhs"
DIR="experimental/tools/elastic-miter/out/c/miter"
F_CSV="$DIR/viz.csv"
F_DOT="$DIR/model.dot"
F_DOT_POS_TMP="$DIR/model.tmp.dot"
F_DOT_POS="$DIR/model_pos.dot"

./build/bin/cex2csv "$DIR/$KERNEL_NAME.mlir" "$DIR/result.txt" $KERNEL_NAME > "$F_CSV"

# Generate a version of the DOT with positioning information
sed -e 's/splines=spline/splines=ortho/g' "$F_DOT" > "$F_DOT_POS_TMP"
dot -Tdot "$F_DOT_POS_TMP" > "$F_DOT_POS"
exit_on_fail "Failed to add positioning info. to DOT" "Added positioning info. to DOT"
rm "$F_DOT_POS_TMP"

# Launch the dataflow visualizer
echo_info "Launching visualizer..."
./bin/visual-dataflow "--dot=$F_DOT_POS" "--csv=$F_CSV"
exit_on_fail "Failed to run visualizer" "Visualizer closed"