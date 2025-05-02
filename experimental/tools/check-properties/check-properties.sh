KERNEL_NAME=$1
DYNAMATIC_DIR="."
OUT_DIR="$DYNAMATIC_DIR/integration-test/$KERNEL_NAME/out"
MODEL_DIR="$OUT_DIR/hdl"
MLIR_DIR="$OUT_DIR/comp"
FORMAL_DIR="$OUT_DIR/formal"

rm -rf "$FORMAL_DIR" && mkdir -p "$FORMAL_DIR"

#annotate properties
$DYNAMATIC_DIR/bin/dynamatic-opt $MLIR_DIR/handshake_export.mlir --handshake-annotate-properties=json-path=$MLIR_DIR/formal_properties.json > $MLIR_DIR/handshake_export_annotated.mlir

# call the model generator
bash $DYNAMATIC_DIR/tools/dynamatic/scripts/write-hdl.sh $DYNAMATIC_DIR $OUT_DIR $KERNEL_NAME smv

# create the testbench
build/bin/testbench-generator -i $MODEL_DIR --name $KERNEL_NAME -o $MODEL_DIR --mlir $MLIR_DIR/hw.mlir

# use the modelcheker
echo "set verbose_level 0;
set pp_list cpp;
set counter_examples 0;
set dynamic_reorder 1;
set on_failure_script_quits;
set reorder_method sift;
set enable_sexp2bdd_caching 0;
set bdd_static_order_heuristics basic;
set cone_of_influence;
set use_coi_size_sorting 1;
read_model -i $MODEL_DIR/main.smv;
flatten_hierarchy;
encode_variables;
build_flat_model;

build_model -f;
check_invar -s forward;
check_ctlspec;

show_property -o $FORMAL_DIR/property.rpt;
time;
quit" > $FORMAL_DIR/prove.cmd


file=$MODEL_DIR/$KERNEL_NAME.smv
total=$(( $(awk '/-- properties/ {found=1; next} found' "$file" | wc -l) + 26))
bar_length=30
i=0

echo "[INFO] Running nuXmv"
nuXmv -source $FORMAL_DIR/prove.cmd | while IFS= read -r line; do
    ((i++))
    percent=$(( i * 100 / total ))
    filled=$(( i * bar_length / total ))
    bar=$(printf "%-${filled}s" "#" | tr ' ' '#')
    max_line_length=$(( $(tput cols) - bar_length - 20 ))
    display_line="${line:0:max_line_length}"
    printf "\r\033[K[%-${bar_length}s] %3d%% %s" "$bar" "$percent" "$display_line"
done

# parse the results
printf "\n[INFO] Saving formal verification results\n"
python experimental/tools/smv-check-opt/parse_results.py $MLIR_DIR/formal_properties.json $FORMAL_DIR/property.rpt