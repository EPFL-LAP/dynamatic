#!/bin/bash

DYNAMATIC_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3
F_HANDSHAKE_EXPORT=$4

FORMAL_DIR="$OUTPUT_DIR/formal"
MODEL_DIR="$FORMAL_DIR/model"

DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_EXPORT_RTL_BIN="$DYNAMATIC_DIR/bin/export-rtl"

F_FORMAL_HW="$FORMAL_DIR/hw.mlir"
F_FORMAL_PROP="$FORMAL_DIR/formal_properties.json"
F_NUXMV_PROP="$FORMAL_DIR/property.rpt"
F_NUXMV_CMD="$FORMAL_DIR/prove.cmd"


rm -rf "$FORMAL_DIR" && mkdir -p "$FORMAL_DIR"


# Annotate properties
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" \
  --handshake-annotate-properties=json-path=$F_FORMAL_PROP \
  > /dev/null

# handshake level -> hw level
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_FORMAL_HW"

# generate SMV
"$DYNAMATIC_EXPORT_RTL_BIN" "$F_FORMAL_HW" $MODEL_DIR data/rtl-config-smv.json --hdl smv --property-database $F_FORMAL_PROP

# create the testbench
build/bin/rigidification-testbench -i $MODEL_DIR --name $KERNEL_NAME --mlir $F_FORMAL_HW

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
show_property -o $F_NUXMV_PROP;
time;
quit" > $F_NUXMV_CMD


# progress bar animation

# approximately count the number of properties we need to check by counting
# the lines of the generated model
file=$MODEL_DIR/$KERNEL_NAME.smv
total=$(( $(awk '/-- properties/ {found=1; next} found' "$file" | wc -l) + 27))
bar_length=30
i=0

# run nuXmv and increase the counter everytime it completes the check of a property
echo "[INFO] Running nuXmv" >&2
nuXmv -source $F_NUXMV_CMD | while IFS= read -r line; do
  ((i++))
  percent=$(( i * 100 / total ))
  filled=$(( i * bar_length / total ))
  bar=$(printf "%-${filled}s" "#" | tr ' ' '#') >&2
  max_line_length=$(( $(tput cols) - bar_length - 20 ))
  display_line="${line:0:max_line_length}"
  printf "\r\033[K[%-${bar_length}s] %3d%% %s" "$bar" "$percent" "$display_line" >&2
done

# parse the results
printf "\n[INFO] Saving formal verification results\n" >&2
python experimental/tools/rigidification/parse_nuxmv_results.py $F_FORMAL_PROP $F_NUXMV_PROP

# apply rigidification
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --handshake-rigidification=json-path=$F_FORMAL_PROP

