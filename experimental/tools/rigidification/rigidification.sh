#!/bin/bash

DYNAMATIC_DIR=$1
OUTPUT_DIR=$2
KERNEL_NAME=$3
F_HANDSHAKE_EXPORT=$4
F_HANDSHAKE_RIGIDIFIED=$5

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"

FORMAL_DIR="$OUTPUT_DIR/formal"
MODEL_DIR="$FORMAL_DIR/model"

DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_EXPORT_RTL_BIN="$DYNAMATIC_DIR/bin/export-rtl"

F_FORMAL_HW="$FORMAL_DIR/hw.mlir"
F_FORMAL_PROP="$FORMAL_DIR/formal_properties.json"
F_NUXMV_PROP="$FORMAL_DIR/property.rpt"
F_NUXMV_CMD="$FORMAL_DIR/prove.cmd"

NUSMV_BINARY="$DYNAMATIC_DIR/ext/NuSMV"
NUXMV_BINARY="$DYNAMATIC_DIR/ext/nuXmv/bin/nuXmv"

FORMAL_TESTBENCH_GEN="$DYNAMATIC_DIR/build/bin/rigidification-testbench"

RTL_CONFIG_SMV="$DYNAMATIC_DIR/data/rtl-config-smv.json"

SMV_RESULT_PARSER="$DYNAMATIC_DIR/experimental/tools/rigidification/parse_nuxmv_results.py"


rm -rf "$FORMAL_DIR" && mkdir -p "$FORMAL_DIR"


# Annotate properties
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" \
  --handshake-annotate-properties=json-path=$F_FORMAL_PROP \
  > /dev/null

# handshake level -> hw level
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_FORMAL_HW"

# generate SMV
"$DYNAMATIC_EXPORT_RTL_BIN" \
  "$F_FORMAL_HW" \
  "$MODEL_DIR" \
  "$RTL_CONFIG_SMV" \
  --hdl smv \
  --property-database "$F_FORMAL_PROP" \
  --dynamatic-path "$DYNAMATIC_DIR"

# create the testbench
"$FORMAL_TESTBENCH_GEN" \
  -i $MODEL_DIR \
  --name $KERNEL_NAME \
  --mlir $F_FORMAL_HW
exit_on_fail "Failed to create formal testbench" \
  "Created formal testbench"

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
exit_on_fail "Failed to create SMV script" \
  "Created SMV script"

# run nuXmv and increase the counter everytime it completes the check of a property
echo "[INFO] Running nuXmv" >&2
$NUXMV_BINARY -source $F_NUXMV_CMD 
exit_on_fail "Failed to check formal properties" \
  "Performed model checking to verify the formal property"

# parse the results
printf "\n[INFO] Saving formal verification results\n" >&2
python "$SMV_RESULT_PARSER" "$F_FORMAL_PROP" "$F_NUXMV_PROP"

# apply rigidification
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --handshake-rigidification=json-path=$F_FORMAL_PROP > "$F_HANDSHAKE_RIGIDIFIED"
exit_on_fail "Applied formal properties to simplify the circuit" \
  "Failed to apply formal properties to simplify the circuit"

