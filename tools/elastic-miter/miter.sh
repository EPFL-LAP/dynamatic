DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="tools/elastic-miter/out/"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_export.mlir"



"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_HW"
exit_on_fail "Failed to lower to HW" "Lowered to HW"


bin/export-rtl $F_HW $OUT_DIR/miter data/rtl-config-verilog.json --hdl verilog

find $OUT_DIR/miter -name "*.v" > $OUT_DIR/miter/filelist.f

