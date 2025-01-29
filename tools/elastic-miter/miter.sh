DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="tools/elastic-miter/out/"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_miter.mlir"
DOT="$COMP_DIR/miter.dot"


"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_HW"
exit_on_fail "Failed to lower to HW" "Lowered to HW"


"bin/export-dot" $F_HANDSHAKE_EXPORT "--edge-style=spline" > $DOT
dot -Tpng $DOT > $COMP_DIR/visual.png

# python3 "../dot2smv/dot2smv" "./miter.dot"