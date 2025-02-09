DYNAMATIC_DIR=.

source "$DYNAMATIC_DIR/tools/dynamatic/scripts/utils.sh"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

OUT_DIR="experimental/tools/ndwirerer/out"
F_HW="$OUT_DIR/hw.mlir"

COMP_DIR="$OUT_DIR/comp"
DOT="$COMP_DIR/miter.dot"

REWRITES="experimental/test/tools/elastic-miter-generator/rewrites"

# TODO remove this
cd build
ninja
exit_on_fail "Failed to build miter module generator"
cd ..

MOD="c"
F_HANDSHAKE_MITER="$COMP_DIR/ndwirererd_${MOD}_lhs.mlir"

build/bin/ndwirerer --lhs=$REWRITES/${MOD}_rhs.mlir -o $COMP_DIR
exit_on_fail "Failed to create miter module"


"bin/export-dot" $F_HANDSHAKE_MITER "--edge-style=spline" > $DOT
exit_on_fail "Failed to convert to dot"
dot -Tpng $DOT > $COMP_DIR/visual.png

python3 "../dot2smv/dot2smv" $DOT
exit_on_fail "Failed to convert to SMV"

mv $(dirname $DOT)/model.smv $(dirname $DOT)/${MOD}_lhs.smv

python3 experimental/tools/ndwirerer/create_state_wrapper.py --inf --json="experimental/tools/elastic-miter-generator/out/comp/elastic-miter-config.json"> experimental/tools/ndwirerer/out/comp/main_inf.smv
exit_on_fail "Failed to create SMV main file"

N=1
while [ true ]; do
  echo "Checking $N tokens."
  python3 experimental/tools/ndwirerer/create_state_wrapper.py -N $N --json="experimental/tools/elastic-miter-generator/out/comp/elastic-miter-config.json"> experimental/tools/ndwirerer/out/comp/main_$N.smv
  exit_on_fail "Failed to create SMV main file"

  nuXmv -source $COMP_DIR/prove_inf.cmd > "$OUT_DIR/inf_states.txt"
  # TODO we need to automatically create prove_$N.cmd
  nuXmv -source $COMP_DIR/prove_$N.cmd > "$OUT_DIR/$N_states.txt"

  python3 experimental/tools/ndwirerer/get_states.py "$OUT_DIR/inf_states.txt" "$OUT_DIR/$N_states.txt"

  if [[ $? -ne 0 ]]; then
    ((N++))
  else 
    echo $N
    break
  fi
done


