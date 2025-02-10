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

MOD="z"
F_HANDSHAKE_MITER="$COMP_DIR/ndwirererd_${MOD}_lhs.mlir"


"bin/export-dot" $REWRITES/${MOD}_lhs.mlir "--edge-style=spline" > $DOT
exit_on_fail "Failed to convert to dot"
dot -Tpng $DOT > $COMP_DIR/visual.png

python3 "../dot2smv/dot2smv" $DOT
exit_on_fail "Failed to convert to SMV"

# TODO ...
mv $(dirname $DOT)/model.smv $(dirname $DOT)/${MOD}_lhs.smv

python3 experimental/tools/ndwirerer/create_state_wrapper.py --inf --mlir=$REWRITES/${MOD}_lhs.mlir > experimental/tools/ndwirerer/out/comp/main_inf.smv
exit_on_fail "Failed to create SMV main file"

nuXmv -source $COMP_DIR/prove_inf.cmd > "$OUT_DIR/inf_states.txt"
exit_on_fail "Failed to analyise reachable states with infinite tokens."

N=1
while [ true ]; do
  echo "Checking $N tokens."
  python3 experimental/tools/ndwirerer/create_state_wrapper.py -N $N --mlir="$(dirname $DOT)/${MOD}_lhs.smv"> experimental/tools/ndwirerer/out/comp/main_$N.smv
  exit_on_fail "Failed to create SMV main file"

  # TODO we need to automatically create prove_$N.cmd
  nuXmv -source $COMP_DIR/prove_$N.cmd > "$OUT_DIR/$N_states.txt"
  exit_on_fail "Failed to analyise reachable states with $N tokens."

  nr_of_differences=$(python3 experimental/tools/ndwirerer/get_states.py "$OUT_DIR/inf_states.txt" "$OUT_DIR/$N_states.txt")


  if [[ $nr_of_differences -ne 0 ]]; then
    ((N++))
  else 
    echo $N
    break
  fi
done


