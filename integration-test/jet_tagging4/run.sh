set -e

DYNAMATIC_PATH=$(realpath "../..")

run_test () {

echo "testing $2"
time bash $DYNAMATIC_PATH/tools/frontend/llvm-cf-handshake.sh \
  $DYNAMATIC_PATH "$1" "$2"
}

run_test "jet_tagging4.c" "jet_tagging4"
