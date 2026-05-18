#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

DYNAMATIC_DIR=$1
SRC_DIR=$2
KERNEL_NAME=$3
TEST_BENCH_FUNC=$4

CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"

# Create a temporary file which has the 'static_assert' appended to it.
# Since we always append the 'static_assert' rather than making it part of the
# original source code, reduction tools such as 'cvise' cannot circumvent it.
file=$(mktemp --suffix .c)
trap 'rm "$file"' EXIT
cat "$SRC_DIR/$KERNEL_NAME.c" >> $file
echo "static_assert(($TEST_BENCH_FUNC(), true));"  >> $file
"$CLANGXX_BIN" $file -std=c++20 -DHLS_FUZZER_VERIFY \
  -I "$DYNAMATIC_DIR/include" \
  -Wno-deprecated -o /dev/null

exit_on_fail "Failed to verify test bench to be free of UB." "Verified test bench to be free of UB."

