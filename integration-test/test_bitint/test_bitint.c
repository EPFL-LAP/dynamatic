#include "dynamatic/Integration.h"

#define AccumType unsigned _BitInt(32)
#define DataType unsigned _BitInt(16)
#define WeightType unsigned _BitInt(5)
#define MultType unsigned _BitInt(21)

AccumType test_bitint(DataType arr[10], WeightType weight[10]) {
  AccumType acc = 0;
  for (unsigned i = 0; i < 10; i++) {
    // NOTE:
    // In LLVM IR, this expression becomes:
    // %conv = zext i16 %0 to i21
    // %conv3 = zext i5 %1 to i21
    // %mul = mul nuw nsw i21 %conv, %conv3
    // The mul instruction must have the same type for both args and results
    //
    // In Handshake IR, this expression becomes:
    // %21 = extui %dataResult {...} : <i16> to <i21>
    // %22 = extui %dataResult_3 {...} : <i5> to <i21>
    // %23 = muli %21, %22 {...} : <i21>
    // TODO:
    // "handshake.muli" should take channel<i5> and channel<i16> to allow Vivado
    // to pack the multiplication into 1 DSP unit.
    acc += (MultType)arr[i] * weight[i];
  }
  return acc;
}

int main() {
  DataType arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  WeightType weight[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  CALL_KERNEL(test_bitint, arr, weight);
  return 0;
}
