// RUN: %dyn-clang-pragmas %s | opt -S -load-pass-plugin "%shlibdir/ArrayPartition.so" -passes="array-partition" | FileCheck %s

// Check that the pragma is correctly erased after doing an array partition pass

// CHECK-NOT: call void @__dyn_array_partition
// CHECK-NOT: declare void @__dyn_array_partition

void kernel(void) {
#pragma DYN array_partition array=arr dimension=1 style=block factor=4
  int arr[100];
}
