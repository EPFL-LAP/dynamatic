// RUN: %dyn-clang-pragmas %s | FileCheck %s

// Check for the pragma expansion for the partition
// CHECK: c"arr\00"
// CHECK: c"block\00"
// CHECK: call void @__dyn_array_partition(ptr noundef {{.+}}, i32 noundef 1, i32 noundef 4, ptr noundef {{.+}})
// CHECK: declare void @__dyn_array_partition(ptr noundef, i32 noundef, i32 noundef, ptr noundef)


void kernel(void) {
#pragma DYN array_partition array=arr dimension=1 style=block factor=4
  int arr[100];
}
