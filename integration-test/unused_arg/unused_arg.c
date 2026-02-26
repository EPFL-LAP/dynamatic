#include <dynamatic/Integration.h>

int unused_arg(int a[1], int used[1], int b) {
  return used[0];
}

int main() {
  int a[1];
  int b = 5;
  int used[1] = {0};
  CALL_KERNEL(unused_arg, a, used, b);
  return 0;
}
