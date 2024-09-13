#include "external_integration.h"
#include "dynamatic/Integration.h"


int external_integration(in_int_t a) {
  int c = a * 57;
  int d = a * a;
  int e = a - 13;
  port_A(c);
  port_B(d);
  return e;
  
}

int main(void) {
  int a = 10;
  CALL_KERNEL(external_integration, a);
  return 0;
}
