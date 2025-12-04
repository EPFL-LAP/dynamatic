#include "dynamatic/Integration.h"
#include "stdlib.h"

#define N 50

void single_loop_unrolled_160(
    int a[N], int b[N], int c0[N], int c1[N], int c2[N], int c3[N], int c4[N],
    int c5[N], int c6[N], int c7[N], int c8[N], int c9[N], int c10[N],
    int c11[N], int c12[N], int c13[N], int c14[N], int c15[N], int c16[N],
    int c17[N], int c18[N], int c19[N], int c20[N], int c21[N], int c22[N],
    int c23[N], int c24[N], int c25[N], int c26[N], int c27[N], int c28[N],
    int c29[N], int c30[N], int c31[N], int c32[N], int c33[N], int c34[N],
    int c35[N], int c36[N], int c37[N], int c38[N], int c39[N], int c40[N],
    int c41[N], int c42[N], int c43[N], int c44[N], int c45[N], int c46[N],
    int c47[N], int c48[N], int c49[N], int c50[N], int c51[N], int c52[N],
    int c53[N], int c54[N], int c55[N], int c56[N], int c57[N], int c58[N],
    int c59[N], int c60[N], int c61[N], int c62[N], int c63[N], int c64[N],
    int c65[N], int c66[N], int c67[N], int c68[N], int c69[N], int c70[N],
    int c71[N], int c72[N], int c73[N], int c74[N], int c75[N], int c76[N],
    int c77[N], int c78[N], int c79[N], int c80[N], int c81[N], int c82[N],
    int c83[N], int c84[N], int c85[N], int c86[N], int c87[N], int c88[N],
    int c89[N], int c90[N], int c91[N], int c92[N], int c93[N], int c94[N],
    int c95[N], int c96[N], int c97[N], int c98[N], int c99[N], int c100[N],
    int c101[N], int c102[N], int c103[N], int c104[N], int c105[N],
    int c106[N], int c107[N], int c108[N], int c109[N], int c110[N],
    int c111[N], int c112[N], int c113[N], int c114[N], int c115[N],
    int c116[N], int c117[N], int c118[N], int c119[N], int c120[N],
    int c121[N], int c122[N], int c123[N], int c124[N], int c125[N],
    int c126[N], int c127[N], int c128[N], int c129[N], int c130[N],
    int c131[N], int c132[N], int c133[N], int c134[N], int c135[N],
    int c136[N], int c137[N], int c138[N], int c139[N], int c140[N],
    int c141[N], int c142[N], int c143[N], int c144[N], int c145[N],
    int c146[N], int c147[N], int c148[N], int c149[N], int c150[N],
    int c151[N], int c152[N], int c153[N], int c154[N], int c155[N],
    int c156[N], int c157[N], int c158[N], int c159[N]) {
  int i = 0;
  int bound = 50;
  int sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c0[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c1[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c2[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c3[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c4[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c5[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c6[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c7[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c8[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c9[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c10[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c11[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c12[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c13[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c14[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c15[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c16[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c17[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c18[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c19[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c20[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c21[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c22[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c23[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c24[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c25[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c26[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c27[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c28[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c29[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c30[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c31[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c32[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c33[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c34[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c35[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c36[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c37[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c38[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c39[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c40[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c41[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c42[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c43[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c44[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c45[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c46[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c47[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c48[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c49[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c50[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c51[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c52[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c53[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c54[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c55[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c56[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c57[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c58[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c59[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c60[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c61[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c62[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c63[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c64[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c65[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c66[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c67[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c68[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c69[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c70[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c71[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c72[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c73[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c74[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c75[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c76[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c77[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c78[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c79[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c80[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c81[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c82[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c83[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c84[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c85[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c86[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c87[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c88[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c89[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c90[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c91[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c92[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c93[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c94[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c95[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c96[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c97[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c98[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c99[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c100[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c101[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c102[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c103[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c104[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c105[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c106[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c107[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c108[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c109[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c110[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c111[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c112[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c113[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c114[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c115[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c116[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c117[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c118[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c119[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c120[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c121[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c122[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c123[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c124[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c125[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c126[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c127[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c128[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c129[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c130[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c131[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c132[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c133[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c134[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c135[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c136[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c137[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c138[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c139[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c140[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c141[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c142[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c143[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c144[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c145[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c146[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c147[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c148[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c149[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c150[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c151[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c152[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c153[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c154[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c155[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c156[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c157[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c158[i] = sum;
    i++;
  }

  i = 0;
  sum = 0;
  while (sum < bound) {
    sum = a[i] * b[i];
    c159[i] = sum;
    i++;
  }
}

int main(void) {
  int a[N];
  int b[N];
  int c0[N];
  int c1[N];
  int c2[N];
  int c3[N];
  int c4[N];
  int c5[N];
  int c6[N];
  int c7[N];
  int c8[N];
  int c9[N];
  int c10[N];
  int c11[N];
  int c12[N];
  int c13[N];
  int c14[N];
  int c15[N];
  int c16[N];
  int c17[N];
  int c18[N];
  int c19[N];
  int c20[N];
  int c21[N];
  int c22[N];
  int c23[N];
  int c24[N];
  int c25[N];
  int c26[N];
  int c27[N];
  int c28[N];
  int c29[N];
  int c30[N];
  int c31[N];
  int c32[N];
  int c33[N];
  int c34[N];
  int c35[N];
  int c36[N];
  int c37[N];
  int c38[N];
  int c39[N];
  int c40[N];
  int c41[N];
  int c42[N];
  int c43[N];
  int c44[N];
  int c45[N];
  int c46[N];
  int c47[N];
  int c48[N];
  int c49[N];
  int c50[N];
  int c51[N];
  int c52[N];
  int c53[N];
  int c54[N];
  int c55[N];
  int c56[N];
  int c57[N];
  int c58[N];
  int c59[N];
  int c60[N];
  int c61[N];
  int c62[N];
  int c63[N];
  int c64[N];
  int c65[N];
  int c66[N];
  int c67[N];
  int c68[N];
  int c69[N];
  int c70[N];
  int c71[N];
  int c72[N];
  int c73[N];
  int c74[N];
  int c75[N];
  int c76[N];
  int c77[N];
  int c78[N];
  int c79[N];
  int c80[N];
  int c81[N];
  int c82[N];
  int c83[N];
  int c84[N];
  int c85[N];
  int c86[N];
  int c87[N];
  int c88[N];
  int c89[N];
  int c90[N];
  int c91[N];
  int c92[N];
  int c93[N];
  int c94[N];
  int c95[N];
  int c96[N];
  int c97[N];
  int c98[N];
  int c99[N];
  int c100[N];
  int c101[N];
  int c102[N];
  int c103[N];
  int c104[N];
  int c105[N];
  int c106[N];
  int c107[N];
  int c108[N];
  int c109[N];
  int c110[N];
  int c111[N];
  int c112[N];
  int c113[N];
  int c114[N];
  int c115[N];
  int c116[N];
  int c117[N];
  int c118[N];
  int c119[N];
  int c120[N];
  int c121[N];
  int c122[N];
  int c123[N];
  int c124[N];
  int c125[N];
  int c126[N];
  int c127[N];
  int c128[N];
  int c129[N];
  int c130[N];
  int c131[N];
  int c132[N];
  int c133[N];
  int c134[N];
  int c135[N];
  int c136[N];
  int c137[N];
  int c138[N];
  int c139[N];
  int c140[N];
  int c141[N];
  int c142[N];
  int c143[N];
  int c144[N];
  int c145[N];
  int c146[N];
  int c147[N];
  int c148[N];
  int c149[N];
  int c150[N];
  int c151[N];
  int c152[N];
  int c153[N];
  int c154[N];
  int c155[N];
  int c156[N];
  int c157[N];
  int c158[N];
  int c159[N];

  srand(13);
  for (int j = 0; j < N; ++j) {
    a[j] = 2;
    b[j] = j;
    c0[j] = 0;
    c1[j] = 0;
    c2[j] = 0;
    c3[j] = 0;
    c4[j] = 0;
    c5[j] = 0;
    c6[j] = 0;
    c7[j] = 0;
    c8[j] = 0;
    c9[j] = 0;
    c10[j] = 0;
    c11[j] = 0;
    c12[j] = 0;
    c13[j] = 0;
    c14[j] = 0;
    c15[j] = 0;
    c16[j] = 0;
    c17[j] = 0;
    c18[j] = 0;
    c19[j] = 0;
    c20[j] = 0;
    c21[j] = 0;
    c22[j] = 0;
    c23[j] = 0;
    c24[j] = 0;
    c25[j] = 0;
    c26[j] = 0;
    c27[j] = 0;
    c28[j] = 0;
    c29[j] = 0;
    c30[j] = 0;
    c31[j] = 0;
    c32[j] = 0;
    c33[j] = 0;
    c34[j] = 0;
    c35[j] = 0;
    c36[j] = 0;
    c37[j] = 0;
    c38[j] = 0;
    c39[j] = 0;
    c40[j] = 0;
    c41[j] = 0;
    c42[j] = 0;
    c43[j] = 0;
    c44[j] = 0;
    c45[j] = 0;
    c46[j] = 0;
    c47[j] = 0;
    c48[j] = 0;
    c49[j] = 0;
    c50[j] = 0;
    c51[j] = 0;
    c52[j] = 0;
    c53[j] = 0;
    c54[j] = 0;
    c55[j] = 0;
    c56[j] = 0;
    c57[j] = 0;
    c58[j] = 0;
    c59[j] = 0;
    c60[j] = 0;
    c61[j] = 0;
    c62[j] = 0;
    c63[j] = 0;
    c64[j] = 0;
    c65[j] = 0;
    c66[j] = 0;
    c67[j] = 0;
    c68[j] = 0;
    c69[j] = 0;
    c70[j] = 0;
    c71[j] = 0;
    c72[j] = 0;
    c73[j] = 0;
    c74[j] = 0;
    c75[j] = 0;
    c76[j] = 0;
    c77[j] = 0;
    c78[j] = 0;
    c79[j] = 0;
    c80[j] = 0;
    c81[j] = 0;
    c82[j] = 0;
    c83[j] = 0;
    c84[j] = 0;
    c85[j] = 0;
    c86[j] = 0;
    c87[j] = 0;
    c88[j] = 0;
    c89[j] = 0;
    c90[j] = 0;
    c91[j] = 0;
    c92[j] = 0;
    c93[j] = 0;
    c94[j] = 0;
    c95[j] = 0;
    c96[j] = 0;
    c97[j] = 0;
    c98[j] = 0;
    c99[j] = 0;
    c100[j] = 0;
    c101[j] = 0;
    c102[j] = 0;
    c103[j] = 0;
    c104[j] = 0;
    c105[j] = 0;
    c106[j] = 0;
    c107[j] = 0;
    c108[j] = 0;
    c109[j] = 0;
    c110[j] = 0;
    c111[j] = 0;
    c112[j] = 0;
    c113[j] = 0;
    c114[j] = 0;
    c115[j] = 0;
    c116[j] = 0;
    c117[j] = 0;
    c118[j] = 0;
    c119[j] = 0;
    c120[j] = 0;
    c121[j] = 0;
    c122[j] = 0;
    c123[j] = 0;
    c124[j] = 0;
    c125[j] = 0;
    c126[j] = 0;
    c127[j] = 0;
    c128[j] = 0;
    c129[j] = 0;
    c130[j] = 0;
    c131[j] = 0;
    c132[j] = 0;
    c133[j] = 0;
    c134[j] = 0;
    c135[j] = 0;
    c136[j] = 0;
    c137[j] = 0;
    c138[j] = 0;
    c139[j] = 0;
    c140[j] = 0;
    c141[j] = 0;
    c142[j] = 0;
    c143[j] = 0;
    c144[j] = 0;
    c145[j] = 0;
    c146[j] = 0;
    c147[j] = 0;
    c148[j] = 0;
    c149[j] = 0;
    c150[j] = 0;
    c151[j] = 0;
    c152[j] = 0;
    c153[j] = 0;
    c154[j] = 0;
    c155[j] = 0;
    c156[j] = 0;
    c157[j] = 0;
    c158[j] = 0;
    c159[j] = 0;
  }

  CALL_KERNEL(
      single_loop_unrolled_160, a, b, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
      c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24,
      c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
      c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54,
      c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69,
      c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84,
      c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99,
      c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111,
      c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123,
      c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135,
      c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147,
      c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159);
  return 0;
}
