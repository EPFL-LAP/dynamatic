\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x1c0e2f2215eb9aa9
Maximize
 [ 2 sArc_0_1 * varMaxExecs + 2 sArc_1_2 * varMaxExecs
   + 2 sArc_1_3 * varMaxExecs + 2 sArc_2_1 * varMaxExecs
   + 2 sArc_2_3 * varMaxExecs ] / 2 
Subject To
 arch_0_1: 99 sArc_0_1 + varMaxExecs <= 100
 arch_1_2: varMaxExecs <= 100
 arch_1_3: 99 sArc_1_3 + varMaxExecs <= 100
 arch_2_1: varMaxExecs <= 100
 arch_2_3: 100 sArc_2_3 + varMaxExecs <= 100
 oneBackedge: sArc_2_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_2_1 = 0
 out1: - sBB_1 + sArc_1_2 + sArc_1_3 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_1 + sArc_2_3 = 0
 in3: - sBB_3 + sArc_1_3 + sArc_2_3 = 0
 out3: - sBB_3 = 0
Bounds
 varMaxExecs <= 100
Binaries
 sBB_0 sBB_1 sBB_2 sBB_3 sArc_0_1 sArc_1_2 sArc_1_3 sArc_2_1 sArc_2_3
Generals
 varMaxExecs
End
