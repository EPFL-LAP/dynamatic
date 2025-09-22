\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xb5c5afd121793bd4
Maximize
 [ 2 sArc_0_1 * varMaxExecs + 2 sArc_2_2 * varMaxExecs
   + 2 sArc_2_3 * varMaxExecs + 2 sArc_1_2 * varMaxExecs
   + 2 sArc_3_4 * varMaxExecs + 2 sArc_3_1 * varMaxExecs ] / 2 
Subject To
 arch_0_1: 547 sArc_0_1 + varMaxExecs <= 548
 arch_2_2: varMaxExecs <= 548
 arch_2_3: 448 sArc_2_3 + varMaxExecs <= 548
 arch_1_2: 448 sArc_1_2 + varMaxExecs <= 548
 arch_3_4: 547 sArc_3_4 + varMaxExecs <= 548
 arch_3_1: 449 sArc_3_1 + varMaxExecs <= 548
 oneBackedge: sArc_2_2 + sArc_3_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_3_1 = 0
 out1: - sBB_1 + sArc_1_2 = 0
 in2: - sBB_2 + sArc_2_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_2 + sArc_2_3 = 0
 in3: - sBB_3 + sArc_2_3 = 0
 out3: - sBB_3 + sArc_3_4 + sArc_3_1 = 0
 in4: - sBB_4 + sArc_3_4 = 0
 out4: - sBB_4 = 0
Bounds
 varMaxExecs <= 548
Binaries
 sBB_0 sBB_1 sBB_2 sBB_3 sBB_4 sArc_0_1 sArc_2_2 sArc_2_3 sArc_1_2 sArc_3_4
 sArc_3_1
Generals
 varMaxExecs
End
