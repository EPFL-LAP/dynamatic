\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x540760995ed3b1d4
Maximize
 [ 2 sArc_3_4 * varMaxExecs + 2 sArc_2_2 * varMaxExecs
   + 2 sArc_1_2 * varMaxExecs + 2 sArc_3_1 * varMaxExecs
   + 2 sArc_2_3 * varMaxExecs + 2 sArc_0_1 * varMaxExecs ] / 2 
Subject To
 arch_3_4: 99 sArc_3_4 + varMaxExecs <= 100
 arch_2_2: 100 sArc_2_2 + varMaxExecs <= 100
 arch_1_2: varMaxExecs <= 100
 arch_3_1: sArc_3_1 + varMaxExecs <= 100
 arch_2_3: varMaxExecs <= 100
 arch_0_1: 99 sArc_0_1 + varMaxExecs <= 100
 oneBackedge: sArc_2_2 + sArc_3_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_3_1 + sArc_0_1 = 0
 out1: - sBB_1 + sArc_1_2 = 0
 in2: - sBB_2 + sArc_2_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_2 + sArc_2_3 = 0
 in3: - sBB_3 + sArc_2_3 = 0
 out3: - sBB_3 + sArc_3_4 + sArc_3_1 = 0
 in4: - sBB_4 + sArc_3_4 = 0
 out4: - sBB_4 = 0
Bounds
 varMaxExecs <= 100
Binaries
 sBB_3 sBB_4 sBB_2 sBB_1 sBB_0 sArc_3_4 sArc_2_2 sArc_1_2 sArc_3_1 sArc_2_3
 sArc_0_1
Generals
 varMaxExecs
End
