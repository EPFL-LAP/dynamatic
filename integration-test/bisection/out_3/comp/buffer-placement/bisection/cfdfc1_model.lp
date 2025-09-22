\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xaf1bd11957268f1d
Maximize
 [ 2 sArc_2_3 * varMaxExecs + 2 sArc_0_1 * varMaxExecs
   + 2 sArc_2_2 * varMaxExecs + 2 sArc_1_2 * varMaxExecs
   + 2 sArc_3_1 * varMaxExecs + 2 sArc_2_4 * varMaxExecs
   + 2 sArc_4_5 * varMaxExecs + 2 sArc_2_4 * varMaxExecs
   + 2 sArc_2_5 * varMaxExecs + 2 sArc_3_4 * varMaxExecs ] / 2 
Subject To
 arch_2_3: sArc_2_3 + varMaxExecs <= 15
 arch_0_1: 14 sArc_0_1 + varMaxExecs <= 15
 arch_2_2: 15 sArc_2_2 + varMaxExecs <= 15
 arch_1_2: varMaxExecs <= 15
 arch_3_1: sArc_3_1 + varMaxExecs <= 15
 arch_2_4: 14 sArc_2_4 + varMaxExecs <= 15
 arch_4_5: 14 sArc_4_5 + varMaxExecs <= 15
 arch_2_4: 15 sArc_2_4 + varMaxExecs <= 15
 arch_2_5: 15 sArc_2_5 + varMaxExecs <= 15
 arch_3_4: 15 sArc_3_4 + varMaxExecs <= 15
 oneBackedge: sArc_2_2 + sArc_3_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_3_1 = 0
 out1: - sBB_1 + sArc_1_2 = 0
 in2: - sBB_2 + sArc_2_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_3 + sArc_2_2 + sArc_2_4 + sArc_2_4 + sArc_2_5 = 0
 in3: - sBB_3 + sArc_2_3 = 0
 out3: - sBB_3 + sArc_3_1 + sArc_3_4 = 0
 in4: - sBB_4 + sArc_2_4 + sArc_2_4 + sArc_3_4 = 0
 out4: - sBB_4 + sArc_4_5 = 0
 in5: - sBB_5 + sArc_4_5 + sArc_2_5 = 0
 out5: - sBB_5 = 0
Bounds
 varMaxExecs <= 15
Binaries
 sBB_2 sBB_3 sBB_0 sBB_1 sBB_4 sBB_5 sArc_2_3 sArc_0_1 sArc_2_2 sArc_1_2
 sArc_3_1 sArc_2_4 sArc_4_5 sArc_2_4 sArc_2_5 sArc_3_4
Generals
 varMaxExecs
End
