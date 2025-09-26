\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x788643d6beabddf2
Maximize
 [ 2 sArc_0_1 * varMaxExecs + 2 sArc_4_1 * varMaxExecs
   + 2 sArc_3_4 * varMaxExecs + 2 sArc_2_4 * varMaxExecs
   + 2 sArc_4_5 * varMaxExecs + 2 sArc_1_3 * varMaxExecs
   + 2 sArc_1_2 * varMaxExecs ] / 2 
Subject To
 arch_0_1: 10 sArc_0_1 + varMaxExecs <= 11
 arch_4_1: sArc_4_1 + varMaxExecs <= 11
 arch_3_4: varMaxExecs <= 11
 arch_2_4: 11 sArc_2_4 + varMaxExecs <= 11
 arch_4_5: 10 sArc_4_5 + varMaxExecs <= 11
 arch_1_3: varMaxExecs <= 11
 arch_1_2: 11 sArc_1_2 + varMaxExecs <= 11
 oneBackedge: sArc_4_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_4_1 = 0
 out1: - sBB_1 + sArc_1_3 + sArc_1_2 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_4 = 0
 in3: - sBB_3 + sArc_1_3 = 0
 out3: - sBB_3 + sArc_3_4 = 0
 in4: - sBB_4 + sArc_3_4 + sArc_2_4 = 0
 out4: - sBB_4 + sArc_4_1 + sArc_4_5 = 0
 in5: - sBB_5 + sArc_4_5 = 0
 out5: - sBB_5 = 0
Bounds
 varMaxExecs <= 11
Binaries
 sBB_0 sBB_1 sBB_4 sBB_3 sBB_2 sBB_5 sArc_0_1 sArc_4_1 sArc_3_4 sArc_2_4
 sArc_4_5 sArc_1_3 sArc_1_2
Generals
 varMaxExecs
End
