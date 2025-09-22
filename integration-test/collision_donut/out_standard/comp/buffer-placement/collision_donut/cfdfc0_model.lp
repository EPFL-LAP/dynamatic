\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x537e55cf80cd5e2b
Maximize
 [ 2 sArc_1_2 * varMaxExecs + 2 sArc_3_1 * varMaxExecs
   + 2 sArc_0_1 * varMaxExecs + 2 sArc_4_5 * varMaxExecs
   + 2 sArc_3_4 * varMaxExecs + 2 sArc_2_3 * varMaxExecs
   + 2 sArc_1_4 * varMaxExecs + 2 sArc_2_5 * varMaxExecs ] / 2 
Subject To
 arch_1_2: varMaxExecs <= 1000
 arch_3_1: sArc_3_1 + varMaxExecs <= 1000
 arch_0_1: 999 sArc_0_1 + varMaxExecs <= 1000
 arch_4_5: 999 sArc_4_5 + varMaxExecs <= 1000
 arch_3_4: 999 sArc_3_4 + varMaxExecs <= 1000
 arch_2_3: varMaxExecs <= 1000
 arch_1_4: 1000 sArc_1_4 + varMaxExecs <= 1000
 arch_2_5: 1000 sArc_2_5 + varMaxExecs <= 1000
 oneBackedge: sArc_3_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_3_1 + sArc_0_1 = 0
 out1: - sBB_1 + sArc_1_2 + sArc_1_4 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 + sArc_2_3 + sArc_2_5 = 0
 in3: - sBB_3 + sArc_2_3 = 0
 out3: - sBB_3 + sArc_3_1 + sArc_3_4 = 0
 in4: - sBB_4 + sArc_3_4 + sArc_1_4 = 0
 out4: - sBB_4 + sArc_4_5 = 0
 in5: - sBB_5 + sArc_4_5 + sArc_2_5 = 0
 out5: - sBB_5 = 0
Bounds
 varMaxExecs <= 1000
Binaries
 sBB_1 sBB_2 sBB_3 sBB_0 sBB_4 sBB_5 sArc_1_2 sArc_3_1 sArc_0_1 sArc_4_5
 sArc_3_4 sArc_2_3 sArc_1_4 sArc_2_5
Generals
 varMaxExecs
End
