\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xdbc2813ab8de1aa4
Maximize
 [ 2 sArc_1_2 * varMaxExecs + 2 sArc_0_1 * varMaxExecs
   + 2 sArc_1_1 * varMaxExecs ] / 2 
Subject To
 arch_1_2: 98 sArc_1_2 + varMaxExecs <= 99
 arch_0_1: 98 sArc_0_1 + varMaxExecs <= 99
 arch_1_1: varMaxExecs <= 99
 oneBackedge: sArc_1_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_1_1 = 0
 out1: - sBB_1 + sArc_1_2 + sArc_1_1 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 = 0
Bounds
 varMaxExecs <= 99
Binaries
 sBB_1 sBB_2 sBB_0 sArc_1_2 sArc_0_1 sArc_1_1
Generals
 varMaxExecs
End
