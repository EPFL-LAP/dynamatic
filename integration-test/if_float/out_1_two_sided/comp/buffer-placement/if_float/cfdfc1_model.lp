\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xa004706d6857980d
Maximize
 [ 2 sArc_0_1 * varMaxExecs + 2 sArc_1_2 * varMaxExecs
   + 2 sArc_1_1 * varMaxExecs ] / 2 
Subject To
 arch_0_1: varMaxExecs <= 1
 arch_1_2: varMaxExecs <= 1
 arch_1_1: sArc_1_1 + varMaxExecs <= 1
 oneBackedge: sArc_1_1 = 1
 in0: - sBB_0 = 0
 out0: - sBB_0 + sArc_0_1 = 0
 in1: - sBB_1 + sArc_0_1 + sArc_1_1 = 0
 out1: - sBB_1 + sArc_1_2 + sArc_1_1 = 0
 in2: - sBB_2 + sArc_1_2 = 0
 out2: - sBB_2 = 0
Bounds
 varMaxExecs <= 1
Binaries
 sBB_0 sBB_1 sBB_2 sArc_0_1 sArc_1_2 sArc_1_1
Generals
 varMaxExecs
End
