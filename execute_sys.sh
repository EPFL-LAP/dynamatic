~/CI_mapbuf/dynamatic/bin/dynamatic-opt --systolic-unit-generation --allow-unregistered-dialect test.mlir  > systolic_array.mlir
~/CI_mapbuf/dynamatic/bin/dynamatic-opt --systolic-unit-reuse-optimization --allow-unregistered-dialect systolic_array.mlir > systolic_array_opt.mlir 
./polygeist/llvm-project/build/bin/mlir-opt --lower-affine --allow-unregistered-dialect systolic_array_opt.mlir > scf_systolic_array_opt.mlir
./polygeist/llvm-project/build/bin/mlir-opt --canonicalize --allow-unregistered-dialect scf_systolic_array_opt.mlir > scf_clean_systolic_array_opt.mlir 
./polygeist/llvm-project/build/bin/mlir-opt --convert-scf-to-cf --allow-unregistered-dialect scf_clean_systolic_array_opt.mlir > cf_systolic_array_opt.mlir

