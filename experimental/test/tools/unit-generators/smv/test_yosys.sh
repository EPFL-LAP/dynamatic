#!/bin/bash

GENPATH=${GENPATH:-"../../../../tools/unit-generators/smv/"}

test_generator () {
  mkdir -p build
  rm build/*

  OUT="build/$2.smv"
  LOG="build/.log"
  PROPERTIES="build/property.rpt"

  GENERATOR_CALL="python3 ${GENPATH}smv-unit-generator.py -o $OUT"
  TEST_CALL="python3 ./smv-test-generator.py -o build/test_bench.smv"

  $GENERATOR_CALL $@

  REF_FILE="~/dynamatic/data/verilog/handshake/*.v"
  DEP_FILE="~/dynamatic/data/verilog/support/*.v"
  # yosys -p "read_verilog ${REF_FILE}; prep -top $2; aigmap; flatten; write_smv golden_model.smv"
  yosys -p "read_verilog ${REF_FILE} ${DEP_FILE}; prep -top $2; flatten; write_smv build/golden_model.smv" >> $LOG
  sed -i 's/\bIVAR\b/VAR/g' build/golden_model.smv

  $TEST_CALL $@

  nuXmv -source prove.cmd >> $LOG

  if [[ ! -f $PROPERTIES ]]; then
    echo "Error: Expected output file property.rpt not found!"
    exit 1
  fi

  if grep -q "False" $PROPERTIES; then
    echo "Test $2 FAILED!"
    cat $PROPERTIES
    exit 1
  fi
  echo "Test $2 passed!"
}

echo "Testing br..."
test_generator -n br_dataless -t br -p in_port_types='{"ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}' port_types='{"outs":"!handshake.control<>"}'
test_generator -n br -t br -p in_port_types='{"ins":"!handshake.channel<ui32>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}' port_types='{"outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting buffer..."
test_generator -n oehb_dataless -t buffer -p slots=1 in_port_types='{"ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n oehb -t buffer -p slots=1 in_port_types='{"ins":"!handshake.channel<ui32>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n ofifo_dataless -t buffer -p slots=5 in_port_types='{"ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
# test_generator -n ofifo -t buffer -p slots=2 in_port_types='{"ins":"!handshake.channel<ui32>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n tehb_dataless -t buffer -p slots=1 in_port_types='{"ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
test_generator -n tehb -t buffer -p slots=1 in_port_types='{"ins":"!handshake.channel<ui32>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
test_generator -n tfifo_dataless -t buffer -p slots=2 in_port_types='{"ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
# test_generator -n ofifo -t buffer -p slots=2 in_port_types='{"ins":"!handshake.channel<ui32>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'

echo -e "\nTesting cond_br..."
test_generator -n cond_br_dataless -t cond_br -p port_types='{"data":"!handshake.control<>"}' in_port_types='{"data":"!handshake.control<>","condition":"!handshake.channel<ui1>"}' out_port_types='{"trueOut":"!handshake.control<>","falseOut":"!handshake.control<>"}'
test_generator -n cond_br -t cond_br -p port_types='{"data":"!handshake.channel<ui32>"}' in_port_types='{"data":"!handshake.channel<ui32>","condition":"!handshake.channel<ui1>"}' out_port_types='{"trueOut":"!handshake.channel<ui32>","falseOut":"!handshake.channel<ui32>"}'

# echo -e "\nTesting constant..."
test_generator -n constant -t constant -p value=42 port_types='{"outs":"!handshake.channel<ui32>"}' in_port_types='{"ctrl":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.channel<ui32>"}'

# echo -e "\nTesting control_merge..."
test_generator -n control_merge_dataless -t control_merge -p size=2  port_types='{"index":"!handshake.channel<ui1>","outs":"!handshake.control<>"}' in_port_types='{"index":"!handshake.channel<ui1>","ins":"!handshake.control<>"}' out_port_types='{"outs":"!handshake.control<>"}'
# test_generator -t control_merge -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.channel<i16>"}'

# echo -e "\nTesting fork..."
# test_generator -t fork -p size=4 port_types='{"ins":"!handshake.control<>"}'
# test_generator -t fork -p size=2 port_types='{"ins":"!handshake.channel<i16>"}'

# echo -e "\nTesting join..."
# test_generator -t join -p size=3

# echo -e "\nTesting lazy_fork..."
# test_generator -t lazy_fork -p size=4 port_types='{"ins":"!handshake.control<>"}'
# test_generator -t lazy_fork -p size=2 port_types='{"ins":"!handshake.channel<i16>"}'

# echo -e "\nTesting load..."
# test_generator -t load -p port_types='{"dataOut":"!handshake.channel<i16>","addrIn":"!handshake.channel<i16>"}'

# echo -e "\nTesting merge..."
# test_generator -t merge -p size=4 port_types='{"outs":"!handshake.control<>"}'
# test_generator -t merge -p size=2 port_types='{"outs":"!handshake.channel<i16>"}'	

# echo -e "\nTesting mux..."
# test_generator -t mux -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.control<>"}'
# test_generator -t mux -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.channel<i16>"}'

# echo -e "\nTesting select..."
# test_generator -t select -p port_types='{"result":"!handshake.control<>"}'

# echo -e "\nTesting sink..."
# test_generator -t sink -p port_types='{"ins":"!handshake.control<>"}'
# test_generator -t sink -p port_types='{"ins":"!handshake.channel<i16>"}'

# echo -e "\nTesting source..."
# test_generator -t source

# echo -e "\nTesting store..."
# test_generator -t store -p port_types='{"dataIn":"!handshake.channel<i16>","addrIn":"!handshake.channel<i16>"}'

# echo -e "\nTesting absf..."
# test_generator -t absf --abstract-data -p port_types='{"outs":"!handshake.channel<f32>"}' latency=0

# echo -e "\nTesting addf..."
# test_generator -t addf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=9
# test_generator -t addf --abstract-data -p port_types='{"result":"!handshake.channel<f64>"}' latency=12

# echo -e "\nTesting addi..."
# test_generator -t addi -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t addi --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting andi..."
# test_generator -t andi -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t andi --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting cmpi..."
# test_generator -t cmpi -p port_types='{"lhs":"!handshake.channel<i32>"}' latency=0 predicate='"eq"' 
# test_generator -t cmpi -p port_types='{"lhs":"!handshake.channel<i32>"}' latency=0 predicate='"slt"' 
# #test_generator -t cmpi -p port_types='{"lhs":"!handshake.channel<i32>"}' latency=0 predicate='"ult"' 
# #test_generator -t cmpi -p port_types='{"lhs":"!handshake.channel<ui32>"}' latency=0 predicate='"sge"' 
# test_generator -t cmpi -p port_types='{"lhs":"!handshake.channel<ui32>"}' latency=0 predicate='"uge"' 
# test_generator -t cmpi --abstract-data -p port_types='{"lhs":"!handshake.channel<ui32>"}' latency=0 predicate='"uge"' 

# echo -e "\nTesting cmpf..."
# test_generator -t cmpf --abstract-data -p port_types='{"lhs":"!handshake.channel<f32>"}' latency=0 predicate='"oeq"' 

# echo -e "\nTesting divf..."
# test_generator -t divf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=29
# test_generator -t divf --abstract-data -p port_types='{"result":"!handshake.channel<f64>"}' latency=36

# echo -e "\nTesting divsi..."
# test_generator -t divsi -p port_types='{"result":"!handshake.channel<i32>"}' latency=35
# test_generator -t divsi --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=35

# echo -e "\nTesting divui..."
# test_generator -t divui -p port_types='{"result":"!handshake.channel<i32>"}' latency=35
# test_generator -t divui --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=35

# echo -e "\nTesting extf..."
# test_generator -t extf --abstract-data -p port_types='{"ins":"!handshake.channel<f32>","outs":"!handshake.channel<f64>"}' latency=0

# echo -e "\nTesting extsi..."
# test_generator -t extsi -p port_types='{"ins":"!handshake.channel<i32>","outs":"!handshake.channel<i64>"}' latency=0
# test_generator -t extsi --abstract-data -p port_types='{"ins":"!handshake.channel<i32>","outs":"!handshake.channel<i64>"}' latency=0

# echo -e "\nTesting extui..."
# test_generator -t extui -p port_types='{"ins":"!handshake.channel<ui32>","outs":"!handshake.channel<ui64>"}' latency=0
# test_generator -t extui --abstract-data -p port_types='{"ins":"!handshake.channel<ui32>","outs":"!handshake.channel<ui64>"}' latency=0

# echo -e "\nTesting fptosi..."
# test_generator -t fptosi --abstract-data -p port_types='{"ins":"!handshake.channel<f32>","outs":"!handshake.channel<i32>"}' latency=5

# echo -e "\nTesting maximumf..."
# test_generator -t maximumf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=0

# echo -e "\nTesting minimumf..."
# test_generator -t minimumf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=0

# echo -e "\nTesting mulf..."
# test_generator -t mulf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=4

# echo -e "\nTesting muli..."
# test_generator -t muli -p port_types='{"result":"!handshake.channel<i32>"}' latency=4
# test_generator -t muli --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=4

# echo -e "\nTesting negf..."
# test_generator -t negf --abstract-data -p port_types='{"outs":"!handshake.channel<f32>"}' latency=0

# echo -e "\nTesting not..."
# test_generator -t not -p port_types='{"outs":"!handshake.channel<i32>"}' latency=0
# test_generator -t not --abstract-data -p port_types='{"outs":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting ori..."
# test_generator -t ori -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t ori --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting shli..."
# test_generator -t shli -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t shli --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting shrsi..."
# test_generator -t shrsi -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t shrsi -p port_types='{"result":"!handshake.channel<ui32>"}' latency=0
# test_generator -t shrsi --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t shrsi --abstract-data -p port_types='{"result":"!handshake.channel<ui32>"}' latency=0

# echo -e "\nTesting shrui..."
# test_generator -t shrui -p port_types='{"result":"!handshake.channel<ui32>"}' latency=0
# test_generator -t shrui -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t shrui --abstract-data -p port_types='{"result":"!handshake.channel<ui32>"}' latency=0
# test_generator -t shrui --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting sitofp..."
# test_generator -t sitofp --abstract-data -p port_types='{"ins":"!handshake.channel<i32>","outs":"!handshake.channel<f32>"}' latency=5

# echo -e "\nTesting subf..."
# test_generator -t subf --abstract-data -p port_types='{"result":"!handshake.channel<f32>"}' latency=9
# test_generator -t subf --abstract-data -p port_types='{"result":"!handshake.channel<f64>"}' latency=12

# echo -e "\nTesting subi..."
# test_generator -t subi -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t subi --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting truncf..."
# test_generator -t truncf --abstract-data -p port_types='{"ins":"!handshake.channel<f64>","outs":"!handshake.channel<f32>"}' latency=0

# echo -e "\nTesting trunci..."
# test_generator -t trunci -p port_types='{"ins":"!handshake.channel<i64>","outs":"!handshake.channel<i32>"}' latency=0
# test_generator -t trunci --abstract-data -p port_types='{"ins":"!handshake.channel<i64>","outs":"!handshake.channel<i32>"}' latency=0

# echo -e "\nTesting xori..."
# test_generator -t xori -p port_types='{"result":"!handshake.channel<i32>"}' latency=0
# test_generator -t xori --abstract-data -p port_types='{"result":"!handshake.channel<i32>"}' latency=0


# rm work/*
