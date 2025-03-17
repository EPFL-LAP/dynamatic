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
  yosys -p "read_verilog ${REF_FILE} ${DEP_FILE}; prep -top $2; memory_map; memory_unpack; flatten; write_smv build/golden_model.smv" >> $LOG
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
  else
    echo "Test $2 passed!"
  fi
}

echo "Testing br..."
test_generator -n br_dataless -t br -p ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}'
test_generator -n br -t br -p ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting buffer..."
test_generator -n oehb_dataless -t buffer -p slots=1 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n oehb -t buffer -p slots=1 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n ofifo_dataless -t buffer -p slots=5 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n ofifo -t buffer -p slots=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:1,D:0,V:0}>"'
test_generator -n tehb_dataless -t buffer -p slots=1 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
test_generator -n tehb -t buffer -p slots=1 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
test_generator -n tfifo_dataless -t buffer -p slots=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'
test_generator -n tfifo -t buffer -p slots=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}' timing='"#handshake.timing<{R:0,D:1,V:1}>"'

echo -e "\nTesting cond_br..."
test_generator -n cond_br_dataless -t cond_br -p ports='[{"name":"data","direction":"in","bitwidth":0,"count":1},{"name":"condition","direction":"in","bitwidth":1,"count":1},{"name":"trueOut","direction":"out","bitwidth":0,"count":1},{"name":"falseOut","direction":"out","bitwidth":0,"count":1}]' port_types='{"data":"!handshake.control<>"}'
test_generator -n cond_br -t cond_br -p ports='[{"name":"data","direction":"in","bitwidth":32,"count":1},{"name":"condition","direction":"in","bitwidth":1,"count":1},{"name":"trueOut","direction":"out","bitwidth":32,"count":1},{"name":"falseOut","direction":"out","bitwidth":32,"count":1}]' port_types='{"data":"!handshake.channel<ui32>"}'

echo -e "\nTesting constant..."
test_generator -n constant -t constant -p value=42 ports='[{"name":"ctrl","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting control_merge..."
test_generator -n control_merge_dataless -t control_merge -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":2},{"name":"outs","direction":"out","bitwidth":0,"count":1},{"name":"index","direction":"out","bitwidth":1,"count":1}]' port_types='{"index":"!handshake.channel<ui1>","outs":"!handshake.control<>"}'
test_generator -n control_merge -t control_merge -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":2},{"name":"outs","direction":"out","bitwidth":32,"count":1},{"name":"index","direction":"out","bitwidth":1,"count":1}]' port_types='{"index":"!handshake.channel<ui1>","outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting fork..."
test_generator -n fork_dataless -t fork -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":2}]' port_types='{"ins":"!handshake.control<>"}'
test_generator -n fork -t fork -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":2}]' port_types='{"ins":"!handshake.channel<ui32>"}'

echo -e "\nTesting join..."
test_generator -n join -t join -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":2},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}'

echo -e "\nTesting lazy_fork..."
test_generator -n lazy_fork_dataless -t lazy_fork -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":2}]' port_types='{"ins":"!handshake.control<>"}'
test_generator -n lazy_fork -t lazy_fork -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":2}]' port_types='{"ins":"!handshake.channel<ui32>"}'

echo -e "\nTesting load..."
test_generator -n load -t load -p ports='[{"name":"addrIn","direction":"in","bitwidth":32,"count":1},{"name":"dataFromMem","direction":"in","bitwidth":32,"count":1},{"name":"addrOut","direction":"out","bitwidth":32,"count":1},{"name":"dataOut","direction":"out","bitwidth":32,"count":1}]' port_types='{"dataOut":"!handshake.channel<ui32>","addrIn":"!handshake.channel<ui32>"}'

echo -e "\nTesting merge..."
test_generator -n merge_dataless -t merge -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":2},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"outs":"!handshake.control<>"}'
test_generator -n merge -t merge -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":2},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting mux..."
test_generator -n mux_dataless -t mux -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":0,"count":2},{"name":"index","direction":"in","bitwidth":2,"count":1},{"name":"outs","direction":"out","bitwidth":0,"count":1}]' port_types='{"index":"!handshake.channel<ui2>","outs":"!handshake.control<>"}'
test_generator -n mux -t mux -p size=2 ports='[{"name":"ins","direction":"in","bitwidth":32,"count":2},{"name":"index","direction":"in","bitwidth":2,"count":1},{"name":"outs","direction":"out","bitwidth":32,"count":1}]' port_types='{"index":"!handshake.channel<ui2>","outs":"!handshake.channel<ui32>"}'

echo -e "\nTesting select..."
test_generator -n select -t select -p ports='[{"name":"condition","direction":"in","bitwidth":1,"count":1},{"name":"trueValue","direction":"in","bitwidth":32,"count":1},{"name":"falseValue","direction":"in","bitwidth":32,"count":1},{"name":"result","direction":"outs","bitwidth":32,"count":1}]' port_types='{"result":"!handshake.channel<ui32>"}'

echo -e "\nTesting sink..."
test_generator -n sink_dataless -t sink -p ports='[{"name":"ins","direction":"in","bitwidth":0,"count":1}]' port_types='{"ins":"!handshake.control<>"}'
test_generator -n sink -t sink -p ports='[{"name":"ins","direction":"in","bitwidth":32,"count":1}]' port_types='{"ins":"!handshake.channel<ui32>"}'

echo -e "\nTesting source..."
test_generator -n source -t source -p ports='[{"name":"outs","direction":"out","bitwidth":0,"count":1}]'

echo -e "\nTesting store..."
test_generator -n store -t store -p ports='[{"name":"dataIn","direction":"in","bitwidth":32,"count":1},{"name":"addrIn","direction":"in","bitwidth":32,"count":1},{"name":"dataToMem","direction":"out","bitwidth":32,"count":1},{"name":"addrOut","direction":"out","bitwidth":32,"count":1}]' port_types='{"dataIn":"!handshake.channel<ui32>","addrIn":"!handshake.channel<ui32>"}'
