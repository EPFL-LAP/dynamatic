#!/bin/bash

GENPATH=${GENPATH:-"./"}
NUXMV=${NUXMV:-1}

test_generator () {
  if [[ $NUXMV -eq 0 ]]; then
    OUT="/dev/null"
  else
    OUT="module.smv"
  fi
  GENERATOR_CALL="python3 ${GENPATH}smv-unit-generator.py -n test_module -o $OUT"
  PARAMS=$1

  $GENERATOR_CALL $@ > $OUT

  if [[ $NUXMV -eq 1 ]]; then
    nuXmv -pre cpp -source <(echo "set on_failure_script_quits; read_model -i $OUT; quit") > /dev/null
    if [ $? != 0 ]; then
      echo "Syntax checking failed for smv file $OUT! Exiting.."
      exit 1
    else
      echo "Passed!"
    fi
  fi
}

echo "Testing br..."
test_generator -t br -p port_types='{"outs":"!handshake.control<>"}'
test_generator -t br -p port_types='{"outs":"!handshake.channel<i16>"}'

echo -e "\nTesting buffer..."
test_generator -t buffer -p slots=1 port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1}>"'
test_generator -t buffer -p slots=1 port_types='{"outs":"!handshake.channel<i16>"}' timing='"#handshake.timing<{R:1}>"'
test_generator -t buffer -p slots=5 port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:1}>"'
test_generator -t buffer -p slots=5 port_types='{"outs":"!handshake.channel<i16>"}' timing='"#handshake.timing<{R:1}>"'
test_generator -t buffer -p slots=1 port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0}>"'
test_generator -t buffer -p slots=1 port_types='{"outs":"!handshake.channel<i16>"}' timing='"#handshake.timing<{R:0}>"'
test_generator -t buffer -p slots=5 port_types='{"outs":"!handshake.control<>"}' timing='"#handshake.timing<{R:0}>"'
test_generator -t buffer -p slots=5 port_types='{"outs":"!handshake.channel<i16>"}' timing='"#handshake.timing<{R:0}>"'

echo -e "\nTesting cond_br..."
test_generator -t cond_br -p port_types='{"data":"!handshake.control<>"}'
test_generator -t cond_br -p port_types='{"data":"!handshake.channel<i16>"}'

echo -e "\nTesting constant..."
test_generator -t constant -p value=42 port_types='{"outs":"!handshake.control<>"}'

echo -e "\nTesting control_merge..."
test_generator -t control_merge -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.control<>"}'
test_generator -t control_merge -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.channel<i16>"}'

echo -e "\nTesting fork..."
test_generator -t fork -p size=4 port_types='{"ins":"!handshake.control<>"}'
test_generator -t fork -p size=2 port_types='{"ins":"!handshake.channel<i16>"}'

echo -e "\nTesting join..."
test_generator -t join -p size=3

echo -e "\nTesting lazy_fork..."
test_generator -t lazy_fork -p size=4 port_types='{"ins":"!handshake.control<>"}'
test_generator -t lazy_fork -p size=2 port_types='{"ins":"!handshake.channel<i16>"}'

echo -e "\nTesting load..."
test_generator -t load -p port_types='{"data_out":"!handshake.channel<i16>","addr_in":"!handshake.channel<i16>"}'

echo -e "\nTesting merge..."
test_generator -t merge -p size=4 port_types='{"outs":"!handshake.control<>"}'
test_generator -t merge -p size=2 port_types='{"outs":"!handshake.channel<i16>"}'	

echo -e "\nTesting mux..."
test_generator -t mux -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.control<>"}'
test_generator -t mux -p size=2 port_types='{"index":"!handshake.channel<i16>","outs":"!handshake.channel<i16>"}'

echo -e "\nTesting select..."
test_generator -t select -p port_types='{"result":"!handshake.control<>"}'

echo -e "\nTesting sink..."
test_generator -t sink -p port_types='{"ins":"!handshake.control<>"}'
test_generator -t sink -p port_types='{"ins":"!handshake.channel<i16>"}'

echo -e "\nTesting source..."
test_generator -t source

echo -e "\nTesting store..."
test_generator -t store -p port_types='{"data_in":"!handshake.channel<i16>","addr_in":"!handshake.channel<i16>"}'

rm $OUT