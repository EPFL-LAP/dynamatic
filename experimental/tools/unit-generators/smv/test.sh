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

  $GENERATOR_CALL $@

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
test_generator -t br -p bitwidth=0
test_generator -t br -p bitwidth=16

echo -e "\nTesting buffer..."
test_generator -t buffer -p num_slots=1 bitwidth=0 transparent=0
test_generator -t buffer -p num_slots=1 bitwidth=16 transparent=0
test_generator -t buffer -p num_slots=5 bitwidth=0 transparent=0
test_generator -t buffer -p num_slots=5 bitwidth=16 transparent=0
test_generator -t buffer -p num_slots=1 bitwidth=0 transparent=1
test_generator -t buffer -p num_slots=1 bitwidth=16 transparent=1
test_generator -t buffer -p num_slots=5 bitwidth=0 transparent=1
test_generator -t buffer -p num_slots=5 bitwidth=16 transparent=1

echo -e "\nTesting cond_br..."
test_generator -t cond_br -p bitwidth=0
test_generator -t cond_br -p bitwidth=16

echo -e "\nTesting constant..."
test_generator -t constant -p value=42 bitwidth=32

echo -e "\nTesting control_merge..."
test_generator -t control_merge -p size=2 data_bitwidth=0 index_bitwidth=16
test_generator -t control_merge -p size=2 data_bitwidth=16 index_bitwidth=16

echo -e "\nTesting fork..."
test_generator -t fork -p size=4 bitwidth=0
test_generator -t fork -p size=2 bitwidth=16

echo -e "\nTesting join..."
test_generator -t join -p size=3

echo -e "\nTesting lazy_fork..."
test_generator -t lazy_fork -p size=4 bitwidth=0
test_generator -t lazy_fork -p size=2 bitwidth=16

echo -e "\nTesting load..."
test_generator -t load -p data_bitwidth=16 addr_bitwidth=16

echo -e "\nTesting merge..."
test_generator -t merge -p size=4 bitwidth=0
test_generator -t merge -p size=2 bitwidth=16

echo -e "\nTesting mux..."
test_generator -t mux -p size=2 data_bitwidth=0 index_bitwidth=16
test_generator -t mux -p size=2 data_bitwidth=16 index_bitwidth=16

echo -e "\nTesting select..."
test_generator -t select -p bitwidth=0

echo -e "\nTesting sink..."
test_generator -t sink -p bitwidth=0
test_generator -t sink -p bitwidth=16

echo -e "\nTesting source..."
test_generator -t source

echo -e "\nTesting store..."
test_generator -t store -p data_bitwidth=16 addr_bitwidth=16

echo -e "\nTesting absf..."
test_generator -t absf --abstract-data -p is_double=0 latency=0

echo -e "\nTesting addf..."
test_generator -t addf --abstract-data -p is_double=0 latency=9
test_generator -t addf --abstract-data -p is_double=1 latency=12

echo -e "\nTesting addi..."
test_generator -t addi -p bitwidth=32 latency=0
test_generator -t addi --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting andi..."
test_generator -t andi -p bitwidth=32 latency=0
test_generator -t andi --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting cmpi..."
test_generator -t cmpi -p bitwidth=32 latency=0 predicate='"eq"' 
test_generator -t cmpi -p bitwidth=32 latency=0 predicate='"uge"' 
test_generator -t cmpi --abstract-data -p bitwidth=32 latency=0 predicate='"uge"' 

echo -e "\nTesting cmpf..."
test_generator -t cmpf --abstract-data -p is_double=0 latency=0 predicate='"oeq"' 

echo -e "\nTesting divf..."
test_generator -t divf --abstract-data -p is_double=0 latency=29
test_generator -t divf --abstract-data -p is_double=1 latency=36

echo -e "\nTesting divsi..."
test_generator -t divsi -p bitwidth=32 latency=35
test_generator -t divsi --abstract-data -p bitwidth=32 latency=35

echo -e "\nTesting divui..."
test_generator -t divui -p bitwidth=32 latency=35
test_generator -t divui --abstract-data -p bitwidth=32 latency=35

echo -e "\nTesting extf..."
test_generator -t extf --abstract-data -p latency=0

echo -e "\nTesting extsi..."
test_generator -t extsi -p input_bitwidth=32 output_bitwidth=64 latency=0
test_generator -t extsi --abstract-data -p input_bitwidth=32 output_bitwidth=64 latency=0

echo -e "\nTesting extui..."
test_generator -t extui -p input_bitwidth=32 output_bitwidth=64 latency=0
test_generator -t extui --abstract-data -p input_bitwidth=32 output_bitwidth=64 latency=0

echo -e "\nTesting fptosi..."
test_generator -t fptosi --abstract-data -p bitwidth=32 latency=5

echo -e "\nTesting maximumf..."
test_generator -t maximumf --abstract-data -p is_double=0 latency=0

echo -e "\nTesting minimumf..."
test_generator -t minimumf --abstract-data -p is_double=0 latency=0

echo -e "\nTesting mulf..."
test_generator -t mulf --abstract-data -p is_double=0 latency=4

echo -e "\nTesting muli..."
test_generator -t muli -p bitwidth=32 latency=4
test_generator -t muli --abstract-data -p bitwidth=32 latency=4

echo -e "\nTesting negf..."
test_generator -t negf --abstract-data -p is_double=0 latency=0

echo -e "\nTesting not..."
test_generator -t not -p bitwidth=32 latency=0
test_generator -t not --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting ori..."
test_generator -t ori -p bitwidth=32 latency=0
test_generator -t ori --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting shli..."
test_generator -t shli -p bitwidth=32 latency=0
test_generator -t shli --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting shrsi..."
test_generator -t shrsi -p bitwidth=32 latency=0
test_generator -t shrsi --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting shrui..."
test_generator -t shrui -p bitwidth=32 latency=0
test_generator -t shrui --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting sitofp..."
test_generator -t sitofp --abstract-data -p bitwidth=32 latency=5

echo -e "\nTesting subf..."
test_generator -t subf --abstract-data -p is_double=0 latency=9
test_generator -t subf --abstract-data -p is_double=1 latency=12

echo -e "\nTesting subi..."
test_generator -t subi -p bitwidth=32 latency=0
test_generator -t subi --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting truncf..."
test_generator -t truncf --abstract-data -p latency=0

echo -e "\nTesting trunci..."
test_generator -t trunci -p input_bitwidth=64 output_bitwidth=32 latency=0
test_generator -t trunci --abstract-data -p input_bitwidth=64 output_bitwidth=32 latency=0

echo -e "\nTesting xori..."
test_generator -t xori -p bitwidth=32 latency=0
test_generator -t xori --abstract-data -p bitwidth=32 latency=0

echo -e "\nTesting memory_controller..."
test_generator -t memory_controller -p num_loads=0 num_stores=1 num_controls=1 data_bitwidth=16 addr_bitwidth=16
test_generator -t memory_controller -p num_loads=0 num_stores=5 num_controls=1 data_bitwidth=16 addr_bitwidth=16
test_generator -t memory_controller -p num_loads=0 num_stores=5 num_controls=5 data_bitwidth=16 addr_bitwidth=16
test_generator -t memory_controller -p num_loads=1 num_stores=0 num_controls=0 data_bitwidth=16 addr_bitwidth=16
test_generator -t memory_controller -p num_loads=5 num_stores=0 num_controls=0 data_bitwidth=16 addr_bitwidth=16
test_generator -t memory_controller -p num_loads=5 num_stores=5 num_controls=5 data_bitwidth=16 addr_bitwidth=16

rm $OUT
