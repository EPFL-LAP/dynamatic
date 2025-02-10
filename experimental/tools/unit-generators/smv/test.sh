#!/bin/bash

GENPATH=${GENPATH:-"./"}
NUXMVPAT=${NUXMVPAT:-"../../../../../nuXmv-2.0.0-Linux/bin/"}
NUXMV=${NUXMV:-1}

if [[ $NUXMV -eq 0 ]]; then
  OUT="/dev/null"
else
  OUT="./module.smv"
fi

check_smv_syntax () {
  if [[ $NUXMV -eq 1 ]]; then
    MODEL=$1
    ${NUXMVPAT}nuXmv -pre cpp -source <(echo "set on_failure_script_quits; read_model -i $MODEL; quit") > /dev/null
    if [ $? != 0 ]; then
      echo "Syntax checking failed for smv file $MODEL! Exiting.."
      exit 1
    else
      echo "Passed!"
    fi
  fi
}


echo "Testing br..."
python ${GENPATH}smv-unit-generator.py -n test_module -t br -p data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t br -p data_type='"!handshake.channel<i16>"' > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting buffer..."
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type='"!handshake.control<>"' timing='"#handshake.timing< {R: 1}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type='"!handshake.channel<i16>"' timing='"#handshake.timing< {R: 1}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type='"!handshake.control<>"' timing='"#handshake.timing< {R: 1}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type='"!handshake.channel<i16>"' timing='"#handshake.timing< {R: 1}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type='"!handshake.control<>"' timing='"#handshake.timing< {R: 0}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type='"!handshake.channel<i16>"' timing='"#handshake.timing< {R: 0}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type='"!handshake.control<>"' timing='"#handshake.timing< {R: 0}>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type='"!handshake.channel<i16>"' timing='"#handshake.timing< {R: 0}>"' > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting cond_br..."
python ${GENPATH}smv-unit-generator.py -n test_module -t cond_br -p data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t cond_br -p data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting constant..."
python ${GENPATH}smv-unit-generator.py -n test_module -t constant -p value=42 data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting control_merge..."
python ${GENPATH}smv-unit-generator.py -n test_module -t control_merge -p size=2 index_type='"!handshake.channel<i16>"' data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t control_merge -p size=2 index_type='"!handshake.channel<i16>"' data_type='"!handshake.channel<i16>"' > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting fork..."
python ${GENPATH}smv-unit-generator.py -n test_module -t fork -p size=4 data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t fork -p size=2 data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting join..."
python ${GENPATH}smv-unit-generator.py -n test_module -t join -p size=3 > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting lazy_fork..."
python ${GENPATH}smv-unit-generator.py -n test_module -t lazy_fork -p size=4 data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t lazy_fork -p size=2 data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting load..."
python ${GENPATH}smv-unit-generator.py -n test_module -t load -p data_type='"!handshake.channel<i16>"' addr_type='"!handshake.channel<i16>"' > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting merge..."
python ${GENPATH}smv-unit-generator.py -n test_module -t merge -p size=4 data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t merge -p size=2 data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting mux..."
python ${GENPATH}smv-unit-generator.py -n test_module -t mux -p size=2 select_type='"!handshake.channel<i16>"' data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t mux -p size=2 select_type='"!handshake.channel<i16>"' data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting select..."
python ${GENPATH}smv-unit-generator.py -n test_module -t select -p data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting sink..."
python ${GENPATH}smv-unit-generator.py -n test_module -t sink -p data_type='"!handshake.control<>"' > $OUT
check_smv_syntax ./module.smv
python ${GENPATH}smv-unit-generator.py -n test_module -t sink -p data_type='"!handshake.channel<i16>"' > $OUT	
check_smv_syntax ./module.smv

echo -e "\nTesting source..."
python ${GENPATH}smv-unit-generator.py -n test_module -t source > $OUT
check_smv_syntax ./module.smv

echo -e "\nTesting store..."
python ${GENPATH}smv-unit-generator.py -n test_module -t store -p data_type='"!handshake.channel<i16>"' addr_type='"!handshake.channel<i16>"' > $OUT
check_smv_syntax ./module.smv

rm ./module.smv