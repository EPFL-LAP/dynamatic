#!/bin/bash
yosys -p "read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/arith/addi.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/join.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/logic.v
        chparam -set DATA_TYPE 12 addi
        hierarchy -top addi;
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/addi/12/addi_12_yosys.blif" > /dev/null
