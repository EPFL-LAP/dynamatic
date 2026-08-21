#!/bin/bash
yosys -p "read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/arith/subi.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/join.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/logic.v
        chparam -set DATA_TYPE 3 subi
        hierarchy -top subi;
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/3/subi_3_yosys.blif" > /dev/null
