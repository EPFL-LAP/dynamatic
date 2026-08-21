#!/bin/bash
yosys -p "read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/arith/subi.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/join.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/logic.v
        chparam -set DATA_TYPE 5 subi
        hierarchy -top subi;
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/5/subi_5_yosys.blif" > /dev/null
