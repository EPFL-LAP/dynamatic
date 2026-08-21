#!/bin/bash
yosys -p "read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/arith/muli.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/join.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/logic.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/delay_buffer.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/dataless/oehb.v
        chparam -set DATA_TYPE 3 muli
        hierarchy -top muli;
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/muli/3/muli_3_yosys.blif" > /dev/null
