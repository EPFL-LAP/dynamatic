#!/bin/bash
yosys -p "read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/handshake/join.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/verilog/support/logic.v
        read_verilog -defer /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/cmpi/12/cmpi.v
        chparam -set DATA_TYPE 12 cmpi
        hierarchy -top cmpi;
        proc;
        opt -nodffe -nosdff;
        memory -nomap;
        techmap;
        flatten;
        clean;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/cmpi/12/cmpi_12_yosys.blif" > /dev/null
