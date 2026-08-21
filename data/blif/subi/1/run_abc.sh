#!/bin/bash
abc -c "read_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/1/subi_1_yosys.blif;
        strash;
        rewrite;
        b;
        refactor;
        b;
        rewrite;
        b;
        refactor;
        b;
        rewrite;
        b;
        refactor;
        b;
        rewrite;
        b;
        refactor;
        b;
        rewrite;
        b;
        refactor;
        b;
        rewrite;
        b;
        refactor;
        b;
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/1/subi.blif"
