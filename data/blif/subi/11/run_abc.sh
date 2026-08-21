#!/bin/bash
abc -c "read_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/11/subi_11_yosys.blif;
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
        write_blif /local/home/crizzi/CI_mapbuf/dynamatic/data/blif/subi/11/subi.blif"
