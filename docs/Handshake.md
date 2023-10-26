# Handshake-level IR

## Supported `Handshake` operations for VHDL export

- handshake.func
- handshake.buffer
- handshake.fork
- handshake.lazy_fork
- handshake.merge
- handshake.mux
- handshake.control_merge
- handshake.br
- handshake.cond_br
- handshake.sink
- handshake.source
- handshake.constant
- handshake.join
- handshake.mem_contoller
- handshake.d_load
- handshake.d_store
- handshake.d_return
- handshake.end

## Supported `arith` operations for VHDL export

### Integer

- arith.addi
- arith.andi
- arith.cmpi
- arith.divsi
- arith.divui
- arith.extsi
- arith.extui
- arith.muli
- arith.ori
- arith.remui*
- arith.remsi*
- arith.select
- arith.shli
- arith.shrsi
- arith.shrui
- arith.sitofp
- arith.subi
- arith.trunci
- arith.uitofp
- arith.xori

### Floating-point

- arith.addf
- arith.cmpf
- arith.divf
- arith.extf
- arith.mulf
- arith.remf
- arith.subf
- arith.truncf
