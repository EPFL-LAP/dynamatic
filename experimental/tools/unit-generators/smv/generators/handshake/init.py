def generate_init(name, _):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  reg : boolean;
  full : boolean;
  ASSIGN
  init(full) := TRUE;
  next(full) := outs_valid & !outs_ready;
  init(reg) := FALSE;
  next(reg) := enable ? ins : reg;
  DEFINE
  enable := ins_ready & ins_valid & !outs_ready;

  // output
  outs := full ? reg : ins;
  outs_valid := ins_valid | full;
  ins_ready := !full;
"""
