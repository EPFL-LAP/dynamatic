def generate_eager_fork_register_block(name):
    return f"""
MODULE {name}(ins_valid, outs_ready, backpressure)
  VAR
  reg_value : boolean;

  ASSIGN
  init(reg_value) := TRUE;
  next(reg_value) := block_stop | !backpressure;

  //output
  DEFINE
  block_stop := !outs_ready & reg_value;
  outs_valid := reg_value & ins_valid;
"""
