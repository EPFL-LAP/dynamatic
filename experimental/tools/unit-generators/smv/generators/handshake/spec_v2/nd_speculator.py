def generate_spec_v2_nd_speculator(name, _):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR interpolate : boolean;
  ASSIGN
  init(interpolate) := {{TRUE, FALSE}};
  next(interpolate) := case
    transfer : {{TRUE, FALSE}};
    TRUE : interpolate;
  esac;
  DEFINE
  transfer := interpolate ? outs_ready : ins_valid & outs_ready;

  // output
  outs := interpolate ? TRUE : ins;
  outs_valid := interpolate ? TRUE : ins_valid;
  ins_ready := interpolate ? FALSE : outs_ready;

  FAIRNESS !interpolate
"""
