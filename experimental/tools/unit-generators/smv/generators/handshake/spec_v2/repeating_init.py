def generate_spec_v2_repeating_init(name, _):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR
  emit_init : boolean;
  ASSIGN
  init(emit_init) := TRUE;
  next(emit_init) := case
    outs_valid & outs_ready : !outs;
    TRUE : emit_init;
  esac;
  DEFINE
  outs := emit_init ? TRUE : ins;
  outs_valid := emit_init | ins_valid;
  ins_ready := !emit_init & outs_ready;
"""
