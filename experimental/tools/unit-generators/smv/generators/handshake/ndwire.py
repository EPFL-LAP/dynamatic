from generators.support.utils import *


def generate_ndwire(name, params):
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["outs"])

  if data_type.bitwidth == 0:
    return _generate_ndwire_dataless(name)
  else:
    return _generate_ndwire(name, data_type)


def _generate_ndwire_dataless(name):
  return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR state : {{SLEEPING, RUNNING}};

  ASSIGN
  init(state) := {{SLEEPING, RUNNING}};
  next(state) := case
    state = SLEEPING : {{SLEEPING, RUNNING}};
    ins_valid and outs_ready : {{SLEEPING, RUNNING}};
    TRUE : state;
  esac;

  // output
  DEFINE  
  outs_valid :=  ins_valid & (state = RUNNING);
  ins_ready  :=  outs_ready & (state = RUNNING);
"""


def _generate_ndwire(name, data_type):
  return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  VAR inner_ndwire : {name}__ndwire_dataless(ins_valid, outs_ready)

  // output
  DEFINE
  outs := ins;
  outs_valid :=  inner_ndwire.outs_valid;
  ins_ready  :=  inner_ndwire.ins_ready;
"""
