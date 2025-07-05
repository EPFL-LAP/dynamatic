from generators.support.utils import *


def generate_ndsource(name, _):
    return f"""
MODULE {name}(result_ready)
  VAR val : boolean;
  ASSIGN
  init(val) := {{TRUE, FALSE}};
  next(val) := case
    transfer : {{TRUE, FALSE}};
    TRUE : val;
  esac;
  DEFINE
  transfer := result_ready;
  // output
  result := val;
  result_valid := TRUE;
"""
