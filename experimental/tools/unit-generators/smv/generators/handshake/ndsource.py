from generators.support.utils import *


def generate_ndsource(name, _):
    return f"""
MODULE {name}(result_ready)
  VAR result_valid : boolean;
  VAR counter : 0..31;
  FROZENVAR exact_tokens : 0..31;
  ASSIGN
  init(result_valid) := exact_tokens > 0;
  next(result_valid) := case
    result_ready & counter + 1 < exact_tokens : TRUE;
    result_ready : FALSE;
    TRUE : result_valid;
  esac;
  init(counter) := 0;
  next(counter) := case
    result_valid & result_ready & counter < exact_tokens & counter < 31 : counter + 1;
    TRUE : counter;
  esac;
"""
