from generators.support.utils import *


def generate_select(name, params):
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    return _generate_select(name, data_type)


def _generate_select(name, data_type):
    return f"""
MODULE {name} (condition, condition_valid, trueValue, trueValue_valid, falseValue, falseValue_valid, result_ready)
  VAR
  inner_antitoken : antitoken__{name}(falseValue_valid, trueValue_valid, g1, g0);

  DEFINE
  ee := condition_valid & ((!condition & falseValue_valid) | (condition & trueValue_valid));
  valid_internal := ee & !inner_antitoken.stop_valid;
  g0 := !trueValue_valid & valid_internal & result_ready;
  g1 := !falseValue_valid & valid_internal & result_ready;

  -- output
  DEFINE
  trueValue_ready := !trueValue_valid | (valid_internal & result_ready) | inner_antitoken.kill_0;
  falseValue_ready := !falseValue_valid | (valid_internal & result_ready) | inner_antitoken.kill_1;
  condition_ready := !condition_valid | (valid_internal & result_ready);
  result_valid := valid_internal;
  result := condition ? trueValue : falseValue;


MODULE antitoken__{name} (ins_valid_0, ins_valid_1, generate_at_0, generate_at_1)
  VAR
  reg_out_0 : boolean;
  reg_out_1 : boolean;

  DEFINE
  reg_in_0 := !ins_valid_0 & (generate_at_0 | reg_out_0);
  reg_in_1 := !ins_valid_1 & (generate_at_1 | reg_out_1);

  ASSIGN
  init(reg_out_0) := FALSE;
  next(reg_out_0) := reg_in_0;
  init(reg_out_1) := FALSE;
  next(reg_out_1) := reg_in_1;

  -- output
  DEFINE
  stop_valid := reg_out_0 | reg_out_1;
  kill_0 := generate_at_0 | reg_out_0;
  kill_1 := generate_at_1 | reg_out_1;
"""
