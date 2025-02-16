from generators.support.utils import *


def generate_select(name, params):
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

  return _generate_select(name, data_type)


def _generate_select(name, data_type):
  return f"""
MODULE {name} (condition, condition_valid, true_value, true_value_valid, false_value, false_value_valid, result_ready)
  VAR
  inner_antitoken : antitoken__{name}(false_value_valid, true_value_valid, g1, g0);

  DEFINE
  ee := condition_valid & ((!condition & false_value_valid) | (condition & true_value_valid));
  valid_internal := ee & !inner_antitoken.stop_valid;
  g0 := !true_value_valid & valid_internal & result_ready;
  g1 := !false_value_valid & valid_internal & result_ready;

  // output
  DEFINE
  true_value_ready := !true_value_valid | (valid_internal & result_ready) | inner_antitoken.kill_0;
  false_value_ready := !false_value_valid | (valid_internal & result_ready) | inner_antitoken.kill_1;
  condition_ready := !condition_valid | (valid_internal & result_ready);
  result_valid := valid_internal;
  result := condition ? true_value : false_value;


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

  // output
  DEFINE
  stop_valid := reg_out_0 | reg_out_1;
  kill_0 := generate_at_0 | reg_out_0;
  kill_1 := generate_at_1 | reg_out_1;
"""


if __name__ == "__main__":
  print(_generate_select("test_select", {
        "data_type": "!handshake.channel<i32>"}))
