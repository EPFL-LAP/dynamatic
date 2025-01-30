from generators.handshake.join import generate_join
from generators.support.utils import mlir_type_to_smv_type


def generate_cond_br(name, params):
  if "data_type" not in params or params["data_type"] == "!handshake.control<>":
    return _generate_cond_br_dataless(name)
  else:
    return _generate_cond_br(name, mlir_type_to_smv_type(params["data_type"]))


def _generate_cond_br_dataless(name):
  return f"""
MODULE {name}(data_valid, condition, condition_valid, trueOut_ready, falseOut_ready)
  VAR inner_join : {name}__join(data_valid, condition_valid, branch_ready);

  DEFINE branch_ready := (falseOut_ready & !condition) | (trueOut_ready & condition);

  // output
  DEFINE data_ready := inner_join.ins_ready_0;
  DEFINE condition_ready := inner_join.ins_ready_1;
  DEFINE trueOut_valid := condition & inner_join.outs_valid_0;
  DEFINE falseOut_valid := !condition & inner_join.outs_valid_0;

{generate_join(f"{name}__join", {"size": 2})}
"""


def _generate_cond_br(name, data_type):
  return f"""
MODULE {name}(data, data_valid, condition, condition_valid, trueOut_ready, falseOut_ready)
  VAR inner_br : cond_br_dataless(data_valid, condition, condition_valid, trueOut_ready, falseOut_ready);

  // output
  DEFINE data_ready := inner_br.data_ready;
  DEFINE condition_ready := inner_br.condition_ready;
  DEFINE trueOut_valid := inner_br.trueOut_valid;
  DEFINE falseOut_valid := inner_br.falseOut_valid;
  DEFINE trueOut := data;
  DEFINE falseOut := data;

{_generate_cond_br_dataless(f"{name}__cond_br_dataless")}
"""


if __name__ == "__main__":
  print(generate_cond_br("test_cond_br_dataless", {}))
  print(generate_cond_br("test_cond_br", {
        "data_type": "!handshake.channel<i32>"}))
