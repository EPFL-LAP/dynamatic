from generators.handshake.join import generate_join
from generators.support.utils import *


def generate_spec_v2_resolver(name, _):
    return f"""
MODULE {name}(actualCondition, actualCondition_valid, generatedCondition, generatedCondition_valid, confirmSpec_ready)
  VAR
  inner_join : {name}__join(actualCondition_valid, generatedCondition_valid, confirmSpec_ready);
  idle : boolean;
  ASSIGN
  init(idle) := TRUE;
  next(idle) := case
    transfer & idle & !actualCondition & generatedCondition : FALSE;
    transfer & !idle & !generatedCondition : TRUE;
    TRUE : idle;
  esac;
  DEFINE
  transfer := actualCondition_valid & generatedCondition_valid & confirmSpec_ready;
  // output
  confirmSpec := idle;
  confirmSpec_valid := inner_join.outs_valid;
  actualCondition_ready := inner_join.ins_0_ready;
  generatedCondition_ready := inner_join.ins_1_ready;

{generate_join(f"{name}__join", {ATTR_SIZE: 2})}
"""
