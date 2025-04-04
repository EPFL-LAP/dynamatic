from generators.support.utils import *
from generators.support.ofifo import generate_ofifo
from generators.support.mc_control import generate_mc_control
from generators.handshake.ndwire import generate_ndwire
from generators.handshake.join import generate_join
from generators.handshake.sink import generate_sink
import json

def generate_lsq(name, params):
  with open(params["config_file"], "r") as file:
    config = json.load(file)

  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["stAddr_0"])
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["ldData_0"])
  num_load_ports = config["numLoadPorts"]
  num_store_ports = config["numStorePorts"]
  capacity = config["fifoDepth"]
  if config["master"]:
    return _generate_lsq_master(name, addr_type, data_type, num_load_ports, num_store_ports, capacity)


def _generate_lsq_master(name, addr_type, data_type, num_load_ports, num_store_ports, capacity):
  ctrl = [f"ctrl_{n}_valid" for n in range(num_load_ports + num_store_ports)]
  load_addr = [f"ldAddr_{n}, ldAddr_{n}_valid" for n in range(num_load_ports)]
  store_addr = [f"stAddr_{n}, stAddr_{n}_valid, stData_{n}, stData_{n}_valid" for n in range(num_store_ports)]
  load_data = [f"ldData_{n}_ready" for n in range(num_load_ports)]
  lsq_in_ports = ", ".join(["loadData", "memStart_valid"] + ctrl + load_addr + store_addr + ["ctrlEnd_valid"] + load_data + ["memEnd_ready"])
  return f"""
MODULE {name} ({lsq_in_ports})

  {"\n  ".join([f"VAR inner_load_port_{n} : {name}__nd_load_port(ctrl_{n}_valid, ldAddr_{n}, ldAddr_{n}_valid, ldData_{n}_ready, loadData);" for n in range(num_load_ports)])}
  {"\n  ".join([f"VAR inner_store_port_{n} : {name}__nd_store_port(ctrl_{n + num_load_ports}_valid, stAddr_{n}, stAddr_{n}_valid, stData_{n}, stData_{n}_valid);" for n in range(num_load_ports)])}
  VAR inner_mc_control : {name}__mc_control(memStart_valid, ctrlEnd_valid, memEnd_ready, all_requests_done);

  -- all_requests_done is true when all the load and store ports have completed the operation
  DEFINE all_requests_done := {" & ".join([f"ctrl_{n}_ready" for n in range(num_load_ports + num_store_ports)])};

  -- output
  DEFINE
  -- We decide not to model these signals as they are connected only to memory, but
  -- we are now simulating the memory behavior through ndwires
  storeData := {data_type.format_constant(0)};
  storeAddr := {addr_type.format_constant(0)};
  storeEn := FALSE;
  loadAddr := {addr_type.format_constant(0)};
  loadEn := FALSE;

  -- non-deterministic signals from the non-deterministic ports
  {"\n  ".join([f"ctrl_{n}_ready := inner_load_port_{n}.ctrl_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldAddr_{n}_ready := inner_load_port_{n}.ldAddr_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n} := inner_load_port_{n}.ldData;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_load_port_{n}.ldData_valid;" for n in range(num_load_ports)])}

  {"\n  ".join([f"ctrl_{n + num_load_ports}_ready := inner_store_port_{n}.ctrl_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stAddr_{n}_ready := inner_store_port_{n}.stAddr_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stData_{n}_ready := inner_store_port_{n}.stData_ready;" for n in range(num_store_ports)])}

  -- control signals from mc_control
  memStart_ready := inner_mc_control.memStart_ready;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;

  {generate_mc_control(f"{name}__mc_control")}
  {_generate_nd_load_port(f"{name}__nd_load_port", capacity, addr_type, data_type)}
  {_generate_nd_store_port(f"{name}__nd_store_port", capacity, addr_type, data_type)}
"""


def _generate_nd_load_port(name, capacity, addr_type, data_type):
    return f"""
MODULE {name} (ctrl_valid, ldAddr, ldAddr_valid, ldData_ready, data_from_mem)
  VAR inner_input_ndw : {name}__in_ndwire(ldAddr, ldAddr_valid, inner_capacity.ins_ready);
  VAR inner_capacity : {name}__ofifo(inner_input_ndw.outs, inner_input_ndw.outs_valid, inner_output_ndw.ins_ready);
  VAR inner_output_ndw : {name}__out_ndwire(data_from_mem, inner_capacity.outs_valid, ldData_ready);

  -- ctrl_valid tells the port when it can start running and ctrl_ready
  -- informs the outside that the operation is done. This is equivalent 
  -- to the memory controller ctrl signal of size 1

  -- output
  DEFINE
  ldAddr_ready := inner_input_ndw.ins_ready & ctrl_valid;
  ldData := inner_output_ndw.outs;
  ldData_valid := inner_output_ndw.outs_valid & ctrl_valid;
  ctrl_ready := ldData_valid;

  {generate_ndwire(f"{name}__in_ndwire", {ATTR_PORT_TYPES: {"outs": addr_type.mlir_type}})}
  {generate_ofifo(f"{name}__ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__out_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
"""

def _generate_nd_store_port(name, capacity, addr_type, data_type):
    return f"""
MODULE {name} (ctrl_valid, stAddr, stAddr_valid, stData, stData_valid)
  VAR inner_addr_ndw : {name}__addr_ndwire(stAddr, stAddr_valid, inner_addr_capacity.ins_ready);
  VAR inner_addr_capacity : {name}__addr_ofifo(inner_addr_ndw.outs, inner_addr_ndw.outs_valid, inner_join.ins_0_ready);
  VAR inner_data_ndw : {name}__data_ndwire(stData, stData_valid, inner_data_capacity.ins_ready);
  VAR inner_data_capacity : {name}__data_ofifo(inner_data_ndw.outs, inner_data_ndw.outs_valid, inner_join.ins_1_ready);
  VAR inner_join : {name}__join(inner_addr_capacity.outs_valid, inner_data_capacity.outs_valid, inner_sink_ndw.ins_valid);
  VAR inner_sink_ndw : {name}__data_ndwire(inner_data_ndw.outs, inner_join.outs_valid, inner_sink.ins_ready);
  VAR inner_sink : {name}__sink(inner_data_ndw.outs, inner_sink_ndw.outs_valid);

  -- output
  DEFINE
  stAddr_ready := inner_addr_ndw.ins_ready & ctrl_valid;
  stData_ready := inner_data_ndw.ins_ready & ctrl_valid;
  ctrl_ready := stAddr_ready & stData_ready;

  {generate_ndwire(f"{name}__addr_ndwire", {ATTR_PORT_TYPES: {"outs": addr_type.mlir_type}})}
  {generate_ofifo(f"{name}__addr_ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__data_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
  {generate_ofifo(f"{name}__data_ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: data_type.mlir_type})}
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_sink(f"{name}__sink", {ATTR_PORT_TYPES: {"ins": data_type.mlir_type}})}
"""
