from generators.support.utils import *
from generators.support.oehb import generate_oehb
from generators.handshake.ndwire import generate_ndwire
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
  load_addr = [f"io_ctrl_{n}_valid, io_ldAddr_{n}_bits, io_ldAddr_{n}_valid" for n in range(num_load_ports)]
  store_addr = [f"io_ctrl_{n + num_load_ports}_valid, io_stAddr_{n}_bits, io_stAddr_{n}_valid, io_stData_{n}_bits, io_stData_{n}_valid" for n in range(num_store_ports)]
  load_data = [f"io_ldData_{n}_ready" for n in range(num_load_ports)]
  lsq_in_ports = ", ".join(["io_loadData", "io_memStart_valid"] + load_addr + store_addr + ["io_ctrlEnd_valid"] + load_data + ["io_memEnd_ready"])
  return f"""
MODULE {name} ({lsq_in_ports})

  {"\n  ".join([f"VAR inner_load_port_{n} : {name}__nd_load_port(io_ctrl_{n}_valid, io_ldAddr_{n}_bits, io_ldAddr_{n}_valid, io_ldData_{n}_ready, io_loadData)" for n in range(num_load_ports)])}
  {"\n  ".join([f"VAR inner_store_port_{n} : {name}__nd_store_port(io_ctrl_{n + num_load_ports}_valid, io_stAddr_{n}_bits, io_stAddr_{n}_valid, io_stData_{n}_bits, io_stData_{n}_valid)" for n in range(num_load_ports)])}
  VAR inner_mc_control : {name}__mc_control(io_memStart_valid, io_ctrlEnd_valid, io_memEnd_ready, all_requests_done);

  -- all_requests_done is true when all the load and store ports have completed the operation
  DEFINE all_requests_done := {" & ".join([f"io_ctrl_{n}_ready" for n in range(num_load_ports + num_store_ports)])};

  -- output
  DEFINE
  -- We decide not to model these signals as they are connected only to memory, but
  -- we are now simulating the memory behavior through ndwires
  io_storeData := {data_type.format_constant(0)};
  io_storeAddress := {addr_type.format_constant(0)};
  io_storeEn := FALSE;
  io_loadAddr := {addr_type.format_constant(0)};
  io_loadEn := FALSE;

  -- non-deterministic signals from the non-deterministic ports
  {"\n  ".join([f"io_ctrl_{n}_ready := inner_load_port_{n}.ctrl_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldAddr_{n}_ready := inner_load_port_{n}.ldAddr_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_bits := inner_load_port_{n}.ldData;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_valid := inner_load_port_{n}.ldData_valid;" for n in range(num_load_ports)])}

  {"\n  ".join([f"io_ctrl_{n + num_load_ports}_ready := inner_store_port_{n}.ctrl_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"io_stAddr_{n}_ready := inner_store_port_{n}.stAddr_ready;;" for n in range(num_store_ports)])}
  {"\n  ".join([f"io_stData_{n}_ready := inner_store_port_{n}.stData_ready;;" for n in range(num_store_ports)])}

  -- control signals from mc_control
  io_memStart_ready := inner_mc_control.memStart_ready;
  io_ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;
  io_memEnd_valid := inner_mc_control.memEnd_valid;

  {_generate_nd_load_port(f"{name}__nd_load_port", capacity, addr_type, data_type)}
  {_generate_nd_store_port(f"{name}__nd_store_port", capacity, addr_type, data_type)}
"""


def _generate_nd_load_port(name, capacity, addr_type, data_type):
    return f"""
MODULE {name} (ctrl_valid, ldAddr, ldAddr_valid, ldData_ready, data_from_mem)
  -- what is the capacity: elastic fifo, ofifo, tfifo???
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
  {generate_oehb(f"{name}__ofifo", {ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__out_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
"""

def _generate_nd_store_port(name, capacity, addr_type, data_type):
    return f"""
MODULE {name} (ctrl_valid, stAddr, stAddr_valid, stData, stData_valid)
  VAR inner_addr_ndw : {name}__addr_ndwire(stAddr, stAddr_valid, inner_addr_capacity.ins_ready);
  VAR inner_addr_capacity : {name}__addr_ofifo(inner_addr_ndw.stAddr, inner_addr_ndw.outs_valid, inner_join.ins_0_ready);
  VAR inner_data_ndw : {name}__data_ndwire(stData, stData_valid, inner_data_capacity.ins_ready);
  VAR inner_data_capacity : {name}__data_ofifo(inner_data_ndw.stAddr, inner_data_ndw.outs_valid, inner_join.ins_1_ready);
  VAR inner_join : join_dataless(inner_addr_capacity.outs_valid, inner_data_capacity.outs_valid);
  VAR inner_sink_ndw : {name}__data_ndwire(inner_data_ndw.outs, inner_join.outs_valid, sink_ins_ready);
  VAR inner_sink : sink(inner_data_ndw.outs, inner_sink_ndw.outs_valid);

  -- output
  DEFINE
  stAddr_ready := inner_addr_ndw.ins_ready & ctrl_valid;
  stData_ready := inner_data_ndw.ins_ready & ctrl_valid;
  ctrl_ready := stAddr_ready & stData_ready;

  {generate_ndwire(f"{name}__addr_ndwire", {ATTR_PORT_TYPES: {"outs": addr_type.mlir_type}})}
  {generate_oehb(f"{name}__addr_ofifo", {ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__data_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
  {generate_oehb(f"{name}__data_ofifo", {ATTR_DATA_TYPE: data_type.mlir_type})}
"""
