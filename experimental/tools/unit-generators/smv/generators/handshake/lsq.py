from generators.support.utils import *
import json

def generate_lsq(name, params):
  with open(params["config_file"], "r") as file:
    config = json.load(file)

  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["stAddr_0"])
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["ldData_0"])
  num_load_ports = config["numLoadPorts"]
  num_store_ports = config["numStorePorts"]
  if config["master"]:
    return _generate_lsq_master(name, addr_type, data_type, num_load_ports, num_store_ports)


def _generate_lsq_master(name, addr_type, data_type, num_load_ports, num_store_ports):
  load_addr = [f"io_ctrl_{n}_valid, io_ldAddr_{n}_bits, io_ldAddr_{n}_valid" for n in range(num_load_ports)]
  store_addr = [f"io_ctrl_{n + num_load_ports}_valid, io_stAddr_{n}_bits, io_stAddr_{n}_valid, io_stData_{n}_bits, io_stData_{n}_valid" for n in range(num_store_ports)]
  load_data = [f"io_ldData_{n}_ready" for n in range(num_load_ports)]
  lsq_in_ports = ", ".join(["io_loadData", "io_memStart_valid"] + load_addr + store_addr + ["io_ctrlEnd_valid"] + load_data + ["io_memEnd_ready"])
  return f"""
MODULE {name} ({lsq_in_ports})

  {"\n  ".join([f"VAR inner_load_port_{n} : nd_load_port_{n}(io_ctrl_{n}_valid, io_ldAddr_{n}_bits, io_ldAddr_{n}_valid, io_ldData_{n}_ready, io_loadData)" for n in range(num_load_ports)])}
  {"\n  ".join([f"VAR inner_store_port_{n} : nd_store_port_{n}(io_ctrl_{n + num_load_ports}_valid, io_stAddr_{n}_bits, io_stAddr_{n}_valid, io_stData_{n}_bits, io_stData_{n}_valid)" for n in range(num_load_ports)])}

  -- output
  DEFINE
  -- this comes from an mc control 
  io_storeData := {data_type.format_constant(0)};
  io_storeAddress := {addr_type.format_constant(0)};
  io_storeEn := FALSE;
  io_loadAddr := {addr_type.format_constant(0)};
  io_loadEn := FALSE;
  -- this we create nondeterministic component like in the paper for each port
  {"\n  ".join([f"io_ctrl_{n}_ready := TRUE;" for n in range(num_load_ports + num_store_ports)])}
  {"\n  ".join([f"io_ldAddr_{n}_ready := TRUE;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_bits := {data_type.format_constant(0)};" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_valid := TRUE;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_stAddr_{n}_ready := TRUE;" for n in range(num_store_ports)])}
  {"\n  ".join([f"io_stData_{n}_ready := TRUE;" for n in range(num_store_ports)])}
  io_memStart_ready :=
  io_ctrlEnd_ready :=
  io_memEnd_valid :=
"""

def _generate_nd_load_port(name, capacity):
    return f"""
MODULE {name} (ctrl_valid, ldAddr, ldAddr_valid, ldData_ready, data_from_mem)
  VAR inner_input_ndw : nd_wire(ldAddr, ldAddr_valid, inner_capacity.ins_ready);
  VAR inner_capacity : capacity(inner_input_ndw.outs, inner_input_ndw.outs_valid, inner_output_ndw.ins_ready);
  VAR inner_output_ndw : nd_wire(data_from_mem, inner_capacity.outs_valid, ldData_ready);

  -- output
  DEFINE
  ldAddr_ready := inner_input_ndw.ins_ready;
  ldData := inner_output_ndw.outs;
  ldData_valid := inner_output_ndw.outs_valid;
"""

def _generate_nd_store_port(name, capacity):
    return f"""
MODULE {name} (ctrl_valid, stAddr, stAddr_valid, stData, stData_valid)
  VAR inner_addr_ndw : nd_wire(stAddr, stAddr_valid, inner_addr_capacity.ins_ready);
  VAR inner_addr_capacity : capacity(inner_addr_ndw.stAddr, inner_addr_ndw.outs_valid, inner_join.ins_0_ready);
  VAR inner_data_ndw : nd_wire(stData, stData_valid, inner_data_capacity.ins_ready);
  VAR inner_data_capacity : capacity(inner_data_ndw.stAddr, inner_data_ndw.outs_valid, inner_join.ins_1_ready);
  VAR inner_join : join_dataless(inner_addr_capacity.outs_valid, inner_data_capacity.outs_valid);
  VAR inner_sink_ndw : nd_wire(inner_data_ndw.outs, inner_join.outs_valid, sink_ins_ready);
  VAR inner_sink : sink(inner_data_ndw.outs, inner_sink_ndw.outs_valid);

  -- output
  DEFINE
  stAddr_ready := inner_addr_ndw.ins_ready;
  stData_ready := inner_data_ndw.ins_ready;
"""
