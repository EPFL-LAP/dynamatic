from generators.support.utils import *
import json

def generate_lsq(name, params):
  with open(params["config_file"], "r") as file:
    config = json.load(file)

  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["stAddr_0"])
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["ldData_0"])
  num_load_ports = config["numLoadPorts"]
  num_store_ports = config["numStorePorts"]

  return _generate_lsq(name, addr_type, data_type, num_load_ports, num_store_ports)


def _generate_lsq(name, addr_type, data_type, num_load_ports, num_store_ports):
  load_addr = [f"io_ctrl_{n}_valid, io_ldAddr_{n}_bits, io_ldAddr_{n}_valid" for n in range(num_load_ports)]
  store_addr = [f"io_ctrl_{n + num_load_ports}_valid, io_stAddr_{n}_bits, io_stAddr_{n}_valid, io_stData_{n}_bits, io_stData_{n}_valid" for n in range(num_store_ports)]
  load_data = [f"io_ldData_{n}_ready" for n in range(num_load_ports)]
  lsq_in_ports = ", ".join(["io_loadData", "io_memStart_valid"] + load_addr + store_addr + ["io_ctrlEnd_valid"] + load_data + ["io_memEnd_ready"])
  return f"""
MODULE {name} ({lsq_in_ports})

  // output
  DEFINE
  io_storeData := {data_type.format_constant(0)};
  io_storeAddress := {addr_type.format_constant(0)};
  io_storeEn := FALSE;
  io_loadAddr := {addr_type.format_constant(0)};
  io_loadEn := FALSE;
  {"\n  ".join([f"io_ctrl_{n}_ready := TRUE;" for n in range(num_load_ports + num_store_ports)])}
  {"\n  ".join([f"io_ldAddr_{n}_ready := TRUE;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_bits := TRUE;" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_ldData_{n}_valid := {data_type.format_constant(0)};" for n in range(num_load_ports)])}
  {"\n  ".join([f"io_stAddr_{n}_ready := TRUE;" for n in range(num_store_ports)])}
  {"\n  ".join([f"io_stData_{n}_ready := TRUE;" for n in range(num_store_ports)])}
  io_memStart_ready :=
  io_ctrlEnd_ready :=
  io_memEnd_valid :=
"""