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
  num_bbs = config["numBBs"]
  capacity = config["fifoDepth"]
  load_groups = config["numLoads"]
  store_groups = config["numStores"]
  lsq_in_signals = get_lsq_ports(
      params[ATTR_PORT_TYPES], num_load_ports, num_store_ports, num_bbs, config["master"])

  if config["master"]:
    return _generate_lsq_master(name, lsq_in_signals, addr_type, data_type, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity)
  else:
    return _generate_lsq_slave(name, lsq_in_signals, addr_type, data_type, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity)


def group_index(index, group_list):
  partial_sum = 0
  for i, num_el in enumerate(group_list):
    partial_sum += num_el
    if index < partial_sum:
      return i

# I could try to generate it from config only, but even HandshakeInterfaceOps.cpp doesn't do it.
# I don't know what the order between loads and stores in a group should be, so I get the order form
# port types


def get_lsq_ports(port_types, num_load_ports, num_store_ports, num_bbs, is_master):
  # Compute the number of signals coming from input channels
  # Load ports: ldAddr ldAddr_valid
  # Store ports: stAddr stAddr_valid stData stData_valid
  # Ctrl: ctrl
  # If master lsq: loadData memStart_valid ctrlEnd_valid
  # If slave lsq: ldDataFromMC ldDataFromMC_valid
  num_input_signals = num_load_ports * 2 + \
      num_store_ports * 4 + num_bbs + (3 if is_master else 2)

  lsq_ports = []
  for channel, handshake_type in port_types.items():
    if len(lsq_ports) < num_input_signals:
      if "handshake" not in handshake_type:
        lsq_ports += [channel]
      else:
        if SmvScalarType(handshake_type).bitwidth != 0:
          lsq_ports += [channel]
        lsq_ports += [f"{channel}_valid"]
    else:
      if "handshake" in handshake_type:
        lsq_ports += [f"{channel}_ready"]
  return lsq_ports


def _generate_lsq_master(name, lsq_in_signals, addr_type, data_type, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity):
  lsq_in_ports = ", ".join(lsq_in_signals)
  return f"""
MODULE {name} ({lsq_in_ports})
  VAR inner_mc_control : {name}__mc_control(memStart_valid, ctrlEnd_valid, memEnd_ready, all_requests_done);
  DEFINE all_requests_done :=  {" & ".join([f"load_requests_{n} = 0" for n in range(num_load_ports)] + [f"store_requests_{n} = 0" for n in range(num_store_ports)])};

  {_generate_lsq_logic(name, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity)}

  storeData := {data_type.format_constant(0)};
  storeAddr := {addr_type.format_constant(0)};
  storeEn := in_storeEn;
  loadAddr := {addr_type.format_constant(0)};
  loadEn := in_loadEn;

  -- control signals from mc_control
  memStart_ready := inner_mc_control.memStart_ready;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;

  {generate_mc_control(f"{name}__mc_control")}
  {_generate_nd_load_port(f"{name}__nd_load_port", capacity, addr_type, data_type)}
  {_generate_nd_store_port(f"{name}__nd_store_port", capacity, addr_type, data_type)}
"""


def _generate_lsq_slave(name, lsq_in_signals, addr_type, data_type, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity):
  lsq_in_ports = ", ".join(lsq_in_signals)
  return f"""
MODULE {name} ({lsq_in_ports})

  DEFINE loadData := ldDataFromMC;
  {_generate_lsq_logic(name, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity)}

  stDataToMC := {data_type.format_constant(0)};
  stDataToMC_valid := in_storeEn;
  stAddrToMC := {addr_type.format_constant(0)};
  stAddrToMC_valid := in_storeEn;
  ldAddrToMC := {addr_type.format_constant(0)};
  ldAddrToMC_valid := in_loadEn;
  ldDataFromMC_ready := TRUE;

  {_generate_nd_load_port(f"{name}__nd_load_port", capacity, addr_type, data_type)}
  {_generate_nd_store_port(f"{name}__nd_store_port", capacity, addr_type, data_type)}
"""


def _generate_lsq_logic(name, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, capacity):
  return f"""
  -------- Load queue --------
  VAR
  -- Number of available slots in the load queue
  available_load_slots : {-capacity}..{capacity};
  -- Number of the pending requests for each load port
  {"\n  ".join([f"load_requests_{n} : 0..{capacity};" for n in range(num_load_ports)])}


  -- Every time a load s group is allocated the requests increase by one. Once
  -- the request is completed (i.e. the load data has reached memory) the
  -- pending requests decrease by one.
  -- load_port_*_ctrl signals that the load can access memory (it has at least 1 request)

  {"\n  ".join([f"""ASSIGN
  init(load_requests_{n}) := 0;
  next(load_requests_{n}) := load_requests_{n} + toint(ctrl_{n}_valid) - toint(inner_load_port_{n}.ldData_valid);
  DEFINE load_port_{n}_ctrl := load_requests_{n} > 0;""" for n in range(num_load_ports)])}
  
  -- If a new group is allocated with the ctrl signal we decrease the available slots by the number of loads in the group
  -- if a load happened we deallocate it increasing the available slots by one.
  ASSIGN
  init(available_load_slots) := {capacity};
  next(available_load_slots) := case
    {"\n  ".join([f"ctrl_{n}_valid : available_load_slots - (available_load_slots > 0 ? {load_groups[n]} : 0) + (load_mem_access_happened ? 1 : 0);" for n in range(num_load_ports)])}
    TRUE : available_load_slots + (load_mem_access_happened ? 1 : 0);
  esac;

  -- Checks if at least one load port executed a load
  DEFINE
  load_mem_access_happened := {" | ".join([f"inner_load_port_{n}.ldData_valid" for n in range(num_load_ports)])};
  in_loadEn := load_mem_access_happened;

  -- Non-deterministic load port
  VAR
  {"\n  ".join([f"inner_load_port_{n} : {name}__nd_load_port(load_port_{n}_ctrl, ldAddr_{n}, ldAddr_{n}_valid, ldData_{n}_ready, loadData);" for n in range(num_load_ports)])}
  ----- end load queue ------


  -------- Store queue --------
  VAR
  -- Number of available slots in the store queue
  available_store_slots : {-capacity}..{capacity};
  -- Number of the pending requests for each store port
  {"\n  ".join([f"store_requests_{n} : 0..{capacity};" for n in range(num_store_ports)])}


  -- Every time a store s group is allocated the requests increase by one. Once
  -- the request is completed (i.e. the store reached memory) the
  -- pending requests decrease by one.
  -- store_port_*_ctrl signals that the store can access memory (it has at least 1 request)

  {"\n  ".join([f"""ASSIGN
  init(store_requests_{n}) := 0;
  next(store_requests_{n}) := store_requests_{n} + toint(ctrl_{n}_valid) - toint(inner_store_port_{n}.memData_valid);
  DEFINE store_port_{n}_ctrl := store_requests_{n} > 0;""" for n in range(num_store_ports)])}
  
  -- If a new group is allocated with the ctrl signal we decrease the available slots by the number of stores in the group
  -- if a store happened we deallocate it increasing the available slots by one.
  ASSIGN
  init(available_store_slots) := {capacity};
  next(available_store_slots) := case
    {"\n  ".join([f"ctrl_{n}_valid : available_store_slots - (available_store_slots > 0 ? {store_groups[n]} : 0) + (store_mem_access_happened ? 1 : 0);" for n in range(num_store_ports)])}
    TRUE : available_store_slots + (store_mem_access_happened ? 1 : 0);
  esac;

  -- Checks if at least one store port executed a store
  DEFINE
  store_mem_access_happened := {" | ".join([f"inner_store_port_{n}.memData_valid" for n in range(num_store_ports)])};
  in_storeEn := store_mem_access_happened;

  -- Non-deterministic store port
  VAR
  {"\n  ".join([f"inner_store_port_{n} : {name}__nd_store_port(store_port_{n}_ctrl, stAddr_{n}, stAddr_{n}_valid, stData_{n}, stData_{n}_valid);" for n in range(num_store_ports)])}

  ----- end store queue ------

  -- output
  -- non-deterministic signals from the non-deterministic ports
  DEFINE
  {"\n  ".join([f"ctrl_{n}_ready := available_load_slots > 0 & available_store_slots > 0;" for n in range(num_bbs)])}

  {"\n  ".join([f"ldAddr_{n}_ready := inner_load_port_{n}.ldAddr_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n} := inner_load_port_{n}.ldData;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_load_port_{n}.ldData_valid;" for n in range(num_load_ports)])}

  {"\n  ".join([f"stAddr_{n}_ready := inner_store_port_{n}.stAddr_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stData_{n}_ready := inner_store_port_{n}.stData_ready;" for n in range(num_store_ports)])}
"""


def _generate_nd_load_port(name, capacity, addr_type, data_type):
  return f"""
MODULE {name} (ctrl_valid, ldAddr, ldAddr_valid, ldData_ready, data_from_mem)
  VAR inner_input_ndw : {name}__in_ndwire(ldAddr, ldAddr_valid & ctrl_valid, inner_capacity.ins_ready);
  VAR inner_capacity : {name}__ofifo(inner_input_ndw.outs, inner_input_ndw.outs_valid, inner_output_ndw.ins_ready);
  VAR inner_output_ndw : {name}__out_ndwire(data_from_mem, inner_capacity.outs_valid, ldData_ready);

  -- ctrl_valid tells the port when it can start running

  -- output
  DEFINE
  ldAddr_ready := inner_input_ndw.ins_ready;
  ldData := inner_output_ndw.outs;
  ldData_valid := inner_output_ndw.outs_valid;

  {generate_ndwire(f"{name}__in_ndwire", {ATTR_PORT_TYPES: {"outs": addr_type.mlir_type}})}
  {generate_ofifo(f"{name}__ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__out_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
"""


def _generate_nd_store_port(name, capacity, addr_type, data_type):
  return f"""
MODULE {name} (ctrl_valid, stAddr, stAddr_valid, stData, stData_valid)
  VAR inner_addr_ndw : {name}__addr_ndwire(stAddr, stAddr_valid & ctrl_valid, inner_addr_capacity.ins_ready);
  VAR inner_addr_capacity : {name}__addr_ofifo(inner_addr_ndw.outs, inner_addr_ndw.outs_valid, inner_join.ins_0_ready);
  VAR inner_data_ndw : {name}__data_ndwire(stData, stData_valid & ctrl_valid, inner_data_capacity.ins_ready);
  VAR inner_data_capacity : {name}__data_ofifo(inner_data_ndw.outs, inner_data_ndw.outs_valid, inner_join.ins_1_ready);
  VAR inner_join : {name}__join(inner_addr_capacity.outs_valid, inner_data_capacity.outs_valid, inner_sink_ndw.ins_valid);
  VAR inner_sink_ndw : {name}__data_ndwire(inner_data_ndw.outs, inner_join.outs_valid, inner_sink.ins_ready);
  VAR inner_sink : {name}__sink(inner_data_ndw.outs, inner_sink_ndw.outs_valid);

  -- output
  DEFINE
  stAddr_ready := inner_addr_ndw.ins_ready;
  stData_ready := inner_data_ndw.ins_ready;
  memData_valid := inner_sink_ndw.outs_valid;

  {generate_ndwire(f"{name}__addr_ndwire", {ATTR_PORT_TYPES: {"outs": addr_type.mlir_type}})}
  {generate_ofifo(f"{name}__addr_ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: addr_type.mlir_type})}
  {generate_ndwire(f"{name}__data_ndwire", {ATTR_PORT_TYPES: {"outs": data_type.mlir_type}})}
  {generate_ofifo(f"{name}__data_ofifo", {ATTR_SLOTS: capacity, ATTR_DATA_TYPE: data_type.mlir_type})}
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_sink(f"{name}__sink", {ATTR_PORT_TYPES: {"ins": data_type.mlir_type}})}
"""


def _generate_lsq_slave(name, addr_type, data_type, num_load_ports, num_store_ports, capacity):
  ctrl = [f"ctrl_{n}_valid" for n in range(num_load_ports + num_store_ports)]
  load_addr = [f"ldAddr_{n}, ldAddr_{n}_valid" for n in range(num_load_ports)]
  store_addr_data = [
      f"stAddr_{n}, stAddr_{n}_valid, stData_{n}, stData_{n}_valid" for n in range(num_store_ports)]
  load_data = [f"ldData_{n}_ready" for n in range(num_load_ports)]
  mc_in_signals = ["ldDataFromMC, ldDataFromMC_valid"]
  mc_out_signals = ["ldAddrToMC_ready, stAddrToMC_ready, stDataToMC_ready"]
  lsq_in_ports = ", ".join(
      ctrl + load_addr + store_addr_data + mc_in_signals + load_data + mc_out_signals)
  return f"""
MODULE {name} ({lsq_in_ports})


  -- output
  DEFINE
  -- We decide not to model these signals as they are connected only to memory, but
  -- we are now simulating the memory behavior through ndwires
  stDataToMC := {data_type.format_constant(0)};
  stDataToMC_valid := TRUE;
  stAddrToMC := {addr_type.format_constant(0)};
  stAddrToMC_valid := TRUE;
  loadAddrfromMC := {addr_type.format_constant(0)};
  loadAddrfromMC_valid := TRUE;
  ldDataFromMC_ready := TRUE;

  -- non-deterministic signals from the non-deterministic ports
  {"\n  ".join([f"ctrl_{n}_ready := inner_load_port_{n}.ctrl_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldAddr_{n}_ready := inner_load_port_{n}.ldAddr_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n} := inner_load_port_{n}.ldData;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_load_port_{n}.ldData_valid;" for n in range(num_load_ports)])}

  {"\n  ".join([f"ctrl_{n + num_load_ports}_ready := inner_store_port_{n}.ctrl_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stAddr_{n}_ready := inner_store_port_{n}.stAddr_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stData_{n}_ready := inner_store_port_{n}.stData_ready;" for n in range(num_store_ports)])}

"""
