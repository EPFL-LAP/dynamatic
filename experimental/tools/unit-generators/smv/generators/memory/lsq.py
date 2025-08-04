from generators.support.utils import *
from generators.handshake.buffers.fifo_break_dv import generate_fifo_break_dv
from generators.support.mc_control import generate_mc_control
from generators.handshake.ndwire import generate_ndwire
from generators.handshake.join import generate_join
from generators.handshake.sink import generate_sink
import json


def generate_lsq(name, params):
    with open(params["config_file"], "r") as file:
        config = json.load(file)

    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    addr_type = SmvScalarType(params[ATTR_ADDR_BITWIDTH])
    num_load_ports = config["numLoadPorts"]
    num_store_ports = config["numStorePorts"]
    num_bbs = config["numBBs"]
    fifo_depth = config["fifoDepth"]
    load_groups = config["numLoads"]
    store_groups = config["numStores"]

    # the "master" config field determines if the LSQ is connected to a memory controller (slave LSQ) or not (master LSQ)
    if config["master"]:
        return _generate_lsq_master(
            name,
            addr_type,
            data_type,
            num_load_ports,
            num_store_ports,
            num_bbs,
            load_groups,
            store_groups,
            fifo_depth,
        )
    else:
        return _generate_lsq_slave(
            name,
            addr_type,
            data_type,
            num_load_ports,
            num_store_ports,
            num_bbs,
            load_groups,
            store_groups,
            fifo_depth,
        )


def group_index(index, group_list):
    partial_sum = 0
    for i, num_elements_in_group in enumerate(group_list):
        partial_sum += num_elements_in_group
        if index < partial_sum:
            return i


def _generate_lsq_master(
    name,
    addr_type,
    data_type,
    num_load_ports,
    num_store_ports,
    num_bbs,
    load_groups,
    store_groups,
    fifo_depth,
):
    # the signal order in the interface is different from HandshakeInterfaces.cpp because export-rtl.cpp
    # groups signals with the same name together
    ctrl = [f"ctrl_{n}_valid" for n in range(num_bbs)]
    load_addr = [f"ldAddr_{n}" for n in range(num_load_ports)] + [
        f"ldAddr_{n}_valid" for n in range(num_load_ports)
    ]
    store_addr = [f"stAddr_{n}" for n in range(num_store_ports)] + [
        f"stAddr_{n}_valid" for n in range(num_store_ports)
    ]
    store_data = [f"stData_{n}" for n in range(num_store_ports)] + [
        f"stData_{n}_valid" for n in range(num_store_ports)
    ]
    load_data = [f"ldData_{n}_ready" for n in range(num_load_ports)]
    lsq_in_ports = ", ".join(
        ["loadData", "memStart_valid"]
        + ctrl
        + load_addr
        + store_addr
        + store_data
        + ["ctrlEnd_valid"]
        + load_data
        + ["memEnd_ready"]
    )

    return f"""
MODULE {name} ({lsq_in_ports})
  VAR inner_mc_control : {name}__mc_control(memStart_valid, ctrlEnd_valid, memEnd_ready, all_requests_done);

  VAR all_requests_done : boolean;
  ASSIGN
  init(all_requests_done) := FALSE;
  next(all_requests_done) := all_requests_done ? TRUE : {{FALSE, TRUE}};

  {_generate_lsq_core(name, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, fifo_depth)}

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
{_generate_nd_load_access(f"{name}__nd_load_access", fifo_depth, addr_type, data_type)}
{_generate_nd_store_access(f"{name}__nd_store_access", fifo_depth, addr_type, data_type)}
"""


def _generate_lsq_slave(
    name,
    addr_type,
    data_type,
    num_load_ports,
    num_store_ports,
    num_bbs,
    load_groups,
    store_groups,
    fifo_depth,
):
    # the signal order in the interface is different from HandshakeInterfaces.cpp because export-rtl.cpp
    # groups signals with the same name together
    # Example:
    # this generator generates: ldAddr_0, ldAddr_1, ldAddr_0_valid, ldAddr_1_valid
    # export-rtl generates: ldAddr_0, ldAddr_0_valid, ldAddr_1, ldAddr_1_valid
    ctrl = [f"ctrl_{n}_valid" for n in range(num_bbs)]
    load_addr = [f"ldAddr_{n}" for n in range(num_load_ports)] + [
        f"ldAddr_{n}_valid" for n in range(num_load_ports)
    ]
    store_addr = [f"stAddr_{n}" for n in range(num_store_ports)] + [
        f"stAddr_{n}_valid" for n in range(num_store_ports)
    ]
    store_data = [f"stData_{n}" for n in range(num_store_ports)] + [
        f"stData_{n}_valid" for n in range(num_store_ports)
    ]
    load_data = [f"ldData_{n}_ready" for n in range(num_load_ports)]
    lsq_in_ports = ", ".join(
        ctrl
        + store_addr
        + store_data
        + load_addr
        + ["ldDataFromMC, ldDataFromMC_valid"]
        + load_data
        + ["ldAddrToMC_ready, stAddrToMC_ready, stDataToMC_ready"]
    )

    return f"""
MODULE {name} ({lsq_in_ports})

  DEFINE loadData := ldDataFromMC;
  {_generate_lsq_core(name, num_load_ports, num_store_ports, num_bbs, load_groups, store_groups, fifo_depth)}

  stDataToMC := {data_type.format_constant(0)};
  stDataToMC_valid := in_storeEn;
  stAddrToMC := {addr_type.format_constant(0)};
  stAddrToMC_valid := in_storeEn;
  ldAddrToMC := {addr_type.format_constant(0)};
  ldAddrToMC_valid := in_loadEn;
  ldDataFromMC_ready := TRUE;

{_generate_nd_load_access(f"{name}__nd_load_access", fifo_depth, addr_type, data_type)}
{_generate_nd_store_access(f"{name}__nd_store_access", fifo_depth, addr_type, data_type)}
"""


def _generate_lsq_core(
    name,
    num_load_ports,
    num_store_ports,
    num_bbs,
    load_groups,
    store_groups,
    fifo_depth,
):
    return f"""
  -- Non-deterministic access: entity that models all possible latencies that can occur when accessing memory
  VAR
  {"\n  ".join([f"inner_load_access_{n} : {name}__nd_load_access(ldAddr_{n}, ldAddr_{n}_valid, ldData_{n}_ready, loadData);" for n in range(num_load_ports)])}
  {"\n  ".join([f"inner_store_access_{n} : {name}__nd_store_access(stAddr_{n}, stAddr_{n}_valid, stData_{n}, stData_{n}_valid);" for n in range(num_store_ports)])}


  -- Checks if at least one access has happened
  DEFINE
  in_loadEn := {" | ".join([f"inner_load_access_{n}.ldData_valid" for n in range(num_load_ports)])};
  in_storeEn := {" | ".join([f"inner_store_access_{n}.memData_valid" for n in range(num_store_ports)])};
  
  -- output
  DEFINE
  -- for faster model checking we ignore the ctrl signal and set it to constant TRUEs
  {"\n  ".join([f"ctrl_{n}_ready := TRUE;" for n in range(num_bbs)])}

  {"\n  ".join([f"ldAddr_{n}_ready := inner_load_access_{n}.ldAddr_ready;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n} := inner_load_access_{n}.ldData;" for n in range(num_load_ports)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_load_access_{n}.ldData_valid;" for n in range(num_load_ports)])}

  {"\n  ".join([f"stAddr_{n}_ready := inner_store_access_{n}.stAddr_ready;" for n in range(num_store_ports)])}
  {"\n  ".join([f"stData_{n}_ready := inner_store_access_{n}.stData_ready;" for n in range(num_store_ports)])}
"""


def _generate_nd_load_access(name, fifo_depth, addr_type, data_type):
    # generates a non-deterministic load access, that simulates any stall that can happen from a read operation (memory stall or memory dependency)
    return f"""
MODULE {name} (ldAddr, ldAddr_valid, ldData_ready, data_from_mem)
  VAR inner_input_ndw : {name}__in_ndwire(ldAddr, ldAddr_valid, inner_fifo.ins_ready);
  VAR inner_fifo : {name}__fifo_break_dv(inner_input_ndw.outs_valid, inner_output_ndw.ins_ready);
  VAR inner_output_ndw : {name}__out_ndwire(data_from_mem, inner_fifo.outs_valid, ldData_ready);

  -- output
  DEFINE
  ldAddr_ready := inner_input_ndw.ins_ready;
  ldData := inner_output_ndw.outs;
  ldData_valid := inner_output_ndw.outs_valid;

  {generate_ndwire(f"{name}__in_ndwire", {ATTR_BITWIDTH: addr_type.bitwidth})}
  {generate_fifo_break_dv(f"{name}__fifo_break_dv", {ATTR_SLOTS: fifo_depth, ATTR_BITWIDTH: 0})}
  {generate_ndwire(f"{name}__out_ndwire", {ATTR_BITWIDTH: data_type.bitwidth})}
"""


def _generate_nd_store_access(name, fifo_depth, addr_type, data_type):
    # generates a non-deterministic store access, that simulates any stall that can happen from a write operation (memory stall or memory dependency)
    return f"""
MODULE {name} (stAddr, stAddr_valid, stData, stData_valid)
  VAR inner_addr_ndw : {name}__addr_ndwire(stAddr, stAddr_valid, inner_addr_fifo.ins_ready);
  VAR inner_addr_fifo : {name}__addr_fifo_break_dv(inner_addr_ndw.outs_valid, inner_join.ins_0_ready);
  VAR inner_data_ndw : {name}__data_ndwire(stData, stData_valid, inner_data_fifo.ins_ready);
  VAR inner_data_fifo : {name}__data_fifo_break_dv(inner_data_ndw.outs_valid, inner_join.ins_1_ready);
  VAR inner_join : {name}__join(inner_addr_fifo.outs_valid, inner_data_fifo.outs_valid, inner_sink_ndw.ins_valid);
  VAR inner_sink_ndw : {name}__data_ndwire(inner_data_ndw.outs, inner_join.outs_valid, inner_sink.ins_ready);
  VAR inner_sink : {name}__sink(inner_data_ndw.outs, inner_sink_ndw.outs_valid);

  -- output
  DEFINE
  stAddr_ready := inner_addr_ndw.ins_ready;
  stData_ready := inner_data_ndw.ins_ready;
  memData_valid := inner_sink_ndw.outs_valid;

  {generate_ndwire(f"{name}__addr_ndwire", {ATTR_BITWIDTH: addr_type.bitwidth})}
  {generate_fifo_break_dv(f"{name}__addr_fifo_break_dv", {ATTR_SLOTS: fifo_depth, ATTR_BITWIDTH: 0})}
  {generate_ndwire(f"{name}__data_ndwire", {ATTR_BITWIDTH: data_type.bitwidth})}
  {generate_fifo_break_dv(f"{name}__data_fifo_break_dv", {ATTR_SLOTS: fifo_depth, ATTR_BITWIDTH: 0})}
  {generate_join(f"{name}__join", {"size": 2})}
  {generate_sink(f"{name}__sink", {ATTR_BITWIDTH: data_type.bitwidth})}
"""
