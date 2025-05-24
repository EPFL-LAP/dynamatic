from generators.support.utils import *


def generate_memory_controller(name, params):
    data_type = SmvScalarType(params[ATTR_DATA_BITWIDTH])
    addr_type = SmvScalarType(params[ATTR_ADDR_BITWIDTH])
    ctrl_type = SmvScalarType(32)

    num_loads = params["num_loads"]
    num_stores = params["num_stores"]
    num_controls = params["num_controls"]

    if num_loads == 0:
        return _generate_mem_controller_loadless(name, num_stores, num_controls, data_type, addr_type, ctrl_type)
    elif num_stores == 0:
        return _generate_mem_controller_storeless(name, num_loads, data_type, addr_type)
    else:
        return _generate_mem_controller(name, num_loads, num_stores, num_controls, data_type, addr_type, ctrl_type)


def _generate_mem_controller_loadless(name, num_stores, num_controls, data_type, addr_type, ctrl_type):

    control_ports = [f"ctrl_{n}" for n in range(
        num_controls)] + [f"ctrl_{n}_valid" for n in range(num_controls)]
    store_address_ports = [f"stAddr_{n}" for n in range(
        num_stores)] + [f"stAddr_{n}_valid" for n in range(num_stores)]
    store_data_ports = [f"stData_{n}" for n in range(
        num_stores)] + [f"stData_{n}_valid" for n in range(num_stores)]
    mc_in_ports = ", ".join(["loadData", "memStart_valid"] + control_ports +
                            store_address_ports + store_data_ports + ["ctrlEnd_valid"] + ["memEnd_ready"])

    p_valid_ports = [
        f"stAddr_{n}_valid & stData_{n}_valid" for n in range(num_stores)]
    address_ports = [f"stAddr_{n}" for n in range(num_stores)]
    data_ports = [f"stData_{n}" for n in range(num_stores)]
    n_valid_ports = [f"TRUE" for _ in range(num_stores)]
    arbiter_args = ", ".join(
        p_valid_ports + address_ports + data_ports + n_valid_ports)

    return f"""
MODULE {name}({mc_in_ports})

  VAR
  inner_arbiter : {name}__write_memory_arbiter({arbiter_args});
  inner_mc_control : {name}__mc_control(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done);
  remainingStores : {ctrl_type.smv_type};

  ASSIGN
  init(remainingStores) := {ctrl_type.format_constant(0)};
  next(remainingStores) := case
    {"\n    ".join([f"ctrl_{n}_valid = TRUE : remainingStores + ctrl_{n} - stores_done;" for n in range(num_controls)])}
    TRUE : remainingStores - stores_done;
  esac;

  DEFINE
  stores_done := storeEn ? {ctrl_type.format_constant(1)} : {ctrl_type.format_constant(0)};
  all_requests_done := (remainingStores = {ctrl_type.format_constant(0)}) & {" & ".join([f"(ctrl_{n}_valid = FALSE)" for n in range(num_controls)])};

  // output
  DEFINE
  memStart_ready := inner_mc_control.memStart_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;

  -- ctrl_*_ready: ready signal of the channel that informs that a BB 
  -- has "ctrl_*" number of stores. The memory controller s internal counter is
  -- counted up every time when it receives a token from this channel. 
  -- We assume that the number never overflows, therefore this ready 
  -- signal is always TRUE.
  {"\n  ".join([f"ctrl_{n}_ready := TRUE;" for n in range(num_controls)])}


  -- stAddr_*_ready: ready signal for the store address ports. This signal 
  -- is derived from the arbiter s decision (the store port is ready 
  -- only if the store port is selected by the arbiter).
  {"\n  ".join([f"stAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_stores)])}

  -- stData_*_ready: ready signal for the store data ports. This signal
  -- is activated the same way as stAddr_*_ready.
  {"\n  ".join([f"stData_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_stores)])}

  loadEn := FALSE; -- in a loadless memory controller there are no loads
  loadAddr := {addr_type.format_constant(0)};
  storeEn := inner_arbiter.write_enable;
  storeAddr := inner_arbiter.write_address;
  storeData := inner_arbiter.data_to_memory;

  {_generate_write_memory_arbiter(f"{name}__write_memory_arbiter", num_stores, data_type, addr_type)}
  {_generate_mc_control(f"{name}__mc_control")}
"""


def _generate_mem_controller_storeless(name, num_loads, data_type, addr_type):
    load_address_ports = [f"ldAddr_{n}" for n in range(
        num_loads)] + [f"ldAddr_{n}_valid" for n in range(num_loads)]
    load_data_ports = [f"ldData_{n}_ready" for n in range(num_loads)]
    mc_in_ports = ", ".join(["loadData", "memStart_valid"] + load_address_ports + [
                            "ctrlEnd_valid"] + load_data_ports + ["memEnd_ready"])

    p_valid_ports = [f"ldAddr_{n}_valid" for n in range(num_loads)]
    address_ports = [f"ldAddr_{n}" for n in range(num_loads)]
    n_valid_ports = [f"ldData_{n}_ready" for n in range(num_loads)]
    arbiter_args = ", ".join(
        p_valid_ports + address_ports + n_valid_ports + ["loadData"])

    return f"""
MODULE {name}({mc_in_ports})

  VAR
  inner_arbiter : {name}__read_memory_arbiter({arbiter_args});
  inner_mc_control : {name}__mc_control(memStart_valid, memEnd_ready, ctrlEnd_valid, TRUE);

  // output
  DEFINE
  memStart_ready := inner_mc_control.memStart_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;

  -- ldAddr_*_ready: ready signal for the store address ports. This signal 
  -- is derived from the arbiter s decision (the load port is ready 
  -- only if the load port is selected by the arbiter).
  {"\n  ".join([f"ldAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_loads)])}

  -- ldData_* and ldData_*_valid: data and valid signals from memory that the
  -- arbiter selected for this port.
  {"\n  ".join([f"ldData_{n} := inner_arbiter.data_out_{n};" for n in range(num_loads)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_arbiter.valid_{n};" for n in range(num_loads)])}

  loadEn := inner_arbiter.read_enable;
  loadAddr := inner_arbiter.read_address;
  storeEn := FALSE;
  storeAddr := {addr_type.format_constant(0)};
  storeData := {data_type.format_constant(0)};

  {_generate_read_memory_arbiter(f"{name}__read_memory_arbiter", num_loads, data_type, addr_type)}
  {_generate_mc_control(f"{name}__mc_control")}
"""


def _generate_mem_controller(name, num_loads, num_stores, num_controls, data_type, addr_type, ctrl_type):
    control_ports = [f"ctrl_{n}" for n in range(
        num_controls)] + [f"ctrl_{n}_valid" for n in range(num_controls)]
    load_address_ports = [f"ldAddr_{n}" for n in range(
        num_loads)] + [f"ldAddr_{n}_valid" for n in range(num_loads)]
    load_data_ports = [f"ldData_{n}_ready" for n in range(num_loads)]
    store_address_ports = [f"stAddr_{n}" for n in range(
        num_stores)] + [f"stAddr_{n}_valid" for n in range(num_stores)]
    store_data_ports = [f"stData_{n}" for n in range(
        num_stores)] + [f"stData_{n}_valid" for n in range(num_stores)]
    mc_in_ports = ", ".join(["loadData", "memStart_valid"] + control_ports + load_address_ports +
                            store_address_ports + store_data_ports + ["ctrlEnd_valid"] + load_data_ports + ["memEnd_ready"])
    mc_loadless_in_ports = ", ".join(["loadData", "memStart_valid"] + control_ports +
                                     store_address_ports + store_data_ports + ["ctrlEnd_valid"] + ["memEnd_ready"])

    p_valid_ports = [f"ldAddr_{n}_valid" for n in range(num_loads)]
    address_ports = [f"ldAddr_{n}" for n in range(num_loads)]
    n_valid_ports = [f"ldData_{n}_ready" for n in range(num_loads)]
    arbiter_args = ", ".join(
        p_valid_ports + address_ports + n_valid_ports + ["loadData"])

    return f"""
MODULE {name}({mc_in_ports})

  VAR
  inner_mc_loadless : {name}__mc_loadless({mc_loadless_in_ports});
  inner_arbiter : {name}__read_memory_arbiter({arbiter_args});

  // outputs
  DEFINE
  memStart_ready := inner_mc_loadless.memStart_ready;
  memEnd_valid := inner_mc_loadless.memEnd_valid;
  ctrlEnd_ready := inner_mc_loadless.ctrlEnd_ready;

  -- ctrl_*_ready: ready signal of the channel that informs that a BB 
  -- has "ctrl_*" number of stores.
  {"\n  ".join([f"ctrl_{n}_ready := inner_mc_loadless.ctrl_{n}_ready;" for n in range(num_controls)])}

  -- ldAddr_*_ready: ready signal for the store address ports. This signal 
  -- is derived from the arbiter s decision (the load port is ready 
  -- only if the load port is selected by the arbiter).
  {"\n  ".join([f"ldAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_loads)])}

  -- ldData_* and ldData_*_valid: data and valid signals from memory that the
  -- arbiter selected for this port.
  {"\n  ".join([f"ldData_{n} := inner_arbiter.data_out_{n};" for n in range(num_loads)])}
  {"\n  ".join([f"ldData_{n}_valid := inner_arbiter.valid_{n};" for n in range(num_loads)])}
  
  -- stAddr_*_ready: ready signal for the store address ports. This signal 
  -- is derived from the arbiter s decision in inner_mc_loadless.
  {"\n  ".join([f"stAddr_{n}_ready := inner_mc_loadless.stAddr_{n}_ready;" for n in range(num_stores)])}

  -- stData_*_ready: ready signal for the store data ports. This signal
  -- is activated the same way as stAddr_*_ready.
  {"\n  ".join([f"stData_{n}_ready := inner_mc_loadless.stData_{n}_ready;" for n in range(num_stores)])}

  loadEn := inner_arbiter.read_enable;
  loadAddr := inner_arbiter.read_address;
  storeEn := inner_mc_loadless.storeEn;
  storeAddr := inner_mc_loadless.storeAddr;
  storeData := inner_mc_loadless.storeData;

  {_generate_mem_controller_loadless(f"{name}__mc_loadless", num_stores, num_controls, data_type, addr_type, ctrl_type)}
  {_generate_read_memory_arbiter(f"{name}__read_memory_arbiter", num_loads, data_type, addr_type)}
"""


def _generate_mc_control(name):
    return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done)
  -- The mc_control manages the signals that control when the circuit is allowed to access memory: it controls
  -- start and end of memory transactions and acknowledges completion.

  -- Handshake Signals:
  -- - memStart_valid / memStart_ready: Controls the start of memory operations.
  -- - memEnd_valid / memEnd_ready: Indicates the completion of memory requests.
  -- - ctrlEnd_valid / ctrlEnd_ready: Used to signal and acknowledge that no more memory requests will be issued.
  -- - allRequestsDone: Flags that all pending memory operations have completed.
  
  VAR
  memStart_ready_in : boolean;
  memEnd_valid_in : boolean;
  ctrlEnd_ready_in : boolean;

  ASSIGN
  init(memStart_ready_in) := TRUE;
  next(memStart_ready_in) := case
    memEnd_valid_in & memEnd_ready : TRUE;
    memStart_valid & memStart_ready_in : FALSE;
    TRUE : memStart_ready_in;
  esac;
  init(memEnd_valid_in) := FALSE;
  next(memEnd_valid_in) := case
    memEnd_valid_in & memEnd_ready : FALSE;
    ctrlEnd_valid & all_requests_done : TRUE;
    TRUE : memEnd_valid_in;
  esac;
  init(ctrlEnd_ready_in) := FALSE;
  next(ctrlEnd_ready_in) := case
    ctrlEnd_valid & ctrlEnd_ready_in : FALSE;
    ctrlEnd_valid & all_requests_done : TRUE;
    TRUE : ctrlEnd_ready_in;
  esac;

  // outputs
  DEFINE
  memStart_ready := memStart_ready_in;
  memEnd_valid := memEnd_valid_in;
  ctrlEnd_ready := ctrlEnd_ready_in;
"""


def _generate_write_memory_arbiter(name, num_stores, data_type, addr_type):
    p_valid_ports = [f"pValid_{n}" for n in range(num_stores)]
    address_ports = [f"address_in_{n}" for n in range(num_stores)]
    data_ports = [f"data_in_{n}" for n in range(num_stores)]
    n_valid_ports = [f"nReady_{n}" for n in range(num_stores)]
    arbiter_in_ports = ", ".join(
        p_valid_ports + address_ports + data_ports + n_valid_ports)

    return f"""
MODULE {name}({arbiter_in_ports})
  -- The write_memory arbiter selects a write address and data based on a priority list
  -- (writes that are ready to be executed are prioritized). 
  VAR
  priority_gen : {name}__priority({", ".join([f"pValid_{n}, nReady_{n}" for n in range(num_stores)])});
  {"\n  ".join([f"valid_{n}_in : boolean;" for n in range(num_stores)])}

  ASSIGN
  {"\n  ".join([f"init(valid_{n}_in) := FALSE;" for n in range(num_stores)])}
  {"\n  ".join([f"next(valid_{n}_in) :=  priority_gen.priority_{n};" for n in range(num_stores)])}


  DEFINE
  write_addr_in := case
    {"\n    ".join([f"priority_gen.priority_{n} = TRUE : address_in_{n};" for n in range(num_stores)])}
    TRUE: {addr_type.format_constant(0)};
  esac;
  write_data_in := case
    {"\n    ".join([f"priority_gen.priority_{n} = TRUE : data_in_{n};" for n in range(num_stores)])}
    TRUE: {data_type.format_constant(0)};
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ready_{n} := priority_gen.priority_{n} & nReady_{n};" for n in range(num_stores)])}
  {"\n  ".join([f"valid_{n} := valid_{n}_in;" for n in range(num_stores)])}
  write_address := write_addr_in;
  data_to_memory := write_data_in;
  write_enable := {" | ".join([f"priority_gen.priority_{n}" for n in range(num_stores)])};
  enable := write_enable;

{_generate_priority(f"{name}__priority", num_stores)}
"""


def _generate_read_memory_arbiter(name, num_loads, data_type, addr_type):
    p_valid_ports = [f"pValid_{n}" for n in range(num_loads)]
    address_ports = [f"address_in_{n}" for n in range(num_loads)]
    n_valid_ports = [f"nReady_{n}" for n in range(num_loads)]
    arbiter_in_ports = ", ".join(
        p_valid_ports + address_ports + n_valid_ports + ["data_from_memory"])

    return f"""
MODULE {name}({arbiter_in_ports})
  -- The read_memory arbiter selects a read address based on a priority list
  -- (reads that are ready to be executed are prioritized). The memory arbiter internally
  -- saves the previous selection and the respective data in the case that the circuit
  -- is not ready to recieve data from memory.

  VAR
  priority_gen : {name}__priority({", ".join([f"pValid_{n}, nReady_{n}" for n in range(num_loads)])});
  {"\n  ".join([f"valid_{n}_in : boolean;" for n in range(num_loads)])}
  {"\n  ".join([f"sel_prev{n} : boolean;" for n in range(num_loads)])}
  {"\n  ".join([f"out_reg_{n} : {data_type.smv_type};" for n in range(num_loads)])}

  ASSIGN
  {"\n  ".join([f"init(valid_{n}_in) := FALSE;" for n in range(num_loads)])}
  {"\n  ".join([f"next(valid_{n}_in) :=  case\n    priority_gen.priority_{n} : TRUE;\n    nReady_{n} : FALSE;\n    TRUE : valid_{n}_in;\n  esac;" for n in range(num_loads)])}

  {"\n  ".join([f"init(sel_prev{n}) := FALSE;" for n in range(num_loads)])}
  {"\n  ".join([f"next(sel_prev{n}) := priority_gen.priority_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"init(out_reg_{n}) := {data_type.format_constant(0)};" for n in range(num_loads)])}
  {"\n  ".join([f"next(out_reg_{n}) := sel_prev{n} ? data_from_memory : out_reg_{n};" for n in range(num_loads)])}

  // output
  DEFINE
  {"\n  ".join([f"ready_{n} := priority_gen.priority_{n} & nReady_{n};" for n in range(num_loads)])}
  {"\n  ".join([f"valid_{n} := valid_{n}_in;" for n in range(num_loads)])}
  {"\n  ".join([f"data_out_{n} := sel_prev{n} ? data_from_memory : out_reg_{n};" for n in range(num_loads)])}
  
  read_enable := {" | ".join([f"priority_gen.priority_{n}" for n in range(num_loads)])};
  read_address := case
    {"\n    ".join([f"priority_gen.priority_{n} = TRUE : address_in_{n};" for n in range(num_loads)])}
    TRUE: {addr_type.format_constant(0)};
  esac;

{_generate_priority(f"{name}__priority", num_loads)}
"""


def _generate_priority(name, size):
    return f"""
MODULE {name}({", ".join([f"req_{n}, data_ready_{n}" for n in range(size)])})
  -- Generates the priority list for the memory arbiters. An index can be selected only
  -- if the respective inputs are ready.

  DEFINE
  que_el_{0} := req_{0} & data_ready_{0};
  prior_0 := que_el_{0};
  {"\n  ".join([f"prior_{n + 1} := que_el_{n} | prior_{n};\n  que_el_{n + 1} := req_{n + 1} & data_ready_{n + 1} & !prior_{n + 1};" for n in range(size - 1)])}

  // output
  DEFINE
  {"\n  ".join([f"priority_{n} := que_el_{n};" for n in range(size)])}

"""
