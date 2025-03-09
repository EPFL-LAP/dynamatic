from generators.support.utils import *


def generate_memory_controller(name, params):
  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["stAddr"]) if "stAddr" in params[ATTR_PORT_TYPES].keys() else SmvScalarType(params[ATTR_PORT_TYPES]["ldAddr"])
  data_type = SmvScalarType(params[ATTR_PORT_TYPES]["ldData"]) if "ldData" in params[ATTR_PORT_TYPES].keys() else None
  num_loads = params["num_loads"]
  num_stores = params["num_stores"]
  num_controls = params["num_controls"]

  if num_loads == 0:
    return _generate_mem_controller_loadless(name, num_stores, num_controls, addr_type)
  elif num_stores == 0:
    return _generate_mem_controller_storeless(name, num_loads, data_type, addr_type)
  else:
    return _generate_mem_controller(name, num_loads, num_stores, num_controls, data_type, addr_type)


def _generate_mem_controller_loadless(name, num_stores, num_controls, addr_type):
  return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, {", ".join([f"ctrl_{n}, ctrl_{n}_valid" for n in range(num_controls)])}, 
  {", ".join([f"stAddr_{n}, stAddr_{n}_valid" for n in range(num_stores)])}, {", ".join([f"stData_{n}, stData_{n}_valid" for n in range(num_stores)])}, loadData)

  VAR
  inner_arbiter : {name}__write_memory_arbiter({", ".join([f"stAddr_{n}_valid & stData_{n}_valid" for n in range(num_stores)])},
    {", ".join([f"stAddr_{n}" for n in range(num_stores)])}, {", ".join([f"stData_{n}" for n in range(num_stores)])}, {", ".join([f"TRUE" for _ in range(num_stores)])});
  inner_mc_control : {name}__mc_control(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done);
  remainingStores : integer;

  ASSIGN
  init(remainingStores) := 0;
  next(remainingStores) := case
    {"\n    ".join([f"ctrl_{n}_valid = TRUE : remainingStores + toint(ctrl_{n}) - stores_done;" for n in range(num_controls)])}
    TRUE : remainingStores - stores_done;
  esac;

  DEFINE
  stores_done := storeEn ? 1 : 0;
  all_requests_done := (remainingStores = 0) & {" & ".join([f"(ctrl_{n}_valid = FALSE)" for n in range(num_controls)])};

  // output
  DEFINE
  memStart_ready := inner_mc_control.memStart_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;

  {"\n  ".join([f"ctrl_{n}_ready := TRUE;" for n in range(num_controls)])}

  {"\n  ".join([f"stAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_stores)])}

  {"\n  ".join([f"stData_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_stores)])}

  loadEn := FALSE;
  loadAddr := {addr_type.format_constant(0)};
  storeEn := inner_arbiter.write_enable;
  storeAddr := inner_arbiter.write_address;
  storeData := inner_arbiter.data_to_memory;

  {_generate_write_memory_arbiter(f"{name}__write_memory_arbiter", num_stores)}
  {_generate_mc_control(f"{name}__mc_control")}
"""

def _generate_mem_controller_storeless(name, num_loads, data_type, addr_type):
  return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, {", ".join([f"ldAddr_{n}, ldAddr_{n}_valid" for n in range(num_loads)])}, {", ".join([f"ldData_{n}_ready" for n in range(num_loads)])}, loadData)

  VAR
  inner_arbiter : {name}__read_memory_arbiter({", ".join([f"ldAddr_{n}_valid" for n in range(num_loads)])}, {", ".join([f"ldAddr_{n}" for n in range(num_loads)])},
    {", ".join([f"ldData_{n}_ready" for n in range(num_loads)])}, {", ".join([f"ldData_{n}" for n in range(num_loads)])}, loadData);
  inner_mc_control : {name}__mc_control(memStart_valid, memEnd_ready, ctrlEnd_valid, TRUE);

  // output
  DEFINE
  memStart_ready := inner_mc_control.memStart_ready;
  memEnd_valid := inner_mc_control.memEnd_valid;
  ctrlEnd_ready := inner_mc_control.ctrlEnd_ready;

  {"\n  ".join([f"ldAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"ldData_{n} := inner_arbiter.data_out_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"ldData_{n}_valid := inner_arbiter.valid_{n};" for n in range(num_loads)])}

  loadEn := inner_arbiter.read_enable;
  loadAddr := inner_arbiter.read_address;
  storeEn := FALSE;
  storeAddr := {addr_type.format_constant(0)};
  storeData := {addr_type.format_constant(0)};

  {_generate_read_memory_arbiter(f"{name}__read_memory_arbiter", num_loads, data_type)}
  {_generate_mc_control(f"{name}__mc_control")}
"""

def _generate_mem_controller(name, num_loads, num_stores, num_controls, data_type, addr_type):
  return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, {", ".join([f"ctrl_{n}, ctrl_{n}_valid" for n in range(num_controls)])},
  {", ".join([f"ldAddr_{n}, ldAddr_{n}_valid" for n in range(num_loads)])}, {", ".join([f"ldData_{n}_ready" for n in range(num_loads)])},
  {", ".join([f"stAddr_{n}, stAddr_{n}_valid" for n in range(num_stores)])}, {", ".join([f"stData_{n}, stData_{n}_valid" for n in range(num_stores)])}, loadData)

  VAR
  inner_mc_loadless : {name}__mc_loadless(memStart_valid, memEnd_ready, ctrlEnd_valid, {", ".join([f"ctrl_{n}, ctrl_{n}_valid" for n in range(num_controls)])},
    {", ".join([f"stAddr_{n}, stAddr_{n}_valid" for n in range(num_stores)])}, {", ".join([f"stData_{n}, stData_{n}_valid" for n in range(num_stores)])}, loadData);
  inner_arbiter : {name}__read_memory_arbiter({", ".join([f"ldAddr_{n}_valid" for n in range(num_loads)])}, {", ".join([f"ldAddr_{n}" for n in range(num_loads)])},
    {", ".join([f"ldData_{n}_ready" for n in range(num_loads)])}, {", ".join([f"ldData_{n}" for n in range(num_loads)])}, loadData);

  // outputs
  DEFINE
  memStart_ready := inner_mc_loadless.memStart_ready;
  memEnd_valid := inner_mc_loadless.memEnd_valid;
  ctrlEnd_ready := inner_mc_loadless.ctrlEnd_ready;

  {"\n  ".join([f"ctrl_{n}_ready := inner_mc_loadless.ctrl_{n}_ready;" for n in range(num_controls)])}

  {"\n  ".join([f"ldAddr_{n}_ready := inner_arbiter.ready_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"ldData_{n} := inner_arbiter.data_out_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"ldData_{n}_valid := inner_arbiter.valid_{n};" for n in range(num_loads)])}
  
  {"\n  ".join([f"stAddr_{n}_ready := inner_mc_loadless.ready_{n};" for n in range(num_stores)])}

  {"\n  ".join([f"stData_{n}_ready := inner_mc_loadless.ready_{n};" for n in range(num_stores)])}

  loadEn := inner_arbiter.read_enable;
  loadAddr := inner_arbiter.read_address;
  storeEn := inner_mc_loadless.write_enable;
  storeAddr := inner_mc_loadless.write_address;
  storeData := inner_mc_loadless.data_to_memory;

  {_generate_mem_controller_loadless(f"{name}__mc_loadless", num_stores, num_controls, addr_type)}
  {_generate_read_memory_arbiter(f"{name}__read_memory_arbiter", num_loads, data_type)}
"""

def _generate_mc_control(name):
  return f"""
MODULE {name}(memStart_valid, memEnd_ready, ctrlEnd_valid, all_requests_done)

  VAR
  memStart_ready_in : boolean;
  memEnd_valid_in : boolean;
  ctrlEnd_ready_in : boolean;

  ASSIGN
  init(memStart_ready_in) := TRUE;
  next(memStart_ready_in) := case
    memStart_valid = TRUE & memStart_ready_in = TRUE : FALSE;
    memEnd_valid_in = TRUE & memEnd_ready = TRUE : TRUE;
    TRUE : memStart_ready_in;
  esac;
  init(memEnd_valid_in) := FALSE;
  next(memEnd_valid_in) := case
    ctrlEnd_valid = TRUE & all_requests_done = TRUE : TRUE;
    memEnd_valid_in = TRUE & memEnd_ready = TRUE : FALSE;
    TRUE : memEnd_valid_in;
  esac;
  init(ctrlEnd_ready_in) := FALSE;
  next(ctrlEnd_ready_in) := case
    ctrlEnd_valid = TRUE & all_requests_done = TRUE : TRUE;
    ctrlEnd_valid = TRUE & ctrlEnd_ready_in = TRUE : FALSE;
    TRUE : ctrlEnd_ready_in;
  esac;

  // outputs
  DEFINE
  memStart_ready := memStart_ready_in;
  memEnd_valid := memEnd_valid_in;
  ctrlEnd_ready := ctrlEnd_ready_in;
"""

def _generate_write_memory_arbiter(name, num_stores):
  return f"""
MODULE {name}({", ".join([f"pValid_{n}" for n in range(num_stores)])}, {", ".join([f"address_in_{n}" for n in range(num_stores)])},
  {", ".join([f"data_in_{n}" for n in range(num_stores)])}, {", ".join([f"nReady_{n}" for n in range(num_stores)])})

  VAR
  {"\n  ".join([f"valid_{n}_in : boolean;" for n in range(num_stores)])}

  ASSIGN
  {"\n  ".join([f"init(valid_{n}_in) := FALSE;" for n in range(num_stores)])}
  {"\n  ".join([f"next(valid_{n}_in) :=  que_el_{n};" for n in range(num_stores)])}


  DEFINE
  que_el_{0} := pValid_{0} & nReady_{0};
  {"\n  ".join([f"prior_{n + 1} := que_el_{n} | prior_{n};\n  que_el_{n + 1} := pValid_{n + 1} & nReady_{n + 1} & !prior_{n + 1};" for n in range(num_stores - 1)])}
  write_addr_in := case
    {"\n    ".join([f"que_el_{n} = TRUE : address_in_{n};" for n in range(num_stores)])}
    TRUE: 0;
  esac;
  write_data_in := case
    {"\n    ".join([f"que_el_{n} = TRUE : data_in_{n};" for n in range(num_stores)])}
    TRUE: 0;
  esac;

  // output
  DEFINE
  {"\n  ".join([f"ready_{n} := que_el_{n} & nReady_{n};" for n in range(num_stores)])}
  {"\n  ".join([f"valid_{n} := valid_{n}_in;" for n in range(num_stores)])}
  write_address := write_addr_in;
  data_to_memory := write_data_in;
  write_enable := {" | ".join([f"que_el_{n}" for n in range(num_stores)])};
  enable := write_enable;
"""

def _generate_read_memory_arbiter(name, num_loads, data_type):
  return f"""
MODULE {name}({", ".join([f"pValid_{n}" for n in range(num_loads)])}, {", ".join([f"address_in_{n}" for n in range(num_loads)])}, {", ".join([f"nReady_{n}" for n in range(num_loads)])}, data_from_memory)

  VAR
  {"\n  ".join([f"valid_{n}_in : boolean;" for n in range(num_loads)])}
  {"\n  ".join([f"sel_prev{n} : boolean;" for n in range(num_loads)])}
  {"\n  ".join([f"out_reg_{n} : {data_type.smv_type};" for n in range(num_loads)])}

  ASSIGN
  {"\n  ".join([f"init(valid_{n}_in) := FALSE;" for n in range(num_loads)])}
  {"\n  ".join([f"next(valid_{n}_in) :=  case\n    que_el_{n} : TRUE;\n    nReady_{n} : FALSE;\n    TRUE : valid_{n}_in;\n  esac;" for n in range(num_loads)])}

  {"\n  ".join([f"init(sel_prev{n}) := FALSE;" for n in range(num_loads)])}
  {"\n  ".join([f"next(sel_prev{n}) := que_el_{n};" for n in range(num_loads)])}

  {"\n  ".join([f"init(out_reg_{n}) := {data_type.format_constant(0)};" for n in range(num_loads)])}
  {"\n  ".join([f"next(out_reg_{n}) := sel_prev{n} ? data_from_memory : out_reg_{n};" for n in range(num_loads)])}


  DEFINE
  que_el_{0} := pValid_{0} & nReady_{0};
  {"\n  ".join([f"prior_{n + 1} := que_el_{n} | prior_{n};\n  que_el_{n + 1} := pValid_{n + 1} & nReady_{n + 1} & !prior_{n + 1};" for n in range(num_loads - 1)])}

  // output
  DEFINE
  {"\n  ".join([f"ready_{n} := que_el_{n} & nReady_{n};" for n in range(num_loads)])}
  {"\n  ".join([f"valid_{n} := valid_{n}_in;" for n in range(num_loads)])}
  {"\n  ".join([f"data_out_{n} := sel_prev{n} ? data_from_memory : out_reg_{n};" for n in range(num_loads)])}
  
  read_enable := {" | ".join([f"que_el_{n}" for n in range(num_loads)])};
  read_addr := case
    {"\n    ".join([f"que_el_{n} = TRUE : address_in_{n};" for n in range(num_loads)])}
    TRUE: 0;
  esac;

"""
