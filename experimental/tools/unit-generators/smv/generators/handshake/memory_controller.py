from generators.support.utils import *


def generate_memory_controller(name, params):
  addr_type = SmvScalarType(params[ATTR_PORT_TYPES]["stAddr"])
  num_loads = params["num_loads"]
  num_stores = params["num_stores"]
  num_controls = params["num_controls"]

  if num_loads == 0:
    return _generate_mem_controller_loadless(name, num_stores, num_controls, addr_type)
  elif num_stores == 0:
    return _generate_mem_controller_storeless(name)
  else:
    _generate_mem_controller(name)


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

  {"\n  ".join([f"stAddr_{n}_ready := inner_arbiter.ready;" for n in range(num_stores)])}

  {"\n  ".join([f"stData_{n}_ready := inner_arbiter.ready;" for n in range(num_stores)])}

  loadEn := FALSE;
  loadAddr := {addr_type.format_constant(0)};
  storeEn := inner_arbiter.write_enable;
  storeAddr := inner_arbiter.write_address;
  storeData := inner_arbiter.data_to_memory;

  {_generate_write_memory_arbiter(f"{name}__write_memory_arbiter", num_stores)}
  {_generate_mc_control(f"{name}__mc_control")}
"""

def _generate_mem_controller_storeless(name):
  return f"""
MODULE {name}()

"""

def _generate_mem_controller(name):
  return f"""
MODULE {name}()

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
