from generators.support.utils import *


def generate_elastic_fifo_inner(name, params):
    slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
    data_type = SmvScalarType(params[ATTR_BITWIDTH])

    if data_type.bitwidth == 0:
        return _generate_elastic_fifo_inner_dataless(name, slots)
    else:
        return _generate_elastic_fifo_inner(name, slots, data_type)


def _generate_elastic_fifo_inner_dataless(name, slots):
    return f"""
MODULE {name}(ins_valid, outs_ready)
  VAR
  full : boolean;
  empty : boolean;
  head : 0..{slots - 1};
  tail : 0..{slots - 1};

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  ASSIGN
  init(tail) := 0;
  next(tail) := case
    {"\n    ".join([f"write_en & (tail = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : tail;
  esac;

  init(head) := 0;
  next(head) := case
    {"\n    ".join([f"read_en & (head = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : head;
  esac;

  init(full) := FALSE;
  next(full) := case
    write_en & !read_en : case
      tail < {slots - 1} & head = tail + 1: TRUE;
      tail = {slots - 1} & head = 0 : TRUE;
      TRUE : full;
    esac;
    !write_en & read_en : FALSE;
    TRUE : full;
  esac;

  init(empty) := TRUE;
  next(empty) := case
    !write_en & read_en : case
      head < {slots - 1} & tail = head + 1: TRUE;
      head = {slots - 1} & tail = 0 : TRUE;
      TRUE : empty;
    esac;
    write_en & !read_en : FALSE;
    TRUE : empty;
  esac;

  -- output
  DEFINE
  ins_ready := !full | outs_ready;
  outs_valid := !empty;
"""


def _generate_elastic_fifo_inner(name, slots, data_type):
    return f"""
MODULE {name}(ins, ins_valid, outs_ready)
  {"\n  ".join([f"VAR mem_{n} : {data_type};" for n in range(slots)])}
  VAR
  full : boolean;
  empty : boolean;
  head : 0..{slots - 1};
  tail : 0..{slots - 1};

  DEFINE
  read_en := outs_ready & !empty;
  write_en := ins_valid & (!full | outs_ready);

  ASSIGN
  init(tail) := 0;
  next(tail) := case
    {"\n    ".join([f"write_en & (tail = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : tail;
  esac;

  init(head) := 0;
  next(head) := case
    {"\n    ".join([f"read_en & (head = {n}) : {(n + 1) % slots};" for n in range(slots)])}
    TRUE : head;
  esac;

  {"\n  ".join([f"""ASSIGN
  init(mem_{n}) := {data_type.format_constant(0)};
  next(mem_{n}) := write_en & (tail = {n}) ? ins : mem_{n};""" for n in range(slots)])}

  init(full) := FALSE;
  next(full) := case
    write_en & !read_en : case
      tail < {slots - 1} & head = tail + 1: TRUE;
      tail = {slots - 1} & head = 0 : TRUE;
      TRUE : full;
    esac;
    !write_en & read_en : FALSE;
    TRUE : full;
  esac;

  init(empty) := TRUE;
  next(empty) := case
    !write_en & read_en : case
      head < {slots - 1} & tail = head + 1: TRUE;
      head = {slots - 1} & tail = 0 : TRUE;
      TRUE : empty;
    esac;
    write_en & !read_en : FALSE;
    TRUE : empty;
  esac;

  -- output
  DEFINE
  ins_ready := !full | outs_ready;
  outs_valid := !empty;
  outs := case
    {"\n    ".join([f"head = {n} : mem_{n};" for n in range(slots)])}
    TRUE : mem_0;
  esac;
"""
