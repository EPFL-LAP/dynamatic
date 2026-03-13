from generators.support.utils import *

def generate_elastic_fifo_inner(name, params):
  slots = params[ATTR_SLOTS] if ATTR_SLOTS in params else 1
  data_type = SmvScalarType(params[ATTR_DATA_TYPE])

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

def generate_free_tags_fifo(name, params):
  bitwidth = params["bitwidth"]
  fifo_depth = params["fifo_depth"]

  return _generate_free_tags_fifo(name, bitwidth, fifo_depth)


def _generate_free_tags_fifo(name, bitwidth, fifo_depth):
  fifo_name = f"{name}_fifo"

  dependencies = \
      generate_elastic_fifo_inner(fifo_name, {
          "size": fifo_depth,
          "bitwidth": bitwidth,
      })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of free_tags_fifo
entity {name} is 
port (
        clk           : in std_logic;
        rst           : in std_logic;
        ins           : in  std_logic_vector({bitwidth} - 1 downto 0);
        outs          : out std_logic_vector({bitwidth} - 1 downto 0);
        ins_valid     : in  std_logic;
        outs_ready    : in  std_logic;
        outs_valid    : out std_logic;
        ins_ready     : out std_logic
);
end entity;
"""
  architecture = f"""
-- Architecture of free_tags_fifo
architecture arch of {name} is
    signal mux_sel : std_logic;
    signal fifo_valid, fifo_ready : STD_LOGIC;
    signal fifo_pvalid, fifo_nready : STD_LOGIC;
    signal fifo_in, fifo_out: std_logic_vector({bitwidth}-1 downto 0);
begin
    
    process (mux_sel, fifo_out, ins) is
        begin
            if (mux_sel = '1') then
                outs <= fifo_out;
            else
                outs <= ins;
            end if;
    end process;

    outs_valid <= ins_valid or fifo_valid;    --fifo_valid is 0 only if fifo is empty
    ins_ready <= fifo_ready or outs_ready;
    fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid); --store in FIFO if next is not ins_ready or FIFO is already outputting something
    mux_sel <= fifo_valid;

    fifo_nready <= outs_ready;
    fifo_in <= ins;

    fifo: entity work.{fifo_name}(arch)
        port map (
        --inputs
            clk => clk, 
            rst => rst,
            ins =>fifo_in, 
            ins_valid  => fifo_pvalid, 
            outs_ready => fifo_nready,    
            
        --outputs
            outs => fifo_out,
            outs_valid => fifo_valid,
            ins_ready => fifo_ready   
        );
end architecture;
"""
  return dependencies + entity + architecture