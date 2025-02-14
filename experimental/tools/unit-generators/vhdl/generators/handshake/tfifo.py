import ast

from generators.support.utils import VhdlScalarType
from generators.support.signal_manager.buffer import generate_buffer_like_signal_manager_full, generate_buffer_like_signal_manager_dataless_full
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner

def generate_tfifo(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])

  if data_type.has_extra_signals():
    if data_type.is_channel():
      return _generate_tfifo_signal_manager(name, params["num_slots"], data_type)
    else:
      return _generate_tfifo_signal_manager_dataless(name, params["num_slots"], data_type)
  elif data_type.is_channel():
    return _generate_tfifo(name, params["num_slots"], data_type.bitwidth)
  else:
    return _generate_tfifo_dataless(name, params["num_slots"])

def _generate_tfifo(name, size, bitwidth):
  fifo_inner_name = f"{name}_fifo"
  dependencies = generate_elastic_fifo_inner(fifo_inner_name, size, bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of tfifo
architecture arch of {name} is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
  signal fifo_in, fifo_out        : std_logic_vector({bitwidth} - 1 downto 0);
begin

  process (mux_sel, fifo_out, ins) is
  begin
    if (mux_sel = '1') then
      outs <= fifo_out;
    else
      outs <= ins;
    end if;
  end process;

  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;

  fifo_nready <= outs_ready;
  fifo_in     <= ins;

  fifo : entity work.{fifo_inner_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => fifo_in,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs       => fifo_out,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;
"""

  return dependencies + entity + architecture

def _generate_tfifo_dataless(name, size):
  fifo_inner_name = f"{name}_fifo"
  dependencies = generate_elastic_fifo_inner(fifo_inner_name, size)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo_dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of tfifo_dataless
architecture arch of {name} is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
begin
  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;
  fifo_nready <= outs_ready;

  fifo : entity work.{fifo_inner_name}(arch) generic map (NUM_SLOTS)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;
"""

  return dependencies + entity + architecture

def _generate_tfifo_signal_manager(name, size, data_type):
  return generate_buffer_like_signal_manager_full(name, size, data_type, _generate_tfifo)

def _generate_tfifo_signal_manager_dataless(name, size, data_type):
  return generate_buffer_like_signal_manager_dataless_full(name, size, data_type, _generate_tfifo)

if __name__ == "__main__":
  print(generate_tfifo("tfifo", {
    "num_slots": 16,
    "data_type": "!handshake.channel<i32, [spec: i1, tag: i8]>"
  }))
  print(generate_tfifo("tfifo", {
    "num_slots": 16,
    "data_type": "!handshake.control<[spec: i1, tag: i8]>"
  }))
