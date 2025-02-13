import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner

def generate_tfifo(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])

  if data_type.has_extra_signals():
    if data_type.is_channel():
      return _generate_tfifo_signal_manager(name, params["size"], data_type)
    else:
      return _generate_tfifo_signal_manager_dataless(name, params["size"], data_type)
  elif data_type.is_channel():
    return _generate_tfifo(name, params["size"], data_type.bitwidth)
  else:
    return _generate_tfifo_dataless(name, params["size"])

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
  bitwidth = data_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for signal_name, signal_bitwidth in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_tfifo(f"{name}_inner", size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo signal manager
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
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

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of tfifo signal manager
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{name}_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_conversion = generate_ins_concat_statements("ins", "ins_inner", extra_signal_mapping, bitwidth)
  outs_conversion = generate_outs_concat_statements("outs", "outs_inner", extra_signal_mapping, bitwidth)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture

def _generate_tfifo_signal_manager_dataless(name, size, data_type):
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)

  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_tfifo(f"{name}_inner", size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo signal manager dataless
entity {name} is
  port (
    clk, rst : in std_logic;
    [EXTRA_SIGNAL_PORTS]
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("ins", "in"), ("outs", "out")
  ], data_type.extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of tfifo signal manager dataless
architecture arch of {name} is
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{name}_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
"""

  ins_conversion = generate_ins_concat_statements_dataless("ins", "ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless("outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
    "  [EXTRA_SIGNAL_LOGIC]",
    ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture

if __name__ == "__main__":
  print(generate_tfifo("tfifo", {
    "size": 16,
    "data_type": "!handshake.channel<i32, [spec: i1, tag: i8]>"
  }))
  print(generate_tfifo("tfifo", {
    "size": 16,
    "data_type": "!handshake.control<[spec: i1, tag: i8]>"
  }))