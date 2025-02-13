import ast

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless
from generators.handshake.tehb import generate_tehb
from generators.support.elastic_fifo_inner import generate_elastic_fifo_inner

def generate_ofifo(name, params):
  port_types = ast.literal_eval(params["port_types"])
  data_type = VhdlScalarType(port_types["ins"])

  if data_type.has_extra_signals():
    if data_type.is_channel():
      return _generate_ofifo_signal_manager(name, params["size"], data_type)
    else:
      return _generate_ofifo_signal_manager_dataless(name, params["size"], data_type)
  elif data_type.is_channel():
    return _generate_ofifo(name, params["size"], data_type.bitwidth)
  else:
    return _generate_ofifo_dataless(name, params["size"])

def _generate_ofifo(name, size, bitwidth):
  tehb_name = f"{name}_tehb"
  fifo_name = f"{name}_fifo"

  dependencies = generate_elastic_fifo_inner(fifo_name, size, bitwidth) + \
    generate_tehb(tehb_name, {
      "port_types": str({
        "ins": f"!handshake.channel<i{bitwidth}>",
        "outs": f"!handshake.channel<i{bitwidth}>",
      })
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo
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
-- Architecture of ofifo
architecture arch of ofifo is
  signal tehb_valid, tehb_ready     : std_logic;
  signal fifo_valid, fifo_ready     : std_logic;
  signal tehb_dataOut, fifo_dataOut : std_logic_vector({bitwidth} - 1 downto 0);
begin
  tehb : entity work.{tehb_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs       => tehb_dataOut,
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins        => tehb_dataOut,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs       => fifo_dataOut,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs       <= fifo_dataOut;
  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
"""

  return dependencies + entity + architecture

def _generate_ofifo_dataless(name, size):
  tehb_name = f"{name}_tehb"
  fifo_name = f"{name}_fifo"

  dependencies = generate_elastic_fifo_inner(fifo_name, size) + \
    generate_tehb(tehb_name, {
      "port_types": str({
        "ins": "!handshake.control<>",
        "outs": "!handshake.control<>",
      })
    })

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo_dataless
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
-- Architecture of ofifo_dataless
architecture arch of {name} is
  signal tehb_valid, tehb_ready : std_logic;
  signal fifo_valid, fifo_ready : std_logic;
begin
  tehb : entity work.{tehb_name}(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => fifo_ready,
      -- outputs
      outs_valid => tehb_valid,
      ins_ready  => tehb_ready
    );

  fifo : entity work.{fifo_name}(arch)
    port map(
      --inputs
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_valid,
      outs_ready => outs_ready,
      --outputs
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );

  outs_valid <= fifo_valid;
  ins_ready  <= tehb_ready;
end architecture;
"""

  return dependencies + entity + architecture

def _generate_ofifo_signal_manager(name, size, data_type):
  inner_name = f"{name}_inner"

  bitwidth = data_type.bitwidth

  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for signal_name, signal_bitwidth in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_ofifo(f"{name}_inner", size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo signal manager
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
-- Architecture of ofifo signal manager
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

def _generate_ofifo_signal_manager_dataless(name, size, data_type):
  inner_name = f"{name}_inner"

  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in data_type.extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)

  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = _generate_ofifo(f"{name}_inner", size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ofifo signal manager dataless
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
-- Architecture of ofifo signal manager dataless
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
