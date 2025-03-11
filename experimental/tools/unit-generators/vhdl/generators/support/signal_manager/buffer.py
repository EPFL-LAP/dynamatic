from collections.abc import Callable

from generators.support.utils import generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless


def generate_buffer_like_signal_manager_full(name: str, size: int, bitwidth: int, extra_signals: dict[str, int], generate_inner: Callable[[str, int, int], str]) -> str:
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  # Specify offset for data bitwidth
  extra_signal_mapping = ExtraSignalMapping(offset=bitwidth)
  for signal_name, signal_type in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_type)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = generate_inner(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of buffer-like signal manager
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
  ], extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of buffer-like signal manager
architecture arch of {name} is
  -- Concatenated data and extra signals
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
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

  # Concatenate data and extra signals based on extra signal mapping
  ins_conversion = generate_ins_concat_statements(
      "ins", "ins_inner", extra_signal_mapping, bitwidth)
  outs_conversion = generate_outs_concat_statements(
      "outs", "outs_inner", extra_signal_mapping, bitwidth)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture


def generate_buffer_like_signal_manager(name: str, bitwidth: int, extra_signals: dict[str, int], generate_inner: Callable[[str, int], str]) -> str:
  return generate_buffer_like_signal_manager_full(
      name, 0, bitwidth, extra_signals,
      lambda name, _, bitwidth: generate_inner(name, bitwidth))


def generate_buffer_like_signal_manager_dataless_full(name: str, size: int, extra_signals: dict[str, int], generate_inner: Callable[[str, int, int], str]) -> str:
  inner_name = f"{name}_inner"

  # Construct extra signal mapping
  extra_signal_mapping = ExtraSignalMapping()
  for signal_name, signal_bitwidth in extra_signals.items():
    extra_signal_mapping.add(signal_name, signal_bitwidth)
  full_bitwidth = extra_signal_mapping.total_bitwidth

  dependencies = generate_inner(inner_name, size, full_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of buffer-like signal manager dataless
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
  ], extra_signals)
  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of buffer-like signal manager dataless
architecture arch of {name} is
  -- Concatenated extra signals
  signal ins_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
  signal outs_inner : std_logic_vector({full_bitwidth} - 1 downto 0);
begin
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
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

  # Concatenate extra signals based on extra signal mapping
  ins_conversion = generate_ins_concat_statements_dataless(
      "ins", "ins_inner", extra_signal_mapping)
  outs_conversion = generate_outs_concat_statements_dataless(
      "outs", "outs_inner", extra_signal_mapping)

  architecture = architecture.replace(
      "  [EXTRA_SIGNAL_LOGIC]",
      ins_conversion + outs_conversion
  )

  return dependencies + entity + architecture


def generate_buffer_like_signal_manager_dataless(name: str, extra_signals: dict[str, int], generate_inner: Callable[[str, int], str]) -> str:
  return generate_buffer_like_signal_manager_dataless_full(
      name, 0, extra_signals,
      lambda name, _, bitwidth: generate_inner(name, bitwidth))
