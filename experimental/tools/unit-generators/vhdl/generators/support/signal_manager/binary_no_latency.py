from collections.abc import Callable

from generators.support.utils import VhdlScalarType, generate_extra_signal_ports, ExtraSignalMapping, generate_ins_concat_statements, generate_ins_concat_statements_dataless, generate_outs_concat_statements, generate_outs_concat_statements_dataless

# todo: can be reusable among various unit generators
extra_signal_logic = {
  "spec": """
  result_spec <= lhs_spec or rhs_spec;
""" # todo: generate_normal_spec_logic(["trueOut", "falseOut"], ["data", "condition"])
}

def generate_binary_no_latency_signal_manager_full(name: str, in_type: VhdlScalarType, out_type: VhdlScalarType, generate_inner: Callable[[str, int, int], str]) -> str:
  inner_name = f"{name}_inner"

  in_bitwidth = in_type.bitwidth
  out_bitwidth = out_type.bitwidth

  dependencies = generate_inner(inner_name, in_bitwidth, out_bitwidth)

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of binary_no_latency signal manager
entity {name} is
  port (
    [EXTRA_SIGNAL_PORTS]
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({in_bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({in_bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector({out_bitwidth} - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  # Add extra signal ports
  extra_signal_ports = generate_extra_signal_ports([
    ("lhs", "in"), ("rhs", "in"), ("result", "out")
  ], in_type.extra_signals)

  entity = entity.replace("    [EXTRA_SIGNAL_PORTS]\n", extra_signal_ports)

  architecture = f"""
-- Architecture of binary_no_latency signal manager
architecture arch of {name} is
begin

  -- list of logic for supported extra signals
  [EXTRA_SIGNAL_LOGIC]

  inner : entity work.{inner_name}(arch)
    port map(
      -- inputs
      clk          => clk,
      rst          => rst,
      lhs          => lhs,
      lhs_valid    => lhs_valid,
      rhs          => rhs,
      rhs_valid    => rhs_valid,
      result_ready => result_ready,
      -- outputs
      result       => result,
      result_valid => result_valid,
      lhs_ready    => lhs_ready,
      rhs_ready    => rhs_ready
    );
end architecture;
"""

  architecture = architecture.replace("  [EXTRA_SIGNAL_LOGIC]", "\n".join([
    extra_signal_logic[name] for name in in_type.extra_signals
  ]))

  return dependencies + entity + architecture

def generate_binary_no_latency_signal_manager(name: str, data_type: VhdlScalarType, generate_inner: Callable[[str, int], str]) -> str:
  return generate_binary_no_latency_signal_manager_full(
    name, data_type, data_type,
    lambda name, in_bitwidth, _: generate_inner(name, in_bitwidth))
