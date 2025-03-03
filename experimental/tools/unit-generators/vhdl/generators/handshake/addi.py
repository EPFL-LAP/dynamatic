from generators.support.utils import VhdlScalarType, generate_extra_signal_ports
from generators.support.signal_manager.binary_no_latency import generate_binary_no_latency_signal_manager
from generators.support.join import generate_join


def generate_addi(name, params):
  port_types = params["port_types"]
  data_type = VhdlScalarType(port_types["result"])

  if data_type.has_extra_signals():
    return _generate_addi_signal_manager(name, data_type)
  else:
    return _generate_addi(name, data_type.bitwidth)


def _generate_addi(name, bitwidth):
  join_name = f"{name}_join"

  dependencies = generate_join(join_name, {"size": 2})

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addi
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

  architecture = f"""
-- Architecture of addi
architecture arch of {name} is
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => result_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  result <= std_logic_vector(unsigned(lhs) + unsigned(rhs));
end architecture;
"""

  return dependencies + entity + architecture


# todo: can be reusable among various unit generators
extra_signal_logic = {
    "spec": """
  result_spec <= lhs_spec or rhs_spec;
"""  # todo: generate_normal_spec_logic(["trueOut", "falseOut"], ["data", "condition"])
}


def _generate_addi_signal_manager(name, data_type):
  return generate_binary_no_latency_signal_manager(name, data_type, _generate_addi)
