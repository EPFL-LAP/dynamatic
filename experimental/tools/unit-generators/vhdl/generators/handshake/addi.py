from generators.support.signal_manager import generate_signal_manager
from generators.handshake.join import generate_join


def generate_addi(name, params):
  bitwidth = params["bitwidth"]
  extra_signals = params.get("extra_signals", None)

  if extra_signals:
    return _generate_addi_signal_manager(name, bitwidth, extra_signals)
  else:
    return _generate_addi(name, bitwidth)


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


def _generate_addi_signal_manager(name, bitwidth, extra_signals):
  return generate_signal_manager(name, {
      "type": "normal",
      "in_ports": [{
          "name": "lhs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }, {
          "name": "rhs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "out_ports": [{
          "name": "result",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "extra_signals": extra_signals
  }, lambda name: _generate_addi(name, bitwidth))
