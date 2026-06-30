from generators.support.utils import data
from generators.support.signal_manager import generate_unary_signal_manager


def generate_ready_remover(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params["extra_signals"]

    def generate_inner(name): return _generate_ready_remover(name, bitwidth)
    def generate(): return generate_inner(name)

    if extra_signals:
        return generate_unary_signal_manager(
            name=name,
            bitwidth=bitwidth,
            extra_signals=extra_signals,
            generate_inner=generate_inner
        )
    else:
        return generate()


def _generate_ready_remover(name, bitwidth):
    potential_input = f"ins          : in std_logic_vector({bitwidth} - 1 downto 0);"
    potential_output = f"outs       : out std_logic_vector({bitwidth} - 1 downto 0);"
    potential_assignment = "outs <= ins;"

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of ready remover
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    {data(potential_input, bitwidth)}
    ins_valid    : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    {data(potential_output, bitwidth)}
    outs_valid : out std_logic;
    ins_ready    : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of ready remover
architecture arch of {name} is
begin
  {data(potential_assignment, bitwidth)}
  outs_valid <= ins_valid;
  ins_ready <= '1';
end architecture;
"""

    return entity + architecture
