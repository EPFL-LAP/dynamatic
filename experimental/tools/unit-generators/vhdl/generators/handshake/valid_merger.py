from generators.support.utils import data
from generators.support.signal_manager.utils.entity import generate_entity
from generators.support.signal_manager.utils.generation import generate_default_mappings


def generate_valid_merger(name, params):
    lhs_bitwidth = params["left_bitwidth"]
    rhs_bitwidth = params["right_bitwidth"]

    lhs_extra_signals = params.get("lhs_extra_signals", None)
    rhs_extra_signals = params.get("rhs_extra_signals", None)

    def generate_inner(name): return _generate_valid_merger(name, lhs_bitwidth, rhs_bitwidth)
    def generate(): return generate_inner(name)

    if lhs_extra_signals or rhs_extra_signals:
        return _generate_valid_merger_signal_manager(
            name,
            lhs_bitwidth,
            rhs_bitwidth,
            generate_inner,
            lhs_extra_signals,
            rhs_extra_signals
        )
    else:
        return generate()


def _generate_valid_merger(name, lhs_bitwidth, rhs_bitwidth):
    possible_lhs_ins = f"lhs_ins          : in std_logic_vector({lhs_bitwidth} - 1 downto 0);"
    possible_rhs_ins = f"rhs_ins          : in std_logic_vector({rhs_bitwidth} - 1 downto 0);"
    possible_lhs_outs = f"lhs_outs         : out std_logic_vector({lhs_bitwidth} - 1 downto 0);"
    possible_rhs_outs = f"rhs_outs         : out std_logic_vector({rhs_bitwidth} - 1 downto 0);"

    possible_lhs_assignment = "lhs_outs <= lhs_ins;"
    possible_rhs_assignment = "rhs_outs <= rhs_ins;"

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of valid_merger
entity {name} is
  port (
    -- inputs
    clk              : in std_logic;
    rst              : in std_logic;
    {data(possible_lhs_ins, lhs_bitwidth)}
    lhs_ins_valid    : in std_logic;
    lhs_outs_ready   : in std_logic;
    {data(possible_rhs_ins, rhs_bitwidth)}
    rhs_ins_valid    : in std_logic;
    rhs_outs_ready   : in std_logic;
    -- outputs
    {data(possible_lhs_outs, lhs_bitwidth)}
    lhs_outs_valid   : out std_logic;
    lhs_ins_ready    : out std_logic;
    {data(possible_rhs_outs, rhs_bitwidth)}
    rhs_outs_valid   : out std_logic;
    rhs_ins_ready    : out std_logic
  );
end entity;
"""
    f"""
-- Architecture of valid_merger
architecture arch of {name} is
begin
  {data(possible_lhs_assignment, lhs_bitwidth)}
  lhs_outs_valid <= lhs_ins_valid;
  lhs_ins_ready <= lhs_outs_ready;

  {data(possible_rhs_assignment, rhs_bitwidth)}
  rhs_outs_valid <= lhs_ins_valid; -- merge happens here
  rhs_ins_ready <= rhs_outs_ready;
end architecture;
"""


def _generate_valid_merger_signal_manager(name,
                                          lhs_bitwidth,
                                          rhs_bitwidth,
                                          generate_inner,
                                          lhs_extra_signals,
                                          rhs_extra_signals):
    inner_name = f"{name}_inner"
    inner = generate_inner(inner_name)

    in_channels = [
        {
            "name": "lhs_in",
            "bitwidth": lhs_bitwidth,
            "extra_signals": lhs_extra_signals
        },
        {
            "name": "rhs_in",
            "bitwidth": rhs_bitwidth,
            "extra_signals": rhs_extra_signals
        }
    ]

    out_channels = [
        {
            "name": "lhs_out",
            "bitwidth": lhs_bitwidth,
            "extra_signals": lhs_extra_signals
        },
        {
            "name": "rhs_out",
            "bitwidth": rhs_bitwidth,
            "extra_signals": rhs_extra_signals
        }
    ]

    entity = generate_entity(
        name,
        in_channels,
        out_channels
    )

    extra_signal_assignments = []
    # directly pass extra signals through the valid merger
    # regardless of how they're normally forwarded
    for extra_signal_name in lhs_extra_signals:
        extra_signal_assignments.append(
            f"lhs_out_{extra_signal_name} <= lhs_in_{extra_signal_name};"
        )

    for extra_signal_name in rhs_extra_signals:
        extra_signal_assignments.append(
            f"rhs_out_{extra_signal_name} <= rhs_in_{extra_signal_name};"
        )

    # Map channels to inner component
    mappings = generate_default_mappings(in_channels + out_channels)

    architecture = f"""
-- Architecture of signal manager (default)
architecture arch of {name} is
begin
-- Forward extra signals to output channels
{"\n  ".join(extra_signal_assignments)}

inner : entity work.{inner_name}(arch)
  port map(
    clk => clk,
    rst => rst,
    {mappings}
  );
end architecture;
"""

    return inner + entity + architecture
