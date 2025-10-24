from generators.support.signal_manager import generate_default_signal_manager


def generate_source(name, params):
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_source_signal_manager(name, extra_signals)
    else:
        return _generate_source(name)


def _generate_source(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of source
entity {name} is
  port (
    clk, rst   : in std_logic;
    -- inputs
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of source
architecture arch of {name} is
begin
  outs_valid <= '1';
end architecture;
"""

    return entity + architecture


def _generate_source_signal_manager(name, extra_signals):
    return generate_default_signal_manager(
        name,
        [],
        [{
            "name": "outs",
            "bitwidth": 0,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_source(name))
