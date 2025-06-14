from generators.support.signal_manager import generate_default_signal_manager
from generators.support.utils import data


def generate_sink(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_sink_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_sink(name, bitwidth)


def _generate_sink(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of sink
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    {data(f"ins       : in  std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    ins_valid : in  std_logic;
    ins_ready : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of sink
architecture arch of {name} is
begin
  ins_ready <= '1';
end architecture;
"""

    return entity + architecture


def _generate_sink_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [],
        extra_signals,
        lambda name: _generate_sink(name, bitwidth))
