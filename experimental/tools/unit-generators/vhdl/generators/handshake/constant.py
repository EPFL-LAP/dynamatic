from generators.support.signal_manager import generate_default_signal_manager


def generate_constant(name, params):
    bitwidth = params["bitwidth"]
    value = params["value"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_constant_signal_manager(name, value, bitwidth, extra_signals)
    else:
        return _generate_constant(name, value, bitwidth)


def _generate_constant(name, value, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of constant
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ctrl_valid : in  std_logic;
    ctrl_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of constant
architecture arch of {name} is
begin
  outs       <= "{value}";
  outs_valid <= ctrl_valid;
  ctrl_ready <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_constant_signal_manager(name, value, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ctrl",
            "bitwidth": 0,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_constant(name, value, bitwidth))
