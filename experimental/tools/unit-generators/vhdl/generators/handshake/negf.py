
from generators.support.signal_manager import generate_default_signal_manager

def generate_negf(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_negf_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_negf(name, bitwidth)


def _generate_negf(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of negf
entity {name} is
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of negf
architecture arch of {name} is
begin
  outs({bitwidth} - 1)          <= ins({bitwidth} - 1) xor '1';
  outs({bitwidth} - 2 downto 0) <= ins({bitwidth} - 2 downto 0);
  outs_valid                  <= ins_valid;
  ins_ready                   <= outs_ready;
end architecture;
"""

    return entity + architecture


def _generate_negf_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_negf(name, bitwidth))
