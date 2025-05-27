from generators.support.signal_manager import generate_signal_manager, get_concat_extra_signals_bitwidth, _get_default_extra_signal_value
from generators.support.utils import data


def generate_non_spec(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params["extra_signals"]

    # Always contains spec signal
    if len(extra_signals) > 1:
        return _generate_non_spec_signal_manager(name, bitwidth, extra_signals)
    return _generate_non_spec(name, bitwidth)


def _generate_non_spec(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of non_spec
entity {name} is
  port (
    clk, rst : in  std_logic;
    {data(f"dataIn : in std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    dataIn_valid : in std_logic;
    dataIn_ready : out std_logic;
    {data(f"dataOut : out std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic;
    dataOut_spec : out std_logic_vector(0 downto 0)
  );
end entity;
"""

    architecture = f"""
-- Architecture of non_spec
architecture arch of {name} is
begin
  {data("dataOut <= dataIn;", bitwidth)}
  dataOut_valid <= dataIn_valid;
  dataIn_ready <= dataOut_ready;
  dataOut_spec <= {_get_default_extra_signal_value("spec")};
end architecture;
"""

    return entity + architecture


def _generate_non_spec_signal_manager(name, bitwidth, extra_signals):
    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
        extra_signals)
    return generate_signal_manager(name, {
        "type": "concat",
        "in_ports": [{
            "name": "dataIn",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        "out_ports": [{
            "name": "dataOut",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals_without_spec,
        }],
        "extra_signals": extra_signals_without_spec
    }, lambda name: _generate_non_spec(name, bitwidth + extra_signals_bitwidth - 1))
