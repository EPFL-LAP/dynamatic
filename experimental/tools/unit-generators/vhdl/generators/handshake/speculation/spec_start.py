from generators.support.signal_manager.spec_units import generate_spec_units_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.support.utils import data
from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value


def generate_spec_start(name, params):
  bitwidth = params["bitwidth"]
  extra_signals = params["extra_signals"]

  # Always contains spec signal
  if len(extra_signals) > 1:
    return _generate_spec_start_signal_manager(name,  bitwidth, extra_signals)
  return _generate_spec_start(name, bitwidth)


def _generate_spec_start(name, bitwidth):

  entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_start
entity {name} is
  port (
    clk, rst : in  std_logic;
    {data(f"ins : in std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    {data(f"outs : out std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : in std_logic_vector(0 downto 0);
  );
end entity;
"""

  architecture = f"""
-- Architecture of spec_start
architecture arch of {name} is
begin
  {data("outs <= ins;", bitwidth)}
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
  outs_spec <= {get_default_extra_signal_value("spec")}
end architecture;
"""

  return entity + architecture


def _generate_spec_start_signal_manager(name, bitwidth, extra_signals):
  extra_signals_without_spec = extra_signals.copy()
  extra_signals_without_spec.pop("spec")

  extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
      extra_signals)
  return generate_spec_units_signal_manager(
      name,
      [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals_without_spec
      }],
      [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      extra_signals_without_spec,
      [],
      lambda name: _generate_spec_start(name, bitwidth + extra_signals_bitwidth - 1))
