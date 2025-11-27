from generators.support.signal_manager.utils.forwarding import get_default_extra_signal_value
from generators.support.signal_manager.utils.concat import ConcatLayout
from generators.support.signal_manager.utils.generation import generate_concat, generate_slice
from generators.support.signal_manager.utils.entity import generate_entity
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
  dataOut_spec <= {get_default_extra_signal_value("spec")};
end architecture;
"""

    return entity + architecture


def _generate_non_spec_signal_manager(name, bitwidth, extra_signals):
    # Concat signals except spec

    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    concat_layout = ConcatLayout(extra_signals_without_spec)
    extra_signals_without_spec_bitwidth = concat_layout.total_bitwidth

    inner_name = f"{name}_inner"
    inner = _generate_non_spec(
        inner_name, bitwidth + extra_signals_without_spec_bitwidth)

    entity = generate_entity(name, [{
        "name": "dataIn",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals_without_spec
    }], [{
        "name": "dataOut",
        "bitwidth": bitwidth,
        "extra_signals": extra_signals
    }])

    assignments = []

    # Concat dataIn data and extra signals (except spec) to create dataIn_concat
    assignments.extend(generate_concat(
        "dataIn", bitwidth, "dataIn_concat", concat_layout))

    # Slice dataOut_concat to create dataOut data and extra signals (except spec)
    assignments.extend(generate_slice(
        "dataOut_concat", "dataOut", bitwidth, concat_layout))

    architecture = f"""
-- Architecture of non_spec signal manager
architecture arch of {name} is
  signal dataIn_concat, dataOut_concat : std_logic_vector({bitwidth + extra_signals_without_spec_bitwidth} - 1 downto 0);
begin
  {"\n  ".join(assignments)}
  inner : entity work.{inner_name}(arch)
    port map(
      clk => clk,
      rst => rst,
      dataIn => dataIn_concat,
      dataIn_valid => dataIn_valid,
      dataIn_ready => dataIn_ready,
      dataOut => dataOut_concat,
      dataOut_valid => dataOut_valid,
      dataOut_ready => dataOut_ready,
      -- Forward spec signal
      dataOut_spec => dataOut_spec
    );
end architecture;
"""

    return inner + entity + architecture
