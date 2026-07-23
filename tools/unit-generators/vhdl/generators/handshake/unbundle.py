from generators.support.signal_manager import generate_default_signal_manager


def generate_unbundle(name, params):
    data_type = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_unbundle_signal_manager(
            name, data_type, extra_signals
        )
    else:
        return _generate_unbundle(name, data_type)


def _generate_unbundle(name, data_type):

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {name} is
  generic (
    DATA_TYPE : integer := {data_type}
  );
  port (
    clk, rst : in std_logic;

    -- data in
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;

    -- data out
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

"""

    architecture = f"""
architecture arch of {name} is
begin

  -- data
  outs       <= ins;
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;

end architecture;
"""

    return entity + architecture


def _generate_unbundle_signal_manager(name, data_type, extra_signals):
    return generate_default_signal_manager(
        name,
        [
            {
                "name": "ins",
                "bitwidth": data_type,
                "extra_signals": extra_signals,
            }
        ],
        [
            {
                "name": "outs",
                "bitwidth": data_type,
                "extra_signals": extra_signals,
            }
        ],
        extra_signals,
        lambda name: _generate_unbundle(name, data_type),
    )