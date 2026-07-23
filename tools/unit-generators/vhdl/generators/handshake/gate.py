from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.top_join import generate_top_join


def generate_gate(name, params):
    size = params["size"]
    data_bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_gate_signal_manager(
            name, size, data_bitwidth, extra_signals
        )
    else:
        return _generate_gate(name, size, data_bitwidth)


def _generate_gate(name, size, data_bitwidth):

    join_module_name = f"{name}_join"
    dependencies = generate_top_join(join_module_name, {"size": size})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

entity {name} is
  generic (
    SIZE      : integer := {size};
    DATA_BITWIDTH : integer := {data_bitwidth}
  );
  port (
    clk, rst : in std_logic;

    -- data input channel
    ins : in data_array(0 downto 0)(DATA_BITWIDTH - 1 downto 0);

    -- control input channels
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);

    -- output channel
    outs : out std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

"""

    architecture = f"""
architecture arch of {name} is
begin

  join_inner : entity work.{join_module_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );

  -- simple data pass-through
  outs <= ins(0);

end architecture;
"""

    return dependencies + entity + architecture


def _generate_gate_signal_manager(name, size, data_bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [
            {{
                "name": "ins",
                "bitwidth": data_bitwidth,
                "extra_signals": extra_signals,
            }}
        ],
        [
            {{
                "name": "outs",
                "bitwidth": data_bitwidth,
                "extra_signals": extra_signals,
            }}
        ],
        extra_signals,
        lambda name: _generate_gate(name, size, data_bitwidth),
    )