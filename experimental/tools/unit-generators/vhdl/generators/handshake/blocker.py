from generators.support.signal_manager import generate_default_signal_manager
from generators.handshake.join import generate_join


def generate_blocker(name, params):
    # Number of input ports
    size = params["size"]

    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_blocker_signal_manager(name, size, bitwidth, extra_signals)
    else:
        return _generate_blocker(name, size, bitwidth)


def _generate_blocker(name, size, bitwidth):
    join_name = f"{name}_join"

    dependencies = generate_join(join_name, {"size": size})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use work.types.all;

-- Entity of blocker
entity {name} is
  port (
    clk          : in std_logic;
    rst          : in std_logic;
    -- input channels
    ins        : in  data_array({size} - 1 downto 0)({bitwidth} - 1 downto 0);
    ins_valid  : in std_logic_vector({size} - 1 downto 0);
    outs_ready : in std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector({size} - 1 downto 0)
  );
end entity;
"""

    architecture = f"""
-- Architecture of blocker
architecture arch of {name} is
begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid    => ins_valid,
      outs_ready   => outs_ready,
      -- outputs
      outs_valid   => outs_valid,
      ins_ready    => ins_ready
    );

  outs <= ins(0);

end architecture;
"""

    return dependencies + entity + architecture


def _generate_blocker_signal_manager(name, size, bitwidth, extra_signals):
    return generate_default_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals,
            "size": size
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name:  _generate_blocker(name, size, bitwidth))
