from generators.support.signal_manager import generate_buffered_signal_manager
from generators.handshake.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.oehb import generate_oehb


def generate_addf(name, params):
    is_double = params["is_double"]
    internal_delay = params["internal_delay"]
    latency = params["latency"]
    extra_signals = params["extra_signals"]

    bitwidth = 64 if is_double else 32
    if extra_signals:
        return _generate_addf_signal_manager(name, bitwidth, internal_delay, latency, extra_signals)
    else:
        return _generate_addf(name, bitwidth, internal_delay, latency)



def _generate_addf(name, bitwidth, internal_delay, latency):
    join_name = f"{name}_join"
    oehb_name = f"{name}_oehb"
    buff_name = f"{name}_buff"

    dependencies = generate_join(join_name, {"size": 2}) + \
        generate_oehb(oehb_name, {"bitwidth": 0}) + \
        generate_delay_buffer(
        buff_name, {"slots": latency - 1})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of addf
entity {name} is
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector({bitwidth} - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector({bitwidth} - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;
"""

    clock_enables = "\n".join(
        [f"ce_1 => oehb_ready,"] +
        [f"        ce_{i} => oehb_ready," for i in range(2, latency+1)]
    )

    architecture = f"""
-- Architecture of addf
architecture arch of {name} is
  signal join_valid : STD_LOGIC;

  signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
  signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);

  --intermediate input signals for float conversion
  signal ip_lhs, ip_rhs : std_logic_vector({bitwidth + 2} - 1 downto 0);

  --intermidiate output signal(s) for float conversion
  signal ip_result : std_logic_vector({bitwidth + 2} - 1 downto 0);

  begin
    join_inputs : entity work.{join_name}(arch)
      port map(
        -- inputs
        ins_valid(0) => lhs_valid,
        ins_valid(1) => rhs_valid,
        outs_ready   => oehb_ready,
        -- outputs
        outs_valid   => join_valid,
        ins_ready(0) => lhs_ready,
        ins_ready(1) => rhs_ready
    );

    buff: entity work.{buff_name}(arch)
      port map(clk,
              rst,
              join_valid,
              oehb_ready,
              buff_valid
    );

    oehb: entity work.{oehb_name}(arch)
        port map(
        clk        => clk,
        rst        => rst,
        ins_valid  => buff_valid,
        outs_ready => result_ready,
        outs_valid => result_valid,
        ins_ready  => oehb_ready
      );

    ieee2nfloat_0: entity work.InputIEEE_{bitwidth}bit(arch)
            port map (
                --input
                X =>lhs,
                --output
                R => ip_lhs
            );

    ieee2nfloat_1: entity work.InputIEEE_{bitwidth}bit(arch)
            port map (
                --input
                X => rhs,
                --output
                R => ip_rhs
            );

    nfloat2ieee : entity work.OutputIEEE_{bitwidth}bit(arch)
            port map (
                --input
                X => ip_result,
                --ouput
                R => result
            );

    operator : entity work.FloatingPointAdder_{bitwidth}_{internal_delay}(arch)
    port map (
        clk   => clk,
        {clock_enables}
        X     => ip_lhs,
        Y     => ip_rhs,
        R     => ip_result
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_addf_signal_manager(name, bitwidth, internal_delay, latency, extra_signals):
    return generate_buffered_signal_manager(
        name,
        [{
            "name": "lhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "rhs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        [{
            "name": "result",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals,
        lambda name: _generate_addf(name, bitwidth, internal_delay, latency),
        latency)
