from generators.support.signal_manager import generate_default_signal_manager
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.join import generate_join


def generate_divsi(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_divsi_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_divsi(name, bitwidth)


def _generate_divsi(name, bitwidth):
    join_name = f"{name}_join"
    buff_name = f"{name}_buff"

    dependencies = generate_join(join_name, {"size": 2}) + \
            generate_delay_buffer(
        buff_name, {"slots": 35 - 1})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of divsi
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

    architecture = f"""
-- Architecture of divsi

architecture arch of {name} is
  component array_RAM_sdiv_32ns_32ns_32_36_1 is
    generic (
      ID         : integer;
      NUM_STAGE  : integer;
      din0_TYPE : integer;
      din1_TYPE : integer;
      dout_TYPE : integer);
    port (
      clk   : in  std_logic;
      reset : in  std_logic;
      ce    : in  std_logic;
      din0  : in  std_logic_vector(din0_TYPE - 1 downto 0);
      din1  : in  std_logic_vector(din1_TYPE - 1 downto 0);
      dout  : out std_logic_vector(dout_TYPE - 1 downto 0));
  end component;

  signal join_valid : std_logic;

begin
  join_inputs : entity work.{join_name}(arch)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  array_RAM_sdiv_32ns_32ns_32_36_1_U1 : component array_RAM_sdiv_32ns_32ns_32_36_1
    generic map(
      ID         => 1,
      NUM_STAGE  => 36,
      din0_TYPE => 32,
      din1_TYPE => 32,
      dout_TYPE => 32)
    port map(
      clk   => clk,
      reset => rst,
      ce    => result_ready,
      din0  => lhs,
      din1  => rhs,
      dout  => result
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      result_ready,
      result_valid
    );

end architecture;
"""

    return dependencies + entity + architecture


def _generate_divsi_signal_manager(name, bitwidth, extra_signals):
    return generate_default_signal_manager(
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
        lambda name: _generate_divsi(name, bitwidth))
