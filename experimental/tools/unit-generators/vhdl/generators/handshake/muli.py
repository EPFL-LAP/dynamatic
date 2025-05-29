from generators.support.signal_manager import generate_buffered_signal_manager
from generators.handshake.join import generate_join
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.oehb import generate_oehb


def generate_muli(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_muli_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_muli(name, bitwidth)


def _get_latency():
    return 4


def _generate_mul_4_stage(name, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of mul_4_stage
entity {name} is
  port (
    clk : in  std_logic;
    ce  : in  std_logic;
    a   : in  std_logic_vector({bitwidth} - 1 downto 0);
    b   : in  std_logic_vector({bitwidth} - 1 downto 0);
    p   : out std_logic_vector({bitwidth} - 1 downto 0));
end entity;
"""

    architecture = f"""
-- Architecture of mul_4_stage
architecture behav of {name} is

  signal a_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal b_reg : std_logic_vector({bitwidth} - 1 downto 0);
  signal q0    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q1    : std_logic_vector({bitwidth} - 1 downto 0);
  signal q2    : std_logic_vector({bitwidth} - 1 downto 0);
  signal mul   : std_logic_vector({bitwidth} - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), {bitwidth}));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;
end architecture;
"""

    return entity + architecture


def _generate_muli(name, bitwidth):
    join_name = f"{name}_join"
    mul_4_stage_name = f"{name}_mul_4_stage"
    buff_name = f"{name}_buff"
    oehb_name = f"{name}_oehb"

    dependencies = \
        generate_join(join_name, {"size": 2}) + \
        _generate_mul_4_stage(mul_4_stage_name, bitwidth) + \
        generate_delay_buffer(buff_name, {"slots": _get_latency() - 1}) + \
        generate_oehb(oehb_name, {"bitwidth": bitwidth})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of muli
entity {name} is
  port (
    -- inputs
    clk, rst     : in std_logic;
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
-- Architecture of muli
architecture arch of {name} is
  signal join_valid                         : std_logic;
  signal buff_valid, oehb_valid, oehb_ready : std_logic;
  signal oehb_dataOut, oehb_datain          : std_logic_vector({bitwidth} - 1 downto 0);
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

  multiply_unit : entity work.{mul_4_stage_name}(behav)
    port map(
      clk => clk,
      ce  => oehb_ready,
      a   => lhs,
      b   => rhs,
      p   => result
    );

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      join_valid,
      oehb_ready,
      buff_valid
    );

  oehb : entity work.{oehb_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => result_ready,
      outs_valid => result_valid,
      ins_ready  => oehb_ready,
      ins        => oehb_datain,
      outs       => oehb_dataOut
    );
end architecture;
"""

    return dependencies + entity + architecture


def _generate_muli_signal_manager(name, bitwidth, extra_signals):
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
        lambda name: _generate_muli(name, bitwidth),
        _get_latency())
