from generators.support.signal_manager import generate_buffered_signal_manager
from generators.support.delay_buffer import generate_delay_buffer
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv


def generate_sitofp(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params["extra_signals"]

    if extra_signals:
        return _generate_sitofp_signal_manager(name, bitwidth, extra_signals)
    else:
        return _generate_sitofp(name, bitwidth)


def _generate_sitofp(name, bitwidth):
    one_slot_break_dv_name = f"{name}_one_slot_break_dv"
    buff_name = f"{name}_buff"

    dependencies = generate_one_slot_break_dv(one_slot_break_dv_name, {"bitwidth": bitwidth}) + \
        generate_delay_buffer(
        buff_name, {"slots": 4})

    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

-- Entity of sitofp
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
begin
  assert {bitwidth}=32
  report "sitofp currently only supports 32-bit floating point operands"
  severity failure;
end entity;
"""

    architecture = f"""
-- Architecture of sitofp
architecture arch of {name} is
  signal converted : std_logic_vector({bitwidth} - 1 downto 0);
  signal q0 : std_logic_vector({bitwidth} - 1 downto 0);
  signal q1 : std_logic_vector({bitwidth} - 1 downto 0);
  signal q2 : std_logic_vector({bitwidth} - 1 downto 0);
  signal q3 : std_logic_vector({bitwidth} - 1 downto 0);
  signal q4 : std_logic_vector({bitwidth} - 1 downto 0);
  signal buff_valid, one_slot_break_dv_ready : std_logic;
  signal one_slot_break_dv_dataOut, one_slot_break_dv_datain          : std_logic_vector({bitwidth} - 1 downto 0);
  signal float_value : float32;
begin

  float_value <= to_float(signed(ins));
  converted <= to_std_logic_vector(float_value);
  outs <= q4;

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (rst) then
        q0 <= (others => '0');
        q1 <= (others => '0');
        q2 <= (others => '0');
        q3 <= (others => '0');
        q4 <= (others => '0');
      elsif (one_slot_break_dv_ready) then
        q0 <= converted;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
        q4 <= q3;
      end if;
    end if;
  end process;

  buff : entity work.{buff_name}(arch)
    port map(
      clk,
      rst,
      ins_valid,
      one_slot_break_dv_ready,
      buff_valid
    );

  one_slot_break_dv : entity work.{one_slot_break_dv_name}(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => buff_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => one_slot_break_dv_ready,
      ins        => one_slot_break_dv_datain,
      outs       => one_slot_break_dv_dataOut
    );
  ins_ready <= one_slot_break_dv_ready;

end architecture;
"""

    return dependencies + entity + architecture


def _generate_sitofp_signal_manager(name, bitwidth, extra_signals):
    return generate_buffered_signal_manager(
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
        lambda name: _generate_sitofp(name, bitwidth),
        5)
