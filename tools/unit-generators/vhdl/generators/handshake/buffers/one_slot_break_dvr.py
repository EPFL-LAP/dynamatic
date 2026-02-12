
from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_one_slot_break_dvr(name, params):
    bitwidth = params["bitwidth"]
    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_one_slot_break_dvr_signal_manager(name, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_one_slot_break_dvr_dataless(name)
    else:
        return _generate_one_slot_break_dvr(name, bitwidth)


def _generate_one_slot_break_dvr_dataless(name):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dvr_dataless
entity {name} is 
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of one_slot_break_dvr_dataless
architecture arch of {name} is

  signal enable, stop : std_logic;
  signal outputValid, inputReady : std_logic;

begin

  p_ready : process(clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        inputReady <= '1';
      else
        inputReady <= (not stop) and (not enable);
      end if;
    end if;
  end process; 

  p_valid : process(clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outputValid <= '0';
      else
        outputValid <= enable or stop;
      end if;
    end if;
  end process;

  enable <= ins_valid and inputReady;
  stop <= outputValid and not outs_ready;
  ins_ready <= inputReady;
  outs_valid <= outputValid;

end architecture;

"""

    return entity + architecture


def _generate_one_slot_break_dvr(name, bitwidth):
    inner_name = f"{name}_inner"

    dependencies = _generate_one_slot_break_dvr_dataless(inner_name)

    entity = f"""
    library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dvr
entity {name} is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector({bitwidth} - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector({bitwidth} - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of one_slot_break_dvr
architecture arch of {name} is
  signal enable, inputReady : std_logic;
  signal dataReg: std_logic_vector({bitwidth} - 1 downto 0);
begin

  control : entity work.{inner_name}
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  p_data : process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        dataReg <= (others => '0');
      elsif (enable) then
        dataReg <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  enable <= ins_valid and inputReady;
  outs <= dataReg;

end architecture;
"""

    return dependencies + entity + architecture


def _generate_one_slot_break_dvr_signal_manager(name, bitwidth, extra_signals):
    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(extra_signals)
    return generate_concat_signal_manager(
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
        lambda name: _generate_one_slot_break_dvr(name, bitwidth + extra_signals_bitwidth))
