
from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth


def generate_shift_reg_break_dv(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]

    extra_signals = params.get("extra_signals", None)

    if extra_signals:
        return _generate_shift_reg_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals)
    if bitwidth == 0:
        return _generate_shift_reg_break_dv_dataless(name, num_slots)
    else:
        return _generate_shift_reg_break_dv(name, num_slots, bitwidth)


def _generate_shift_reg_break_dv_dataless(name, num_slots):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of shift_reg_break_dv_dataless
entity {name} is
  port(
    -- inputs
    clk, rst   : in std_logic;
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;
"""

    architecture = f"""
    -- Architecture of shift_reg_break_dv_dataless
    architecture arch of {name} is

    signal regEn      : std_logic;
    type REG_VALID is array (0 to {num_slots} - 1) of std_logic;
    signal valid_reg  : REG_VALID;

    begin
    -- See 'docs/Specs/Buffering/Buffering.md'
    -- All the slots share a single handshake control and thus 
    -- accept or stall inputs together.
    process(clk) is
    begin
        if (rising_edge(clk)) then
        if (rst = '1') then
            for i in 0 to {num_slots} - 1 loop
            valid_reg(i) <= '0';
            end loop;
        else
            if (regEn) then
            for i in 1 to {num_slots} - 1 loop
                valid_reg(i) <= valid_reg(i - 1);
            end loop;
            valid_reg(0) <= ins_valid;
            end if;               
        end if;
        end if;
    end process; 

    outs_valid <= valid_reg({num_slots} - 1);
    regEn <= not outs_valid or outs_ready;
    ins_ready <= regEn;

    end architecture;
    """
    return entity + architecture


def _generate_shift_reg_break_dv(name, num_slots, bitwidth):
    inner_name = f"{name}_inner"

    dependencies = _generate_shift_reg_break_dv_dataless(inner_name)

    entity = f"""

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of shift_reg_break_dv

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
-- Architecture of shift_reg_break_dv

architecture arch of {name} is

  signal regEn, inputReady : std_logic;
  type REG_MEMORY is array (0 to {num_slots} - 1) of std_logic_vector({bitwidth} - 1 downto 0);
  signal Memory  : REG_MEMORY;

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

  -- If valid is reset, the data in this slot becomes obsolete.
  -- Hence, there is no need to reset the data as well.
  -- If reset is required, then add the following lines:
  -- if (rst = '1') then
  --   for i in 0 to {num_slots} - 1 loop
  --     Memory(i) <= (others => '0');
  --   end loop;

  -- See 'docs/Specs/Buffering/Buffering.md'
  -- All the slots share a single handshake control and thus 
  -- accept or stall inputs together.
  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (regEn) then
        for i in 1 to {num_slots} - 1 loop
          Memory(i) <= Memory(i - 1);
        end loop;
        Memory(0) <= ins;
      end if;
    end if;
  end process;

  regEn <= inputReady;
  ins_ready <= inputReady;
  outs <= Memory({num_slots} - 1);

end architecture;
"""

    return dependencies + entity + architecture


def _generate_shift_reg_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals):
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
        lambda name: _generate_shift_reg_break_dv(name, num_slots, bitwidth + extra_signals_bitwidth))
