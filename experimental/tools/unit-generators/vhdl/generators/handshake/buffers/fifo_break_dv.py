from generators.support.signal_manager import generate_concat_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.handshake.buffers.one_slot_break_dv import generate_one_slot_break_dv

def generate_fifo_break_dv(name, params):
    bitwidth = params["bitwidth"]
    num_slots = params["num_slots"]
    extra_signals = params.get("extra_signals", None)

    if num_slots == 1:
        return generate_one_slot_break_dv(name, params)

    if extra_signals:
        return _generate_fifo_break_dv_signal_manager(name, num_slots, bitwidth, extra_signals)
    elif bitwidth == 0:
        return _generate_fifo_break_dv_dataless(name, num_slots)
    else:
        return _generate_fifo_break_dv(name, num_slots, bitwidth)


def _generate_fifo_break_dv_dataless(name, num_slots):
    entity = f"""

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_dv_dataless
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
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
-- Architecture of fifo_break_dv_dataless
architecture arch of {name} is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to {num_slots} - 1;
  signal Head       : natural range 0 to {num_slots} - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;

begin

  -- ready if there is space in the fifo
  ins_ready <= not Full or outs_ready;

  -- read if next can accept and there is sth in fifo to read
  ReadEn <= (outs_ready and not Empty);

  outs_valid <= not Empty;

  WriteEn <= ins_valid and (not Full or outs_ready);

  -- valid
  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fifo_valid <= '0';
      else
        if (ReadEn = '1') then
          fifo_valid <= '1';
        elsif (outs_ready = '1') then
          fifo_valid <= '0';
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating tail
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Tail <= 0;
      else
        if (WriteEn = '1') then
          Tail <= (Tail + 1) mod {num_slots};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Head <= 0;
      else
        if (ReadEn = '1') then
          Head <= (Head + 1) mod {num_slots};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Full <= '0';
      else
        -- if only filling but not emptying
        if (WriteEn = '1') and (ReadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail + 1) mod {num_slots} = Head) then
            Full <= '1';
          end if;
          -- if only emptying but not filling
        elsif (WriteEn = '0') and (ReadEn = '1') then
          Full <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then
          -- if new head index will reach tail index
          if ((Head + 1) mod {num_slots} = Tail) then
            Empty <= '1';
          end if;
          -- if only filling but not emptying
        elsif (WriteEn = '1') and (ReadEn = '0') then
          Empty <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture



def _generate_fifo_break_dv(name, num_slots, bitwidth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fifo_break_dv
entity {name} is
  port (
    clk : in std_logic;
    rst : in std_logic;
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
-- Architecture of fifo_break_dv
architecture arch of {name} is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to {num_slots} - 1;
  signal Head       : natural range 0 to {num_slots} - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;
  type FIFO_Memory is array (0 to {num_slots} - 1) of std_logic_vector ({bitwidth} - 1 downto 0);
  signal Memory : FIFO_Memory;

begin

  -- ready if there is space in the fifo
  ins_ready <= not Full or outs_ready;
  -- read if next can accept and there is sth in fifo to read
  ReadEn <= (outs_ready and not Empty);
  outs_valid <= not Empty;
  outs <= Memory(Head);
  WriteEn <= ins_valid and (not Full or outs_ready);

  -- valid
  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fifo_valid <= '0';
      else
        if (ReadEn = '1') then
          fifo_valid <= '1';
        elsif (outs_ready = '1') then
          fifo_valid <= '0';
        end if;
      end if;
    end if;
  end process;

  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        for i in Memory'range loop
          Memory(i) <= (others => '0');
        end loop;
      else
        if (WriteEn = '1') then
          -- Write Data to Memory
          Memory(Tail) <= ins;
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating tail
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Tail <= 0;
      else
        if (WriteEn = '1') then
          Tail <= (Tail + 1) mod {num_slots};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Head <= 0;
      else
        if (ReadEn = '1') then
          Head <= (Head + 1) mod {num_slots};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if (rst = '1') then
        Full <= '0';
      else
        -- if only filling but not emptying
        if (WriteEn = '1') and (ReadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail + 1) mod {num_slots} = Head) then
            Full <= '1';
          end if;
          -- if only emptying but not filling
        elsif (WriteEn = '0') and (ReadEn = '1') then
          Full <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (WriteEn = '0') and (ReadEn = '1') then
          -- if new head index will reach tail index
          if ((Head + 1) mod {num_slots} = Tail) then
            Empty <= '1';
          end if;
          -- if only filling but not emptying
        elsif (WriteEn = '1') and (ReadEn = '0') then
          Empty <= '0';
          -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_fifo_break_dv_signal_manager(name, size, bitwidth, extra_signals):
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
        lambda name: _generate_fifo_break_dv(name, size, bitwidth + extra_signals_bitwidth))
