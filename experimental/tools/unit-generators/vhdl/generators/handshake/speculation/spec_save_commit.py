from generators.support.signal_manager import generate_spec_units_signal_manager
from generators.support.signal_manager.utils.concat import get_concat_extra_signals_bitwidth
from generators.support.utils import data


def generate_spec_save_commit(name, params):
    bitwidth = params["bitwidth"]
    fifo_depth = params["fifo_depth"]
    extra_signals = params["extra_signals"]

    # Always contains spec signal
    if len(extra_signals) > 1:
        return _generate_spec_save_commit_signal_manager(name, bitwidth, fifo_depth, extra_signals)
    return _generate_spec_save_commit(name, bitwidth, fifo_depth)


def _generate_spec_save_commit(name, bitwidth, fifo_depth):
    entity = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_save_commit
entity {name} is
  port (
    clk, rst : in std_logic;
    -- inputs
    {data(f"ins : in std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0); -- not used
    ctrl : in std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    ctrl_valid : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    {data(f"outs : out std_logic_vector({bitwidth} - 1 downto 0);", bitwidth)}
    outs_valid : out std_logic;
    outs_spec : out std_logic_vector(0 downto 0);
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;
"""

    architecture = f"""
-- Architecture of spec_save_commit
architecture arch of {name} is
  signal HeadEn : std_logic := '0';
  signal TailEn : std_logic := '0';
  signal CurrEn : std_logic := '0';

  signal Tail : natural range 0 to {fifo_depth} - 1;
  signal Head : natural range 0 to {fifo_depth} - 1;
  signal Curr : natural range 0 to {fifo_depth} - 1;

  signal CurrEmpty : std_logic;
  signal Empty     : std_logic;
  signal Full      : std_logic;

  {data(f"type FIFO_Memory is array (0 to {fifo_depth} - 1) of STD_LOGIC_VECTOR ({bitwidth} - 1 downto 0);", bitwidth)}
  {data("signal Memory : FIFO_Memory;", bitwidth)}
begin
  ins_ready <= not Full;

  -------------------------------------------
  -- comb process for all signals
  -------------------------------------------
  signal_proc : process (
    ctrl_valid, ctrl,
    CurrEmpty, Empty, Full,
    {data("Memory, ins,", bitwidth)}
    Head, Curr,
    outs_ready, ins_valid)
  begin
    TailEn <= not Full and ins_valid;
    HeadEn <= '0';
    CurrEn <= '0';

    ctrl_ready <= '0';
    outs_valid <= '0';
    {data("outs <= Memory(Head);", bitwidth)}
    outs_spec <= "0";

    if ctrl_valid = '1' and ctrl = "000" then
      -- PASS
      if CurrEmpty = '1' then
        -- Curr = Tail. Perform bypassing.

        -- Consider the condition required for TailEn = '1' (not Full and ins_valid).
        CurrEn <= outs_ready and ins_valid and not Full;

        ctrl_ready <= outs_ready and ins_valid and not Full;
        outs_valid <= ins_valid and not Full;
        {data("outs <= ins;", bitwidth)}
      else
        -- Curr < Tail.
        CurrEn <= outs_ready;

        ctrl_ready <= outs_ready;
        outs_valid <= '1';
        {data("outs <= Memory(Curr);", bitwidth)}
      end if;
      outs_spec <= "1";
    elsif ctrl_valid = '1' and ctrl = "001" then
      -- KILL
      if Head = Curr then
        -- Exceptional case. See my report.
        -- `not Empty` ensures Curr < Tail.
        CurrEn <= not Empty;
        HeadEn <= not Empty;
        ctrl_ready <= not Empty;
      else
        -- Head < Curr.
        HeadEn <= '1';
        ctrl_ready <= '1';
      end if;
    elsif ctrl_valid = '1' and ctrl = "011" then
      -- PASS_KILL
      -- Head < Curr is assumed from the specification.
      if CurrEmpty = '1' then
        -- Curr = Tail. Perform bypassing.

        -- Consider the condition required for TailEn = '1' (not Full and ins_valid).
        CurrEn <= outs_ready and ins_valid and not Full;
        HeadEn <= outs_ready and ins_valid and not Full;

        ctrl_ready <= outs_ready and ins_valid and not Full;
        outs_valid <= ins_valid and not Full;
        {data("outs <= ins;", bitwidth)}
      else
        -- Curr < Tail.
        CurrEn <= outs_ready;
        HeadEn <= outs_ready;

        ctrl_ready <= outs_ready;
        outs_valid <= '1';
        {data("outs <= Memory(Curr);", bitwidth)}
      end if;
      outs_spec <= "1";
    elsif ctrl_valid = '1' and ctrl = "010" then
      -- RESEND
      -- Head < Curr is assumed from the specification.
      HeadEn <= outs_ready;

      ctrl_ready <= outs_ready;
      outs_valid <= '1';
      {data("outs <= Memory(Head);", bitwidth)}
      outs_spec <= "0";
    elsif ctrl_valid = '1' and ctrl = "100" then
      -- NO_CMP
      -- TODO: When Empty = '1', input data should be bypassed,
      --       just like when PASS or PASS_KILL, for better performance.
      -- Head = Curr is assumed from the specification.
      -- `not Empty` ensures Curr < Tail.
      CurrEn <= outs_ready and not Empty;
      HeadEn <= outs_ready and not Empty;

      ctrl_ready <= outs_ready and not Empty;
      outs_valid <= not Empty;
      {data("outs <= Memory(Head);", bitwidth)}
      outs_spec <= "0";
    end if;
  end process;

  ----------------------------------------------------------------
  -- Sequential Process
  ----------------------------------------------------------------

  {data("""
  -------------------------------------------
  -- process for writing data
  -------------------------------------------
  fifo_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        -- TODO: Nothing??
      else
        if TailEn = '1' then
          -- Write Data to Memory
          Memory(Tail) <= ins;
        end if;
      end if;
    end if;
  end process;
  """, bitwidth)}

  -------------------------------------------
  -- process for updating tail
  -------------------------------------------
  TailUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Tail <= 0;
      else
        if (TailEn = '1') then
          Tail <= (Tail + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating head
  -------------------------------------------
  HeadUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Head <= 0;
      else
        if (HeadEn = '1') then
          Head  <= (Head + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating curr
  -------------------------------------------
  CurrUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Curr <= 0;
      else
        if (CurrEn = '1') then
          Curr <= (Curr + 1) mod {fifo_depth};
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating full
  -------------------------------------------
  FullUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Full <= '0';
      else
        -- if only filling but not emptying
        if (TailEn = '1') and (HeadEn = '0') then
          -- if new tail index will reach head index
          if ((Tail +1) mod {fifo_depth} = Head) then
            Full  <= '1';
          end if;
        elsif (TailEn = '0') and (HeadEn = '1') then
          -- if only emptying but not filling
          Full <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating empty
  -------------------------------------------
  EmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        Empty <= '1';
      else
        -- if only emptying but not filling
        if (TailEn = '0') and (HeadEn = '1') then
          -- if new head index will reach tail index
          if ((Head +1) mod {fifo_depth} = Tail) then
            Empty  <= '1';
          end if;
        elsif (TailEn = '1') and (HeadEn = '0') then
          -- if only filling but not emptying
          Empty <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;

  -------------------------------------------
  -- process for updating curr empty
  -------------------------------------------
  CurrEmptyUpdate_proc : process (clk)
  begin
    if rising_edge(clk) then
      if rst = '1' then
        CurrEmpty <= '1';
      else
        -- if only emptying but not filling
        if (TailEn = '0') and (CurrEn = '1') then
          -- if new head index will reach tail index
          if ((Curr +1) mod {fifo_depth} = Tail) then
            CurrEmpty  <= '1';
          end if;
        elsif (TailEn = '1') and (CurrEn = '0') then
          -- if only filling but not emptying
          CurrEmpty <= '0';
        -- otherwise, nothing is happening or simultaneous read and write
        end if;
      end if;
    end if;
  end process;
end architecture;
"""

    return entity + architecture


def _generate_spec_save_commit_signal_manager(name, bitwidth, fifo_depth, extra_signals):
    extra_signals_without_spec = extra_signals.copy()
    extra_signals_without_spec.pop("spec")

    extra_signals_bitwidth = get_concat_extra_signals_bitwidth(
        extra_signals)
    return generate_spec_units_signal_manager(
        name,
        [{
            "name": "ins",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }, {
            "name": "ctrl",
            "bitwidth": 3
        }],
        [{
            "name": "outs",
            "bitwidth": bitwidth,
            "extra_signals": extra_signals
        }],
        extra_signals_without_spec,
        ["ctrl"],
        lambda name: _generate_spec_save_commit(name, bitwidth + extra_signals_bitwidth - 1, fifo_depth))
