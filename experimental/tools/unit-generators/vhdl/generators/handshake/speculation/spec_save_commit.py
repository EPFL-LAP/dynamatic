from generators.support.signal_manager import generate_signal_manager, get_concat_extra_signals_bitwidth
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

  signal PassEn   : std_logic := '0';
  signal KillEn   : std_logic := '0';
  signal ResendEn : std_logic := '0';
  signal NoCmpEn  : std_logic := '0';

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

  -------------------
  -- comb process for pointers
  -------------------
  pointer_proc : process (Full, Empty, CurrEmpty, ResendEn, KillEn, PassEn, NoCmpEn, ins_valid, outs_ready, ctrl_ready)
  begin
    TailEn <= not Full and ins_valid;

    if ResendEn = '1' then
      HeadEn <= outs_ready and not Empty;
    elsif KillEn = '1' then
      HeadEn <= not Empty;
    else
      HeadEn <= '0';
    end if;

    -- Note: PassEn and KillEn can be simultaneously '1'
    -- In that case, PassEn is currently prioritized
    -- TODO: I don't think the assumption above is correct.
    if PassEn = '1' then
      if CurrEmpty = '1' then
        -- Bypass
        CurrEn <= outs_ready and ins_valid;
      else
        CurrEn <= outs_ready;
      end if;
    elsif NoCmpEn = '1' then
      -- TODO: When Empty = '1', input data should be bypassed,
      --       just like with PassEn.
      CurrEn <= outs_ready and not Empty;
    elsif KillEn = '1' and Curr = Head then
      -- Exceptional case (Curr = Head). See my report.

      -- Ensuring non-empty is necessary to avoid a transient Curr > Tail situation,
      -- as the token has not arrived and Tail may not have been incremented yet.
      CurrEn <= not Empty;
    else
      CurrEn <= '0';
    end if;
  end process;

  -------------------
  -- comb process for control en
  -------------------
  en_proc : process (ins_valid, ctrl_valid, ctrl)
  begin
    PassEn <= '0';
    KillEn <= '0';
    ResendEn <= '0';
    NoCmpEn <= '0';

    if ctrl_valid = '1' and ctrl = "000" then
      -- PASS
      PassEn <= '1';
    elsif ctrl_valid = '1' and ctrl = "001" then
      -- KILL
      KillEn <= '1';
    elsif ctrl_valid = '1' and ctrl = "010" then
      -- RESEND
      ResendEn <= '1';
    elsif ctrl_valid = '1' and ctrl = "011" then
      -- PASS_KILL
      PassEn <= '1';
      KillEn <= '1';
    elsif ctrl_valid = '1' and ctrl = "100" then
      -- NO_CMP: Special case of RESEND. Curr must be incremented as well.
      ResendEn <= '1';
      NoCmpEn <= '1';
    end if;
  end process;

  -------------------------------------------
  -- comb process for control ready
  -------------------------------------------
  ctrl_ready_proc : process (PassEn, KillEn, ResendEn, CurrEmpty, Empty, outs_ready, ins_valid)
  begin
    -- Note: PassEn and KillEn can be simultaneously '1'
    -- In that case, PassEn is currently prioritized
    -- TODO: I don't think the assumption above is correct.
    if PassEn = '1' then
      if CurrEmpty = '1' then
        -- Bypass
        ctrl_ready <= outs_ready and ins_valid;
      else
        ctrl_ready <= outs_ready;
      end if;
    elsif ResendEn = '1' then
      ctrl_ready <= not Empty and outs_ready;
    elsif KillEn = '1' then
      ctrl_ready <= not Empty;
    else
      ctrl_ready <= '0';
    end if;
  end process;

  -------------------------------------------
  -- comb process for outs_valid
  -------------------------------------------
  outs_valid_proc : process (PassEn, ResendEn, CurrEmpty, Empty, ins_valid)
  begin
    if PassEn = '1' then
      if CurrEmpty = '1' then
        -- Bypass
        outs_valid <= ins_valid;
      else
        outs_valid <= '1'; -- TODO: "not Empty"?
      end if;
    elsif ResendEn = '1' then
      outs_valid <= not Empty;
    else
      outs_valid <= '0';
    end if;
  end process;

  -------------------------------------------
  -- comb process for output data
  -------------------------------------------
  output_proc : process (PassEn, Curr, {data("ins, Memory, ", bitwidth)}Head)
  begin
    if PassEn = '1' then
      {data("""
      if CurrEmpty = '1' then
        -- Bypass
        outs <= ins;
      else
        outs <= Memory(Curr);
      end if;
      """, bitwidth)}
      outs_spec <= "1";
    else
      {data(f"outs <= Memory(Head);", bitwidth)}
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
  return generate_signal_manager(name, {
      "type": "concat",
      "in_ports": [{
          "name": "ins",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }, {
          "name": "ctrl",
          "bitwidth": 3
      }],
      "out_ports": [{
          "name": "outs",
          "bitwidth": bitwidth,
          "extra_signals": extra_signals
      }],
      "extra_signals": extra_signals_without_spec,
      "ignore_ports": ["ctrl"]
  }, lambda name: _generate_spec_save_commit(name, bitwidth + extra_signals_bitwidth - 1, fifo_depth))
