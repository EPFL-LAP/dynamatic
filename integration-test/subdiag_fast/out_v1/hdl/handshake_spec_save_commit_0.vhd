-- handshake_spec_save_commit_0 : spec_save_commit({'fifo_depth': 16, 'bitwidth': 32, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of spec_save_commit
entity handshake_spec_save_commit_0 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- inputs
    ins : in std_logic_vector(32 - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec : in std_logic_vector(0 downto 0); -- not used
    ctrl : in std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    ctrl_valid : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec : out std_logic_vector(0 downto 0);
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end entity;

-- Architecture of spec_save_commit
architecture arch of handshake_spec_save_commit_0 is
  signal HeadEn : std_logic := '0';
  signal TailEn : std_logic := '0';
  signal CurrEn : std_logic := '0';

  signal Tail : natural range 0 to 16 - 1;
  signal Head : natural range 0 to 16 - 1;
  signal Curr : natural range 0 to 16 - 1;

  signal CurrEmpty : std_logic;
  signal Empty     : std_logic;
  signal Full      : std_logic;

  type FIFO_Memory is array (0 to 16 - 1) of STD_LOGIC_VECTOR (32 - 1 downto 0);
  signal Memory : FIFO_Memory;
begin
  ins_ready <= not Full;

  -------------------------------------------
  -- comb process for all signals
  -------------------------------------------
  signal_proc : process (
    ctrl_valid, ctrl,
    CurrEmpty, Empty, Full,
    Memory, ins,
    Head, Curr,
    outs_ready, ins_valid)
  begin
    TailEn <= not Full and ins_valid;
    HeadEn <= '0';
    CurrEn <= '0';

    ctrl_ready <= '0';
    outs_valid <= '0';
    outs <= Memory(Head);
    outs_spec <= "0";

    if ctrl_valid = '1' and ctrl = "000" then
      -- PASS
      if CurrEmpty = '1' then
        -- Curr = Tail. Perform bypassing.

        -- Consider the condition required for TailEn = '1' (not Full and ins_valid).
        CurrEn <= outs_ready and ins_valid and not Full;

        ctrl_ready <= outs_ready and ins_valid and not Full;
        outs_valid <= ins_valid and not Full;
        outs <= ins;
      else
        -- Curr < Tail.
        CurrEn <= outs_ready;

        ctrl_ready <= outs_ready;
        outs_valid <= '1';
        outs <= Memory(Curr);
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
        outs <= ins;
      else
        -- Curr < Tail.
        CurrEn <= outs_ready;
        HeadEn <= outs_ready;

        ctrl_ready <= outs_ready;
        outs_valid <= '1';
        outs <= Memory(Curr);
      end if;
      outs_spec <= "1";
    elsif ctrl_valid = '1' and ctrl = "010" then
      -- RESEND
      -- Head < Curr is assumed from the specification.
      HeadEn <= outs_ready;

      ctrl_ready <= outs_ready;
      outs_valid <= '1';
      outs <= Memory(Head);
      outs_spec <= "0";
    elsif ctrl_valid = '1' and ctrl = "100" then
      -- NO_CMP
      -- Head = Curr is assumed from the specification.
      if Empty = '1' then
        CurrEn <= outs_ready and ins_valid and not Full;
        HeadEn <= outs_ready and ins_valid and not Full;

        ctrl_ready <= outs_ready and ins_valid and not Full;
        outs_valid <= ins_valid and not Full;
        outs <= ins;
      else
        -- `Empty = '0'` ensures Curr < Tail.
        CurrEn <= outs_ready;
        HeadEn <= outs_ready;

        ctrl_ready <= outs_ready;
        outs_valid <= '1';
        outs <= Memory(Head);
      end if;
      outs_spec <= "0";
    end if;
  end process;

  ----------------------------------------------------------------
  -- Sequential Process
  ----------------------------------------------------------------

  
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
          Tail <= (Tail + 1) mod 16;
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
          Head  <= (Head + 1) mod 16;
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
          Curr <= (Curr + 1) mod 16;
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
          if ((Tail + 2) mod 16 = Head) then
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
          if ((Head + 1) mod 16 = Tail) then
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
          if ((Curr + 1) mod 16 = Tail) then
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

