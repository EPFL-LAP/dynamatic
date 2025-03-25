
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of elastic_fifo_inner
entity spec_commit_inner_fifo_disc_fifo is
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(1 - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

-- Architecture of elastic_fifo_inner
architecture arch of spec_commit_inner_fifo_disc_fifo is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to 1 - 1;
  signal Head       : natural range 0 to 1 - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;
  type FIFO_Memory is array (0 to 1 - 1) of std_logic_vector (1 - 1 downto 0);
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
          Tail <= (Tail + 1) mod 1;
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
          Head <= (Head + 1) mod 1;
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
          if ((Tail + 1) mod 1 = Head) then
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
          if ((Head + 1) mod 1 = Tail) then
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo
entity spec_commit_inner_fifo_disc is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(1 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tfifo
architecture arch of spec_commit_inner_fifo_disc is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
  signal fifo_in, fifo_out        : std_logic_vector(1 - 1 downto 0);
begin

  process (mux_sel, fifo_out, ins) is
  begin
    if (mux_sel = '1') then
      outs <= fifo_out;
    else
      outs <= ins;
    end if;
  end process;

  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;

  fifo_nready <= outs_ready;
  fifo_in     <= ins;

  fifo : entity work.spec_commit_inner_fifo_disc_fifo(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => fifo_in,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs       => fifo_out,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of and_n
entity spec_commit_inner_cond_br_inner_join_and_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of and_n
architecture arch of spec_commit_inner_cond_br_inner_join_and_n is
  signal all_ones : std_logic_vector(2 - 1 downto 0) := (others => '1');
begin
  outs <= '1' when ins = all_ones else '0';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of join
entity spec_commit_inner_cond_br_inner_join is
  port (
    -- inputs
    ins_valid  : in std_logic_vector(2 - 1 downto 0);
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of join
architecture arch of spec_commit_inner_cond_br_inner_join is
  signal allValid : std_logic;
begin
  allValidAndGate : entity work.spec_commit_inner_cond_br_inner_join_and_n port map(ins_valid, allValid);
  outs_valid <= allValid;

  process (ins_valid, outs_ready)
    variable singlePValid : std_logic_vector(2 - 1 downto 0);
  begin
    for i in 0 to 2 - 1 loop
      singlePValid(i) := '1';
      for j in 0 to 2 - 1 loop
        if (i /= j) then
          singlePValid(i) := (singlePValid(i) and ins_valid(j));
        end if;
      end loop;
    end loop;
    for i in 0 to 2 - 1 loop
      ins_ready(i) <= (singlePValid(i) and outs_ready);
    end loop;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br_dataless
entity spec_commit_inner_cond_br_inner is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

-- Architecture of cond_br_dataless
architecture arch of spec_commit_inner_cond_br_inner is
  signal branchInputs_valid, branch_ready : std_logic;
begin

  join : entity work.spec_commit_inner_cond_br_inner_join(arch)
    port map(
      -- input channels
      ins_valid(0) => data_valid,
      ins_valid(1) => condition_valid,
      ins_ready(0) => data_ready,
      ins_ready(1) => condition_ready,
      -- output channel
      outs_valid => branchInputs_valid,
      outs_ready => branch_ready
    );

  trueOut_valid  <= condition(0) and branchInputs_valid;
  falseOut_valid <= (not condition(0)) and branchInputs_valid;
  branch_ready   <= (falseOut_ready and not condition(0)) or (trueOut_ready and condition(0));
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of cond_br
entity spec_commit_inner_cond_br is
  port (
    clk, rst : in std_logic;
    -- data input channel
    data       : in  std_logic_vector(12 - 1 downto 0);
    data_valid : in  std_logic;
    data_ready : out std_logic;
    -- condition input channel
    condition       : in  std_logic_vector(0 downto 0);
    condition_valid : in  std_logic;
    condition_ready : out std_logic;
    -- true output channel
    trueOut       : out std_logic_vector(12 - 1 downto 0);
    trueOut_valid : out std_logic;
    trueOut_ready : in  std_logic;
    -- false output channel
    falseOut       : out std_logic_vector(12 - 1 downto 0);
    falseOut_valid : out std_logic;
    falseOut_ready : in  std_logic
  );
end entity;

-- Architecture of cond_br
architecture arch of spec_commit_inner_cond_br is
begin
  control : entity work.spec_commit_inner_cond_br_inner
    port map(
      clk             => clk,
      rst             => rst,
      data_valid      => data_valid,
      data_ready      => data_ready,
      condition       => condition,
      condition_valid => condition_valid,
      condition_ready => condition_ready,
      trueOut_valid   => trueOut_valid,
      trueOut_ready   => trueOut_ready,
      falseOut_valid  => falseOut_valid,
      falseOut_ready  => falseOut_ready
    );

  trueOut  <= data;
  falseOut <= data;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of elastic_fifo_inner
entity spec_commit_inner_buff_fifo is
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(12 - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

-- Architecture of elastic_fifo_inner
architecture arch of spec_commit_inner_buff_fifo is

  signal ReadEn     : std_logic := '0';
  signal WriteEn    : std_logic := '0';
  signal Tail       : natural range 0 to 1 - 1;
  signal Head       : natural range 0 to 1 - 1;
  signal Empty      : std_logic;
  signal Full       : std_logic;
  signal Bypass     : std_logic;
  signal fifo_valid : std_logic;
  type FIFO_Memory is array (0 to 1 - 1) of std_logic_vector (12 - 1 downto 0);
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
          Tail <= (Tail + 1) mod 1;
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
          Head <= (Head + 1) mod 1;
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
          if ((Tail + 1) mod 1 = Head) then
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
          if ((Head + 1) mod 1 = Tail) then
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tfifo
entity spec_commit_inner_buff is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(12 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tfifo
architecture arch of spec_commit_inner_buff is
  signal mux_sel                  : std_logic;
  signal fifo_valid, fifo_ready   : std_logic;
  signal fifo_pvalid, fifo_nready : std_logic;
  signal fifo_in, fifo_out        : std_logic_vector(12 - 1 downto 0);
begin

  process (mux_sel, fifo_out, ins) is
  begin
    if (mux_sel = '1') then
      outs <= fifo_out;
    else
      outs <= ins;
    end if;
  end process;

  outs_valid  <= ins_valid or fifo_valid;
  ins_ready   <= fifo_ready or outs_ready;
  fifo_pvalid <= ins_valid and (not outs_ready or fifo_valid);
  mux_sel     <= fifo_valid;

  fifo_nready <= outs_ready;
  fifo_in     <= ins;

  fifo : entity work.spec_commit_inner_buff_fifo(arch)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => fifo_in,
      ins_valid  => fifo_pvalid,
      outs_ready => fifo_nready,
      -- outputs
      outs       => fifo_out,
      outs_valid => fifo_valid,
      ins_ready  => fifo_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge_notehb
entity spec_commit_inner_merge_inner is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(12 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge_notehb
architecture arch of spec_commit_inner_merge_inner is
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(12 - 1 downto 0);
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(2 - 1 downto 0);
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');

    for I in 0 to (2 - 1) loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;

    outs <= std_logic_vector(resize(tmp_data_out, 12));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb_dataless
entity spec_commit_inner_merge_tehb_dataless is
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

-- Architecture of tehb_dataless
architecture arch of spec_commit_inner_merge_tehb_dataless is
  signal fullReg, outputValid : std_logic;
begin
  outputValid <= ins_valid or fullReg;

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        fullReg <= '0';
      else
        fullReg <= outputValid and not outs_ready;
      end if;
    end if;
  end process;

  ins_ready  <= not fullReg;
  outs_valid <= outputValid;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb
entity spec_commit_inner_merge_tehb is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(12 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tehb
architecture arch of spec_commit_inner_merge_tehb is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(12 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.spec_commit_inner_merge_tehb_dataless
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => regNotFull,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        dataReg <= (others => '0');
      elsif (regEnable) then
        dataReg <= ins;
      end if;
    end if;
  end process;

  process (regNotFull, dataReg, ins) is
  begin
    if (regNotFull) then
      outs <= ins;
    else
      outs <= dataReg;
    end if;
  end process;

  ins_ready <= regNotFull;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity spec_commit_inner_merge is
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(12 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge
architecture arch of spec_commit_inner_merge is
  signal tehb_data_in : std_logic_vector(12 - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin

  merge_ins : entity work.spec_commit_inner_merge_inner(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => tehb_ready,
      ins_ready  => ins_ready,
      outs       => tehb_data_in,
      outs_valid => tehb_pvalid
    );

  tehb : entity work.spec_commit_inner_merge_tehb(arch)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of spec_commit
entity spec_commit_inner is
  port (
    clk, rst : in  std_logic;
    ins : in std_logic_vector(12 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(0 downto 0);
    ctrl : in std_logic_vector(0 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    outs : out std_logic_vector(12 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic
  );
end entity;

-- Architecture of spec_commit
architecture arch of spec_commit_inner is
signal fifo_disc_outs : std_logic_vector(0 downto 0);
signal fifo_disc_outs_valid : std_logic;
signal fifo_disc_outs_ready : std_logic;

signal branch_in_condition : std_logic_vector(0 downto 0);
signal branch_in_condition_ready : std_logic;
signal branch_in_trueOut : std_logic_vector(12 - 1 downto 0);
signal branch_in_trueOut_valid : std_logic;
signal branch_in_trueOut_ready : std_logic;
signal branch_in_falseOut : std_logic_vector(12 - 1 downto 0);
signal branch_in_falseOut_valid : std_logic;
signal branch_in_falseOut_ready : std_logic;

signal buff_outs : std_logic_vector(12 - 1 downto 0);
signal buff_outs_valid : std_logic;
signal buff_outs_ready : std_logic;

signal branch_disc_trueOut : std_logic_vector(12 - 1 downto 0);
signal branch_disc_trueOut_valid : std_logic;
signal branch_disc_trueOut_ready : std_logic;
signal branch_disc_falseOut : std_logic_vector(12 - 1 downto 0);
signal branch_disc_falseOut_valid : std_logic;
signal branch_disc_falseOut_ready : std_logic;

signal merge_ins : data_array(1 downto 0)(12 - 1 downto 0);
signal merge_ins_valid : std_logic_vector(1 downto 0);
signal merge_ins_ready : std_logic_vector(1 downto 0);

begin

-- Design taken directly from the Speculation 2019 paper

fifo_disc: entity work.spec_commit_inner_fifo_disc(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => ctrl,
    ins_valid => ctrl_valid,
    ins_ready => ctrl_ready,
    outs => fifo_disc_outs,
    outs_valid => fifo_disc_outs_valid,
    outs_ready => fifo_disc_outs_ready
  );

branch_in_condition <= ins_spec;
branch_in: entity work.spec_commit_inner_cond_br(arch)
  port map (
    clk => clk,
    rst => rst,
    data => ins,
    data_valid => ins_valid,
    data_ready => ins_ready,
    condition => branch_in_condition,
    condition_valid => '1', -- always valid
    condition_ready => branch_in_condition_ready,
    trueOut => branch_in_trueOut,
    trueOut_valid => branch_in_trueOut_valid,
    trueOut_ready => branch_in_trueOut_ready,
    falseOut => branch_in_falseOut,
    falseOut_valid => branch_in_falseOut_valid,
    falseOut_ready => branch_in_falseOut_ready
  );

buff: entity work.spec_commit_inner_buff(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => branch_in_trueOut,
    ins_valid => branch_in_trueOut_valid,
    ins_ready => branch_in_trueOut_ready,
    outs => buff_outs,
    outs_valid => buff_outs_valid,
    outs_ready => buff_outs_ready
  );

branch_disc_trueOut_ready <= '1'; -- sink
branch_disc: entity work.spec_commit_inner_cond_br(arch)
  port map (
    clk => clk,
    rst => rst,
    data => buff_outs,
    data_valid => buff_outs_valid,
    data_ready => buff_outs_ready,
    condition => fifo_disc_outs,
    condition_valid => fifo_disc_outs_valid,
    condition_ready => fifo_disc_outs_ready,
    trueOut => branch_disc_trueOut,
    trueOut_valid => branch_disc_trueOut_valid,
    trueOut_ready => branch_disc_trueOut_ready,
    falseOut => branch_disc_falseOut,
    falseOut_valid => branch_disc_falseOut_valid,
    falseOut_ready => branch_disc_falseOut_ready
  );

merge_ins <= (branch_disc_falseOut, branch_in_falseOut);
merge_ins_valid <= (branch_disc_falseOut_valid, branch_in_falseOut_valid);
branch_disc_falseOut_ready <= merge_ins_ready(1);
branch_in_falseOut_ready <= merge_ins_ready(0);

merge_out: entity work.spec_commit_inner_merge(arch)
  port map (
    clk => clk,
    rst => rst,
    ins => merge_ins,
    ins_valid => merge_ins_valid,
    ins_ready => merge_ins_ready,
    outs => outs,
    outs_valid => outs_valid,
    outs_ready => outs_ready
  );

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity spec_commit is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in std_logic_vector(4 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0);
    ins_tag : in std_logic_vector(8 - 1 downto 0);
    ctrl : in std_logic_vector(1 - 1 downto 0);
    ctrl_valid : in std_logic;
    ctrl_ready : out std_logic;
    outs : out std_logic_vector(4 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_tag : out std_logic_vector(8 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (concat)
architecture arch of spec_commit is
  -- Concatenated data and extra signals
  signal ins_inner : std_logic_vector(12 - 1 downto 0);
  signal outs_inner : std_logic_vector(12 - 1 downto 0);
begin
  -- Concatenate data and extra signals
  ins_inner(4 - 1 downto 0) <= ins;
  ins_inner(11 downto 4) <= ins_tag;
  outs <= outs_inner(4 - 1 downto 0);
  outs_tag <= outs_inner(11 downto 4);

  inner : entity work.spec_commit_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      ins_spec => ins_spec,
      ctrl => ctrl,
      ctrl_valid => ctrl_valid,
      ctrl_ready => ctrl_ready,
      outs => outs_inner,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );
end architecture;
-- Signal manager generation info: spec_commit, {'type': 'concat', 'in_ports': [{'name': 'ins', 'bitwidth': 4, 'extra_signals': {'spec': 1, 'tag': 8}}, {'name': 'ctrl', 'bitwidth': 1}], 'out_ports': [{'name': 'outs', 'bitwidth': 4, 'extra_signals': {'tag': 8}}], 'extra_signals': {'tag': 8}, 'simple_ports': ['ctrl']}

