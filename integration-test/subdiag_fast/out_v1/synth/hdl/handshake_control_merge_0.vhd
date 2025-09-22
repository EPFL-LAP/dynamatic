-- handshake_control_merge_0 : control_merge({'size': 2, 'data_bitwidth': 0, 'index_bitwidth': 1, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of merge
entity handshake_control_merge_0_inner_merge is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(1 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge
architecture arch of handshake_control_merge_0_inner_merge is
begin
  process (ins_valid, ins, outs_ready)
    variable tmp_data_out  : unsigned(1 - 1 downto 0);
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

    outs <= std_logic_vector(resize(tmp_data_out, 1));
    outs_valid  <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_r_dataless
entity handshake_control_merge_0_inner_one_slot_break_r_dataless is
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

-- Architecture of one_slot_break_r_dataless
architecture arch of handshake_control_merge_0_inner_one_slot_break_r_dataless is
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

-- Entity of one_slot_break_r
entity handshake_control_merge_0_inner_one_slot_break_r is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(2 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(2 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of one_slot_break_r
architecture arch of handshake_control_merge_0_inner_one_slot_break_r is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(2 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_control_merge_0_inner_one_slot_break_r_dataless
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

-- Entity of or_n
entity handshake_control_merge_0_inner_fork_or_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of or_n
architecture arch of handshake_control_merge_0_inner_fork_or_n is
  signal all_zeros : std_logic_vector(2 - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of eager_fork_register_block
entity handshake_control_merge_0_inner_fork_regblock is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- inputs
    ins_valid    : in std_logic;
    outs_ready   : in std_logic;
    backpressure : in std_logic;
    -- outputs
    outs_valid : out std_logic;
    blockStop  : out std_logic
  );
end entity;

-- Architecture of eager_fork_register_block
architecture arch of handshake_control_merge_0_inner_fork_regblock is
  signal transmitValue, keepValue : std_logic;
begin
  keepValue <= (not outs_ready) and transmitValue;

  process (clk)
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        transmitValue <= '1';
      else
        transmitValue <= keepValue or (not backpressure);
      end if;
    end if;
  end process;

  outs_valid <= transmitValue and ins_valid;
  blockStop  <= keepValue;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of fork_dataless
entity handshake_control_merge_0_inner_fork is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(2 - 1 downto 0);
    outs_ready : in  std_logic_vector(2 - 1 downto 0)
  );
end entity;

-- Architecture of fork_dataless
architecture arch of handshake_control_merge_0_inner_fork is
  signal blockStopArray : std_logic_vector(2 - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.handshake_control_merge_0_inner_fork_or_n
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in 2 - 1 downto 0 generate
    regblock : entity work.handshake_control_merge_0_inner_fork_regblock(arch)
      port map(
        -- inputs
        clk          => clk,
        rst          => rst,
        ins_valid    => ins_valid,
        outs_ready   => outs_ready(i),
        backpressure => backpressure,
        -- outputs
        outs_valid => outs_valid(i),
        blockStop  => blockStopArray(i)
      );
  end generate;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of control_merge
entity handshake_control_merge_0_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    ins       : in  data_array(2 - 1 downto 0)(1 - 1 downto 0);
    ins_valid : in  std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    -- data output channel
    outs       : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector(1 - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
  );
end entity;

-- Architecture of control_merge
architecture arch of handshake_control_merge_0_inner is
  signal merge_outs : std_logic_vector(1 - 1 downto 0);
  signal merge_outs_valid : std_logic;
  signal buf_ins_ready, buf_outs_valid : std_logic;
  signal fork_ins_ready : std_logic;
  signal index_internal : std_logic_vector(1 - 1 downto 0);
begin
  process (ins_valid)
  begin
    index_internal <= (1 - 1 downto 0 => '0');
    for i in 0 to (2 - 1) loop
      if (ins_valid(i) = '1') then
        index_internal <= std_logic_vector(to_unsigned(i, 1));
        exit;
      end if;
    end loop;
  end process;

  merge_ins : entity work.handshake_control_merge_0_inner_merge(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins,
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      outs => merge_outs,
      outs_valid => merge_outs_valid,
      outs_ready => buf_ins_ready
    );

  one_slot_break_r : entity work.handshake_control_merge_0_inner_one_slot_break_r(arch)
    port map(
      clk => clk,
      rst => rst,
      ins(0 downto 0) => index_internal,
      ins(1 downto 1) => merge_outs,
      ins_valid => merge_outs_valid,
      ins_ready => buf_ins_ready,
      outs(0 downto 0) => index,
      outs(1 downto 1) => outs,
      outs_valid => buf_outs_valid,
      outs_ready => fork_ins_ready
    );

  fork_valid : entity work.handshake_control_merge_0_inner_fork(arch)
    port map(
      clk => clk,
      rst => rst,
      ins_valid => buf_outs_valid,
      ins_ready => fork_ins_ready,
      outs_valid(0) => outs_valid,
      outs_valid(1) => index_valid,
      outs_ready(0) => outs_ready,
      outs_ready(1) => index_ready
    );
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_control_merge_0 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins_valid : in std_logic_vector(2 - 1 downto 0);
    ins_ready : out std_logic_vector(2 - 1 downto 0);
    ins_0_spec : in std_logic_vector(1 - 1 downto 0);
    ins_1_spec : in std_logic_vector(1 - 1 downto 0);
    index : out std_logic_vector(1 - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in std_logic;
    index_spec : out std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (cmerge)
architecture arch of handshake_control_merge_0 is
  signal ins_inner : data_array(2 - 1 downto 0)(0 + 1 - 1 downto 0);
  signal ins_inner_valid, ins_inner_ready : std_logic_vector(2 - 1 downto 0);
  signal outs_inner : std_logic_vector(0 + 1 - 1 downto 0);
  signal outs_inner_valid, outs_inner_ready : std_logic;
begin
  -- Concat/slice data and extra signals
  ins_inner(0)(0 downto 0) <= ins_0_spec;
  ins_inner(1)(0 downto 0) <= ins_1_spec;
  ins_inner_valid <= ins_valid;
  ins_ready <= ins_inner_ready;
  outs_spec <= outs_inner(0 downto 0);
  outs_valid <= outs_inner_valid;
  outs_inner_ready <= outs_ready;

  -- Assign index extra signals
    index_spec <= "0";

  inner : entity work.handshake_control_merge_0_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_inner,
      ins_valid => ins_inner_valid,
      ins_ready => ins_inner_ready,
      outs => outs_inner,
      outs_valid => outs_inner_valid,
      outs_ready => outs_inner_ready,
      index => index,
      index_valid => index_valid,
      index_ready => index_ready
    );
end architecture;

