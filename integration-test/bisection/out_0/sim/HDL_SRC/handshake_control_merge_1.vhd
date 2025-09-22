-- handshake_control_merge_1 : control_merge({'size': 3, 'data_bitwidth': 0, 'index_bitwidth': 2, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of merge_dataless
entity handshake_control_merge_1_merge is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    ins_valid : in  std_logic_vector(3 - 1 downto 0);
    ins_ready : out std_logic_vector(3 - 1 downto 0);
    -- output channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of merge_dataless
architecture arch of handshake_control_merge_1_merge is
begin
  process (ins_valid, outs_ready)
    variable tmp_valid_out : std_logic;
    variable tmp_ready_out : std_logic_vector(3 - 1 downto 0);
  begin
    tmp_valid_out := '0';
    tmp_ready_out := (others => '0');

    for i in 0 to (3 - 1) loop
      if (ins_valid(i) = '1') then
        tmp_valid_out := '1';
        tmp_ready_out(i) := outs_ready;
        exit;
      end if;
    end loop;

    outs_valid <= tmp_valid_out;
    ins_ready <= tmp_ready_out;
  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_r_dataless
entity handshake_control_merge_1_one_slot_break_r_dataless is
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
architecture arch of handshake_control_merge_1_one_slot_break_r_dataless is
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
entity handshake_control_merge_1_one_slot_break_r is
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
architecture arch of handshake_control_merge_1_one_slot_break_r is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(2 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_control_merge_1_one_slot_break_r_dataless
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
entity handshake_control_merge_1_fork_or_n is
  port (
    -- inputs
    ins : in std_logic_vector(2 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of or_n
architecture arch of handshake_control_merge_1_fork_or_n is
  signal all_zeros : std_logic_vector(2 - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of eager_fork_register_block
entity handshake_control_merge_1_fork_regblock is
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
architecture arch of handshake_control_merge_1_fork_regblock is
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
entity handshake_control_merge_1_fork is
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
architecture arch of handshake_control_merge_1_fork is
  signal blockStopArray : std_logic_vector(2 - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.handshake_control_merge_1_fork_or_n
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in 2 - 1 downto 0 generate
    regblock : entity work.handshake_control_merge_1_fork_regblock(arch)
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
entity handshake_control_merge_1 is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channels
    
    ins_valid : in  std_logic_vector(3 - 1 downto 0);
    ins_ready : out std_logic_vector(3 - 1 downto 0);
    -- data output channel
    
    outs_valid : out std_logic;
    outs_ready : in  std_logic;
    -- index output channel
    index       : out std_logic_vector(2 - 1 downto 0);
    index_valid : out std_logic;
    index_ready : in  std_logic
  );
end entity;

-- Architecture of control_merge
architecture arch of handshake_control_merge_1 is
  
  signal merge_outs_valid : std_logic;
  signal buf_ins_ready, buf_outs_valid : std_logic;
  signal fork_ins_ready : std_logic;
  signal index_internal : std_logic_vector(2 - 1 downto 0);
begin
  process (ins_valid)
  begin
    index_internal <= (2 - 1 downto 0 => '0');
    for i in 0 to (3 - 1) loop
      if (ins_valid(i) = '1') then
        index_internal <= std_logic_vector(to_unsigned(i, 2));
        exit;
      end if;
    end loop;
  end process;

  merge_ins : entity work.handshake_control_merge_1_merge(arch)
    port map(
      clk => clk,
      rst => rst,
      
      ins_valid => ins_valid,
      ins_ready => ins_ready,
      
      outs_valid => merge_outs_valid,
      outs_ready => buf_ins_ready
    );

  one_slot_break_r : entity work.handshake_control_merge_1_one_slot_break_r(arch)
    port map(
      clk => clk,
      rst => rst,
      ins(1 downto 0) => index_internal,
      
      ins_valid => merge_outs_valid,
      ins_ready => buf_ins_ready,
      outs(1 downto 0) => index,
      
      outs_valid => buf_outs_valid,
      outs_ready => fork_ins_ready
    );

  fork_valid : entity work.handshake_control_merge_1_fork(arch)
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

