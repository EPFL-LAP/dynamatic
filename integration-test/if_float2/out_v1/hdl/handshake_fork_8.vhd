-- handshake_fork_8 : fork({'size': 8, 'bitwidth': 1, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of or_n
entity handshake_fork_8_inner_inner_or_n is
  port (
    -- inputs
    ins : in std_logic_vector(8 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of or_n
architecture arch of handshake_fork_8_inner_inner_or_n is
  signal all_zeros : std_logic_vector(8 - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of eager_fork_register_block
entity handshake_fork_8_inner_inner_regblock is
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
architecture arch of handshake_fork_8_inner_inner_regblock is
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
entity handshake_fork_8_inner_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(8 - 1 downto 0);
    outs_ready : in  std_logic_vector(8 - 1 downto 0)
  );
end entity;

-- Architecture of fork_dataless
architecture arch of handshake_fork_8_inner_inner is
  signal blockStopArray : std_logic_vector(8 - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.handshake_fork_8_inner_inner_or_n
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in 8 - 1 downto 0 generate
    regblock : entity work.handshake_fork_8_inner_inner_regblock(arch)
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
use work.types.all;

-- Entity of fork
entity handshake_fork_8_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(2 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array(8 - 1 downto 0)(2 - 1 downto 0);
    outs_valid : out std_logic_vector(8 - 1 downto 0);
    outs_ready : in  std_logic_vector(8 - 1 downto 0)
  );
end entity;

-- Architecture of fork
architecture arch of handshake_fork_8_inner is
begin
  control : entity work.handshake_fork_8_inner_inner
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => ins_ready,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (ins)
  begin
    for i in 0 to 8 - 1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_fork_8 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in std_logic_vector(1 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0);
    outs : out data_array(8 - 1 downto 0)(1 - 1 downto 0);
    outs_valid : out std_logic_vector(8 - 1 downto 0);
    outs_ready : in std_logic_vector(8 - 1 downto 0);
    outs_0_spec : out std_logic_vector(1 - 1 downto 0);
    outs_1_spec : out std_logic_vector(1 - 1 downto 0);
    outs_2_spec : out std_logic_vector(1 - 1 downto 0);
    outs_3_spec : out std_logic_vector(1 - 1 downto 0);
    outs_4_spec : out std_logic_vector(1 - 1 downto 0);
    outs_5_spec : out std_logic_vector(1 - 1 downto 0);
    outs_6_spec : out std_logic_vector(1 - 1 downto 0);
    outs_7_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (concat)
architecture arch of handshake_fork_8 is
  signal ins_concat : std_logic_vector(1 downto 0);
  signal ins_concat_valid : std_logic;
  signal ins_concat_ready : std_logic;
  signal outs_concat : data_array(7 downto 0)(1 downto 0);
  signal outs_concat_valid : std_logic_vector(7 downto 0);
  signal outs_concat_ready : std_logic_vector(7 downto 0);
begin
  -- Concate/slice data and extra signals
  ins_concat(1 - 1 downto 0) <= ins;
  ins_concat(1 downto 1) <= ins_spec;
  ins_concat_valid <= ins_valid;
  ins_ready <= ins_concat_ready;
  outs(0) <= outs_concat(0)(1 - 1 downto 0);
  outs_0_spec <= outs_concat(0)(1 downto 1);
  outs(1) <= outs_concat(1)(1 - 1 downto 0);
  outs_1_spec <= outs_concat(1)(1 downto 1);
  outs(2) <= outs_concat(2)(1 - 1 downto 0);
  outs_2_spec <= outs_concat(2)(1 downto 1);
  outs(3) <= outs_concat(3)(1 - 1 downto 0);
  outs_3_spec <= outs_concat(3)(1 downto 1);
  outs(4) <= outs_concat(4)(1 - 1 downto 0);
  outs_4_spec <= outs_concat(4)(1 downto 1);
  outs(5) <= outs_concat(5)(1 - 1 downto 0);
  outs_5_spec <= outs_concat(5)(1 downto 1);
  outs(6) <= outs_concat(6)(1 - 1 downto 0);
  outs_6_spec <= outs_concat(6)(1 downto 1);
  outs(7) <= outs_concat(7)(1 - 1 downto 0);
  outs_7_spec <= outs_concat(7)(1 downto 1);
  outs_valid <= outs_concat_valid;
  outs_concat_ready <= outs_ready;

  inner : entity work.handshake_fork_8_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => ins_concat,
      ins_valid => ins_concat_valid,
      ins_ready => ins_concat_ready,
      outs => outs_concat,
      outs_valid => outs_concat_valid,
      outs_ready => outs_concat_ready
    );
end architecture;

