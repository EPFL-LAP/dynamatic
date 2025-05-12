-- handshake_fork_7 : fork({'size': 3, 'port_types': {'ins': '!handshake.channel<i1>', 'outs_0': '!handshake.channel<i1>', 'outs_1': '!handshake.channel<i1>', 'outs_2': '!handshake.channel<i1>'}, 'bitwidth': 1, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;

-- Entity of or_n
entity handshake_fork_7_inner_or_n is
  port (
    -- inputs
    ins : in std_logic_vector(3 - 1 downto 0);
    -- outputs
    outs : out std_logic
  );
end entity;

-- Architecture of or_n
architecture arch of handshake_fork_7_inner_or_n is
  signal all_zeros : std_logic_vector(3 - 1 downto 0) := (others => '0');
begin
  outs <= '0' when ins = all_zeros else '1';
end architecture;

library ieee;
use ieee.std_logic_1164.all;

-- Entity of eager_fork_register_block
entity handshake_fork_7_inner_regblock is
  port (
    clk, rst : in std_logic;
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
architecture arch of handshake_fork_7_inner_regblock is
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
entity handshake_fork_7_inner is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs_valid : out std_logic_vector(3 - 1 downto 0);
    outs_ready : in  std_logic_vector(3 - 1 downto 0)
  );
end entity;

-- Architecture of fork_dataless
architecture arch of handshake_fork_7_inner is
  signal blockStopArray : std_logic_vector(3 - 1 downto 0);
  signal anyBlockStop   : std_logic;
  signal backpressure   : std_logic;
begin
  anyBlockFull : entity work.handshake_fork_7_inner_or_n
    port map(
      blockStopArray,
      anyBlockStop
    );

  ins_ready    <= not anyBlockStop;
  backpressure <= ins_valid and anyBlockStop;

  generateBlocks : for i in 3 - 1 downto 0 generate
    regblock : entity work.handshake_fork_7_inner_regblock(arch)
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
entity handshake_fork_7 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(1 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channels
    outs       : out data_array(3 - 1 downto 0)(1 - 1 downto 0);
    outs_valid : out std_logic_vector(3 - 1 downto 0);
    outs_ready : in  std_logic_vector(3 - 1 downto 0)
  );
end entity;

-- Architecture of fork
architecture arch of handshake_fork_7 is
begin
  control : entity work.handshake_fork_7_inner
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
    for i in 0 to 3 - 1 loop
      outs(i) <= ins;
    end loop;
  end process;
end architecture;

