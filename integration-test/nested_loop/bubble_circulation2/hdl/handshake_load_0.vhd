-- handshake_load_0 : load({'addr_bitwidth': 10, 'data_bitwidth': 32, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of tehb_dataless
entity handshake_load_0_addr_tehb_dataless is
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
architecture arch of handshake_load_0_addr_tehb_dataless is
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
entity handshake_load_0_addr_tehb is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(10 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(10 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tehb
architecture arch of handshake_load_0_addr_tehb is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(10 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_load_0_addr_tehb_dataless
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

-- Entity of tehb_dataless
entity handshake_load_0_data_tehb_dataless is
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
architecture arch of handshake_load_0_data_tehb_dataless is
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
entity handshake_load_0_data_tehb is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(32 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of tehb
architecture arch of handshake_load_0_data_tehb is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(32 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_load_0_data_tehb_dataless
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

-- Entity of load
entity handshake_load_0 is
  port (
    clk, rst : in std_logic;
    -- address from circuit channel
    addrIn       : in  std_logic_vector(10 - 1 downto 0);
    addrIn_valid : in  std_logic;
    addrIn_ready : out std_logic;
    -- address to interface channel
    addrOut       : out std_logic_vector(10 - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in  std_logic;
    -- data from interface channel
    dataFromMem       : in  std_logic_vector(32 - 1 downto 0);
    dataFromMem_valid : in  std_logic;
    dataFromMem_ready : out std_logic;
    -- data from memory channel
    dataOut       : out std_logic_vector(32 - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in  std_logic
  );
end entity;

-- Architecture of load
architecture arch of handshake_load_0 is
begin
  addr_tehb : entity work.handshake_load_0_addr_tehb(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => addrIn,
      ins_valid => addrIn_valid,
      ins_ready => addrIn_ready,
      -- output channel
      outs       => addrOut,
      outs_valid => addrOut_valid,
      outs_ready => addrOut_ready
    );

  data_tehb : entity work.handshake_load_0_data_tehb(arch)
    port map(
      clk => clk,
      rst => rst,
      -- input channel
      ins       => dataFromMem,
      ins_valid => dataFromMem_valid,
      ins_ready => dataFromMem_ready,
      -- output channel
      outs       => dataOut,
      outs_valid => dataOut_valid,
      outs_ready => dataOut_ready
    );
end architecture;

