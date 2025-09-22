-- handshake_load_2 : load({'addr_bitwidth': 10, 'data_bitwidth': 32, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_r_dataless
entity handshake_load_2_inner_addr_one_slot_break_r_dataless is
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
architecture arch of handshake_load_2_inner_addr_one_slot_break_r_dataless is
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
entity handshake_load_2_inner_addr_one_slot_break_r is
  port (
    clk : in std_logic;
    rst : in std_logic;
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

-- Architecture of one_slot_break_r
architecture arch of handshake_load_2_inner_addr_one_slot_break_r is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(10 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_load_2_inner_addr_one_slot_break_r_dataless
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

-- Entity of one_slot_break_r_dataless
entity handshake_load_2_inner_data_one_slot_break_r_dataless is
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
architecture arch of handshake_load_2_inner_data_one_slot_break_r_dataless is
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
entity handshake_load_2_inner_data_one_slot_break_r is
  port (
    clk : in std_logic;
    rst : in std_logic;
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

-- Architecture of one_slot_break_r
architecture arch of handshake_load_2_inner_data_one_slot_break_r is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(32 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_load_2_inner_data_one_slot_break_r_dataless
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
entity handshake_load_2_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
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
architecture arch of handshake_load_2_inner is
begin
  addr_one_slot_break_r : entity work.handshake_load_2_inner_addr_one_slot_break_r(arch)
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

  data_one_slot_break_r : entity work.handshake_load_2_inner_data_one_slot_break_r(arch)
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv_dataless
entity handshake_load_2_fifo_break_dv_inner is
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

-- Architecture of one_slot_break_dv_dataless
architecture arch of handshake_load_2_fifo_break_dv_inner is
  signal outputValid : std_logic;
begin
  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outputValid <= '0';
      else
        outputValid <= ins_valid or (outputValid and not outs_ready);
      end if;
    end if;
  end process;

  ins_ready  <= not outputValid or outs_ready;
  outs_valid <= outputValid;

  
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv
entity handshake_load_2_fifo_break_dv is
  port (
    clk : in std_logic;
    rst : in std_logic;
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

-- Architecture of one_slot_break_dv
architecture arch of handshake_load_2_fifo_break_dv is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.handshake_load_2_fifo_break_dv_inner
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins_ready  => inputReady,
      outs_valid => outs_valid,
      outs_ready => outs_ready
    );

  process (clk) is
  begin
    if (rising_edge(clk)) then
      if (rst = '1') then
        outs <= (others => '0');
      elsif (regEn) then
        outs <= ins;
      end if;
    end if;
  end process;

  ins_ready <= inputReady;
  regEn     <= inputReady and ins_valid;

  
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of signal manager
entity handshake_load_2 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    addrIn : in std_logic_vector(10 - 1 downto 0);
    addrIn_valid : in std_logic;
    addrIn_ready : out std_logic;
    addrIn_spec : in std_logic_vector(1 - 1 downto 0);
    dataFromMem : in std_logic_vector(32 - 1 downto 0);
    dataFromMem_valid : in std_logic;
    dataFromMem_ready : out std_logic;
    addrOut : out std_logic_vector(10 - 1 downto 0);
    addrOut_valid : out std_logic;
    addrOut_ready : in std_logic;
    dataOut : out std_logic_vector(32 - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic;
    dataOut_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of load signal manager
architecture arch of handshake_load_2 is
  signal signals_pre_buffer, signals_post_buffer : std_logic_vector(1 - 1 downto 0);
  signal transfer_in, transfer_out : std_logic;
begin
  -- Transfer signal assignments
  transfer_in <= addrIn_valid and addrIn_ready;
  transfer_out <= dataOut_valid and dataOut_ready;

  -- Concat/slice extra signals
  signals_pre_buffer(0 downto 0) <= addrIn_spec;
  dataOut_spec <= signals_post_buffer(0 downto 0);

  -- Buffer to store extra signals for in-flight memory requests
  -- LoadOp is assumed to be connected to a memory controller
  -- Use fifo_break_dv with latency 1 (MC latency)
  fifo_break_dv : entity work.handshake_load_2_fifo_break_dv(arch)
    port map(
      clk => clk,
      rst => rst,
      ins => signals_pre_buffer,
      ins_valid => transfer_in,
      ins_ready => open,
      outs => signals_post_buffer,
      outs_valid => open,
      outs_ready => transfer_out
    );

  inner : entity work.handshake_load_2_inner(arch)
    port map(
      clk => clk,
      rst => rst,
      addrIn => addrIn,
      addrIn_valid => addrIn_valid,
      addrIn_ready => addrIn_ready,
      addrOut => addrOut,
      addrOut_valid => addrOut_valid,
      addrOut_ready => addrOut_ready,
      dataFromMem => dataFromMem,
      dataFromMem_valid => dataFromMem_valid,
      dataFromMem_ready => dataFromMem_ready,
      dataOut => dataOut,
      dataOut_valid => dataOut_valid,
      dataOut_ready => dataOut_ready
    );
end architecture;

