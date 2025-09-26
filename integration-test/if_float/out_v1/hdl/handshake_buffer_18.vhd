-- handshake_buffer_18 : buffer({'num_slots': 1, 'bitwidth': 0, 'buffer_type': 'ONE_SLOT_BREAK_R', 'extra_signals': {'spec': 1}, 'debug_counter': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_r_dataless
entity handshake_buffer_18_inner_dataless is
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
architecture arch of handshake_buffer_18_inner_dataless is
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
entity handshake_buffer_18_inner is
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

-- Architecture of one_slot_break_r
architecture arch of handshake_buffer_18_inner is
  signal regEnable, regNotFull : std_logic;
  signal dataReg               : std_logic_vector(1 - 1 downto 0);
begin
  regEnable <= regNotFull and ins_valid and not outs_ready;

  control : entity work.handshake_buffer_18_inner_dataless
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

-- Entity of signal manager
entity handshake_buffer_18 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (concat)
architecture arch of handshake_buffer_18 is
  signal ins_concat : std_logic_vector(0 downto 0);
  signal ins_concat_valid : std_logic;
  signal ins_concat_ready : std_logic;
  signal outs_concat : std_logic_vector(0 downto 0);
  signal outs_concat_valid : std_logic;
  signal outs_concat_ready : std_logic;
begin
  -- Concate/slice data and extra signals
  ins_concat(0 downto 0) <= ins_spec;
  ins_concat_valid <= ins_valid;
  ins_ready <= ins_concat_ready;
  outs_spec <= outs_concat(0 downto 0);
  outs_valid <= outs_concat_valid;
  outs_concat_ready <= outs_ready;

  inner : entity work.handshake_buffer_18_inner(arch)
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

