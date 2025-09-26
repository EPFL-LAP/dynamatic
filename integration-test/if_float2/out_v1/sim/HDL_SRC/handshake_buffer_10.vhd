-- handshake_buffer_10 : buffer({'num_slots': 1, 'bitwidth': 32, 'buffer_type': 'ONE_SLOT_BREAK_DV', 'extra_signals': {'spec': 1}, 'debug_counter': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv_dataless
entity handshake_buffer_10_inner_inner is
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
architecture arch of handshake_buffer_10_inner_inner is
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
entity handshake_buffer_10_inner is
  port (
    clk : in std_logic;
    rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(33 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(33 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of one_slot_break_dv
architecture arch of handshake_buffer_10_inner is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.handshake_buffer_10_inner_inner
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
entity handshake_buffer_10 is
  port(
    clk : in std_logic;
    rst : in std_logic;
    ins : in std_logic_vector(32 - 1 downto 0);
    ins_valid : in std_logic;
    ins_ready : out std_logic;
    ins_spec : in std_logic_vector(1 - 1 downto 0);
    outs : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in std_logic;
    outs_spec : out std_logic_vector(1 - 1 downto 0)
  );
end entity;

-- Architecture of signal manager (concat)
architecture arch of handshake_buffer_10 is
  signal ins_concat : std_logic_vector(32 downto 0);
  signal ins_concat_valid : std_logic;
  signal ins_concat_ready : std_logic;
  signal outs_concat : std_logic_vector(32 downto 0);
  signal outs_concat_valid : std_logic;
  signal outs_concat_ready : std_logic;
begin
  -- Concate/slice data and extra signals
  ins_concat(32 - 1 downto 0) <= ins;
  ins_concat(32 downto 32) <= ins_spec;
  ins_concat_valid <= ins_valid;
  ins_ready <= ins_concat_ready;
  outs <= outs_concat(32 - 1 downto 0);
  outs_spec <= outs_concat(32 downto 32);
  outs_valid <= outs_concat_valid;
  outs_concat_ready <= outs_ready;

  inner : entity work.handshake_buffer_10_inner(arch)
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

