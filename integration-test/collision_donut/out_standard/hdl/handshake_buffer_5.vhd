-- handshake_buffer_5 : buffer({'num_slots': 1, 'bitwidth': 32, 'buffer_type': 'ONE_SLOT_BREAK_DV', 'extra_signals': {}, 'debug_counter': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv_dataless
entity handshake_buffer_5_inner is
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
architecture arch of handshake_buffer_5_inner is
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
entity handshake_buffer_5 is
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

-- Architecture of one_slot_break_dv
architecture arch of handshake_buffer_5 is
  signal regEn, inputReady : std_logic;
begin

  control : entity work.handshake_buffer_5_inner
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

