-- handshake_buffer_9 : buffer({'num_slots': 1, 'bitwidth': 0, 'buffer_type': 'ONE_SLOT_BREAK_DV', 'extra_signals': {}, 'debug_counter': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_dv_dataless
entity handshake_buffer_9 is
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
architecture arch of handshake_buffer_9 is
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

