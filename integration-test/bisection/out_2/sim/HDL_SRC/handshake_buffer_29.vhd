-- handshake_buffer_29 : buffer({'num_slots': 1, 'bitwidth': 0, 'buffer_type': 'ONE_SLOT_BREAK_R', 'extra_signals': {}, 'debug_counter': 0})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of one_slot_break_r_dataless
entity handshake_buffer_29 is
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
architecture arch of handshake_buffer_29 is
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

