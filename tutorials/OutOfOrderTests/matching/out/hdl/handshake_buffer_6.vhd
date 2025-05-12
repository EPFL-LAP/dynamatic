-- handshake_buffer_6 : buffer({'num_slots': 1, 'port_types': {'ins': '!handshake.control<>', 'outs': '!handshake.control<>'}, 'timing': '#handshake<timing {D: 1, V: 1, R: 0}>', 'bitwidth': 0, 'transparent': False, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of oehb_dataless
entity handshake_buffer_6 is
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

-- Architecture of oehb_dataless
architecture arch of handshake_buffer_6 is
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

