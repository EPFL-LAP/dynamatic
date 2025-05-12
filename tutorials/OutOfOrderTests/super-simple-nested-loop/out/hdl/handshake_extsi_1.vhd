-- handshake_extsi_1 : extsi({'port_types': {'ins': '!handshake.channel<i6>', 'outs': '!handshake.channel<i7>'}, 'input_bitwidth': 6, 'output_bitwidth': 7, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi
entity handshake_extsi_1 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(6 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(7 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of extsi
architecture arch of handshake_extsi_1 is
begin
  outs(7 - 1 downto 6) <= (7 - 6 - 1 downto 0 => ins(6 - 1));
  outs(6 - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

