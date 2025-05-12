-- handshake_extsi_3 : extsi({'port_types': {'ins': '!handshake.channel<i2>', 'outs': '!handshake.channel<i7>'}, 'input_bitwidth': 2, 'output_bitwidth': 7, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi
entity handshake_extsi_3 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(2 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(7 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of extsi
architecture arch of handshake_extsi_3 is
begin
  outs(7 - 1 downto 2) <= (7 - 2 - 1 downto 0 => ins(2 - 1));
  outs(2 - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

