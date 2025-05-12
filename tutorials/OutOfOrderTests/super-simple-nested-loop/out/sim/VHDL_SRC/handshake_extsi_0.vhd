-- handshake_extsi_0 : extsi({'port_types': {'ins': '!handshake.channel<i1>', 'outs': '!handshake.channel<i6>'}, 'input_bitwidth': 1, 'output_bitwidth': 6, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of extsi
entity handshake_extsi_0 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(1 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(6 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of extsi
architecture arch of handshake_extsi_0 is
begin
  outs(6 - 1 downto 1) <= (6 - 1 - 1 downto 0 => ins(1 - 1));
  outs(1 - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;

