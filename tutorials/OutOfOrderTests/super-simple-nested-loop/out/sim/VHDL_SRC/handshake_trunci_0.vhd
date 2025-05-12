-- handshake_trunci_0 : trunci({'port_types': {'ins': '!handshake.channel<i6>', 'outs': '!handshake.channel<i5>'}, 'input_bitwidth': 6, 'output_bitwidth': 5, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of trunci
entity handshake_trunci_0 is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(6 - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(5 - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

-- Architecture of trunci
architecture arch of handshake_trunci_0 is
begin
  outs       <= ins(5 - 1 downto 0);
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
end architecture;

