-- handshake_absf_0 : absf({'bitwidth': 32, 'extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Entity of absf
entity handshake_absf_0 is
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(32 - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(32 - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

-- Architecture of absf
architecture arch of handshake_absf_0 is
begin
  outs(32 - 1)          <= '0';
  outs(32 - 2 downto 0) <= ins(32 - 2 downto 0);
  outs_valid                  <= ins_valid;
  ins_ready                   <= outs_ready;
end architecture;

