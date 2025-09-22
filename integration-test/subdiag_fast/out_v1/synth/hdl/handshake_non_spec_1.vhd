-- handshake_non_spec_1 : non_spec({'bitwidth': 32, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of non_spec
entity handshake_non_spec_1 is
  port (
    clk, rst : in  std_logic;
    dataIn : in std_logic_vector(32 - 1 downto 0);
    dataIn_valid : in std_logic;
    dataIn_ready : out std_logic;
    dataOut : out std_logic_vector(32 - 1 downto 0);
    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic;
    dataOut_spec : out std_logic_vector(0 downto 0)
  );
end entity;

-- Architecture of non_spec
architecture arch of handshake_non_spec_1 is
begin
  dataOut <= dataIn;
  dataOut_valid <= dataIn_valid;
  dataIn_ready <= dataOut_ready;
  dataOut_spec <= "0";
end architecture;

