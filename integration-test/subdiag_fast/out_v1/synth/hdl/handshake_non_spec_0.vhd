-- handshake_non_spec_0 : non_spec({'bitwidth': 0, 'extra_signals': {'spec': 1}})


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

-- Entity of non_spec
entity handshake_non_spec_0 is
  port (
    clk, rst : in  std_logic;
    
    dataIn_valid : in std_logic;
    dataIn_ready : out std_logic;
    
    dataOut_valid : out std_logic;
    dataOut_ready : in std_logic;
    dataOut_spec : out std_logic_vector(0 downto 0)
  );
end entity;

-- Architecture of non_spec
architecture arch of handshake_non_spec_0 is
begin
  
  dataOut_valid <= dataIn_valid;
  dataIn_ready <= dataOut_ready;
  dataOut_spec <= "0";
end architecture;

