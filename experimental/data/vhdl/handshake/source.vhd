library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity source is

  generic (
    BITWIDTH : integer
  );

  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic);
end source;

architecture arch of source is

begin

  outs_valid <= '1';

end arch;
