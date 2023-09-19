library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.customTypes.all;
entity sink_node is

  generic (
    BITWIDTH : integer
  );

  port (
    -- inputs
    clk       : in std_logic;
    rst       : in std_logic;
    ins_valid : in std_logic;
    ins       : in std_logic_vector(BITWIDTH - 1 downto 0);
    -- outputs
    ins_ready : out std_logic);
end entity;

architecture arch of sink_node is

begin

  ins_ready <= '1';

end arch;
