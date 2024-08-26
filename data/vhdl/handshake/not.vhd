library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity logic_not is
  generic (
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of logic_not is
begin
  outs       <= not ins;
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
end architecture;
