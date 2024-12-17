library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity negf is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of negf is
begin
  outs(DATA_TYPE - 1)          <= ins(DATA_TYPE - 1) xor '1';
  outs(DATA_TYPE - 2 downto 0) <= ins(DATA_TYPE - 2 downto 0);
  outs_valid                  <= ins_valid;
  ins_ready                   <= outs_ready;
end architecture;
