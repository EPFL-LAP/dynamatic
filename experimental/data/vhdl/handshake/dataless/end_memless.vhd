library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity end_sync_memless_dataless is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic_vector(0 downto 0);
    ins_ready : out std_logic_vector(0 downto 0);
    -- output channel
    outs_valid : out std_logic_vector(0 downto 0);
    outs_ready : in  std_logic_vector(0 downto 0));
end entity;

architecture arch of end_sync_memless_dataless is
  signal valid : std_logic;
begin
  outs_valid(0) <= ins_valid(0);
  ins_ready(0)  <= ins_valid(0) and outs_ready(0);
end architecture;
