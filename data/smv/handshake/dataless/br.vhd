library ieee;
use ieee.std_logic_1164.all;

entity br_dataless is
  port (
    clk, rst : in std_logic;
    -- input channel
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- input channel
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of br_dataless is
begin
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
end architecture;
