library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity source is
  port (
    -- inputs
    clk, rst   : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
  );
end entity;

architecture arch of source is
begin
  outs_valid <= '1';
end arch;

entity source_with_tag is
  port (
    -- inputs
    clk, rst   : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs_valid : out std_logic
    outs_spec_tag : out std_logic
  );
end entity;

architecture arch of source_with_tag is
begin
  outs_valid <= '1';
  outs_spec_tag <= '0';
end arch;
