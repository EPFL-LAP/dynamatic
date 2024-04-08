library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity return is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk, rst   : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of return is
begin
  tehb : entity work.tehb(arch) generic map (BITWIDTH)
    port map(
      -- inputs
      clk        => clk,
      rst        => rst,
      ins        => ins,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      -- outputs
      outs       => outs,
      outs_valid => outs_valid,
      ins_ready  => ins_ready
    );
end architecture;
