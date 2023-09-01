library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity d_return is
  generic (
    BITWIDTH : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end entity;

architecture arch of d_return is

begin

  tehb : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => ins_ready,
      ins        => ins,
      outs       => outs
    );

end architecture;
