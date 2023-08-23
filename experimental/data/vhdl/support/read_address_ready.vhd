library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity read_address_ready is
  generic (
    ARBITER_SIZE : natural
  );
  port (
    sel    : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    nReady : in std_logic_vector(ARBITER_SIZE - 1 downto 0);
    ready  : out std_logic_vector(ARBITER_SIZE - 1 downto 0)
  );
end entity;

architecture arch of read_address_ready is
begin
  GEN1 : for I in 0 to ARBITER_SIZE - 1 generate
    ready(I) <= nReady(I) and sel(I);
  end generate GEN1;
end architecture;
