library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
entity negf_node is
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
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end entity;

architecture arch of negf_node is

  constant msb_mask : std_logic_vector(31 downto 0) := (31 => '1', others => '0');

begin

  outs       <= ins xor msb_mask;
  outs_valid <= ins_valid;
  ins_ready  <= outs_ready;
end architecture;
