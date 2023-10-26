library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity trunci_node is
  generic (
    INPUT_BITWIDTH  : integer;
    OUTPUT_BITWIDTH : integer
  );
  port (
    -- inputs
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in std_logic_vector(INPUT_BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic;
    outs       : out std_logic_vector(OUTPUT_BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end entity;

architecture arch of trunci_node is
begin
  outs       <= ins(OUTPUT_BITWIDTH - 1 downto 0);
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
end architecture;
