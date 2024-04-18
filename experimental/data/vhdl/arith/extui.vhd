library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity extsi is
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
    outs       : out std_logic_vector(OUTPUT_BITWIDTH - 1 downto 0);
    outs_valid : out std_logic;
    ins_ready  : out std_logic
  );
end entity;

architecture arch of extsi is
begin
  outs       <= std_logic_vector(ieee.numeric_std.resize(unsigned(ins), OUTPUT_BITWIDTH));
  outs_valid <= ins_valid;
  ins_ready  <= not ins_valid or (ins_valid and outs_ready);
end architecture;
