-- untagger : untagger({'data_bitwidth': 32, 'tag_bitwidth': 1, 'input_extra_signals': {'tag': 1}, 'output_extra_signals': {}})


library ieee;
use ieee.std_logic_1164.all;
use work.types.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

-- Entity of untagger
entity untagger is
port(
  clk        : in std_logic;
  rst        : in std_logic;
  ins_valid : in std_logic;

  outs_ready : in std_logic; 
  outs_valid : out std_logic;

  ins_ready : out std_logic;

  ins   : in  std_logic_vector(32 - 1 downto 0);
  outs  : out std_logic_vector(32 - 1 downto 0);

  dataOut : out std_logic_vector(1-1 downto 0);
  dataOut_valid : out  std_logic;
  dataOut_ready : in std_logic;

  ins_tag : in std_logic_vector(1-1 downto 0) 
);
end entity;

-- Architecture of untagger
architecture arch of untagger is
begin
    outs_valid<= ins_valid;
    dataOut_valid<= ins_valid;
    ins_ready <= dataOut_ready and outs_ready;
    outs <= ins;
    dataOut <= ins_tag;
end architecture;

