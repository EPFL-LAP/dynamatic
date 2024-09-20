library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity double_to_single is
  port (
    ins  : std_logic_vector(63 downto 0);
    outs : std_logic_vector(31 downto 0)
  )
end entity;

architecture arch of double_to_single is
  signal in_sign           : std_logic;
  signal in_exponent       : std_logic_vector(10 downto 0);
  signal in_mantissa       : std_logic_vector(51 downto 0);
  signal out_sign            : std_logic;
  signal out_exponent        : std_logic_vector(7 downto 0);
  signal out_mantissa        : std_logic_vector(22 downto 0);
begin
  outs <= out_sign & out_exponent & out_mantissa;
  in_sign <= ins(63);
  in_exponent <= ins(62 downto 52);
  in_mantissa <= ins(51 downto 0);
  out_sign <= in_sign;
  -- The exponent of IEEE-754 double has a different bias added to the 
  -- original exponent compared with IEEE-754 single (1023 vs 127).
  -- When converting double to single , we sub the difference 896 = 1023 - 127.
  out_exponent <= std_logic_vector(unsigned(in_exponent) - 896);
  -- TODO: here rounding is not considered.
  out_mantissa <= in_mantissa(51 downto 30);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity truncf is
  generic (
    INPUT_TYPE  : integer;
    OUTPUT_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(INPUT_TYPE - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(OUTPUT_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
begin
  assert INPUT_TYPE=64
  report "Currently we assume that truncf only converts double to single!"
  severity failure;
  assert OUTPUT_TYPE=32
  report "Currently we assume that truncf only converts double to single!"
  severity failure;
end entity;

architecture arch of truncf is
begin
  converter: work.double_to_single(arch)
    port map (
      ins => ins,
      outs => outs
  );
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;



