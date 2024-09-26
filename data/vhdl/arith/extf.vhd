library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.float_pkg.all;

-- TODO: we need a floating-point expert for this unit
-- entity single_to_double is
--   port (
--     ins  : in std_logic_vector(31 downto 0);
--     outs : out std_logic_vector(63 downto 0)
--   );
-- end entity;
-- 
-- architecture arch of single_to_double is
--   signal in_sign            : std_logic;
--   signal in_exponent        : std_logic_vector(7 downto 0);
--   signal in_mantissa        : std_logic_vector(22 downto 0);
--   signal out_sign           : std_logic;
--   signal out_exponent       : std_logic_vector(10 downto 0);
--   signal out_mantissa       : std_logic_vector(51 downto 0);
--   constant mantissa_padding : std_logic_vector(28 downto 0) := (others => '0');
-- begin
--   outs <= out_sign & out_exponent & out_mantissa;
--   in_sign <= ins(31);
--   in_exponent <= ins(30 downto 23);
--   in_mantissa <= ins(22 downto 0);
--   out_sign <= in_sign;
--   -- The exponent of IEEE-754 double has a different bias added to the 
--   -- original exponent compared with IEEE-754 single (1023 vs 127).
--   -- When converting single to double , we add the difference 896 = 1023 - 127.
--   out_exponent <= std_logic_vector(resize(unsigned(in_exponent), 11) + 896);
--   out_mantissa <= in_mantissa & mantissa_padding;
-- end architecture;

entity single_to_double is
  port (
    ins  : in std_logic_vector(31 downto 0);
    outs : out std_logic_vector(63 downto 0)
  );
end entity;

architecture arch of single_to_double is
  signal float_value : float32;
  signal float_extended : float64;
begin
  float_value <= to_float(ins);
  float_extended <= to_float64(float_value);
  outs <= to_std_logic_vector(float_extended);
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity extf is
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
  assert INPUT_TYPE=32
  report "Currently we assume that extf only converts float to double!"
  severity failure;
  assert OUTPUT_TYPE=64
  report "Currently we assume that extf only converts float to double!"
  severity failure;
end entity;

architecture arch of extf is
begin
  converter: entity work.single_to_double(arch)
    port map (
      ins => ins,
      outs => outs
  );
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;



