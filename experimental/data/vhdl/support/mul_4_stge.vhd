library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity mul_4_stage is
  generic (
    INPUTS        : integer;
    OUTPUTS       : integer;
    DATA_SIZE_IN  : integer;
    DATA_SIZE_OUT : integer
  );
  port (
    clk : in std_logic;
    ce  : in std_logic;
    a   : in std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    b   : in std_logic_vector(DATA_SIZE_IN - 1 downto 0);
    p   : out std_logic_vector(DATA_SIZE_OUT - 1 downto 0));
end entity;

architecture behav of mul_4_stage is

  signal a_reg : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
  signal b_reg : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
  signal q0    : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
  signal q1    : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
  signal q2    : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);
  signal mul   : std_logic_vector(DATA_SIZE_OUT - 1 downto 0);

begin

  mul <= std_logic_vector(resize(unsigned(std_logic_vector(signed(a_reg) * signed(b_reg))), DATA_SIZE_OUT));

  process (clk)
  begin
    if (clk'event and clk = '1') then
      if (ce = '1') then
        a_reg <= a;
        b_reg <= b;
        q0    <= mul;
        q1    <= q0;
        q2    <= q1;
      end if;
    end if;
  end process;

  p <= q2;

end architecture;
