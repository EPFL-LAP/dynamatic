library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity maxsi is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of maxsi is
begin
join_inputs : entity work.join(arch) generic map(2)
  port map(
    -- inputs
    ins_valid(0) => lhs_valid,
    ins_valid(1) => rhs_valid,
    outs_ready   => result_ready,
    -- outputs
    outs_valid   => result_valid,
    ins_ready(0) => lhs_ready,
    ins_ready(1) => rhs_ready
  );

process (lhs, rhs)
  variable result_var : signed(result'range);
  variable signed_lhs : signed(lhs'range);
  variable signed_rhs : signed(rhs'range);
begin
  signed_lhs := signed(lhs);
  signed_rhs := signed(rhs);
  if (signed_lhs > signed_rhs) then
    result_var := signed_lhs;
  else
    result_var := signed_rhs;
  end if;
  result <= std_logic_vector(result_var);
end process;
end architecture;
