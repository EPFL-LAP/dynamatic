library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity subi is
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

architecture arch of subi is
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

  result <= std_logic_vector(unsigned(lhs) - unsigned(rhs));
end architecture;

entity subi_with_tag is
  generic (
    DATA_TYPE : integer
  );
  port (
    -- inputs
    clk          : in std_logic;
    rst          : in std_logic;
    lhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    lhs_valid    : in std_logic;
    lhs_spec_tag : in std_logic;
    rhs          : in std_logic_vector(DATA_TYPE - 1 downto 0);
    rhs_valid    : in std_logic;
    rhs_spec_tag : in std_logic;
    result_ready : in std_logic;
    -- outputs
    result       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    result_valid : out std_logic;
    result_spec_tag : out std_logic;
    lhs_ready    : out std_logic;
    rhs_ready    : out std_logic
  );
end entity;

architecture arch of subi_with_tag is
begin
  result_spec_tag <= lhs_spec_tag or rhs_spec_tag;
  subi_inner : entity work.subi(arch) generic map(DATA_TYPE)
    port map(
      clk          => clk,
      rst          => rst,
      lhs          => lhs,
      lhs_valid    => lhs_valid,
      rhs          => rhs,
      rhs_valid    => rhs_valid,
      result_ready => result_ready,
      result       => result,
      result_valid => result_valid,
      lhs_ready    => lhs_ready,
      rhs_ready    => rhs_ready
    );
end architecture;
