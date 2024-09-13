library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity maximumf is
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

architecture arch of maximumf is

  component my_maxf is
    port (
      ap_clk    : in  std_logic;
      ap_rst    : in  std_logic;
      a         : in  std_logic_vector (31 downto 0);
      b         : in  std_logic_vector (31 downto 0);
      ap_return : out std_logic_vector (31 downto 0));
  end component;

  signal join_valid : std_logic;
begin
  join_inputs : entity work.join(arch) generic map(2)
    port map(
      -- inputs
      ins_valid(0) => lhs_valid,
      ins_valid(1) => rhs_valid,
      outs_ready   => result_ready,
      -- outputs
      outs_valid   => join_valid,
      ins_ready(0) => lhs_ready,
      ins_ready(1) => rhs_ready
    );

  my_trunc_U1 : component my_maxf
    port map(
      ap_clk    => clk,
      ap_rst    => rst,
      a         => lhs,
      b         => rhs,
      ap_return => result
    );

  buff : entity work.delay_buffer(arch) generic map(1)
    port map(
      clk,
      rst,
      join_valid,
      result_ready,
      result_valid
    );
end architecture;
