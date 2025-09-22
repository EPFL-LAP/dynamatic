library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bisection_wrapper is
  port (
    a : in std_logic_vector(31 downto 0);
    a_valid : in std_logic;
    b : in std_logic_vector(31 downto 0);
    b_valid : in std_logic;
    tol : in std_logic_vector(31 downto 0);
    tol_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    a_ready : out std_logic;
    b_ready : out std_logic;
    tol_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of bisection_wrapper is

  signal bisection_wrapped_out0 : std_logic_vector(31 downto 0);
  signal bisection_wrapped_out0_valid : std_logic;
  signal bisection_wrapped_out0_ready : std_logic;
  signal bisection_wrapped_end_valid : std_logic;
  signal bisection_wrapped_end_ready : std_logic;

begin

  out0 <= bisection_wrapped_out0;
  out0_valid <= bisection_wrapped_out0_valid;
  bisection_wrapped_out0_ready <= out0_ready;
  end_valid <= bisection_wrapped_end_valid;
  bisection_wrapped_end_ready <= end_ready;

  bisection_wrapped : entity work.bisection(behavioral)
    port map(
      a => a,
      a_valid => a_valid,
      a_ready => a_ready,
      b => b,
      b_valid => b_valid,
      b_ready => b_ready,
      tol => tol,
      tol_valid => tol_valid,
      tol_ready => tol_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => bisection_wrapped_out0,
      out0_valid => bisection_wrapped_out0_valid,
      out0_ready => bisection_wrapped_out0_ready,
      end_valid => bisection_wrapped_end_valid,
      end_ready => bisection_wrapped_end_ready
    );

end architecture;
