library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity iterative_sqrt_wrapper is
  port (
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of iterative_sqrt_wrapper is

  signal iterative_sqrt_wrapped_out0 : std_logic_vector(31 downto 0);
  signal iterative_sqrt_wrapped_out0_valid : std_logic;
  signal iterative_sqrt_wrapped_out0_ready : std_logic;
  signal iterative_sqrt_wrapped_end_valid : std_logic;
  signal iterative_sqrt_wrapped_end_ready : std_logic;

begin

  out0 <= iterative_sqrt_wrapped_out0;
  out0_valid <= iterative_sqrt_wrapped_out0_valid;
  iterative_sqrt_wrapped_out0_ready <= out0_ready;
  end_valid <= iterative_sqrt_wrapped_end_valid;
  iterative_sqrt_wrapped_end_ready <= end_ready;

  iterative_sqrt_wrapped : entity work.iterative_sqrt(behavioral)
    port map(
      n => n,
      n_valid => n_valid,
      n_ready => n_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => iterative_sqrt_wrapped_out0,
      out0_valid => iterative_sqrt_wrapped_out0_valid,
      out0_ready => iterative_sqrt_wrapped_out0_ready,
      end_valid => iterative_sqrt_wrapped_end_valid,
      end_ready => iterative_sqrt_wrapped_end_ready
    );

end architecture;
