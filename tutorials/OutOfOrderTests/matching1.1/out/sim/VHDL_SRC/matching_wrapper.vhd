library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity matching_wrapper is
  port (
    num_edges : in std_logic_vector(31 downto 0);
    num_edges_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    num_edges_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of matching_wrapper is

  signal matching_wrapped_out0 : std_logic_vector(31 downto 0);
  signal matching_wrapped_out0_valid : std_logic;
  signal matching_wrapped_out0_ready : std_logic;
  signal matching_wrapped_end_valid : std_logic;
  signal matching_wrapped_end_ready : std_logic;

begin

  out0 <= matching_wrapped_out0;
  out0_valid <= matching_wrapped_out0_valid;
  matching_wrapped_out0_ready <= out0_ready;
  end_valid <= matching_wrapped_end_valid;
  matching_wrapped_end_ready <= end_ready;

  matching_wrapped : entity work.matching(behavioral)
    port map(
      num_edges => num_edges,
      num_edges_valid => num_edges_valid,
      num_edges_ready => num_edges_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => matching_wrapped_out0,
      out0_valid => matching_wrapped_out0_valid,
      out0_ready => matching_wrapped_out0_ready,
      end_valid => matching_wrapped_end_valid,
      end_ready => matching_wrapped_end_ready
    );

end architecture;
