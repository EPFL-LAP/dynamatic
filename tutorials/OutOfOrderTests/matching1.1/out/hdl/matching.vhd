library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity matching is
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

architecture behavioral of matching is


begin

  out0 <= num_edges;
  out0_valid <= num_edges_valid;
  num_edges_ready <= out0_ready;
  end_valid <= start_valid;
  start_ready <= end_ready;

end architecture;
