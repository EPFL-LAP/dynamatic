library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity golden_ratio_wrapper is
  port (
    x0 : in std_logic_vector(31 downto 0);
    x0_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    end_ready : in std_logic;
    x0_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    end_valid : out std_logic
  );
end entity;

architecture behavioral of golden_ratio_wrapper is

  signal golden_ratio_wrapped_out0 : std_logic_vector(31 downto 0);
  signal golden_ratio_wrapped_out0_valid : std_logic;
  signal golden_ratio_wrapped_out0_ready : std_logic;
  signal golden_ratio_wrapped_end_valid : std_logic;
  signal golden_ratio_wrapped_end_ready : std_logic;

begin

  out0 <= golden_ratio_wrapped_out0;
  out0_valid <= golden_ratio_wrapped_out0_valid;
  golden_ratio_wrapped_out0_ready <= out0_ready;
  end_valid <= golden_ratio_wrapped_end_valid;
  golden_ratio_wrapped_end_ready <= end_ready;

  golden_ratio_wrapped : entity work.golden_ratio(behavioral)
    port map(
      x0 => x0,
      x0_valid => x0_valid,
      x0_ready => x0_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => golden_ratio_wrapped_out0,
      out0_valid => golden_ratio_wrapped_out0_valid,
      out0_ready => golden_ratio_wrapped_out0_ready,
      end_valid => golden_ratio_wrapped_end_valid,
      end_ready => golden_ratio_wrapped_end_ready
    );

end architecture;
