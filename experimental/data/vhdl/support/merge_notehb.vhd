library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity merge_notehb is

  generic (
    INPUTS   : integer;
    BITWIDTH : integer);
  port (
    clk        : in std_logic;
    rst        : in std_logic;
    ins        : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector(INPUTS - 1 downto 0);
    outs_ready : in std_logic;
    outs_valid : out std_logic;
    ins_ready  : out std_logic_vector(INPUTS - 1 downto 0));
end merge_notehb;

architecture arch of merge_notehb is
  signal tehb_data_in : std_logic_vector(BITWIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin

  process (ins_valid, ins)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    for I in INPUTS - 1 downto 0 loop
      if (ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := ins_valid(I);
      end if;
    end loop;

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    tehb_pvalid  <= tmp_valid_out;

  end process;

  process (tehb_ready)
  begin
    for I in 0 to INPUTS - 1 loop
      ins_ready(I) <= tehb_ready;
    end loop;
  end process;

  tehb_ready <= outs_ready;
  outs_valid <= tehb_pvalid;
  outs       <= tehb_data_in;

end arch;
