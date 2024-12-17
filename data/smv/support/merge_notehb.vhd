library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity merge_notehb is
  generic (
    INPUTS   : integer;
    DATA_TYPE : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channels
    ins       : in  data_array(INPUTS - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ins_valid : in  std_logic_vector(INPUTS - 1 downto 0);
    ins_ready : out std_logic_vector(INPUTS - 1 downto 0)
    -- output channel
    outs       : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of merge_notehb is
  signal tehb_data_in : std_logic_vector(DATA_TYPE - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;
begin
  process (ins_valid, ins)
    variable tmp_data_out  : unsigned(DATA_TYPE - 1 downto 0);
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

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, DATA_TYPE));
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
end architecture;
