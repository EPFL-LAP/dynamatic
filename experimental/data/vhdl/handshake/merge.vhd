library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity merge is

  generic (
    INPUTS   : integer;
    BITWIDTH : integer
  );
  port (
    -- inputs
    ins        : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid  : in std_logic_vector(INPUTS - 1 downto 0);
    clk        : in std_logic;
    rst        : in std_logic;
    outs_ready : in std_logic;
    -- outputs
    ins_ready  : out std_logic_vector(INPUTS - 1 downto 0);
    outs       : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid : out std_logic);
end entity;

architecture arch of merge is
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

  tehb1 : entity work.TEHB(arch) generic map (BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => tehb_pvalid,
      outs_ready => outs_ready,
      outs_valid => outs_valid,
      ins_ready  => tehb_ready,
      ins        => tehb_data_in,
      outs       => outs
    );
end architecture;
