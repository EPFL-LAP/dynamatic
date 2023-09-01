library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;
use IEEE.math_real.all;

entity mux is
  generic (
    NUM_INPUTS    : integer;
    BITWIDTH      : integer;
    COND_BITWIDTH : integer
  );
  port (
    -- NUM_INPUTS
    clk              : in std_logic;
    rst              : in std_logic;
    select_ind       : in std_logic_vector(COND_BITWIDTH - 1 downto 0);
    select_ind_valid : in std_logic;
    ins              : in data_array(NUM_INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid        : in std_logic_vector(NUM_INPUTS downto 0);
    outs_ready       : in std_logic;
    -- outputs
    select_ind_ready : out std_logic;
    ins_ready        : out std_logic_vector(NUM_INPUTS - 1 downto 0);
    outs             : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid       : out std_logic);
end mux;

architecture arch of mux is

  signal tehb_data_in : std_logic_vector(BITWIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin
  process (ins, ins_valid, outs_ready, select_ind, tehb_ready)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    for I in NUM_INPUTS - 1 downto 0 loop
      if (unsigned(select_ind) = to_unsigned(I, select_ind'length) and select_ind_valid = '1' and ins_valid(I + 1) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
      end if;

      if ((unsigned(select_ind) = to_unsigned(I, select_ind'length) and select_ind_valid = '1' and tehb_ready = '1' and ins_valid(I + 1) = '1') or ins_valid(I + 1) = '0') then
        ins_ready(I + 1) <= '1';
      else
        ins_ready(I + 1) <= '0';
      end if;
    end loop;

    if (select_ind_valid = '0' or (tmp_valid_out = '1' and tehb_ready = '1')) then
      select_ind_ready <= '1';
    else
      select_ind_ready <= '0';
    end if;

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    tehb_pvalid  <= tmp_valid_out;
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
end arch;
