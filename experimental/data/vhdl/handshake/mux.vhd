library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.types.all;

entity mux is
  generic (
    SIZE         : integer;
    DATA_WIDTH   : integer;
    SELECT_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- data input channels
    ins       : in  data_array(SIZE - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    ins_valid : in  std_logic_vector(SIZE - 1 downto 0);
    ins_ready : out std_logic_vector(SIZE - 1 downto 0);
    -- index input channel
    index       : in  std_logic_vector(SELECT_WIDTH - 1 downto 0);
    index_valid : in  std_logic;
    index_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of mux is

  signal tehb_data_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal tehb_pvalid  : std_logic;
  signal tehb_ready   : std_logic;

begin
  process (ins, ins_valid, outs_ready, index, tehb_ready)
    variable tmp_data_out  : unsigned(DATA_WIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;
  begin
    tmp_data_out  := unsigned(ins(0));
    tmp_valid_out := '0';
    for I in SIZE - 1 downto 0 loop
      if (unsigned(index) = to_unsigned(I, index'length) and index_valid = '1' and ins_valid(I) = '1') then
        tmp_data_out  := unsigned(ins(I));
        tmp_valid_out := '1';
      end if;

      if ((unsigned(index) = to_unsigned(I, index'length) and index_valid = '1' and tehb_ready = '1' and ins_valid(I) = '1') or ins_valid(I) = '0') then
        ins_ready(I) <= '1';
      else
        ins_ready(I) <= '0';
      end if;
    end loop;

    if (index_valid = '0' or (tmp_valid_out = '1' and tehb_ready = '1')) then
      index_ready <= '1';
    else
      index_ready <= '0';
    end if;

    tehb_data_in <= std_logic_vector(resize(tmp_data_out, DATA_WIDTH));
    tehb_pvalid  <= tmp_valid_out;
  end process;

  tehb : entity work.tehb(arch) generic map (DATA_WIDTH)
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
