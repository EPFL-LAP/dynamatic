library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity control_merge is
  generic (
    INPUTS        : integer;
    BITWIDTH      : integer;
    COND_BITWIDTH : integer
  );
  port (
    -- inputs
    ins             : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
    ins_valid       : in std_logic_vector(INPUTS - 1 downto 0);
    clk             : in std_logic;
    rst             : in std_logic;
    outs_ready      : in std_logic;
    condition_ready : in std_logic;
    -- outputs
    ins_ready       : out std_logic_vector(INPUTS - 1 downto 0);
    outs            : out std_logic_vector(BITWIDTH - 1 downto 0);
    outs_valid      : out std_logic;
    condition       : out std_logic_vector(COND_BITWIDTH - 1 downto 0);
    condition_valid : out std_logic
  );
end entity;

architecture arch of control_merge is

  signal phi_C1_readyArray   : std_logic_vector (INPUTS - 1 downto 0);
  signal phi_C1_validArray   : std_logic;
  signal phi_C1_dataOutArray : std_logic_vector (COND_BITWIDTH - 1 downto 0);

  signal fork_C1_readyArray   : std_logic;
  signal fork_C1_dataOutArray : data_array(1 downto 0)(0 downto 0);
  signal fork_C1_validArray   : std_logic_vector (1 downto 0);

  signal all_ones                 : std_logic_vector (COND_BITWIDTH - 1 downto 0) := (others => '1');
  signal index, oehb1_dataOut     : std_logic_vector (COND_BITWIDTH - 1 downto 0);
  signal oehb1_valid, oehb1_ready : std_logic;

begin
  ins_ready <= phi_C1_readyArray;

  phi_C1 : entity work.merge_notehb(arch) generic map (INPUTS, COND_BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => ins_valid,
      ins        => (INPUTS - 1 downto 0 => all_ones),
      outs_ready => oehb1_ready,
      outs       => phi_C1_dataOutArray,
      ins_ready  => phi_C1_readyArray,
      outs_valid => phi_C1_validArray);

  process (ins_valid)
  begin
    index <= (COND_BITWIDTH - 1 downto 0 => '0');
    for i in 0 to (INPUTS - 1) loop
      if (ins_valid(i) = '1') then
        index <= std_logic_vector(to_unsigned(i, COND_BITWIDTH));
        exit;
      end if;
    end loop;
  end process;

  oehb1 : entity work.tehb(arch) generic map (COND_BITWIDTH)
    port map(
      clk        => clk,
      rst        => rst,
      ins_valid  => phi_C1_validArray,
      outs_ready => fork_C1_readyArray,
      outs_valid => oehb1_valid,
      ins_ready  => oehb1_ready,
      ins        => index,
      outs       => oehb1_dataOut
    );

  fork_C1 : entity work.fork(arch) generic map (2, 1)
    port map(
      clk           => clk,
      rst           => rst,
      ins_valid     => oehb1_valid,
      ins(0)        => '1',
      outs_ready(0) => outs_ready,
      outs_ready(1) => condition_ready,
      outs          => fork_C1_dataOutArray,
      ins_ready     => fork_C1_readyArray,
      outs_valid    => fork_C1_validArray);
end architecture;
