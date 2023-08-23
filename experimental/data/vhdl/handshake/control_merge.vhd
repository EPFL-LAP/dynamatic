library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;

entity control_merge is generic (
  INPUTS        : integer;
  BITWIDTH      : integer;
  COND_BITWIDTH : integer
);
port (
  clk, rst     : in std_logic;
  pValidArray  : in std_logic_vector(INPUTS - 1 downto 0);
  nReadyArray  : in std_logic_vector(1 downto 0);
  validArray   : out std_logic_vector(1 downto 0);
  readyArray   : out std_logic_vector(INPUTS - 1 downto 0);
  dataInArray  : in data_array(INPUTS - 1 downto 0)(BITWIDTH - 1 downto 0);
  dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
  condition    : out std_logic_vector(COND_BITWIDTH - 1 downto 0));
end control_merge;

architecture arch of control_merge is

  signal phi_C1_readyArray   : std_logic_vector (INPUTS - 1 downto 0);
  signal phi_C1_validArray   : std_logic;
  signal phi_C1_dataOutArray : std_logic_vector(COND_BITWIDTH - 1 downto 0);

  signal fork_C1_readyArray   : std_logic;
  signal fork_C1_dataOutArray : data_array(1 downto 0)(0 downto 0);
  signal fork_C1_validArray   : std_logic_vector (1 downto 0);

  signal all_ones                 : std_logic_vector (COND_BITWIDTH - 1 downto 0) := (others => '1');
  signal index, oehb1_dataOut     : std_logic_vector (COND_BITWIDTH - 1 downto 0);
  signal oehb1_valid, oehb1_ready : std_logic;

begin
  readyArray <= phi_C1_readyArray;

  phi_C1 : entity work.merge_notehb(arch) generic map (INPUTS, COND_BITWIDTH)
    port map(
      --inputs
      clk            => clk,         --clk
      rst            => rst,         --rst
      pValidArray    => pValidArray, --pValidArray
      dataInArray => (INPUTS - 1 downto 0 => all_ones),
      nReadyArray(0) => oehb1_ready, --outputs
      dataOutArray   => phi_C1_dataOutArray,
      readyArray     => phi_C1_readyArray, --readyArray
      validArray     => phi_C1_validArray  --validArray
    );

  process (pValidArray)
  begin
    index <= (COND_BITWIDTH - 1 downto 0 => '0');
    for i in 0 to (INPUTS - 1) loop
      if (pValidArray(i) = '1') then
        index <= std_logic_vector(to_unsigned(i, COND_BITWIDTH));
        exit;
      end if;
    end loop;
  end process;

  oehb1 : entity work.TEHB(arch) generic map (COND_BITWIDTH)
    port map(
      --inputspValidArray
      clk            => clk,
      rst            => rst,
      pValidArray(0) => phi_C1_validArray,
      nReadyArray(0) => fork_C1_readyArray,
      validArray(0)  => oehb1_valid,
      --outputs
      readyArray(0)     => oehb1_ready,
      dataInArray(0)(0) => index,
      dataOutArray      => oehb1_dataOut
    );

  fork_C1 : entity work.fork(arch) generic map (2, 1)
    port map(
      --inputs
      clk             => clk,         --clk
      rst             => rst,         --rst
      pValidArray(0)  => oehb1_valid, --pValidArray
      dataInArray (0) => "1",
      nReadyArray     => nReadyArray, --nReadyArray
      --outputs
      dataOutArray => fork_C1_dataOutArray,
      readyArray   => fork_C1_readyArray, --readyArray
      validArray   => fork_C1_validArray  --validArray
    );
  validArray <= fork_C1_validArray;
  condition  <= oehb1_dataOut;

end architecture;
