library ieee;
use ieee.std_logic_1164.all;
use work.customTypes.all;
use ieee.numeric_std.all;

entity end_node is
  generic (
    MEM_INPUTS : integer;
    BITWIDTH   : integer
  );

  port (
    clk, rst     : in std_logic;
    dataInArray  : in std_logic_vector(BITWIDTH - 1 downto 0);
    dataOutArray : out std_logic_vector(BITWIDTH - 1 downto 0);
    ready        : out std_logic;
    valid        : out std_logic;
    nReady       : in std_logic;
    pValid       : in std_logic;
    eReadyArray  : out std_logic_vector(MEM_INPUTS - 1 downto 0);
    eValidArray  : in std_logic_vector(MEM_INPUTS - 1 downto 0) := (others => '1'));
end end_node;

architecture arch of end_node is
  signal allPValid : std_logic;
  signal nReady    : std_logic;
  signal valid     : std_logic;
  signal mem_valid : std_logic;
  signal joinValid : std_logic;
  signal joinReady : std_logic;

begin

  -- process for the return data
  -- there may be multiple return points, check if any is valid and output its data
  process (pValid, dataInArray)
    variable tmp_data_out  : unsigned(BITWIDTH - 1 downto 0);
    variable tmp_valid_out : std_logic;

  begin
    tmp_data_out  := unsigned(dataInArray);
    tmp_valid_out := '0';

    if (pValid = '1') then
      tmp_data_out  := unsigned(dataInArray);
      tmp_valid_out := pValid;
    end if;

    dataOutArray <= std_logic_vector(resize(tmp_data_out, BITWIDTH));
    valid        <= tmp_valid_out;
  end process;

  -- check if all mem controllers are done (and of all valids from memory)
  mem_and : entity work.andN(vanilla) generic map (MEM_INPUTS)
    port map(eValidArray, mem_valid);

  -- join for return data and memory--we exit only in case the first process gets
  -- a single valid and if the AND of all memories is set
  j : entity work.join(arch) generic map(2)
    port map(
    (valid, mem_valid),
      nReady,
      joinValid,
      joinReady);

  -- valid to successor (set by join)
  valid <= joinValid;

  -- join sends ready to predecessors
  -- not needed for eReady (because memory never reads it)
  process (joinReady)
  begin
    ready <= joinReady;
  end process;
end architecture;
