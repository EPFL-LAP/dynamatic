library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_storeless is
  generic (
    NUM_LOADS : integer;
    DATA_WIDTH : integer;
    ADDR_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- load address input channels
    ldAddr       : in  data_array (NUM_LOADS - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    ldAddr_valid : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    ldAddr_ready : out std_logic_vector(NUM_LOADS - 1 downto 0);
    -- load data output channels
    ldData       : out data_array (NUM_LOADS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    ldData_valid : out std_logic_vector(NUM_LOADS - 1 downto 0);
    ldData_ready : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    --- memory done channel
    memDone_valid : out std_logic;
    memDone_ready : in  std_logic;
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    storeData : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );
end entity;

architecture arch of mem_controller_storeless is
begin
  -- no stores will ever be issued
  storeAddr <= (others => '0');
  storeData <= (others => '0');
  storeEn   <= '0';

  -- MC is "always done with stores"
  memDone_valid <= '1';

  read_arbiter : entity work.read_memory_arbiter
    generic map(
      ARBITER_SIZE => NUM_LOADS,
      ADDR_WIDTH   => ADDR_WIDTH,
      DATA_WIDTH   => DATA_WIDTH
    )
    port map(
      rst              => rst,
      clk              => clk,
      pValid           => ldAddr_valid,
      ready            => ldAddr_ready,
      address_in       => ldAddr,
      nReady           => ldData_ready,
      valid            => ldData_valid,
      data_out         => ldData,
      read_enable      => loadEn,
      read_address     => loadAddr,
      data_from_memory => loadData
    );
end architecture;
