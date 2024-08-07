library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller is
  generic (
    NUM_CONTROLS  : integer;
    NUM_LOADS  : integer;
    NUM_STORES : integer;
    DATA_WIDTH  : integer;
    ADDR_WIDTH  : integer
  );
  port (
    clk, rst : in std_logic;
    -- control input channels
    ctrl       : in  data_array (NUM_CONTROLS - 1 downto 0)(31 downto 0);
    ctrl_valid : in  std_logic_vector(NUM_CONTROLS - 1 downto 0);
    ctrl_ready : out std_logic_vector(NUM_CONTROLS - 1 downto 0);
    -- load address input channels
    ldAddr       : in  data_array (NUM_LOADS - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    ldAddr_valid : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    ldAddr_ready : out std_logic_vector(NUM_LOADS - 1 downto 0);
    -- load data output channels
    ldData       : out data_array (NUM_LOADS - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    ldData_valid : out std_logic_vector(NUM_LOADS - 1 downto 0);
    ldData_ready : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array (NUM_STORES - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    stAddr_valid : in  std_logic_vector(NUM_STORES - 1 downto 0);
    stAddr_ready : out std_logic_vector(NUM_STORES - 1 downto 0);
    -- store data input channels
    stData       : in  data_array (NUM_STORES - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    stData_valid : in  std_logic_vector(NUM_STORES - 1 downto 0);
    stData_ready : out std_logic_vector(NUM_STORES - 1 downto 0);
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

architecture arch of mem_controller is
  signal dropLoadAddr : std_logic_vector(ADDR_WIDTH - 1 downto 0);
  signal dropLoadData : std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal dropLoadEn   : std_logic;
begin

  stores : entity work.mem_controller_loadless
    generic map(
      NUM_CONTROLS  => NUM_CONTROLS,
      NUM_STORES => NUM_STORES,
      DATA_WIDTH  => DATA_WIDTH,
      ADDR_WIDTH  => ADDR_WIDTH)
    port map(
      clk           => clk,
      rst           => rst,
      ctrl          => ctrl,
      ctrl_valid    => ctrl_valid,
      ctrl_ready    => ctrl_ready,
      stAddr        => stAddr,
      stAddr_valid  => stAddr_valid,
      stAddr_ready  => stAddr_ready,
      stData        => stData,
      stData_valid  => stData_valid,
      stData_ready  => stData_ready,
      memDone_valid => memDone_valid,
      memDone_ready => memDone_ready,
      loadData      => dropLoadData,
      loadEn        => dropLoadEn,
      loadAddr      => dropLoadAddr,
      storeEn       => storeEn,
      storeAddr     => storeAddr,
      storeData     => storeData
    );

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
