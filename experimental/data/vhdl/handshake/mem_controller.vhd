library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller is
  generic (
    CTRL_COUNT    : integer;
    LOAD_COUNT    : integer;
    STORE_COUNT   : integer;
    DATA_BITWIDTH : integer;
    ADDR_BITWIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- control input channels
    ctrl       : in  data_array (CTRL_COUNT - 1 downto 0)(31 downto 0);
    ctrl_valid : in  std_logic_vector(CTRL_COUNT - 1 downto 0);
    ctrl_ready : out std_logic_vector(CTRL_COUNT - 1 downto 0);
    -- load address input channels
    ldAddr       : in  data_array (LOAD_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
    ldAddr_valid : in  std_logic_vector(LOAD_COUNT - 1 downto 0);
    ldAddr_ready : out std_logic_vector(LOAD_COUNT - 1 downto 0);
    -- load data output channels
    ldData       : out data_array (LOAD_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
    ldData_valid : out std_logic_vector(LOAD_COUNT - 1 downto 0);
    ldData_ready : in  std_logic_vector(LOAD_COUNT - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array (STORE_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
    stAddr_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    stAddr_ready : out std_logic_vector(STORE_COUNT - 1 downto 0);
    -- store data input channels
    stData       : in  data_array (STORE_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
    stData_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    stData_ready : out std_logic_vector(STORE_COUNT - 1 downto 0);
    --- memory done channel
    memDone_valid : out std_logic;
    memDone_ready : in  std_logic;
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(DATA_BITWIDTH - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
    storeData : out std_logic_vector(DATA_BITWIDTH - 1 downto 0)
  );
end entity;

architecture arch of mem_controller is
  signal dropLoadAddr, mcLoadAddrOut : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  signal mcLoadDataIn                : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
begin
  stores : entity work.mem_controller_loadless generic map (CTRL_COUNT, STORE_COUNT, DATA_BITWIDTH, ADDR_BITWIDTH)
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
      loadData      => (others => '0'),
      loadEn        => open,
      loadAddr      => open,
      storeEn       => storeEn,
      storeAddr     => storeAddr,
      storeData     => storeData
    );

  read_arbiter : entity work.read_memory_arbiter
    generic map(
      ARBITER_SIZE => LOAD_COUNT,
      ADDR_WIDTH   => ADDR_BITWIDTH,
      DATA_WIDTH   => DATA_BITWIDTH
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
      read_address     => mcLoadAddrOut,
      data_from_memory => mcLoadDataIn
    );

  loadAddr     <= mcLoadAddrOut;
  mcLoadDataIn <= loadData;
end architecture;
