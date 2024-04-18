library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_no_store is
  generic (
    LOAD_COUNT    : integer;
    DATA_BITWIDTH : integer;
    ADDR_BITWIDTH : integer
  );
  port (
    -- inputs
    loadData      : in std_logic_vector(DATA_BITWIDTH downto 0);
    ldAddr        : in data_array (LOAD_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
    ldAddr_valid  : in std_logic_vector(LOAD_COUNT - 1 downto 0);
    clk           : in std_logic;
    rst           : in std_logic;
    ldData_ready  : in std_logic_vector(LOAD_COUNT - 1 downto 0);
    memDone_ready : in std_logic;
    -- outputs
    ldAddr_ready  : out std_logic_vector(LOAD_COUNT - 1 downto 0);
    ldData        : out data_array (LOAD_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
    ldData_valid  : out std_logic_vector(LOAD_COUNT - 1 downto 0);
    memDone       : out std_logic_vector(0 downto 0);
    memDone_valid : out std_logic;
    loadEn        : out std_logic;
    loadAddr      : out std_logic_vector(ADDR_BITWIDTH downto 0);
    storeEn       : out std_logic;
    storeAddr     : out std_logic_vector(ADDR_BITWIDTH downto 0);
    storeData     : out std_logic_vector(DATA_BITWIDTH downto 0)
  );
end entity;

architecture arch of mem_controller_no_store is
  signal counter1  : std_logic_vector(31 downto 0);
  constant zero_32 : std_logic_vector(31 downto 0) := (others => '0');

  signal mcStoreDataOut : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal mcStoreAddrOut : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  signal mcLoadDataIn   : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal mcLoadAddrOut  : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);

begin
  storeData    <= std_logic_vector (resize(unsigned(mcStoreDataOut), storeData'length));
  storeAddr    <= std_logic_vector (resize(unsigned(mcStoreAddrOut), storeAddr'length));
  mcLoadDataIn <= std_logic_vector (resize(unsigned(loadData), mcLoadDataIn'length));
  loadAddr     <= std_logic_vector (resize(unsigned(mcLoadAddrOut), loadAddr'length));

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

  Counterp : process (CLK)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');
    elsif rising_edge(CLK) then
      counter1 <= counter;
    end if;
  end process;

  memDone_valid <= '1' when (counter1 = zero_32) else '0';

end architecture;
