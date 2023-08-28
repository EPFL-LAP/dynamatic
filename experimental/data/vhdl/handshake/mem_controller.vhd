library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.customTypes.all;

entity mem_controller is generic (
  DATA_BITWIDTH : natural;
  ADDR_BITWIDTH : natural;
  LOAD_COUNT    : natural;
  STORE_COUNT   : natural);
port (
  -- inputs
  inLoadData   : in std_logic_vector(31 downto 0);
  ctrl         : in std_logic_vector(31 downto 0);
  ctrl_valid   : in std_logic;
  ldAddr       : in data_array (LOAD_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
  ldAddr_valid : in std_logic_vector(LOAD_COUNT - 1 downto 0);
  stAddr       : in data_array (STORE_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
  stAddr_valid : in std_logic_vector(STORE_COUNT - 1 downto 0);
  stData       : in data_array (STORE_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
  stData_valid : in std_logic_vector(STORE_COUNT - 1 downto 0);
  clk          : in std_logic;
  rst          : in std_logic;
  ldData_ready : in std_logic_vector(LOAD_COUNT - 1 downto 0);
  done_ready   : in std_logic;
  -- outputs
  bbReadyToPrevs : out std_logic;
  ldAddr_ready   : out std_logic_vector(LOAD_COUNT - 1 downto 0);
  stAddr_ready   : out std_logic_vector(STORE_COUNT - 1 downto 0);
  stData_ready   : out std_logic_vector(STORE_COUNT - 1 downto 0)
  ldData         : out data_array (LOAD_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
  ldData_valid   : out std_logic_vector(LOAD_COUNT - 1 downto 0);
  done_valid     : out std_logic;
  loadEnable     : out std_logic;
  loadAddrOut    : out std_logic_vector(31 downto 0);
  storeEnable    : out std_logic;
  storeAddrOut   : out std_logic_vector(31 downto 0);
  storeDataOut   : out std_logic_vector(31 downto 0));

end entity;
architecture arch of mem_controller is
  signal counter1 : std_logic_vector(31 downto 0);
  signal valid_WR : std_logic_vector(STORE_COUNT - 1 downto 0);
  constant zero   : std_logic := (others => '0');

  signal mcStoreDataOut : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal mcStoreAddrOut : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);
  signal mcLoadDataIn   : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal mcLoadAddrOut  : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);

begin
  stData_ready <= stAddr_ready;

  storeDataOut <= std_logic_vector (resize(unsigned(mcStoreDataOut), storeDataOut'length));
  storeAddrOut <= std_logic_vector (resize(unsigned(mcStoreAddrOut), storeDataOut'length));
  mcLoadDataIn <= std_logic_vector (resize(unsigned(inLoadData), mcLoadDataIn'length));
  loadAddrOut  <= std_logic_vector (resize(unsigned(mcLoadAddrOut), loadAddrOut'length));

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
      read_enable      => loadEnable,
      read_address     => mcLoadAddrOut,
      data_from_memory => mcLoadDataIn
    );

  write_arbiter : entity work.write_memory_arbiter
    generic map(
      ARBITER_SIZE => STORE_COUNT,
      ADDR_WIDTH   => ADDR_BITWIDTH,
      DATA_WIDTH   => DATA_BITWIDTH
    )
    port map(
      rst            => rst,
      clk            => clk,
      pValid         => stAddr_valid,
      ready          => stAddr_ready,
      address_in     => stAddr,
      data_in        => stData,
      nReady => (others => '1'),
      valid          => valid_WR,
      write_enable   => storeEnable,
      write_address  => mcStoreAddrOut,
      data_to_memory => mcStoreDataOut
    );

  Counter : process (CLK)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');

    elsif rising_edge(CLK) then
      if (ctrl_valid(I) = '1') then
        counter := std_logic_vector(unsigned(counter) + unsigned(ctrl));
      end if;
      if (StoreEnable = '1') then
        counter := std_logic_vector(unsigned(counter) - 1);
      end if;

      counter1 <= counter;
    end if;

  end process;
  done_valid <= '1' when (counter1 = (31 downto 0 => '0') and (ctrl_valid(0 downto 0) = zero)) else
    '0';

  bbReadyToPrevs <= (others => '1');

end architecture;
