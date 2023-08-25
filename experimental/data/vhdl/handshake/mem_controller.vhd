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
  io_inLoadData   : in std_logic_vector(31 downto 0);
  io_ctrl         : in std_logic_vector(31 downto 0);
  io_ctrl_valid   : in std_logic;
  io_ldAddr       : in data_array (LOAD_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
  io_ldAddr_valid : in std_logic_vector(LOAD_COUNT - 1 downto 0);
  io_stAddr       : in data_array (STORE_COUNT - 1 downto 0)(ADDR_BITWIDTH - 1 downto 0);
  io_stAddr_valid : in std_logic_vector(STORE_COUNT - 1 downto 0);
  io_stData       : in data_array (STORE_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
  io_stData_valid : in std_logic_vector(STORE_COUNT - 1 downto 0);
  clk             : in std_logic;
  rst             : in std_logic;
  io_ldData_ready : in std_logic_vector(LOAD_COUNT - 1 downto 0);
  io_done_ready   : in std_logic;
  -- outputs
  io_bbReadyToPrevs : out std_logic;
  io_ldAddr_ready   : out std_logic_vector(LOAD_COUNT - 1 downto 0);
  io_stAddr_ready   : out std_logic_vector(STORE_COUNT - 1 downto 0);
  io_stData_ready   : out std_logic_vector(STORE_COUNT - 1 downto 0)
  io_ldData         : out data_array (LOAD_COUNT - 1 downto 0)(DATA_BITWIDTH - 1 downto 0);
  io_ldData_valid   : out std_logic_vector(LOAD_COUNT - 1 downto 0);
  io_done_valid     : out std_logic;
  io_loadEnable     : out std_logic;
  io_loadAddrOut    : out std_logic_vector(31 downto 0);
  io_storeEnable    : out std_logic;
  io_storeAddrOut   : out std_logic_vector(31 downto 0);
  io_storeDataOut   : out std_logic_vector(31 downto 0);
);

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
  io_stData_ready <= io_stAddr_ready;

  io_storeDataOut <= std_logic_vector (resize(unsigned(mcStoreDataOut), io_storeDataOut'length));
  io_storeAddrOut <= std_logic_vector (resize(unsigned(mcStoreAddrOut), io_storeDataOut'length));
  mcLoadDataIn    <= std_logic_vector (resize(unsigned(io_inLoadData), mcLoadDataIn'length));
  io_loadAddrOut  <= std_logic_vector (resize(unsigned(mcLoadAddrOut), io_loadAddrOut'length));

  read_arbiter : entity work.read_memory_arbiter
    generic map(
      ARBITER_SIZE => LOAD_COUNT,
      ADDR_WIDTH   => ADDR_BITWIDTH,
      DATA_WIDTH   => DATA_BITWIDTH
    )
    port map(
      rst              => rst,
      clk              => clk,
      pValid           => io_ldAddr_valid,
      ready            => io_ldAddr_ready,
      address_in       => io_ldAddr,
      nReady           => io_ldData_ready,
      valid            => io_ldData_valid,
      data_out         => io_ldData,
      read_enable      => io_loadEnable,
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
      pValid         => io_stAddr_valid,
      ready          => io_stAddr_ready,
      address_in     => io_stAddr,
      data_in        => io_stData,
      nReady => (others => '1'),
      valid          => valid_WR,
      write_enable   => io_storeEnable,
      write_address  => mcStoreAddrOut,
      data_to_memory => mcStoreDataOut
    );

  Counter : process (CLK)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');

    elsif rising_edge(CLK) then
      if (io_ctrl_valid(I) = '1') then
        counter := std_logic_vector(unsigned(counter) + unsigned(io_ctrl));
      end if;
      if (io_StoreEnable = '1') then
        counter := std_logic_vector(unsigned(counter) - 1);
      end if;

      counter1 <= counter;
    end if;

  end process;
  io_done_valid <= '1' when (counter1 = (31 downto 0 => '0') and (io_ctrl_valid(0 downto 0) = zero)) else
    '0';

  io_bbReadyToPrevs <= (others => '1');

end architecture;
