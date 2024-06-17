library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_loadless is
  generic (
    CTRL_COUNT    : integer;
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

architecture arch of mem_controller_loadless is
  signal counter1    : std_logic_vector(31 downto 0);
  signal valid_WR    : std_logic_vector(STORE_COUNT - 1 downto 0);
  constant zero_32   : std_logic_vector(31 downto 0)             := (others => '0');
  constant zero_ctrl : std_logic_vector(CTRL_COUNT - 1 downto 0) := (others => '0');

  signal mcStoreDataOut : std_logic_vector(DATA_BITWIDTH - 1 downto 0);
  signal mcStoreAddrOut : std_logic_vector(ADDR_BITWIDTH - 1 downto 0);

begin
  loadEn   <= '0';
  loadAddr <= (others => '0');

  stData_ready <= stAddr_ready;
  storeAddr    <= mcStoreAddrOut;
  storeData    <= mcStoreDataOut;

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
      nReady         => (others => '1'),
      valid          => valid_WR,
      write_enable   => storeEn,
      write_address  => mcStoreAddrOut,
      data_to_memory => mcStoreDataOut
    );

  Counterp : process (CLK)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');

    elsif rising_edge(CLK) then
      for I in 0 to CTRL_COUNT - 1 loop
        if (ctrl_valid(I) = '1') then
          counter := std_logic_vector(unsigned(counter) + unsigned(ctrl(I)));
        end if;
      end loop;
      if (storeEn = '1') then
        counter := std_logic_vector(unsigned(counter) - 1);
      end if;

      counter1 <= counter;
    end if;
  end process;

  memDone_valid <= '1' when (counter1 = zero_32 and (ctrl_valid = zero_ctrl)) else '0';
  ctrl_ready    <= (others => '1');
end architecture;
