library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_loadless is
  generic (
    CTRL_COUNT  : integer;
    STORE_COUNT : integer;
    DATA_WIDTH  : integer;
    ADDR_WIDTH  : integer
  );
  port (
    clk, rst : in std_logic;
    -- control input channels
    ctrl       : in  data_array (CTRL_COUNT - 1 downto 0)(31 downto 0);
    ctrl_valid : in  std_logic_vector(CTRL_COUNT - 1 downto 0);
    ctrl_ready : out std_logic_vector(CTRL_COUNT - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array (STORE_COUNT - 1 downto 0)(ADDR_WIDTH - 1 downto 0);
    stAddr_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    stAddr_ready : out std_logic_vector(STORE_COUNT - 1 downto 0);
    -- store data input channels
    stData       : in  data_array (STORE_COUNT - 1 downto 0)(DATA_WIDTH - 1 downto 0);
    stData_valid : in  std_logic_vector(STORE_COUNT - 1 downto 0);
    stData_ready : out std_logic_vector(STORE_COUNT - 1 downto 0);
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

architecture arch of mem_controller_loadless is
  signal remainingStores                    : std_logic_vector(31 downto 0);
  signal storePorts_valid, storePorts_ready : std_logic_vector(STORE_COUNT - 1 downto 0);
  constant zeroStore                        : std_logic_vector(31 downto 0)             := (others => '0');
  constant zeroCtrl                         : std_logic_vector(CTRL_COUNT - 1 downto 0) := (others => '0');

begin
  loadEn   <= '0';
  loadAddr <= (others => '0');

  write_arbiter : entity work.write_memory_arbiter
    generic map(
      ARBITER_SIZE => STORE_COUNT,
      ADDR_WIDTH   => ADDR_WIDTH,
      DATA_WIDTH   => DATA_WIDTH
    )
    port map(
      rst            => rst,
      clk            => clk,
      pValid         => stAddr_valid,
      ready          => storePorts_ready,
      address_in     => stAddr,
      data_in        => stData,
      nReady         => (others => '1'),
      valid          => storePorts_valid,
      write_enable   => storeEn,
      write_address  => storeAddr,
      data_to_memory => storeData
    );

  stData_ready <= storePorts_ready;
  stAddr_ready <= storePorts_ready;

  count_stores : process (clk)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');
    elsif rising_edge(clk) then
      for i in 0 to CTRL_COUNT - 1 loop
        if ctrl_valid(i) then
          counter := std_logic_vector(unsigned(counter) + unsigned(ctrl(i)));
        end if;
      end loop;
      if storeEn then
        counter := std_logic_vector(unsigned(counter) - 1);
      end if;
      remainingStores <= counter;
    end if;
  end process;

  memDone_valid <= '1' when (remainingStores = zeroStore and (ctrl_valid = zeroCtrl)) else '0';
  ctrl_ready    <= (others => '1');
end architecture;
