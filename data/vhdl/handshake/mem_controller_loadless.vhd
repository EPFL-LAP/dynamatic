library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_loadless is
  generic (
    NUM_CONTROLS : integer;
    NUM_STORES   : integer;
    DATA_TYPE   : integer;
    ADDR_TYPE   : integer
  );
  port (
    clk, rst : in std_logic;
    -- start input control
    memStart_valid : in  std_logic;
    memStart_ready : out std_logic;
    -- end output control
    memEnd_valid : out std_logic;
    memEnd_ready : in  std_logic;
    -- "no more requests" input control
    ctrlEnd_valid : in  std_logic;
    ctrlEnd_ready : out std_logic;
    -- control input channels
    ctrl       : in  data_array (NUM_CONTROLS - 1 downto 0)(31 downto 0);
    ctrl_valid : in  std_logic_vector(NUM_CONTROLS - 1 downto 0);
    ctrl_ready : out std_logic_vector(NUM_CONTROLS - 1 downto 0);
    -- store address input channels
    stAddr       : in  data_array (NUM_STORES - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    stAddr_valid : in  std_logic_vector(NUM_STORES - 1 downto 0);
    stAddr_ready : out std_logic_vector(NUM_STORES - 1 downto 0);
    -- store data input channels
    stData       : in  data_array (NUM_STORES - 1 downto 0)(DATA_TYPE - 1 downto 0);
    stData_valid : in  std_logic_vector(NUM_STORES - 1 downto 0);
    stData_ready : out std_logic_vector(NUM_STORES - 1 downto 0);
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    storeData : out std_logic_vector(DATA_TYPE - 1 downto 0)
  );
end entity;

architecture arch of mem_controller_loadless is
  signal remainingStores                    : std_logic_vector(31 downto 0);
  signal storePorts_valid, storePorts_ready : std_logic_vector(NUM_STORES - 1 downto 0);
  signal allRequestsDone                    : std_logic;

  constant zeroStore : std_logic_vector(31 downto 0)               := (others => '0');
  constant zeroCtrl  : std_logic_vector(NUM_CONTROLS - 1 downto 0) := (others => '0');

begin
  loadEn   <= '0';
  loadAddr <= (others => '0');

  write_arbiter : entity work.write_memory_arbiter
    generic map(
      ARBITER_SIZE => NUM_STORES,
      ADDR_TYPE   => ADDR_TYPE,
      DATA_TYPE   => DATA_TYPE
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
  ctrl_ready   <= (others => '1');

  count_stores : process (rst, clk)
    variable counter : std_logic_vector(31 downto 0);
  begin
    if (rst = '1') then
      counter := (31 downto 0 => '0');
    elsif rising_edge(clk) then
      for i in 0 to NUM_CONTROLS - 1 loop
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

  -- NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  -- we should also check that there are no loads pending as well. To achieve 
  -- this the control signals could simply start indicating the total number
  -- of accesses in the block instead of just the number of stores.
  allRequestsDone <= '1' when (remainingStores = zeroStore) and (ctrl_valid = zeroCtrl) else '0';

  control : entity work.mc_control
    port map(
      rst             => rst,
      clk             => clk,
      memStart_valid  => memStart_valid,
      memStart_ready  => memStart_ready,
      memEnd_valid    => memEnd_valid,
      memEnd_ready    => memEnd_ready,
      ctrlEnd_valid   => ctrlEnd_valid,
      ctrlEnd_ready   => ctrlEnd_ready,
      allRequestsDone => allRequestsDone
    );

end architecture;
