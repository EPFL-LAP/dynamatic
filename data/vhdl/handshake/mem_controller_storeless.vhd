library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity mem_controller_storeless is
  generic (
    NUM_LOADS  : integer;
    DATA_TYPE : integer;
    ADDR_TYPE : integer
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
    -- load address input channels
    ldAddr       : in  data_array (NUM_LOADS - 1 downto 0)(ADDR_TYPE - 1 downto 0);
    ldAddr_valid : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    ldAddr_ready : out std_logic_vector(NUM_LOADS - 1 downto 0);
    -- load data output channels
    ldData       : out data_array (NUM_LOADS - 1 downto 0)(DATA_TYPE - 1 downto 0);
    ldData_valid : out std_logic_vector(NUM_LOADS - 1 downto 0);
    ldData_ready : in  std_logic_vector(NUM_LOADS - 1 downto 0);
    -- interface to dual-port BRAM
    loadData  : in  std_logic_vector(DATA_TYPE - 1 downto 0);
    loadEn    : out std_logic;
    loadAddr  : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    storeEn   : out std_logic;
    storeAddr : out std_logic_vector(ADDR_TYPE - 1 downto 0);
    storeData : out std_logic_vector(DATA_TYPE - 1 downto 0)
  );
end entity;

architecture arch of mem_controller_storeless is
  signal allRequestsDone : std_logic;
begin
  -- no stores will ever be issued
  storeAddr <= (others => '0');
  storeData <= (others => '0');
  storeEn   <= '0';

  read_arbiter : entity work.read_memory_arbiter
    generic map(
      ARBITER_SIZE => NUM_LOADS,
      ADDR_TYPE   => ADDR_TYPE,
      DATA_TYPE   => DATA_TYPE
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

  -- NOTE: (lucas-rami) In addition to making sure there are no stores pending,
  -- we should also check that there are no loads pending as well. To achieve 
  -- this the control signals could simply start indicating the total number
  -- of accesses in the block instead of just the number of stores.
  allRequestsDone <= '1';

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
