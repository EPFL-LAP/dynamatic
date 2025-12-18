library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity get_tanh_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    addr_din0 : in std_logic_vector(31 downto 0);
    addr_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    addr_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    addr_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    addr_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    addr_end_valid : out std_logic;
    end_valid : out std_logic;
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(9 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(9 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0);
    addr_ce0 : out std_logic;
    addr_we0 : out std_logic;
    addr_address0 : out std_logic_vector(9 downto 0);
    addr_dout0 : out std_logic_vector(31 downto 0);
    addr_ce1 : out std_logic;
    addr_we1 : out std_logic;
    addr_address1 : out std_logic_vector(9 downto 0);
    addr_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of get_tanh_wrapper is

  signal mem_to_bram_converter_addr_ce0 : std_logic;
  signal mem_to_bram_converter_addr_we0 : std_logic;
  signal mem_to_bram_converter_addr_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_addr_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_addr_ce1 : std_logic;
  signal mem_to_bram_converter_addr_we1 : std_logic;
  signal mem_to_bram_converter_addr_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_addr_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_addr_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal get_tanh_wrapped_A_end_valid : std_logic;
  signal get_tanh_wrapped_A_end_ready : std_logic;
  signal get_tanh_wrapped_addr_end_valid : std_logic;
  signal get_tanh_wrapped_addr_end_ready : std_logic;
  signal get_tanh_wrapped_end_valid : std_logic;
  signal get_tanh_wrapped_end_ready : std_logic;
  signal get_tanh_wrapped_A_loadEn : std_logic;
  signal get_tanh_wrapped_A_loadAddr : std_logic_vector(9 downto 0);
  signal get_tanh_wrapped_A_storeEn : std_logic;
  signal get_tanh_wrapped_A_storeAddr : std_logic_vector(9 downto 0);
  signal get_tanh_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal get_tanh_wrapped_addr_loadEn : std_logic;
  signal get_tanh_wrapped_addr_loadAddr : std_logic_vector(9 downto 0);
  signal get_tanh_wrapped_addr_storeEn : std_logic;
  signal get_tanh_wrapped_addr_storeAddr : std_logic_vector(9 downto 0);
  signal get_tanh_wrapped_addr_storeData : std_logic_vector(31 downto 0);

begin

  A_end_valid <= get_tanh_wrapped_A_end_valid;
  get_tanh_wrapped_A_end_ready <= A_end_ready;
  addr_end_valid <= get_tanh_wrapped_addr_end_valid;
  get_tanh_wrapped_addr_end_ready <= addr_end_ready;
  end_valid <= get_tanh_wrapped_end_valid;
  get_tanh_wrapped_end_ready <= end_ready;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;
  addr_ce0 <= mem_to_bram_converter_addr_ce0;
  addr_we0 <= mem_to_bram_converter_addr_we0;
  addr_address0 <= mem_to_bram_converter_addr_address0;
  addr_dout0 <= mem_to_bram_converter_addr_dout0;
  addr_ce1 <= mem_to_bram_converter_addr_ce1;
  addr_we1 <= mem_to_bram_converter_addr_we1;
  addr_address1 <= mem_to_bram_converter_addr_address1;
  addr_dout1 <= mem_to_bram_converter_addr_dout1;

  mem_to_bram_converter_addr : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => get_tanh_wrapped_addr_loadEn,
      loadAddr => get_tanh_wrapped_addr_loadAddr,
      storeEn => get_tanh_wrapped_addr_storeEn,
      storeAddr => get_tanh_wrapped_addr_storeAddr,
      storeData => get_tanh_wrapped_addr_storeData,
      din0 => addr_din0,
      din1 => addr_din1,
      ce0 => mem_to_bram_converter_addr_ce0,
      we0 => mem_to_bram_converter_addr_we0,
      address0 => mem_to_bram_converter_addr_address0,
      dout0 => mem_to_bram_converter_addr_dout0,
      ce1 => mem_to_bram_converter_addr_ce1,
      we1 => mem_to_bram_converter_addr_we1,
      address1 => mem_to_bram_converter_addr_address1,
      dout1 => mem_to_bram_converter_addr_dout1,
      loadData => mem_to_bram_converter_addr_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => get_tanh_wrapped_A_loadEn,
      loadAddr => get_tanh_wrapped_A_loadAddr,
      storeEn => get_tanh_wrapped_A_storeEn,
      storeAddr => get_tanh_wrapped_A_storeAddr,
      storeData => get_tanh_wrapped_A_storeData,
      din0 => A_din0,
      din1 => A_din1,
      ce0 => mem_to_bram_converter_A_ce0,
      we0 => mem_to_bram_converter_A_we0,
      address0 => mem_to_bram_converter_A_address0,
      dout0 => mem_to_bram_converter_A_dout0,
      ce1 => mem_to_bram_converter_A_ce1,
      we1 => mem_to_bram_converter_A_we1,
      address1 => mem_to_bram_converter_A_address1,
      dout1 => mem_to_bram_converter_A_dout1,
      loadData => mem_to_bram_converter_A_loadData
    );

  get_tanh_wrapped : entity work.get_tanh(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      addr_loadData => mem_to_bram_converter_addr_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      addr_start_valid => addr_start_valid,
      addr_start_ready => addr_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      A_end_valid => get_tanh_wrapped_A_end_valid,
      A_end_ready => get_tanh_wrapped_A_end_ready,
      addr_end_valid => get_tanh_wrapped_addr_end_valid,
      addr_end_ready => get_tanh_wrapped_addr_end_ready,
      end_valid => get_tanh_wrapped_end_valid,
      end_ready => get_tanh_wrapped_end_ready,
      A_loadEn => get_tanh_wrapped_A_loadEn,
      A_loadAddr => get_tanh_wrapped_A_loadAddr,
      A_storeEn => get_tanh_wrapped_A_storeEn,
      A_storeAddr => get_tanh_wrapped_A_storeAddr,
      A_storeData => get_tanh_wrapped_A_storeData,
      addr_loadEn => get_tanh_wrapped_addr_loadEn,
      addr_loadAddr => get_tanh_wrapped_addr_loadAddr,
      addr_storeEn => get_tanh_wrapped_addr_storeEn,
      addr_storeAddr => get_tanh_wrapped_addr_storeAddr,
      addr_storeData => get_tanh_wrapped_addr_storeData
    );

end architecture;
