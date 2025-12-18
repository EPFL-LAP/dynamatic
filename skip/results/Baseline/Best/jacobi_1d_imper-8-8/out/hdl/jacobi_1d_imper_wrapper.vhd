library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity jacobi_1d_imper_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    B_din0 : in std_logic_vector(31 downto 0);
    B_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    B_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    B_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    B_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    B_end_valid : out std_logic;
    end_valid : out std_logic;
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(6 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(6 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0);
    B_ce0 : out std_logic;
    B_we0 : out std_logic;
    B_address0 : out std_logic_vector(6 downto 0);
    B_dout0 : out std_logic_vector(31 downto 0);
    B_ce1 : out std_logic;
    B_we1 : out std_logic;
    B_address1 : out std_logic_vector(6 downto 0);
    B_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of jacobi_1d_imper_wrapper is

  signal mem_to_bram_converter_B_ce0 : std_logic;
  signal mem_to_bram_converter_B_we0 : std_logic;
  signal mem_to_bram_converter_B_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_ce1 : std_logic;
  signal mem_to_bram_converter_B_we1 : std_logic;
  signal mem_to_bram_converter_B_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal jacobi_1d_imper_wrapped_A_end_valid : std_logic;
  signal jacobi_1d_imper_wrapped_A_end_ready : std_logic;
  signal jacobi_1d_imper_wrapped_B_end_valid : std_logic;
  signal jacobi_1d_imper_wrapped_B_end_ready : std_logic;
  signal jacobi_1d_imper_wrapped_end_valid : std_logic;
  signal jacobi_1d_imper_wrapped_end_ready : std_logic;
  signal jacobi_1d_imper_wrapped_A_loadEn : std_logic;
  signal jacobi_1d_imper_wrapped_A_loadAddr : std_logic_vector(6 downto 0);
  signal jacobi_1d_imper_wrapped_A_storeEn : std_logic;
  signal jacobi_1d_imper_wrapped_A_storeAddr : std_logic_vector(6 downto 0);
  signal jacobi_1d_imper_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal jacobi_1d_imper_wrapped_B_loadEn : std_logic;
  signal jacobi_1d_imper_wrapped_B_loadAddr : std_logic_vector(6 downto 0);
  signal jacobi_1d_imper_wrapped_B_storeEn : std_logic;
  signal jacobi_1d_imper_wrapped_B_storeAddr : std_logic_vector(6 downto 0);
  signal jacobi_1d_imper_wrapped_B_storeData : std_logic_vector(31 downto 0);

begin

  A_end_valid <= jacobi_1d_imper_wrapped_A_end_valid;
  jacobi_1d_imper_wrapped_A_end_ready <= A_end_ready;
  B_end_valid <= jacobi_1d_imper_wrapped_B_end_valid;
  jacobi_1d_imper_wrapped_B_end_ready <= B_end_ready;
  end_valid <= jacobi_1d_imper_wrapped_end_valid;
  jacobi_1d_imper_wrapped_end_ready <= end_ready;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;
  B_ce0 <= mem_to_bram_converter_B_ce0;
  B_we0 <= mem_to_bram_converter_B_we0;
  B_address0 <= mem_to_bram_converter_B_address0;
  B_dout0 <= mem_to_bram_converter_B_dout0;
  B_ce1 <= mem_to_bram_converter_B_ce1;
  B_we1 <= mem_to_bram_converter_B_we1;
  B_address1 <= mem_to_bram_converter_B_address1;
  B_dout1 <= mem_to_bram_converter_B_dout1;

  mem_to_bram_converter_B : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => jacobi_1d_imper_wrapped_B_loadEn,
      loadAddr => jacobi_1d_imper_wrapped_B_loadAddr,
      storeEn => jacobi_1d_imper_wrapped_B_storeEn,
      storeAddr => jacobi_1d_imper_wrapped_B_storeAddr,
      storeData => jacobi_1d_imper_wrapped_B_storeData,
      din0 => B_din0,
      din1 => B_din1,
      ce0 => mem_to_bram_converter_B_ce0,
      we0 => mem_to_bram_converter_B_we0,
      address0 => mem_to_bram_converter_B_address0,
      dout0 => mem_to_bram_converter_B_dout0,
      ce1 => mem_to_bram_converter_B_ce1,
      we1 => mem_to_bram_converter_B_we1,
      address1 => mem_to_bram_converter_B_address1,
      dout1 => mem_to_bram_converter_B_dout1,
      loadData => mem_to_bram_converter_B_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => jacobi_1d_imper_wrapped_A_loadEn,
      loadAddr => jacobi_1d_imper_wrapped_A_loadAddr,
      storeEn => jacobi_1d_imper_wrapped_A_storeEn,
      storeAddr => jacobi_1d_imper_wrapped_A_storeAddr,
      storeData => jacobi_1d_imper_wrapped_A_storeData,
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

  jacobi_1d_imper_wrapped : entity work.jacobi_1d_imper(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      B_loadData => mem_to_bram_converter_B_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      B_start_valid => B_start_valid,
      B_start_ready => B_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      A_end_valid => jacobi_1d_imper_wrapped_A_end_valid,
      A_end_ready => jacobi_1d_imper_wrapped_A_end_ready,
      B_end_valid => jacobi_1d_imper_wrapped_B_end_valid,
      B_end_ready => jacobi_1d_imper_wrapped_B_end_ready,
      end_valid => jacobi_1d_imper_wrapped_end_valid,
      end_ready => jacobi_1d_imper_wrapped_end_ready,
      A_loadEn => jacobi_1d_imper_wrapped_A_loadEn,
      A_loadAddr => jacobi_1d_imper_wrapped_A_loadAddr,
      A_storeEn => jacobi_1d_imper_wrapped_A_storeEn,
      A_storeAddr => jacobi_1d_imper_wrapped_A_storeAddr,
      A_storeData => jacobi_1d_imper_wrapped_A_storeData,
      B_loadEn => jacobi_1d_imper_wrapped_B_loadEn,
      B_loadAddr => jacobi_1d_imper_wrapped_B_loadAddr,
      B_storeEn => jacobi_1d_imper_wrapped_B_storeEn,
      B_storeAddr => jacobi_1d_imper_wrapped_B_storeAddr,
      B_storeData => jacobi_1d_imper_wrapped_B_storeData
    );

end architecture;
