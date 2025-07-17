library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity iterative_sqrt_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    A_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    A_end_valid : out std_logic;
    end_valid : out std_logic;
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(3 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(3 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of iterative_sqrt_wrapper is

  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal iterative_sqrt_wrapped_out0 : std_logic_vector(31 downto 0);
  signal iterative_sqrt_wrapped_out0_valid : std_logic;
  signal iterative_sqrt_wrapped_out0_ready : std_logic;
  signal iterative_sqrt_wrapped_A_end_valid : std_logic;
  signal iterative_sqrt_wrapped_A_end_ready : std_logic;
  signal iterative_sqrt_wrapped_end_valid : std_logic;
  signal iterative_sqrt_wrapped_end_ready : std_logic;
  signal iterative_sqrt_wrapped_A_loadEn : std_logic;
  signal iterative_sqrt_wrapped_A_loadAddr : std_logic_vector(3 downto 0);
  signal iterative_sqrt_wrapped_A_storeEn : std_logic;
  signal iterative_sqrt_wrapped_A_storeAddr : std_logic_vector(3 downto 0);
  signal iterative_sqrt_wrapped_A_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= iterative_sqrt_wrapped_out0;
  out0_valid <= iterative_sqrt_wrapped_out0_valid;
  iterative_sqrt_wrapped_out0_ready <= out0_ready;
  A_end_valid <= iterative_sqrt_wrapped_A_end_valid;
  iterative_sqrt_wrapped_A_end_ready <= A_end_ready;
  end_valid <= iterative_sqrt_wrapped_end_valid;
  iterative_sqrt_wrapped_end_ready <= end_ready;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 4)
    port map(
      loadEn => iterative_sqrt_wrapped_A_loadEn,
      loadAddr => iterative_sqrt_wrapped_A_loadAddr,
      storeEn => iterative_sqrt_wrapped_A_storeEn,
      storeAddr => iterative_sqrt_wrapped_A_storeAddr,
      storeData => iterative_sqrt_wrapped_A_storeData,
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

  iterative_sqrt_wrapped : entity work.iterative_sqrt(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => iterative_sqrt_wrapped_out0,
      out0_valid => iterative_sqrt_wrapped_out0_valid,
      out0_ready => iterative_sqrt_wrapped_out0_ready,
      A_end_valid => iterative_sqrt_wrapped_A_end_valid,
      A_end_ready => iterative_sqrt_wrapped_A_end_ready,
      end_valid => iterative_sqrt_wrapped_end_valid,
      end_ready => iterative_sqrt_wrapped_end_ready,
      A_loadEn => iterative_sqrt_wrapped_A_loadEn,
      A_loadAddr => iterative_sqrt_wrapped_A_loadAddr,
      A_storeEn => iterative_sqrt_wrapped_A_storeEn,
      A_storeAddr => iterative_sqrt_wrapped_A_storeAddr,
      A_storeData => iterative_sqrt_wrapped_A_storeData
    );

end architecture;
