library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity matrix_power_wrapper is
  port (
    mat_din0 : in std_logic_vector(31 downto 0);
    mat_din1 : in std_logic_vector(31 downto 0);
    row_din0 : in std_logic_vector(31 downto 0);
    row_din1 : in std_logic_vector(31 downto 0);
    col_din0 : in std_logic_vector(31 downto 0);
    col_din1 : in std_logic_vector(31 downto 0);
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    mat_start_valid : in std_logic;
    row_start_valid : in std_logic;
    col_start_valid : in std_logic;
    a_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    mat_end_ready : in std_logic;
    row_end_ready : in std_logic;
    col_end_ready : in std_logic;
    a_end_ready : in std_logic;
    end_ready : in std_logic;
    mat_start_ready : out std_logic;
    row_start_ready : out std_logic;
    col_start_ready : out std_logic;
    a_start_ready : out std_logic;
    start_ready : out std_logic;
    mat_end_valid : out std_logic;
    row_end_valid : out std_logic;
    col_end_valid : out std_logic;
    a_end_valid : out std_logic;
    end_valid : out std_logic;
    mat_ce0 : out std_logic;
    mat_we0 : out std_logic;
    mat_address0 : out std_logic_vector(8 downto 0);
    mat_dout0 : out std_logic_vector(31 downto 0);
    mat_ce1 : out std_logic;
    mat_we1 : out std_logic;
    mat_address1 : out std_logic_vector(8 downto 0);
    mat_dout1 : out std_logic_vector(31 downto 0);
    row_ce0 : out std_logic;
    row_we0 : out std_logic;
    row_address0 : out std_logic_vector(4 downto 0);
    row_dout0 : out std_logic_vector(31 downto 0);
    row_ce1 : out std_logic;
    row_we1 : out std_logic;
    row_address1 : out std_logic_vector(4 downto 0);
    row_dout1 : out std_logic_vector(31 downto 0);
    col_ce0 : out std_logic;
    col_we0 : out std_logic;
    col_address0 : out std_logic_vector(4 downto 0);
    col_dout0 : out std_logic_vector(31 downto 0);
    col_ce1 : out std_logic;
    col_we1 : out std_logic;
    col_address1 : out std_logic_vector(4 downto 0);
    col_dout1 : out std_logic_vector(31 downto 0);
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(4 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(4 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of matrix_power_wrapper is

  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_col_ce0 : std_logic;
  signal mem_to_bram_converter_col_we0 : std_logic;
  signal mem_to_bram_converter_col_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_col_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_col_ce1 : std_logic;
  signal mem_to_bram_converter_col_we1 : std_logic;
  signal mem_to_bram_converter_col_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_col_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_col_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_mat_ce0 : std_logic;
  signal mem_to_bram_converter_mat_we0 : std_logic;
  signal mem_to_bram_converter_mat_address0 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_mat_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_mat_ce1 : std_logic;
  signal mem_to_bram_converter_mat_we1 : std_logic;
  signal mem_to_bram_converter_mat_address1 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_mat_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_mat_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_row_ce0 : std_logic;
  signal mem_to_bram_converter_row_we0 : std_logic;
  signal mem_to_bram_converter_row_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_row_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_row_ce1 : std_logic;
  signal mem_to_bram_converter_row_we1 : std_logic;
  signal mem_to_bram_converter_row_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_row_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_row_loadData : std_logic_vector(31 downto 0);
  signal matrix_power_wrapped_mat_end_valid : std_logic;
  signal matrix_power_wrapped_mat_end_ready : std_logic;
  signal matrix_power_wrapped_row_end_valid : std_logic;
  signal matrix_power_wrapped_row_end_ready : std_logic;
  signal matrix_power_wrapped_col_end_valid : std_logic;
  signal matrix_power_wrapped_col_end_ready : std_logic;
  signal matrix_power_wrapped_a_end_valid : std_logic;
  signal matrix_power_wrapped_a_end_ready : std_logic;
  signal matrix_power_wrapped_end_valid : std_logic;
  signal matrix_power_wrapped_end_ready : std_logic;
  signal matrix_power_wrapped_mat_loadEn : std_logic;
  signal matrix_power_wrapped_mat_loadAddr : std_logic_vector(8 downto 0);
  signal matrix_power_wrapped_mat_storeEn : std_logic;
  signal matrix_power_wrapped_mat_storeAddr : std_logic_vector(8 downto 0);
  signal matrix_power_wrapped_mat_storeData : std_logic_vector(31 downto 0);
  signal matrix_power_wrapped_row_loadEn : std_logic;
  signal matrix_power_wrapped_row_loadAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_row_storeEn : std_logic;
  signal matrix_power_wrapped_row_storeAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_row_storeData : std_logic_vector(31 downto 0);
  signal matrix_power_wrapped_col_loadEn : std_logic;
  signal matrix_power_wrapped_col_loadAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_col_storeEn : std_logic;
  signal matrix_power_wrapped_col_storeAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_col_storeData : std_logic_vector(31 downto 0);
  signal matrix_power_wrapped_a_loadEn : std_logic;
  signal matrix_power_wrapped_a_loadAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_a_storeEn : std_logic;
  signal matrix_power_wrapped_a_storeAddr : std_logic_vector(4 downto 0);
  signal matrix_power_wrapped_a_storeData : std_logic_vector(31 downto 0);

begin

  mat_end_valid <= matrix_power_wrapped_mat_end_valid;
  matrix_power_wrapped_mat_end_ready <= mat_end_ready;
  row_end_valid <= matrix_power_wrapped_row_end_valid;
  matrix_power_wrapped_row_end_ready <= row_end_ready;
  col_end_valid <= matrix_power_wrapped_col_end_valid;
  matrix_power_wrapped_col_end_ready <= col_end_ready;
  a_end_valid <= matrix_power_wrapped_a_end_valid;
  matrix_power_wrapped_a_end_ready <= a_end_ready;
  end_valid <= matrix_power_wrapped_end_valid;
  matrix_power_wrapped_end_ready <= end_ready;
  mat_ce0 <= mem_to_bram_converter_mat_ce0;
  mat_we0 <= mem_to_bram_converter_mat_we0;
  mat_address0 <= mem_to_bram_converter_mat_address0;
  mat_dout0 <= mem_to_bram_converter_mat_dout0;
  mat_ce1 <= mem_to_bram_converter_mat_ce1;
  mat_we1 <= mem_to_bram_converter_mat_we1;
  mat_address1 <= mem_to_bram_converter_mat_address1;
  mat_dout1 <= mem_to_bram_converter_mat_dout1;
  row_ce0 <= mem_to_bram_converter_row_ce0;
  row_we0 <= mem_to_bram_converter_row_we0;
  row_address0 <= mem_to_bram_converter_row_address0;
  row_dout0 <= mem_to_bram_converter_row_dout0;
  row_ce1 <= mem_to_bram_converter_row_ce1;
  row_we1 <= mem_to_bram_converter_row_we1;
  row_address1 <= mem_to_bram_converter_row_address1;
  row_dout1 <= mem_to_bram_converter_row_dout1;
  col_ce0 <= mem_to_bram_converter_col_ce0;
  col_we0 <= mem_to_bram_converter_col_we0;
  col_address0 <= mem_to_bram_converter_col_address0;
  col_dout0 <= mem_to_bram_converter_col_dout0;
  col_ce1 <= mem_to_bram_converter_col_ce1;
  col_we1 <= mem_to_bram_converter_col_we1;
  col_address1 <= mem_to_bram_converter_col_address1;
  col_dout1 <= mem_to_bram_converter_col_dout1;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;

  mem_to_bram_converter_a : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => matrix_power_wrapped_a_loadEn,
      loadAddr => matrix_power_wrapped_a_loadAddr,
      storeEn => matrix_power_wrapped_a_storeEn,
      storeAddr => matrix_power_wrapped_a_storeAddr,
      storeData => matrix_power_wrapped_a_storeData,
      din0 => a_din0,
      din1 => a_din1,
      ce0 => mem_to_bram_converter_a_ce0,
      we0 => mem_to_bram_converter_a_we0,
      address0 => mem_to_bram_converter_a_address0,
      dout0 => mem_to_bram_converter_a_dout0,
      ce1 => mem_to_bram_converter_a_ce1,
      we1 => mem_to_bram_converter_a_we1,
      address1 => mem_to_bram_converter_a_address1,
      dout1 => mem_to_bram_converter_a_dout1,
      loadData => mem_to_bram_converter_a_loadData
    );

  mem_to_bram_converter_col : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => matrix_power_wrapped_col_loadEn,
      loadAddr => matrix_power_wrapped_col_loadAddr,
      storeEn => matrix_power_wrapped_col_storeEn,
      storeAddr => matrix_power_wrapped_col_storeAddr,
      storeData => matrix_power_wrapped_col_storeData,
      din0 => col_din0,
      din1 => col_din1,
      ce0 => mem_to_bram_converter_col_ce0,
      we0 => mem_to_bram_converter_col_we0,
      address0 => mem_to_bram_converter_col_address0,
      dout0 => mem_to_bram_converter_col_dout0,
      ce1 => mem_to_bram_converter_col_ce1,
      we1 => mem_to_bram_converter_col_we1,
      address1 => mem_to_bram_converter_col_address1,
      dout1 => mem_to_bram_converter_col_dout1,
      loadData => mem_to_bram_converter_col_loadData
    );

  mem_to_bram_converter_mat : entity work.mem_to_bram(arch) generic map(32, 9)
    port map(
      loadEn => matrix_power_wrapped_mat_loadEn,
      loadAddr => matrix_power_wrapped_mat_loadAddr,
      storeEn => matrix_power_wrapped_mat_storeEn,
      storeAddr => matrix_power_wrapped_mat_storeAddr,
      storeData => matrix_power_wrapped_mat_storeData,
      din0 => mat_din0,
      din1 => mat_din1,
      ce0 => mem_to_bram_converter_mat_ce0,
      we0 => mem_to_bram_converter_mat_we0,
      address0 => mem_to_bram_converter_mat_address0,
      dout0 => mem_to_bram_converter_mat_dout0,
      ce1 => mem_to_bram_converter_mat_ce1,
      we1 => mem_to_bram_converter_mat_we1,
      address1 => mem_to_bram_converter_mat_address1,
      dout1 => mem_to_bram_converter_mat_dout1,
      loadData => mem_to_bram_converter_mat_loadData
    );

  mem_to_bram_converter_row : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => matrix_power_wrapped_row_loadEn,
      loadAddr => matrix_power_wrapped_row_loadAddr,
      storeEn => matrix_power_wrapped_row_storeEn,
      storeAddr => matrix_power_wrapped_row_storeAddr,
      storeData => matrix_power_wrapped_row_storeData,
      din0 => row_din0,
      din1 => row_din1,
      ce0 => mem_to_bram_converter_row_ce0,
      we0 => mem_to_bram_converter_row_we0,
      address0 => mem_to_bram_converter_row_address0,
      dout0 => mem_to_bram_converter_row_dout0,
      ce1 => mem_to_bram_converter_row_ce1,
      we1 => mem_to_bram_converter_row_we1,
      address1 => mem_to_bram_converter_row_address1,
      dout1 => mem_to_bram_converter_row_dout1,
      loadData => mem_to_bram_converter_row_loadData
    );

  matrix_power_wrapped : entity work.matrix_power(behavioral)
    port map(
      mat_loadData => mem_to_bram_converter_mat_loadData,
      row_loadData => mem_to_bram_converter_row_loadData,
      col_loadData => mem_to_bram_converter_col_loadData,
      a_loadData => mem_to_bram_converter_a_loadData,
      mat_start_valid => mat_start_valid,
      mat_start_ready => mat_start_ready,
      row_start_valid => row_start_valid,
      row_start_ready => row_start_ready,
      col_start_valid => col_start_valid,
      col_start_ready => col_start_ready,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      mat_end_valid => matrix_power_wrapped_mat_end_valid,
      mat_end_ready => matrix_power_wrapped_mat_end_ready,
      row_end_valid => matrix_power_wrapped_row_end_valid,
      row_end_ready => matrix_power_wrapped_row_end_ready,
      col_end_valid => matrix_power_wrapped_col_end_valid,
      col_end_ready => matrix_power_wrapped_col_end_ready,
      a_end_valid => matrix_power_wrapped_a_end_valid,
      a_end_ready => matrix_power_wrapped_a_end_ready,
      end_valid => matrix_power_wrapped_end_valid,
      end_ready => matrix_power_wrapped_end_ready,
      mat_loadEn => matrix_power_wrapped_mat_loadEn,
      mat_loadAddr => matrix_power_wrapped_mat_loadAddr,
      mat_storeEn => matrix_power_wrapped_mat_storeEn,
      mat_storeAddr => matrix_power_wrapped_mat_storeAddr,
      mat_storeData => matrix_power_wrapped_mat_storeData,
      row_loadEn => matrix_power_wrapped_row_loadEn,
      row_loadAddr => matrix_power_wrapped_row_loadAddr,
      row_storeEn => matrix_power_wrapped_row_storeEn,
      row_storeAddr => matrix_power_wrapped_row_storeAddr,
      row_storeData => matrix_power_wrapped_row_storeData,
      col_loadEn => matrix_power_wrapped_col_loadEn,
      col_loadAddr => matrix_power_wrapped_col_loadAddr,
      col_storeEn => matrix_power_wrapped_col_storeEn,
      col_storeAddr => matrix_power_wrapped_col_storeAddr,
      col_storeData => matrix_power_wrapped_col_storeData,
      a_loadEn => matrix_power_wrapped_a_loadEn,
      a_loadAddr => matrix_power_wrapped_a_loadAddr,
      a_storeEn => matrix_power_wrapped_a_storeEn,
      a_storeAddr => matrix_power_wrapped_a_storeAddr,
      a_storeData => matrix_power_wrapped_a_storeData
    );

end architecture;
