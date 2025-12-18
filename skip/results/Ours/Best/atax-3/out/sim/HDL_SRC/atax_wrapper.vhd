library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity atax_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    x_din0 : in std_logic_vector(31 downto 0);
    x_din1 : in std_logic_vector(31 downto 0);
    y_din0 : in std_logic_vector(31 downto 0);
    y_din1 : in std_logic_vector(31 downto 0);
    tmp_din0 : in std_logic_vector(31 downto 0);
    tmp_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    x_start_valid : in std_logic;
    y_start_valid : in std_logic;
    tmp_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    x_end_ready : in std_logic;
    y_end_ready : in std_logic;
    tmp_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    x_start_ready : out std_logic;
    y_start_ready : out std_logic;
    tmp_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    x_end_valid : out std_logic;
    y_end_valid : out std_logic;
    tmp_end_valid : out std_logic;
    end_valid : out std_logic;
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(8 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(8 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0);
    x_ce0 : out std_logic;
    x_we0 : out std_logic;
    x_address0 : out std_logic_vector(4 downto 0);
    x_dout0 : out std_logic_vector(31 downto 0);
    x_ce1 : out std_logic;
    x_we1 : out std_logic;
    x_address1 : out std_logic_vector(4 downto 0);
    x_dout1 : out std_logic_vector(31 downto 0);
    y_ce0 : out std_logic;
    y_we0 : out std_logic;
    y_address0 : out std_logic_vector(4 downto 0);
    y_dout0 : out std_logic_vector(31 downto 0);
    y_ce1 : out std_logic;
    y_we1 : out std_logic;
    y_address1 : out std_logic_vector(4 downto 0);
    y_dout1 : out std_logic_vector(31 downto 0);
    tmp_ce0 : out std_logic;
    tmp_we0 : out std_logic;
    tmp_address0 : out std_logic_vector(4 downto 0);
    tmp_dout0 : out std_logic_vector(31 downto 0);
    tmp_ce1 : out std_logic;
    tmp_we1 : out std_logic;
    tmp_address1 : out std_logic_vector(4 downto 0);
    tmp_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of atax_wrapper is

  signal mem_to_bram_converter_y_ce0 : std_logic;
  signal mem_to_bram_converter_y_we0 : std_logic;
  signal mem_to_bram_converter_y_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y_ce1 : std_logic;
  signal mem_to_bram_converter_y_we1 : std_logic;
  signal mem_to_bram_converter_y_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(8 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_ce0 : std_logic;
  signal mem_to_bram_converter_tmp_we0 : std_logic;
  signal mem_to_bram_converter_tmp_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_tmp_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_ce1 : std_logic;
  signal mem_to_bram_converter_tmp_we1 : std_logic;
  signal mem_to_bram_converter_tmp_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_tmp_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_tmp_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_ce0 : std_logic;
  signal mem_to_bram_converter_x_we0 : std_logic;
  signal mem_to_bram_converter_x_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_ce1 : std_logic;
  signal mem_to_bram_converter_x_we1 : std_logic;
  signal mem_to_bram_converter_x_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_loadData : std_logic_vector(31 downto 0);
  signal atax_wrapped_A_end_valid : std_logic;
  signal atax_wrapped_A_end_ready : std_logic;
  signal atax_wrapped_x_end_valid : std_logic;
  signal atax_wrapped_x_end_ready : std_logic;
  signal atax_wrapped_y_end_valid : std_logic;
  signal atax_wrapped_y_end_ready : std_logic;
  signal atax_wrapped_tmp_end_valid : std_logic;
  signal atax_wrapped_tmp_end_ready : std_logic;
  signal atax_wrapped_end_valid : std_logic;
  signal atax_wrapped_end_ready : std_logic;
  signal atax_wrapped_A_loadEn : std_logic;
  signal atax_wrapped_A_loadAddr : std_logic_vector(8 downto 0);
  signal atax_wrapped_A_storeEn : std_logic;
  signal atax_wrapped_A_storeAddr : std_logic_vector(8 downto 0);
  signal atax_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal atax_wrapped_x_loadEn : std_logic;
  signal atax_wrapped_x_loadAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_x_storeEn : std_logic;
  signal atax_wrapped_x_storeAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_x_storeData : std_logic_vector(31 downto 0);
  signal atax_wrapped_y_loadEn : std_logic;
  signal atax_wrapped_y_loadAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_y_storeEn : std_logic;
  signal atax_wrapped_y_storeAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_y_storeData : std_logic_vector(31 downto 0);
  signal atax_wrapped_tmp_loadEn : std_logic;
  signal atax_wrapped_tmp_loadAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_tmp_storeEn : std_logic;
  signal atax_wrapped_tmp_storeAddr : std_logic_vector(4 downto 0);
  signal atax_wrapped_tmp_storeData : std_logic_vector(31 downto 0);

begin

  A_end_valid <= atax_wrapped_A_end_valid;
  atax_wrapped_A_end_ready <= A_end_ready;
  x_end_valid <= atax_wrapped_x_end_valid;
  atax_wrapped_x_end_ready <= x_end_ready;
  y_end_valid <= atax_wrapped_y_end_valid;
  atax_wrapped_y_end_ready <= y_end_ready;
  tmp_end_valid <= atax_wrapped_tmp_end_valid;
  atax_wrapped_tmp_end_ready <= tmp_end_ready;
  end_valid <= atax_wrapped_end_valid;
  atax_wrapped_end_ready <= end_ready;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;
  x_ce0 <= mem_to_bram_converter_x_ce0;
  x_we0 <= mem_to_bram_converter_x_we0;
  x_address0 <= mem_to_bram_converter_x_address0;
  x_dout0 <= mem_to_bram_converter_x_dout0;
  x_ce1 <= mem_to_bram_converter_x_ce1;
  x_we1 <= mem_to_bram_converter_x_we1;
  x_address1 <= mem_to_bram_converter_x_address1;
  x_dout1 <= mem_to_bram_converter_x_dout1;
  y_ce0 <= mem_to_bram_converter_y_ce0;
  y_we0 <= mem_to_bram_converter_y_we0;
  y_address0 <= mem_to_bram_converter_y_address0;
  y_dout0 <= mem_to_bram_converter_y_dout0;
  y_ce1 <= mem_to_bram_converter_y_ce1;
  y_we1 <= mem_to_bram_converter_y_we1;
  y_address1 <= mem_to_bram_converter_y_address1;
  y_dout1 <= mem_to_bram_converter_y_dout1;
  tmp_ce0 <= mem_to_bram_converter_tmp_ce0;
  tmp_we0 <= mem_to_bram_converter_tmp_we0;
  tmp_address0 <= mem_to_bram_converter_tmp_address0;
  tmp_dout0 <= mem_to_bram_converter_tmp_dout0;
  tmp_ce1 <= mem_to_bram_converter_tmp_ce1;
  tmp_we1 <= mem_to_bram_converter_tmp_we1;
  tmp_address1 <= mem_to_bram_converter_tmp_address1;
  tmp_dout1 <= mem_to_bram_converter_tmp_dout1;

  mem_to_bram_converter_y : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => atax_wrapped_y_loadEn,
      loadAddr => atax_wrapped_y_loadAddr,
      storeEn => atax_wrapped_y_storeEn,
      storeAddr => atax_wrapped_y_storeAddr,
      storeData => atax_wrapped_y_storeData,
      din0 => y_din0,
      din1 => y_din1,
      ce0 => mem_to_bram_converter_y_ce0,
      we0 => mem_to_bram_converter_y_we0,
      address0 => mem_to_bram_converter_y_address0,
      dout0 => mem_to_bram_converter_y_dout0,
      ce1 => mem_to_bram_converter_y_ce1,
      we1 => mem_to_bram_converter_y_we1,
      address1 => mem_to_bram_converter_y_address1,
      dout1 => mem_to_bram_converter_y_dout1,
      loadData => mem_to_bram_converter_y_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 9)
    port map(
      loadEn => atax_wrapped_A_loadEn,
      loadAddr => atax_wrapped_A_loadAddr,
      storeEn => atax_wrapped_A_storeEn,
      storeAddr => atax_wrapped_A_storeAddr,
      storeData => atax_wrapped_A_storeData,
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

  mem_to_bram_converter_tmp : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => atax_wrapped_tmp_loadEn,
      loadAddr => atax_wrapped_tmp_loadAddr,
      storeEn => atax_wrapped_tmp_storeEn,
      storeAddr => atax_wrapped_tmp_storeAddr,
      storeData => atax_wrapped_tmp_storeData,
      din0 => tmp_din0,
      din1 => tmp_din1,
      ce0 => mem_to_bram_converter_tmp_ce0,
      we0 => mem_to_bram_converter_tmp_we0,
      address0 => mem_to_bram_converter_tmp_address0,
      dout0 => mem_to_bram_converter_tmp_dout0,
      ce1 => mem_to_bram_converter_tmp_ce1,
      we1 => mem_to_bram_converter_tmp_we1,
      address1 => mem_to_bram_converter_tmp_address1,
      dout1 => mem_to_bram_converter_tmp_dout1,
      loadData => mem_to_bram_converter_tmp_loadData
    );

  mem_to_bram_converter_x : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => atax_wrapped_x_loadEn,
      loadAddr => atax_wrapped_x_loadAddr,
      storeEn => atax_wrapped_x_storeEn,
      storeAddr => atax_wrapped_x_storeAddr,
      storeData => atax_wrapped_x_storeData,
      din0 => x_din0,
      din1 => x_din1,
      ce0 => mem_to_bram_converter_x_ce0,
      we0 => mem_to_bram_converter_x_we0,
      address0 => mem_to_bram_converter_x_address0,
      dout0 => mem_to_bram_converter_x_dout0,
      ce1 => mem_to_bram_converter_x_ce1,
      we1 => mem_to_bram_converter_x_we1,
      address1 => mem_to_bram_converter_x_address1,
      dout1 => mem_to_bram_converter_x_dout1,
      loadData => mem_to_bram_converter_x_loadData
    );

  atax_wrapped : entity work.atax(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      x_loadData => mem_to_bram_converter_x_loadData,
      y_loadData => mem_to_bram_converter_y_loadData,
      tmp_loadData => mem_to_bram_converter_tmp_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      x_start_valid => x_start_valid,
      x_start_ready => x_start_ready,
      y_start_valid => y_start_valid,
      y_start_ready => y_start_ready,
      tmp_start_valid => tmp_start_valid,
      tmp_start_ready => tmp_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      A_end_valid => atax_wrapped_A_end_valid,
      A_end_ready => atax_wrapped_A_end_ready,
      x_end_valid => atax_wrapped_x_end_valid,
      x_end_ready => atax_wrapped_x_end_ready,
      y_end_valid => atax_wrapped_y_end_valid,
      y_end_ready => atax_wrapped_y_end_ready,
      tmp_end_valid => atax_wrapped_tmp_end_valid,
      tmp_end_ready => atax_wrapped_tmp_end_ready,
      end_valid => atax_wrapped_end_valid,
      end_ready => atax_wrapped_end_ready,
      A_loadEn => atax_wrapped_A_loadEn,
      A_loadAddr => atax_wrapped_A_loadAddr,
      A_storeEn => atax_wrapped_A_storeEn,
      A_storeAddr => atax_wrapped_A_storeAddr,
      A_storeData => atax_wrapped_A_storeData,
      x_loadEn => atax_wrapped_x_loadEn,
      x_loadAddr => atax_wrapped_x_loadAddr,
      x_storeEn => atax_wrapped_x_storeEn,
      x_storeAddr => atax_wrapped_x_storeAddr,
      x_storeData => atax_wrapped_x_storeData,
      y_loadEn => atax_wrapped_y_loadEn,
      y_loadAddr => atax_wrapped_y_loadAddr,
      y_storeEn => atax_wrapped_y_storeEn,
      y_storeAddr => atax_wrapped_y_storeAddr,
      y_storeData => atax_wrapped_y_storeData,
      tmp_loadEn => atax_wrapped_tmp_loadEn,
      tmp_loadAddr => atax_wrapped_tmp_loadAddr,
      tmp_storeEn => atax_wrapped_tmp_storeEn,
      tmp_storeAddr => atax_wrapped_tmp_storeAddr,
      tmp_storeData => atax_wrapped_tmp_storeData
    );

end architecture;
