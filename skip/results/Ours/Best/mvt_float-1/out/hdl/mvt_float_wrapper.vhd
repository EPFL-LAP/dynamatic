library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mvt_float_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    x1_din0 : in std_logic_vector(31 downto 0);
    x1_din1 : in std_logic_vector(31 downto 0);
    x2_din0 : in std_logic_vector(31 downto 0);
    x2_din1 : in std_logic_vector(31 downto 0);
    y1_din0 : in std_logic_vector(31 downto 0);
    y1_din1 : in std_logic_vector(31 downto 0);
    y2_din0 : in std_logic_vector(31 downto 0);
    y2_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    x1_start_valid : in std_logic;
    x2_start_valid : in std_logic;
    y1_start_valid : in std_logic;
    y2_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    x1_end_ready : in std_logic;
    x2_end_ready : in std_logic;
    y1_end_ready : in std_logic;
    y2_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    x1_start_ready : out std_logic;
    x2_start_ready : out std_logic;
    y1_start_ready : out std_logic;
    y2_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    x1_end_valid : out std_logic;
    x2_end_valid : out std_logic;
    y1_end_valid : out std_logic;
    y2_end_valid : out std_logic;
    end_valid : out std_logic;
    A_ce0 : out std_logic;
    A_we0 : out std_logic;
    A_address0 : out std_logic_vector(9 downto 0);
    A_dout0 : out std_logic_vector(31 downto 0);
    A_ce1 : out std_logic;
    A_we1 : out std_logic;
    A_address1 : out std_logic_vector(9 downto 0);
    A_dout1 : out std_logic_vector(31 downto 0);
    x1_ce0 : out std_logic;
    x1_we0 : out std_logic;
    x1_address0 : out std_logic_vector(4 downto 0);
    x1_dout0 : out std_logic_vector(31 downto 0);
    x1_ce1 : out std_logic;
    x1_we1 : out std_logic;
    x1_address1 : out std_logic_vector(4 downto 0);
    x1_dout1 : out std_logic_vector(31 downto 0);
    x2_ce0 : out std_logic;
    x2_we0 : out std_logic;
    x2_address0 : out std_logic_vector(4 downto 0);
    x2_dout0 : out std_logic_vector(31 downto 0);
    x2_ce1 : out std_logic;
    x2_we1 : out std_logic;
    x2_address1 : out std_logic_vector(4 downto 0);
    x2_dout1 : out std_logic_vector(31 downto 0);
    y1_ce0 : out std_logic;
    y1_we0 : out std_logic;
    y1_address0 : out std_logic_vector(4 downto 0);
    y1_dout0 : out std_logic_vector(31 downto 0);
    y1_ce1 : out std_logic;
    y1_we1 : out std_logic;
    y1_address1 : out std_logic_vector(4 downto 0);
    y1_dout1 : out std_logic_vector(31 downto 0);
    y2_ce0 : out std_logic;
    y2_we0 : out std_logic;
    y2_address0 : out std_logic_vector(4 downto 0);
    y2_dout0 : out std_logic_vector(31 downto 0);
    y2_ce1 : out std_logic;
    y2_we1 : out std_logic;
    y2_address1 : out std_logic_vector(4 downto 0);
    y2_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of mvt_float_wrapper is

  signal mem_to_bram_converter_y2_ce0 : std_logic;
  signal mem_to_bram_converter_y2_we0 : std_logic;
  signal mem_to_bram_converter_y2_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y2_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y2_ce1 : std_logic;
  signal mem_to_bram_converter_y2_we1 : std_logic;
  signal mem_to_bram_converter_y2_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y2_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y2_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x2_ce0 : std_logic;
  signal mem_to_bram_converter_x2_we0 : std_logic;
  signal mem_to_bram_converter_x2_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x2_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x2_ce1 : std_logic;
  signal mem_to_bram_converter_x2_we1 : std_logic;
  signal mem_to_bram_converter_x2_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x2_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x2_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y1_ce0 : std_logic;
  signal mem_to_bram_converter_y1_we0 : std_logic;
  signal mem_to_bram_converter_y1_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y1_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y1_ce1 : std_logic;
  signal mem_to_bram_converter_y1_we1 : std_logic;
  signal mem_to_bram_converter_y1_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_y1_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y1_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x1_ce0 : std_logic;
  signal mem_to_bram_converter_x1_we0 : std_logic;
  signal mem_to_bram_converter_x1_address0 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x1_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x1_ce1 : std_logic;
  signal mem_to_bram_converter_x1_we1 : std_logic;
  signal mem_to_bram_converter_x1_address1 : std_logic_vector(4 downto 0);
  signal mem_to_bram_converter_x1_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x1_loadData : std_logic_vector(31 downto 0);
  signal mvt_float_wrapped_A_end_valid : std_logic;
  signal mvt_float_wrapped_A_end_ready : std_logic;
  signal mvt_float_wrapped_x1_end_valid : std_logic;
  signal mvt_float_wrapped_x1_end_ready : std_logic;
  signal mvt_float_wrapped_x2_end_valid : std_logic;
  signal mvt_float_wrapped_x2_end_ready : std_logic;
  signal mvt_float_wrapped_y1_end_valid : std_logic;
  signal mvt_float_wrapped_y1_end_ready : std_logic;
  signal mvt_float_wrapped_y2_end_valid : std_logic;
  signal mvt_float_wrapped_y2_end_ready : std_logic;
  signal mvt_float_wrapped_end_valid : std_logic;
  signal mvt_float_wrapped_end_ready : std_logic;
  signal mvt_float_wrapped_A_loadEn : std_logic;
  signal mvt_float_wrapped_A_loadAddr : std_logic_vector(9 downto 0);
  signal mvt_float_wrapped_A_storeEn : std_logic;
  signal mvt_float_wrapped_A_storeAddr : std_logic_vector(9 downto 0);
  signal mvt_float_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal mvt_float_wrapped_x1_loadEn : std_logic;
  signal mvt_float_wrapped_x1_loadAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_x1_storeEn : std_logic;
  signal mvt_float_wrapped_x1_storeAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_x1_storeData : std_logic_vector(31 downto 0);
  signal mvt_float_wrapped_x2_loadEn : std_logic;
  signal mvt_float_wrapped_x2_loadAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_x2_storeEn : std_logic;
  signal mvt_float_wrapped_x2_storeAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_x2_storeData : std_logic_vector(31 downto 0);
  signal mvt_float_wrapped_y1_loadEn : std_logic;
  signal mvt_float_wrapped_y1_loadAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_y1_storeEn : std_logic;
  signal mvt_float_wrapped_y1_storeAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_y1_storeData : std_logic_vector(31 downto 0);
  signal mvt_float_wrapped_y2_loadEn : std_logic;
  signal mvt_float_wrapped_y2_loadAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_y2_storeEn : std_logic;
  signal mvt_float_wrapped_y2_storeAddr : std_logic_vector(4 downto 0);
  signal mvt_float_wrapped_y2_storeData : std_logic_vector(31 downto 0);

begin

  A_end_valid <= mvt_float_wrapped_A_end_valid;
  mvt_float_wrapped_A_end_ready <= A_end_ready;
  x1_end_valid <= mvt_float_wrapped_x1_end_valid;
  mvt_float_wrapped_x1_end_ready <= x1_end_ready;
  x2_end_valid <= mvt_float_wrapped_x2_end_valid;
  mvt_float_wrapped_x2_end_ready <= x2_end_ready;
  y1_end_valid <= mvt_float_wrapped_y1_end_valid;
  mvt_float_wrapped_y1_end_ready <= y1_end_ready;
  y2_end_valid <= mvt_float_wrapped_y2_end_valid;
  mvt_float_wrapped_y2_end_ready <= y2_end_ready;
  end_valid <= mvt_float_wrapped_end_valid;
  mvt_float_wrapped_end_ready <= end_ready;
  A_ce0 <= mem_to_bram_converter_A_ce0;
  A_we0 <= mem_to_bram_converter_A_we0;
  A_address0 <= mem_to_bram_converter_A_address0;
  A_dout0 <= mem_to_bram_converter_A_dout0;
  A_ce1 <= mem_to_bram_converter_A_ce1;
  A_we1 <= mem_to_bram_converter_A_we1;
  A_address1 <= mem_to_bram_converter_A_address1;
  A_dout1 <= mem_to_bram_converter_A_dout1;
  x1_ce0 <= mem_to_bram_converter_x1_ce0;
  x1_we0 <= mem_to_bram_converter_x1_we0;
  x1_address0 <= mem_to_bram_converter_x1_address0;
  x1_dout0 <= mem_to_bram_converter_x1_dout0;
  x1_ce1 <= mem_to_bram_converter_x1_ce1;
  x1_we1 <= mem_to_bram_converter_x1_we1;
  x1_address1 <= mem_to_bram_converter_x1_address1;
  x1_dout1 <= mem_to_bram_converter_x1_dout1;
  x2_ce0 <= mem_to_bram_converter_x2_ce0;
  x2_we0 <= mem_to_bram_converter_x2_we0;
  x2_address0 <= mem_to_bram_converter_x2_address0;
  x2_dout0 <= mem_to_bram_converter_x2_dout0;
  x2_ce1 <= mem_to_bram_converter_x2_ce1;
  x2_we1 <= mem_to_bram_converter_x2_we1;
  x2_address1 <= mem_to_bram_converter_x2_address1;
  x2_dout1 <= mem_to_bram_converter_x2_dout1;
  y1_ce0 <= mem_to_bram_converter_y1_ce0;
  y1_we0 <= mem_to_bram_converter_y1_we0;
  y1_address0 <= mem_to_bram_converter_y1_address0;
  y1_dout0 <= mem_to_bram_converter_y1_dout0;
  y1_ce1 <= mem_to_bram_converter_y1_ce1;
  y1_we1 <= mem_to_bram_converter_y1_we1;
  y1_address1 <= mem_to_bram_converter_y1_address1;
  y1_dout1 <= mem_to_bram_converter_y1_dout1;
  y2_ce0 <= mem_to_bram_converter_y2_ce0;
  y2_we0 <= mem_to_bram_converter_y2_we0;
  y2_address0 <= mem_to_bram_converter_y2_address0;
  y2_dout0 <= mem_to_bram_converter_y2_dout0;
  y2_ce1 <= mem_to_bram_converter_y2_ce1;
  y2_we1 <= mem_to_bram_converter_y2_we1;
  y2_address1 <= mem_to_bram_converter_y2_address1;
  y2_dout1 <= mem_to_bram_converter_y2_dout1;

  mem_to_bram_converter_y2 : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => mvt_float_wrapped_y2_loadEn,
      loadAddr => mvt_float_wrapped_y2_loadAddr,
      storeEn => mvt_float_wrapped_y2_storeEn,
      storeAddr => mvt_float_wrapped_y2_storeAddr,
      storeData => mvt_float_wrapped_y2_storeData,
      din0 => y2_din0,
      din1 => y2_din1,
      ce0 => mem_to_bram_converter_y2_ce0,
      we0 => mem_to_bram_converter_y2_we0,
      address0 => mem_to_bram_converter_y2_address0,
      dout0 => mem_to_bram_converter_y2_dout0,
      ce1 => mem_to_bram_converter_y2_ce1,
      we1 => mem_to_bram_converter_y2_we1,
      address1 => mem_to_bram_converter_y2_address1,
      dout1 => mem_to_bram_converter_y2_dout1,
      loadData => mem_to_bram_converter_y2_loadData
    );

  mem_to_bram_converter_x2 : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => mvt_float_wrapped_x2_loadEn,
      loadAddr => mvt_float_wrapped_x2_loadAddr,
      storeEn => mvt_float_wrapped_x2_storeEn,
      storeAddr => mvt_float_wrapped_x2_storeAddr,
      storeData => mvt_float_wrapped_x2_storeData,
      din0 => x2_din0,
      din1 => x2_din1,
      ce0 => mem_to_bram_converter_x2_ce0,
      we0 => mem_to_bram_converter_x2_we0,
      address0 => mem_to_bram_converter_x2_address0,
      dout0 => mem_to_bram_converter_x2_dout0,
      ce1 => mem_to_bram_converter_x2_ce1,
      we1 => mem_to_bram_converter_x2_we1,
      address1 => mem_to_bram_converter_x2_address1,
      dout1 => mem_to_bram_converter_x2_dout1,
      loadData => mem_to_bram_converter_x2_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 10)
    port map(
      loadEn => mvt_float_wrapped_A_loadEn,
      loadAddr => mvt_float_wrapped_A_loadAddr,
      storeEn => mvt_float_wrapped_A_storeEn,
      storeAddr => mvt_float_wrapped_A_storeAddr,
      storeData => mvt_float_wrapped_A_storeData,
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

  mem_to_bram_converter_y1 : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => mvt_float_wrapped_y1_loadEn,
      loadAddr => mvt_float_wrapped_y1_loadAddr,
      storeEn => mvt_float_wrapped_y1_storeEn,
      storeAddr => mvt_float_wrapped_y1_storeAddr,
      storeData => mvt_float_wrapped_y1_storeData,
      din0 => y1_din0,
      din1 => y1_din1,
      ce0 => mem_to_bram_converter_y1_ce0,
      we0 => mem_to_bram_converter_y1_we0,
      address0 => mem_to_bram_converter_y1_address0,
      dout0 => mem_to_bram_converter_y1_dout0,
      ce1 => mem_to_bram_converter_y1_ce1,
      we1 => mem_to_bram_converter_y1_we1,
      address1 => mem_to_bram_converter_y1_address1,
      dout1 => mem_to_bram_converter_y1_dout1,
      loadData => mem_to_bram_converter_y1_loadData
    );

  mem_to_bram_converter_x1 : entity work.mem_to_bram(arch) generic map(32, 5)
    port map(
      loadEn => mvt_float_wrapped_x1_loadEn,
      loadAddr => mvt_float_wrapped_x1_loadAddr,
      storeEn => mvt_float_wrapped_x1_storeEn,
      storeAddr => mvt_float_wrapped_x1_storeAddr,
      storeData => mvt_float_wrapped_x1_storeData,
      din0 => x1_din0,
      din1 => x1_din1,
      ce0 => mem_to_bram_converter_x1_ce0,
      we0 => mem_to_bram_converter_x1_we0,
      address0 => mem_to_bram_converter_x1_address0,
      dout0 => mem_to_bram_converter_x1_dout0,
      ce1 => mem_to_bram_converter_x1_ce1,
      we1 => mem_to_bram_converter_x1_we1,
      address1 => mem_to_bram_converter_x1_address1,
      dout1 => mem_to_bram_converter_x1_dout1,
      loadData => mem_to_bram_converter_x1_loadData
    );

  mvt_float_wrapped : entity work.mvt_float(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      x1_loadData => mem_to_bram_converter_x1_loadData,
      x2_loadData => mem_to_bram_converter_x2_loadData,
      y1_loadData => mem_to_bram_converter_y1_loadData,
      y2_loadData => mem_to_bram_converter_y2_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      x1_start_valid => x1_start_valid,
      x1_start_ready => x1_start_ready,
      x2_start_valid => x2_start_valid,
      x2_start_ready => x2_start_ready,
      y1_start_valid => y1_start_valid,
      y1_start_ready => y1_start_ready,
      y2_start_valid => y2_start_valid,
      y2_start_ready => y2_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      A_end_valid => mvt_float_wrapped_A_end_valid,
      A_end_ready => mvt_float_wrapped_A_end_ready,
      x1_end_valid => mvt_float_wrapped_x1_end_valid,
      x1_end_ready => mvt_float_wrapped_x1_end_ready,
      x2_end_valid => mvt_float_wrapped_x2_end_valid,
      x2_end_ready => mvt_float_wrapped_x2_end_ready,
      y1_end_valid => mvt_float_wrapped_y1_end_valid,
      y1_end_ready => mvt_float_wrapped_y1_end_ready,
      y2_end_valid => mvt_float_wrapped_y2_end_valid,
      y2_end_ready => mvt_float_wrapped_y2_end_ready,
      end_valid => mvt_float_wrapped_end_valid,
      end_ready => mvt_float_wrapped_end_ready,
      A_loadEn => mvt_float_wrapped_A_loadEn,
      A_loadAddr => mvt_float_wrapped_A_loadAddr,
      A_storeEn => mvt_float_wrapped_A_storeEn,
      A_storeAddr => mvt_float_wrapped_A_storeAddr,
      A_storeData => mvt_float_wrapped_A_storeData,
      x1_loadEn => mvt_float_wrapped_x1_loadEn,
      x1_loadAddr => mvt_float_wrapped_x1_loadAddr,
      x1_storeEn => mvt_float_wrapped_x1_storeEn,
      x1_storeAddr => mvt_float_wrapped_x1_storeAddr,
      x1_storeData => mvt_float_wrapped_x1_storeData,
      x2_loadEn => mvt_float_wrapped_x2_loadEn,
      x2_loadAddr => mvt_float_wrapped_x2_loadAddr,
      x2_storeEn => mvt_float_wrapped_x2_storeEn,
      x2_storeAddr => mvt_float_wrapped_x2_storeAddr,
      x2_storeData => mvt_float_wrapped_x2_storeData,
      y1_loadEn => mvt_float_wrapped_y1_loadEn,
      y1_loadAddr => mvt_float_wrapped_y1_loadAddr,
      y1_storeEn => mvt_float_wrapped_y1_storeEn,
      y1_storeAddr => mvt_float_wrapped_y1_storeAddr,
      y1_storeData => mvt_float_wrapped_y1_storeData,
      y2_loadEn => mvt_float_wrapped_y2_loadEn,
      y2_loadAddr => mvt_float_wrapped_y2_loadAddr,
      y2_storeEn => mvt_float_wrapped_y2_storeEn,
      y2_storeAddr => mvt_float_wrapped_y2_storeAddr,
      y2_storeData => mvt_float_wrapped_y2_storeData
    );

end architecture;
