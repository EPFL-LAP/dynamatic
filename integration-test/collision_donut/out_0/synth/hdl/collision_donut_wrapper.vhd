library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity collision_donut_wrapper is
  port (
    x_din0 : in std_logic_vector(31 downto 0);
    x_din1 : in std_logic_vector(31 downto 0);
    y_din0 : in std_logic_vector(31 downto 0);
    y_din1 : in std_logic_vector(31 downto 0);
    x_start_valid : in std_logic;
    y_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    x_end_ready : in std_logic;
    y_end_ready : in std_logic;
    end_ready : in std_logic;
    x_start_ready : out std_logic;
    y_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    x_end_valid : out std_logic;
    y_end_valid : out std_logic;
    end_valid : out std_logic;
    x_ce0 : out std_logic;
    x_we0 : out std_logic;
    x_address0 : out std_logic_vector(9 downto 0);
    x_dout0 : out std_logic_vector(31 downto 0);
    x_ce1 : out std_logic;
    x_we1 : out std_logic;
    x_address1 : out std_logic_vector(9 downto 0);
    x_dout1 : out std_logic_vector(31 downto 0);
    y_ce0 : out std_logic;
    y_we0 : out std_logic;
    y_address0 : out std_logic_vector(9 downto 0);
    y_dout0 : out std_logic_vector(31 downto 0);
    y_ce1 : out std_logic;
    y_we1 : out std_logic;
    y_address1 : out std_logic_vector(9 downto 0);
    y_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of collision_donut_wrapper is

  signal mem_to_bram_converter_y_ce0 : std_logic;
  signal mem_to_bram_converter_y_we0 : std_logic;
  signal mem_to_bram_converter_y_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_y_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y_ce1 : std_logic;
  signal mem_to_bram_converter_y_we1 : std_logic;
  signal mem_to_bram_converter_y_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_y_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_y_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_ce0 : std_logic;
  signal mem_to_bram_converter_x_we0 : std_logic;
  signal mem_to_bram_converter_x_address0 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_x_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_ce1 : std_logic;
  signal mem_to_bram_converter_x_we1 : std_logic;
  signal mem_to_bram_converter_x_address1 : std_logic_vector(9 downto 0);
  signal mem_to_bram_converter_x_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_loadData : std_logic_vector(31 downto 0);
  signal collision_donut_wrapped_out0 : std_logic_vector(31 downto 0);
  signal collision_donut_wrapped_out0_valid : std_logic;
  signal collision_donut_wrapped_out0_ready : std_logic;
  signal collision_donut_wrapped_x_end_valid : std_logic;
  signal collision_donut_wrapped_x_end_ready : std_logic;
  signal collision_donut_wrapped_y_end_valid : std_logic;
  signal collision_donut_wrapped_y_end_ready : std_logic;
  signal collision_donut_wrapped_end_valid : std_logic;
  signal collision_donut_wrapped_end_ready : std_logic;
  signal collision_donut_wrapped_x_loadEn : std_logic;
  signal collision_donut_wrapped_x_loadAddr : std_logic_vector(9 downto 0);
  signal collision_donut_wrapped_x_storeEn : std_logic;
  signal collision_donut_wrapped_x_storeAddr : std_logic_vector(9 downto 0);
  signal collision_donut_wrapped_x_storeData : std_logic_vector(31 downto 0);
  signal collision_donut_wrapped_y_loadEn : std_logic;
  signal collision_donut_wrapped_y_loadAddr : std_logic_vector(9 downto 0);
  signal collision_donut_wrapped_y_storeEn : std_logic;
  signal collision_donut_wrapped_y_storeAddr : std_logic_vector(9 downto 0);
  signal collision_donut_wrapped_y_storeData : std_logic_vector(31 downto 0);

begin

  out0 <= collision_donut_wrapped_out0;
  out0_valid <= collision_donut_wrapped_out0_valid;
  collision_donut_wrapped_out0_ready <= out0_ready;
  x_end_valid <= collision_donut_wrapped_x_end_valid;
  collision_donut_wrapped_x_end_ready <= x_end_ready;
  y_end_valid <= collision_donut_wrapped_y_end_valid;
  collision_donut_wrapped_y_end_ready <= y_end_ready;
  end_valid <= collision_donut_wrapped_end_valid;
  collision_donut_wrapped_end_ready <= end_ready;
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

  mem_to_bram_converter_y : entity work.mem_to_bram_32_10(arch)
    port map(
      loadEn => collision_donut_wrapped_y_loadEn,
      loadAddr => collision_donut_wrapped_y_loadAddr,
      storeEn => collision_donut_wrapped_y_storeEn,
      storeAddr => collision_donut_wrapped_y_storeAddr,
      storeData => collision_donut_wrapped_y_storeData,
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

  mem_to_bram_converter_x : entity work.mem_to_bram_32_10(arch)
    port map(
      loadEn => collision_donut_wrapped_x_loadEn,
      loadAddr => collision_donut_wrapped_x_loadAddr,
      storeEn => collision_donut_wrapped_x_storeEn,
      storeAddr => collision_donut_wrapped_x_storeAddr,
      storeData => collision_donut_wrapped_x_storeData,
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

  collision_donut_wrapped : entity work.collision_donut(behavioral)
    port map(
      x_loadData => mem_to_bram_converter_x_loadData,
      y_loadData => mem_to_bram_converter_y_loadData,
      x_start_valid => x_start_valid,
      x_start_ready => x_start_ready,
      y_start_valid => y_start_valid,
      y_start_ready => y_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      out0 => collision_donut_wrapped_out0,
      out0_valid => collision_donut_wrapped_out0_valid,
      out0_ready => collision_donut_wrapped_out0_ready,
      x_end_valid => collision_donut_wrapped_x_end_valid,
      x_end_ready => collision_donut_wrapped_x_end_ready,
      y_end_valid => collision_donut_wrapped_y_end_valid,
      y_end_ready => collision_donut_wrapped_y_end_ready,
      end_valid => collision_donut_wrapped_end_valid,
      end_ready => collision_donut_wrapped_end_ready,
      x_loadEn => collision_donut_wrapped_x_loadEn,
      x_loadAddr => collision_donut_wrapped_x_loadAddr,
      x_storeEn => collision_donut_wrapped_x_storeEn,
      x_storeAddr => collision_donut_wrapped_x_storeAddr,
      x_storeData => collision_donut_wrapped_x_storeData,
      y_loadEn => collision_donut_wrapped_y_loadEn,
      y_loadAddr => collision_donut_wrapped_y_loadAddr,
      y_storeEn => collision_donut_wrapped_y_storeEn,
      y_storeAddr => collision_donut_wrapped_y_storeAddr,
      y_storeData => collision_donut_wrapped_y_storeData
    );

end architecture;
