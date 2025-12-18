library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity triangular_wrapper is
  port (
    x_din0 : in std_logic_vector(31 downto 0);
    x_din1 : in std_logic_vector(31 downto 0);
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    a_din0 : in std_logic_vector(31 downto 0);
    a_din1 : in std_logic_vector(31 downto 0);
    x_start_valid : in std_logic;
    a_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    x_end_ready : in std_logic;
    a_end_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    x_start_ready : out std_logic;
    a_start_ready : out std_logic;
    start_ready : out std_logic;
    x_end_valid : out std_logic;
    a_end_valid : out std_logic;
    end_valid : out std_logic;
    x_ce0 : out std_logic;
    x_we0 : out std_logic;
    x_address0 : out std_logic_vector(3 downto 0);
    x_dout0 : out std_logic_vector(31 downto 0);
    x_ce1 : out std_logic;
    x_we1 : out std_logic;
    x_address1 : out std_logic_vector(3 downto 0);
    x_dout1 : out std_logic_vector(31 downto 0);
    a_ce0 : out std_logic;
    a_we0 : out std_logic;
    a_address0 : out std_logic_vector(6 downto 0);
    a_dout0 : out std_logic_vector(31 downto 0);
    a_ce1 : out std_logic;
    a_we1 : out std_logic;
    a_address1 : out std_logic_vector(6 downto 0);
    a_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of triangular_wrapper is

  signal mem_to_bram_converter_x_ce0 : std_logic;
  signal mem_to_bram_converter_x_we0 : std_logic;
  signal mem_to_bram_converter_x_address0 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_x_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_ce1 : std_logic;
  signal mem_to_bram_converter_x_we1 : std_logic;
  signal mem_to_bram_converter_x_address1 : std_logic_vector(3 downto 0);
  signal mem_to_bram_converter_x_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_x_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce0 : std_logic;
  signal mem_to_bram_converter_a_we0 : std_logic;
  signal mem_to_bram_converter_a_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_a_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_ce1 : std_logic;
  signal mem_to_bram_converter_a_we1 : std_logic;
  signal mem_to_bram_converter_a_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_a_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_a_loadData : std_logic_vector(31 downto 0);
  signal triangular_wrapped_x_end_valid : std_logic;
  signal triangular_wrapped_x_end_ready : std_logic;
  signal triangular_wrapped_a_end_valid : std_logic;
  signal triangular_wrapped_a_end_ready : std_logic;
  signal triangular_wrapped_end_valid : std_logic;
  signal triangular_wrapped_end_ready : std_logic;
  signal triangular_wrapped_x_loadEn : std_logic;
  signal triangular_wrapped_x_loadAddr : std_logic_vector(3 downto 0);
  signal triangular_wrapped_x_storeEn : std_logic;
  signal triangular_wrapped_x_storeAddr : std_logic_vector(3 downto 0);
  signal triangular_wrapped_x_storeData : std_logic_vector(31 downto 0);
  signal triangular_wrapped_a_loadEn : std_logic;
  signal triangular_wrapped_a_loadAddr : std_logic_vector(6 downto 0);
  signal triangular_wrapped_a_storeEn : std_logic;
  signal triangular_wrapped_a_storeAddr : std_logic_vector(6 downto 0);
  signal triangular_wrapped_a_storeData : std_logic_vector(31 downto 0);

begin

  x_end_valid <= triangular_wrapped_x_end_valid;
  triangular_wrapped_x_end_ready <= x_end_ready;
  a_end_valid <= triangular_wrapped_a_end_valid;
  triangular_wrapped_a_end_ready <= a_end_ready;
  end_valid <= triangular_wrapped_end_valid;
  triangular_wrapped_end_ready <= end_ready;
  x_ce0 <= mem_to_bram_converter_x_ce0;
  x_we0 <= mem_to_bram_converter_x_we0;
  x_address0 <= mem_to_bram_converter_x_address0;
  x_dout0 <= mem_to_bram_converter_x_dout0;
  x_ce1 <= mem_to_bram_converter_x_ce1;
  x_we1 <= mem_to_bram_converter_x_we1;
  x_address1 <= mem_to_bram_converter_x_address1;
  x_dout1 <= mem_to_bram_converter_x_dout1;
  a_ce0 <= mem_to_bram_converter_a_ce0;
  a_we0 <= mem_to_bram_converter_a_we0;
  a_address0 <= mem_to_bram_converter_a_address0;
  a_dout0 <= mem_to_bram_converter_a_dout0;
  a_ce1 <= mem_to_bram_converter_a_ce1;
  a_we1 <= mem_to_bram_converter_a_we1;
  a_address1 <= mem_to_bram_converter_a_address1;
  a_dout1 <= mem_to_bram_converter_a_dout1;

  mem_to_bram_converter_x : entity work.mem_to_bram(arch) generic map(32, 4)
    port map(
      loadEn => triangular_wrapped_x_loadEn,
      loadAddr => triangular_wrapped_x_loadAddr,
      storeEn => triangular_wrapped_x_storeEn,
      storeAddr => triangular_wrapped_x_storeAddr,
      storeData => triangular_wrapped_x_storeData,
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

  mem_to_bram_converter_a : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => triangular_wrapped_a_loadEn,
      loadAddr => triangular_wrapped_a_loadAddr,
      storeEn => triangular_wrapped_a_storeEn,
      storeAddr => triangular_wrapped_a_storeAddr,
      storeData => triangular_wrapped_a_storeData,
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

  triangular_wrapped : entity work.triangular(behavioral)
    port map(
      x_loadData => mem_to_bram_converter_x_loadData,
      n => n,
      n_valid => n_valid,
      n_ready => n_ready,
      a_loadData => mem_to_bram_converter_a_loadData,
      x_start_valid => x_start_valid,
      x_start_ready => x_start_ready,
      a_start_valid => a_start_valid,
      a_start_ready => a_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      x_end_valid => triangular_wrapped_x_end_valid,
      x_end_ready => triangular_wrapped_x_end_ready,
      a_end_valid => triangular_wrapped_a_end_valid,
      a_end_ready => triangular_wrapped_a_end_ready,
      end_valid => triangular_wrapped_end_valid,
      end_ready => triangular_wrapped_end_ready,
      x_loadEn => triangular_wrapped_x_loadEn,
      x_loadAddr => triangular_wrapped_x_loadAddr,
      x_storeEn => triangular_wrapped_x_storeEn,
      x_storeAddr => triangular_wrapped_x_storeAddr,
      x_storeData => triangular_wrapped_x_storeData,
      a_loadEn => triangular_wrapped_a_loadEn,
      a_loadAddr => triangular_wrapped_a_loadAddr,
      a_storeEn => triangular_wrapped_a_storeEn,
      a_storeAddr => triangular_wrapped_a_storeAddr,
      a_storeData => triangular_wrapped_a_storeData
    );

end architecture;
