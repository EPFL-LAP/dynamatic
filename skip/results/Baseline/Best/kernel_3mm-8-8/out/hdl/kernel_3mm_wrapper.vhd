library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity kernel_3mm_wrapper is
  port (
    A_din0 : in std_logic_vector(31 downto 0);
    A_din1 : in std_logic_vector(31 downto 0);
    B_din0 : in std_logic_vector(31 downto 0);
    B_din1 : in std_logic_vector(31 downto 0);
    C_din0 : in std_logic_vector(31 downto 0);
    C_din1 : in std_logic_vector(31 downto 0);
    D_din0 : in std_logic_vector(31 downto 0);
    D_din1 : in std_logic_vector(31 downto 0);
    E_din0 : in std_logic_vector(31 downto 0);
    E_din1 : in std_logic_vector(31 downto 0);
    F_din0 : in std_logic_vector(31 downto 0);
    F_din1 : in std_logic_vector(31 downto 0);
    G_din0 : in std_logic_vector(31 downto 0);
    G_din1 : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    B_start_valid : in std_logic;
    C_start_valid : in std_logic;
    D_start_valid : in std_logic;
    E_start_valid : in std_logic;
    F_start_valid : in std_logic;
    G_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    B_end_ready : in std_logic;
    C_end_ready : in std_logic;
    D_end_ready : in std_logic;
    E_end_ready : in std_logic;
    F_end_ready : in std_logic;
    G_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    B_start_ready : out std_logic;
    C_start_ready : out std_logic;
    D_start_ready : out std_logic;
    E_start_ready : out std_logic;
    F_start_ready : out std_logic;
    G_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    B_end_valid : out std_logic;
    C_end_valid : out std_logic;
    D_end_valid : out std_logic;
    E_end_valid : out std_logic;
    F_end_valid : out std_logic;
    G_end_valid : out std_logic;
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
    B_dout1 : out std_logic_vector(31 downto 0);
    C_ce0 : out std_logic;
    C_we0 : out std_logic;
    C_address0 : out std_logic_vector(6 downto 0);
    C_dout0 : out std_logic_vector(31 downto 0);
    C_ce1 : out std_logic;
    C_we1 : out std_logic;
    C_address1 : out std_logic_vector(6 downto 0);
    C_dout1 : out std_logic_vector(31 downto 0);
    D_ce0 : out std_logic;
    D_we0 : out std_logic;
    D_address0 : out std_logic_vector(6 downto 0);
    D_dout0 : out std_logic_vector(31 downto 0);
    D_ce1 : out std_logic;
    D_we1 : out std_logic;
    D_address1 : out std_logic_vector(6 downto 0);
    D_dout1 : out std_logic_vector(31 downto 0);
    E_ce0 : out std_logic;
    E_we0 : out std_logic;
    E_address0 : out std_logic_vector(6 downto 0);
    E_dout0 : out std_logic_vector(31 downto 0);
    E_ce1 : out std_logic;
    E_we1 : out std_logic;
    E_address1 : out std_logic_vector(6 downto 0);
    E_dout1 : out std_logic_vector(31 downto 0);
    F_ce0 : out std_logic;
    F_we0 : out std_logic;
    F_address0 : out std_logic_vector(6 downto 0);
    F_dout0 : out std_logic_vector(31 downto 0);
    F_ce1 : out std_logic;
    F_we1 : out std_logic;
    F_address1 : out std_logic_vector(6 downto 0);
    F_dout1 : out std_logic_vector(31 downto 0);
    G_ce0 : out std_logic;
    G_we0 : out std_logic;
    G_address0 : out std_logic_vector(6 downto 0);
    G_dout0 : out std_logic_vector(31 downto 0);
    G_ce1 : out std_logic;
    G_we1 : out std_logic;
    G_address1 : out std_logic_vector(6 downto 0);
    G_dout1 : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of kernel_3mm_wrapper is

  signal mem_to_bram_converter_F_ce0 : std_logic;
  signal mem_to_bram_converter_F_we0 : std_logic;
  signal mem_to_bram_converter_F_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_F_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_F_ce1 : std_logic;
  signal mem_to_bram_converter_F_we1 : std_logic;
  signal mem_to_bram_converter_F_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_F_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_F_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_ce0 : std_logic;
  signal mem_to_bram_converter_D_we0 : std_logic;
  signal mem_to_bram_converter_D_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_D_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_ce1 : std_logic;
  signal mem_to_bram_converter_D_we1 : std_logic;
  signal mem_to_bram_converter_D_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_D_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_D_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce0 : std_logic;
  signal mem_to_bram_converter_A_we0 : std_logic;
  signal mem_to_bram_converter_A_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_ce1 : std_logic;
  signal mem_to_bram_converter_A_we1 : std_logic;
  signal mem_to_bram_converter_A_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_A_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_A_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_G_ce0 : std_logic;
  signal mem_to_bram_converter_G_we0 : std_logic;
  signal mem_to_bram_converter_G_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_G_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_G_ce1 : std_logic;
  signal mem_to_bram_converter_G_we1 : std_logic;
  signal mem_to_bram_converter_G_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_G_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_G_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_C_ce0 : std_logic;
  signal mem_to_bram_converter_C_we0 : std_logic;
  signal mem_to_bram_converter_C_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_C_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_C_ce1 : std_logic;
  signal mem_to_bram_converter_C_we1 : std_logic;
  signal mem_to_bram_converter_C_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_C_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_C_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_ce0 : std_logic;
  signal mem_to_bram_converter_B_we0 : std_logic;
  signal mem_to_bram_converter_B_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_ce1 : std_logic;
  signal mem_to_bram_converter_B_we1 : std_logic;
  signal mem_to_bram_converter_B_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_B_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_B_loadData : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_E_ce0 : std_logic;
  signal mem_to_bram_converter_E_we0 : std_logic;
  signal mem_to_bram_converter_E_address0 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_E_dout0 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_E_ce1 : std_logic;
  signal mem_to_bram_converter_E_we1 : std_logic;
  signal mem_to_bram_converter_E_address1 : std_logic_vector(6 downto 0);
  signal mem_to_bram_converter_E_dout1 : std_logic_vector(31 downto 0);
  signal mem_to_bram_converter_E_loadData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_A_end_valid : std_logic;
  signal kernel_3mm_wrapped_A_end_ready : std_logic;
  signal kernel_3mm_wrapped_B_end_valid : std_logic;
  signal kernel_3mm_wrapped_B_end_ready : std_logic;
  signal kernel_3mm_wrapped_C_end_valid : std_logic;
  signal kernel_3mm_wrapped_C_end_ready : std_logic;
  signal kernel_3mm_wrapped_D_end_valid : std_logic;
  signal kernel_3mm_wrapped_D_end_ready : std_logic;
  signal kernel_3mm_wrapped_E_end_valid : std_logic;
  signal kernel_3mm_wrapped_E_end_ready : std_logic;
  signal kernel_3mm_wrapped_F_end_valid : std_logic;
  signal kernel_3mm_wrapped_F_end_ready : std_logic;
  signal kernel_3mm_wrapped_G_end_valid : std_logic;
  signal kernel_3mm_wrapped_G_end_ready : std_logic;
  signal kernel_3mm_wrapped_end_valid : std_logic;
  signal kernel_3mm_wrapped_end_ready : std_logic;
  signal kernel_3mm_wrapped_A_loadEn : std_logic;
  signal kernel_3mm_wrapped_A_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_A_storeEn : std_logic;
  signal kernel_3mm_wrapped_A_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_A_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_B_loadEn : std_logic;
  signal kernel_3mm_wrapped_B_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_B_storeEn : std_logic;
  signal kernel_3mm_wrapped_B_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_B_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_C_loadEn : std_logic;
  signal kernel_3mm_wrapped_C_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_C_storeEn : std_logic;
  signal kernel_3mm_wrapped_C_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_C_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_D_loadEn : std_logic;
  signal kernel_3mm_wrapped_D_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_D_storeEn : std_logic;
  signal kernel_3mm_wrapped_D_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_D_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_E_loadEn : std_logic;
  signal kernel_3mm_wrapped_E_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_E_storeEn : std_logic;
  signal kernel_3mm_wrapped_E_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_E_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_F_loadEn : std_logic;
  signal kernel_3mm_wrapped_F_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_F_storeEn : std_logic;
  signal kernel_3mm_wrapped_F_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_F_storeData : std_logic_vector(31 downto 0);
  signal kernel_3mm_wrapped_G_loadEn : std_logic;
  signal kernel_3mm_wrapped_G_loadAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_G_storeEn : std_logic;
  signal kernel_3mm_wrapped_G_storeAddr : std_logic_vector(6 downto 0);
  signal kernel_3mm_wrapped_G_storeData : std_logic_vector(31 downto 0);

begin

  A_end_valid <= kernel_3mm_wrapped_A_end_valid;
  kernel_3mm_wrapped_A_end_ready <= A_end_ready;
  B_end_valid <= kernel_3mm_wrapped_B_end_valid;
  kernel_3mm_wrapped_B_end_ready <= B_end_ready;
  C_end_valid <= kernel_3mm_wrapped_C_end_valid;
  kernel_3mm_wrapped_C_end_ready <= C_end_ready;
  D_end_valid <= kernel_3mm_wrapped_D_end_valid;
  kernel_3mm_wrapped_D_end_ready <= D_end_ready;
  E_end_valid <= kernel_3mm_wrapped_E_end_valid;
  kernel_3mm_wrapped_E_end_ready <= E_end_ready;
  F_end_valid <= kernel_3mm_wrapped_F_end_valid;
  kernel_3mm_wrapped_F_end_ready <= F_end_ready;
  G_end_valid <= kernel_3mm_wrapped_G_end_valid;
  kernel_3mm_wrapped_G_end_ready <= G_end_ready;
  end_valid <= kernel_3mm_wrapped_end_valid;
  kernel_3mm_wrapped_end_ready <= end_ready;
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
  C_ce0 <= mem_to_bram_converter_C_ce0;
  C_we0 <= mem_to_bram_converter_C_we0;
  C_address0 <= mem_to_bram_converter_C_address0;
  C_dout0 <= mem_to_bram_converter_C_dout0;
  C_ce1 <= mem_to_bram_converter_C_ce1;
  C_we1 <= mem_to_bram_converter_C_we1;
  C_address1 <= mem_to_bram_converter_C_address1;
  C_dout1 <= mem_to_bram_converter_C_dout1;
  D_ce0 <= mem_to_bram_converter_D_ce0;
  D_we0 <= mem_to_bram_converter_D_we0;
  D_address0 <= mem_to_bram_converter_D_address0;
  D_dout0 <= mem_to_bram_converter_D_dout0;
  D_ce1 <= mem_to_bram_converter_D_ce1;
  D_we1 <= mem_to_bram_converter_D_we1;
  D_address1 <= mem_to_bram_converter_D_address1;
  D_dout1 <= mem_to_bram_converter_D_dout1;
  E_ce0 <= mem_to_bram_converter_E_ce0;
  E_we0 <= mem_to_bram_converter_E_we0;
  E_address0 <= mem_to_bram_converter_E_address0;
  E_dout0 <= mem_to_bram_converter_E_dout0;
  E_ce1 <= mem_to_bram_converter_E_ce1;
  E_we1 <= mem_to_bram_converter_E_we1;
  E_address1 <= mem_to_bram_converter_E_address1;
  E_dout1 <= mem_to_bram_converter_E_dout1;
  F_ce0 <= mem_to_bram_converter_F_ce0;
  F_we0 <= mem_to_bram_converter_F_we0;
  F_address0 <= mem_to_bram_converter_F_address0;
  F_dout0 <= mem_to_bram_converter_F_dout0;
  F_ce1 <= mem_to_bram_converter_F_ce1;
  F_we1 <= mem_to_bram_converter_F_we1;
  F_address1 <= mem_to_bram_converter_F_address1;
  F_dout1 <= mem_to_bram_converter_F_dout1;
  G_ce0 <= mem_to_bram_converter_G_ce0;
  G_we0 <= mem_to_bram_converter_G_we0;
  G_address0 <= mem_to_bram_converter_G_address0;
  G_dout0 <= mem_to_bram_converter_G_dout0;
  G_ce1 <= mem_to_bram_converter_G_ce1;
  G_we1 <= mem_to_bram_converter_G_we1;
  G_address1 <= mem_to_bram_converter_G_address1;
  G_dout1 <= mem_to_bram_converter_G_dout1;

  mem_to_bram_converter_F : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_F_loadEn,
      loadAddr => kernel_3mm_wrapped_F_loadAddr,
      storeEn => kernel_3mm_wrapped_F_storeEn,
      storeAddr => kernel_3mm_wrapped_F_storeAddr,
      storeData => kernel_3mm_wrapped_F_storeData,
      din0 => F_din0,
      din1 => F_din1,
      ce0 => mem_to_bram_converter_F_ce0,
      we0 => mem_to_bram_converter_F_we0,
      address0 => mem_to_bram_converter_F_address0,
      dout0 => mem_to_bram_converter_F_dout0,
      ce1 => mem_to_bram_converter_F_ce1,
      we1 => mem_to_bram_converter_F_we1,
      address1 => mem_to_bram_converter_F_address1,
      dout1 => mem_to_bram_converter_F_dout1,
      loadData => mem_to_bram_converter_F_loadData
    );

  mem_to_bram_converter_D : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_D_loadEn,
      loadAddr => kernel_3mm_wrapped_D_loadAddr,
      storeEn => kernel_3mm_wrapped_D_storeEn,
      storeAddr => kernel_3mm_wrapped_D_storeAddr,
      storeData => kernel_3mm_wrapped_D_storeData,
      din0 => D_din0,
      din1 => D_din1,
      ce0 => mem_to_bram_converter_D_ce0,
      we0 => mem_to_bram_converter_D_we0,
      address0 => mem_to_bram_converter_D_address0,
      dout0 => mem_to_bram_converter_D_dout0,
      ce1 => mem_to_bram_converter_D_ce1,
      we1 => mem_to_bram_converter_D_we1,
      address1 => mem_to_bram_converter_D_address1,
      dout1 => mem_to_bram_converter_D_dout1,
      loadData => mem_to_bram_converter_D_loadData
    );

  mem_to_bram_converter_A : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_A_loadEn,
      loadAddr => kernel_3mm_wrapped_A_loadAddr,
      storeEn => kernel_3mm_wrapped_A_storeEn,
      storeAddr => kernel_3mm_wrapped_A_storeAddr,
      storeData => kernel_3mm_wrapped_A_storeData,
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

  mem_to_bram_converter_G : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_G_loadEn,
      loadAddr => kernel_3mm_wrapped_G_loadAddr,
      storeEn => kernel_3mm_wrapped_G_storeEn,
      storeAddr => kernel_3mm_wrapped_G_storeAddr,
      storeData => kernel_3mm_wrapped_G_storeData,
      din0 => G_din0,
      din1 => G_din1,
      ce0 => mem_to_bram_converter_G_ce0,
      we0 => mem_to_bram_converter_G_we0,
      address0 => mem_to_bram_converter_G_address0,
      dout0 => mem_to_bram_converter_G_dout0,
      ce1 => mem_to_bram_converter_G_ce1,
      we1 => mem_to_bram_converter_G_we1,
      address1 => mem_to_bram_converter_G_address1,
      dout1 => mem_to_bram_converter_G_dout1,
      loadData => mem_to_bram_converter_G_loadData
    );

  mem_to_bram_converter_C : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_C_loadEn,
      loadAddr => kernel_3mm_wrapped_C_loadAddr,
      storeEn => kernel_3mm_wrapped_C_storeEn,
      storeAddr => kernel_3mm_wrapped_C_storeAddr,
      storeData => kernel_3mm_wrapped_C_storeData,
      din0 => C_din0,
      din1 => C_din1,
      ce0 => mem_to_bram_converter_C_ce0,
      we0 => mem_to_bram_converter_C_we0,
      address0 => mem_to_bram_converter_C_address0,
      dout0 => mem_to_bram_converter_C_dout0,
      ce1 => mem_to_bram_converter_C_ce1,
      we1 => mem_to_bram_converter_C_we1,
      address1 => mem_to_bram_converter_C_address1,
      dout1 => mem_to_bram_converter_C_dout1,
      loadData => mem_to_bram_converter_C_loadData
    );

  mem_to_bram_converter_B : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_B_loadEn,
      loadAddr => kernel_3mm_wrapped_B_loadAddr,
      storeEn => kernel_3mm_wrapped_B_storeEn,
      storeAddr => kernel_3mm_wrapped_B_storeAddr,
      storeData => kernel_3mm_wrapped_B_storeData,
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

  mem_to_bram_converter_E : entity work.mem_to_bram(arch) generic map(32, 7)
    port map(
      loadEn => kernel_3mm_wrapped_E_loadEn,
      loadAddr => kernel_3mm_wrapped_E_loadAddr,
      storeEn => kernel_3mm_wrapped_E_storeEn,
      storeAddr => kernel_3mm_wrapped_E_storeAddr,
      storeData => kernel_3mm_wrapped_E_storeData,
      din0 => E_din0,
      din1 => E_din1,
      ce0 => mem_to_bram_converter_E_ce0,
      we0 => mem_to_bram_converter_E_we0,
      address0 => mem_to_bram_converter_E_address0,
      dout0 => mem_to_bram_converter_E_dout0,
      ce1 => mem_to_bram_converter_E_ce1,
      we1 => mem_to_bram_converter_E_we1,
      address1 => mem_to_bram_converter_E_address1,
      dout1 => mem_to_bram_converter_E_dout1,
      loadData => mem_to_bram_converter_E_loadData
    );

  kernel_3mm_wrapped : entity work.kernel_3mm(behavioral)
    port map(
      A_loadData => mem_to_bram_converter_A_loadData,
      B_loadData => mem_to_bram_converter_B_loadData,
      C_loadData => mem_to_bram_converter_C_loadData,
      D_loadData => mem_to_bram_converter_D_loadData,
      E_loadData => mem_to_bram_converter_E_loadData,
      F_loadData => mem_to_bram_converter_F_loadData,
      G_loadData => mem_to_bram_converter_G_loadData,
      A_start_valid => A_start_valid,
      A_start_ready => A_start_ready,
      B_start_valid => B_start_valid,
      B_start_ready => B_start_ready,
      C_start_valid => C_start_valid,
      C_start_ready => C_start_ready,
      D_start_valid => D_start_valid,
      D_start_ready => D_start_ready,
      E_start_valid => E_start_valid,
      E_start_ready => E_start_ready,
      F_start_valid => F_start_valid,
      F_start_ready => F_start_ready,
      G_start_valid => G_start_valid,
      G_start_ready => G_start_ready,
      start_valid => start_valid,
      start_ready => start_ready,
      clk => clk,
      rst => rst,
      A_end_valid => kernel_3mm_wrapped_A_end_valid,
      A_end_ready => kernel_3mm_wrapped_A_end_ready,
      B_end_valid => kernel_3mm_wrapped_B_end_valid,
      B_end_ready => kernel_3mm_wrapped_B_end_ready,
      C_end_valid => kernel_3mm_wrapped_C_end_valid,
      C_end_ready => kernel_3mm_wrapped_C_end_ready,
      D_end_valid => kernel_3mm_wrapped_D_end_valid,
      D_end_ready => kernel_3mm_wrapped_D_end_ready,
      E_end_valid => kernel_3mm_wrapped_E_end_valid,
      E_end_ready => kernel_3mm_wrapped_E_end_ready,
      F_end_valid => kernel_3mm_wrapped_F_end_valid,
      F_end_ready => kernel_3mm_wrapped_F_end_ready,
      G_end_valid => kernel_3mm_wrapped_G_end_valid,
      G_end_ready => kernel_3mm_wrapped_G_end_ready,
      end_valid => kernel_3mm_wrapped_end_valid,
      end_ready => kernel_3mm_wrapped_end_ready,
      A_loadEn => kernel_3mm_wrapped_A_loadEn,
      A_loadAddr => kernel_3mm_wrapped_A_loadAddr,
      A_storeEn => kernel_3mm_wrapped_A_storeEn,
      A_storeAddr => kernel_3mm_wrapped_A_storeAddr,
      A_storeData => kernel_3mm_wrapped_A_storeData,
      B_loadEn => kernel_3mm_wrapped_B_loadEn,
      B_loadAddr => kernel_3mm_wrapped_B_loadAddr,
      B_storeEn => kernel_3mm_wrapped_B_storeEn,
      B_storeAddr => kernel_3mm_wrapped_B_storeAddr,
      B_storeData => kernel_3mm_wrapped_B_storeData,
      C_loadEn => kernel_3mm_wrapped_C_loadEn,
      C_loadAddr => kernel_3mm_wrapped_C_loadAddr,
      C_storeEn => kernel_3mm_wrapped_C_storeEn,
      C_storeAddr => kernel_3mm_wrapped_C_storeAddr,
      C_storeData => kernel_3mm_wrapped_C_storeData,
      D_loadEn => kernel_3mm_wrapped_D_loadEn,
      D_loadAddr => kernel_3mm_wrapped_D_loadAddr,
      D_storeEn => kernel_3mm_wrapped_D_storeEn,
      D_storeAddr => kernel_3mm_wrapped_D_storeAddr,
      D_storeData => kernel_3mm_wrapped_D_storeData,
      E_loadEn => kernel_3mm_wrapped_E_loadEn,
      E_loadAddr => kernel_3mm_wrapped_E_loadAddr,
      E_storeEn => kernel_3mm_wrapped_E_storeEn,
      E_storeAddr => kernel_3mm_wrapped_E_storeAddr,
      E_storeData => kernel_3mm_wrapped_E_storeData,
      F_loadEn => kernel_3mm_wrapped_F_loadEn,
      F_loadAddr => kernel_3mm_wrapped_F_loadAddr,
      F_storeEn => kernel_3mm_wrapped_F_storeEn,
      F_storeAddr => kernel_3mm_wrapped_F_storeAddr,
      F_storeData => kernel_3mm_wrapped_F_storeData,
      G_loadEn => kernel_3mm_wrapped_G_loadEn,
      G_loadAddr => kernel_3mm_wrapped_G_loadAddr,
      G_storeEn => kernel_3mm_wrapped_G_storeEn,
      G_storeAddr => kernel_3mm_wrapped_G_storeAddr,
      G_storeData => kernel_3mm_wrapped_G_storeData
    );

end architecture;
