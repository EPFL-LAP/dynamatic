library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity kernel_2mm is
  port (
    alpha : in std_logic_vector(31 downto 0);
    alpha_valid : in std_logic;
    beta : in std_logic_vector(31 downto 0);
    beta_valid : in std_logic;
    tmp_loadData : in std_logic_vector(31 downto 0);
    A_loadData : in std_logic_vector(31 downto 0);
    B_loadData : in std_logic_vector(31 downto 0);
    C_loadData : in std_logic_vector(31 downto 0);
    D_loadData : in std_logic_vector(31 downto 0);
    tmp_start_valid : in std_logic;
    A_start_valid : in std_logic;
    B_start_valid : in std_logic;
    C_start_valid : in std_logic;
    D_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    tmp_end_ready : in std_logic;
    A_end_ready : in std_logic;
    B_end_ready : in std_logic;
    C_end_ready : in std_logic;
    D_end_ready : in std_logic;
    end_ready : in std_logic;
    alpha_ready : out std_logic;
    beta_ready : out std_logic;
    tmp_start_ready : out std_logic;
    A_start_ready : out std_logic;
    B_start_ready : out std_logic;
    C_start_ready : out std_logic;
    D_start_ready : out std_logic;
    start_ready : out std_logic;
    tmp_end_valid : out std_logic;
    A_end_valid : out std_logic;
    B_end_valid : out std_logic;
    C_end_valid : out std_logic;
    D_end_valid : out std_logic;
    end_valid : out std_logic;
    tmp_loadEn : out std_logic;
    tmp_loadAddr : out std_logic_vector(6 downto 0);
    tmp_storeEn : out std_logic;
    tmp_storeAddr : out std_logic_vector(6 downto 0);
    tmp_storeData : out std_logic_vector(31 downto 0);
    A_loadEn : out std_logic;
    A_loadAddr : out std_logic_vector(6 downto 0);
    A_storeEn : out std_logic;
    A_storeAddr : out std_logic_vector(6 downto 0);
    A_storeData : out std_logic_vector(31 downto 0);
    B_loadEn : out std_logic;
    B_loadAddr : out std_logic_vector(6 downto 0);
    B_storeEn : out std_logic;
    B_storeAddr : out std_logic_vector(6 downto 0);
    B_storeData : out std_logic_vector(31 downto 0);
    C_loadEn : out std_logic;
    C_loadAddr : out std_logic_vector(6 downto 0);
    C_storeEn : out std_logic;
    C_storeAddr : out std_logic_vector(6 downto 0);
    C_storeData : out std_logic_vector(31 downto 0);
    D_loadEn : out std_logic;
    D_loadAddr : out std_logic_vector(6 downto 0);
    D_storeEn : out std_logic;
    D_storeAddr : out std_logic_vector(6 downto 0);
    D_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of kernel_2mm is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal lsq2_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq2_ldData_0_valid : std_logic;
  signal lsq2_ldData_0_ready : std_logic;
  signal lsq2_ldData_1 : std_logic_vector(31 downto 0);
  signal lsq2_ldData_1_valid : std_logic;
  signal lsq2_ldData_1_ready : std_logic;
  signal lsq2_memEnd_valid : std_logic;
  signal lsq2_memEnd_ready : std_logic;
  signal lsq2_loadEn : std_logic;
  signal lsq2_loadAddr : std_logic_vector(6 downto 0);
  signal lsq2_storeEn : std_logic;
  signal lsq2_storeAddr : std_logic_vector(6 downto 0);
  signal lsq2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0_valid : std_logic;
  signal mem_controller4_ldData_0_ready : std_logic;
  signal mem_controller4_memEnd_valid : std_logic;
  signal mem_controller4_memEnd_ready : std_logic;
  signal mem_controller4_loadEn : std_logic;
  signal mem_controller4_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller4_storeEn : std_logic;
  signal mem_controller4_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller4_storeData : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0_valid : std_logic;
  signal mem_controller5_ldData_0_ready : std_logic;
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0_valid : std_logic;
  signal lsq3_ldData_0_ready : std_logic;
  signal lsq3_ldData_1 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_1_valid : std_logic;
  signal lsq3_ldData_1_ready : std_logic;
  signal lsq3_memEnd_valid : std_logic;
  signal lsq3_memEnd_ready : std_logic;
  signal lsq3_loadEn : std_logic;
  signal lsq3_loadAddr : std_logic_vector(6 downto 0);
  signal lsq3_storeEn : std_logic;
  signal lsq3_storeAddr : std_logic_vector(6 downto 0);
  signal lsq3_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi30_outs : std_logic_vector(4 downto 0);
  signal extsi30_outs_valid : std_logic;
  signal extsi30_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(4 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer51_outs : std_logic_vector(31 downto 0);
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(0 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(0 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(0 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi29_outs : std_logic_vector(4 downto 0);
  signal extsi29_outs_valid : std_logic;
  signal extsi29_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(4 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(4 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(4 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(4 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(4 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi31_outs : std_logic_vector(6 downto 0);
  signal extsi31_outs_valid : std_logic;
  signal extsi31_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(4 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(4 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(4 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(4 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(4 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal extsi32_outs : std_logic_vector(31 downto 0);
  signal extsi32_outs_valid : std_logic;
  signal extsi32_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(0 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork6_outs_3 : std_logic_vector(0 downto 0);
  signal fork6_outs_3_valid : std_logic;
  signal fork6_outs_3_ready : std_logic;
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal lazy_fork0_outs_2_valid : std_logic;
  signal lazy_fork0_outs_2_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant29_outs : std_logic_vector(2 downto 0);
  signal constant29_outs_valid : std_logic;
  signal constant29_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal addi19_result : std_logic_vector(6 downto 0);
  signal addi19_result_valid : std_logic;
  signal addi19_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(6 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(6 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal extsi28_outs : std_logic_vector(4 downto 0);
  signal extsi28_outs_valid : std_logic;
  signal extsi28_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(4 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(4 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(4 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(4 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(4 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(4 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal extsi33_outs : std_logic_vector(6 downto 0);
  signal extsi33_outs_valid : std_logic;
  signal extsi33_outs_ready : std_logic;
  signal extsi34_outs : std_logic_vector(5 downto 0);
  signal extsi34_outs_valid : std_logic;
  signal extsi34_outs_ready : std_logic;
  signal extsi35_outs : std_logic_vector(31 downto 0);
  signal extsi35_outs_valid : std_logic;
  signal extsi35_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal mux8_outs : std_logic_vector(31 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(31 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal mux10_outs : std_logic_vector(4 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(4 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(4 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(4 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(4 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal extsi36_outs : std_logic_vector(31 downto 0);
  signal extsi36_outs_valid : std_logic;
  signal extsi36_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(31 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(31 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal fork12_outs_2 : std_logic_vector(31 downto 0);
  signal fork12_outs_2_valid : std_logic;
  signal fork12_outs_2_ready : std_logic;
  signal fork12_outs_3 : std_logic_vector(31 downto 0);
  signal fork12_outs_3_valid : std_logic;
  signal fork12_outs_3_ready : std_logic;
  signal fork12_outs_4 : std_logic_vector(31 downto 0);
  signal fork12_outs_4_valid : std_logic;
  signal fork12_outs_4_ready : std_logic;
  signal fork12_outs_5 : std_logic_vector(31 downto 0);
  signal fork12_outs_5_valid : std_logic;
  signal fork12_outs_5_ready : std_logic;
  signal mux11_outs : std_logic_vector(4 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal buffer25_outs : std_logic_vector(4 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(4 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(4 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(4 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork13_outs_2 : std_logic_vector(4 downto 0);
  signal fork13_outs_2_valid : std_logic;
  signal fork13_outs_2_ready : std_logic;
  signal fork13_outs_3 : std_logic_vector(4 downto 0);
  signal fork13_outs_3_valid : std_logic;
  signal fork13_outs_3_ready : std_logic;
  signal extsi37_outs : std_logic_vector(6 downto 0);
  signal extsi37_outs_valid : std_logic;
  signal extsi37_outs_ready : std_logic;
  signal extsi38_outs : std_logic_vector(6 downto 0);
  signal extsi38_outs_valid : std_logic;
  signal extsi38_outs_ready : std_logic;
  signal extsi39_outs : std_logic_vector(6 downto 0);
  signal extsi39_outs_valid : std_logic;
  signal extsi39_outs_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(0 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(0 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(0 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal fork14_outs_3 : std_logic_vector(0 downto 0);
  signal fork14_outs_3_valid : std_logic;
  signal fork14_outs_3_ready : std_logic;
  signal fork14_outs_4 : std_logic_vector(0 downto 0);
  signal fork14_outs_4_valid : std_logic;
  signal fork14_outs_4_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant30_outs : std_logic_vector(4 downto 0);
  signal constant30_outs_valid : std_logic;
  signal constant30_outs_ready : std_logic;
  signal extsi40_outs : std_logic_vector(5 downto 0);
  signal extsi40_outs_valid : std_logic;
  signal extsi40_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant31_outs : std_logic_vector(1 downto 0);
  signal constant31_outs_valid : std_logic;
  signal constant31_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(1 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(1 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal extsi41_outs : std_logic_vector(5 downto 0);
  signal extsi41_outs_valid : std_logic;
  signal extsi41_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(31 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(31 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal fork16_outs_2 : std_logic_vector(31 downto 0);
  signal fork16_outs_2_valid : std_logic;
  signal fork16_outs_2_ready : std_logic;
  signal fork16_outs_3 : std_logic_vector(31 downto 0);
  signal fork16_outs_3_valid : std_logic;
  signal fork16_outs_3_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant32_outs : std_logic_vector(2 downto 0);
  signal constant32_outs_valid : std_logic;
  signal constant32_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(31 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(31 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_2 : std_logic_vector(31 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal fork17_outs_3 : std_logic_vector(31 downto 0);
  signal fork17_outs_3_valid : std_logic;
  signal fork17_outs_3_ready : std_logic;
  signal shli2_result : std_logic_vector(31 downto 0);
  signal shli2_result_valid : std_logic;
  signal shli2_result_ready : std_logic;
  signal buffer28_outs : std_logic_vector(31 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(6 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer29_outs : std_logic_vector(31 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(6 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal addi20_result : std_logic_vector(6 downto 0);
  signal addi20_result_valid : std_logic;
  signal addi20_result_ready : std_logic;
  signal buffer30_outs : std_logic_vector(6 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(6 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal shli4_result : std_logic_vector(31 downto 0);
  signal shli4_result_valid : std_logic;
  signal shli4_result_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(6 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal shli5_result : std_logic_vector(31 downto 0);
  signal shli5_result_valid : std_logic;
  signal shli5_result_ready : std_logic;
  signal buffer32_outs : std_logic_vector(31 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal trunci5_outs : std_logic_vector(6 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal addi21_result : std_logic_vector(6 downto 0);
  signal addi21_result_valid : std_logic;
  signal addi21_result_ready : std_logic;
  signal buffer33_outs : std_logic_vector(6 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal addi4_result : std_logic_vector(6 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal load1_addrOut : std_logic_vector(6 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal muli1_result : std_logic_vector(31 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal shli6_result : std_logic_vector(31 downto 0);
  signal shli6_result_valid : std_logic;
  signal shli6_result_ready : std_logic;
  signal buffer34_outs : std_logic_vector(31 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(6 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal shli7_result : std_logic_vector(31 downto 0);
  signal shli7_result_valid : std_logic;
  signal shli7_result_ready : std_logic;
  signal buffer35_outs : std_logic_vector(31 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(6 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal addi22_result : std_logic_vector(6 downto 0);
  signal addi22_result_valid : std_logic;
  signal addi22_result_ready : std_logic;
  signal buffer36_outs : std_logic_vector(6 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal addi5_result : std_logic_vector(6 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer37_outs : std_logic_vector(6 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(6 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal shli8_result : std_logic_vector(31 downto 0);
  signal shli8_result_valid : std_logic;
  signal shli8_result_ready : std_logic;
  signal buffer39_outs : std_logic_vector(31 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal trunci8_outs : std_logic_vector(6 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal shli9_result : std_logic_vector(31 downto 0);
  signal shli9_result_valid : std_logic;
  signal shli9_result_ready : std_logic;
  signal buffer40_outs : std_logic_vector(31 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal trunci9_outs : std_logic_vector(6 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal addi23_result : std_logic_vector(6 downto 0);
  signal addi23_result_valid : std_logic;
  signal addi23_result_ready : std_logic;
  signal buffer41_outs : std_logic_vector(6 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal addi6_result : std_logic_vector(6 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer38_outs : std_logic_vector(31 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal buffer42_outs : std_logic_vector(6 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(6 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal addi13_result : std_logic_vector(5 downto 0);
  signal addi13_result_valid : std_logic;
  signal addi13_result_ready : std_logic;
  signal buffer43_outs : std_logic_vector(5 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(5 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(5 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal trunci10_outs : std_logic_vector(4 downto 0);
  signal trunci10_outs_valid : std_logic;
  signal trunci10_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer44_outs : std_logic_vector(0 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(0 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(0 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(0 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal fork19_outs_3 : std_logic_vector(0 downto 0);
  signal fork19_outs_3_valid : std_logic;
  signal fork19_outs_3_ready : std_logic;
  signal fork19_outs_4 : std_logic_vector(0 downto 0);
  signal fork19_outs_4_valid : std_logic;
  signal fork19_outs_4_ready : std_logic;
  signal fork19_outs_5 : std_logic_vector(0 downto 0);
  signal fork19_outs_5_valid : std_logic;
  signal fork19_outs_5_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(4 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(4 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(31 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(31 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(31 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(31 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(4 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(4 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(4 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(4 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal extsi42_outs : std_logic_vector(5 downto 0);
  signal extsi42_outs_valid : std_logic;
  signal extsi42_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant33_outs : std_logic_vector(4 downto 0);
  signal constant33_outs_valid : std_logic;
  signal constant33_outs_ready : std_logic;
  signal extsi43_outs : std_logic_vector(5 downto 0);
  signal extsi43_outs_valid : std_logic;
  signal extsi43_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant34_outs : std_logic_vector(1 downto 0);
  signal constant34_outs_valid : std_logic;
  signal constant34_outs_ready : std_logic;
  signal extsi44_outs : std_logic_vector(5 downto 0);
  signal extsi44_outs_valid : std_logic;
  signal extsi44_outs_ready : std_logic;
  signal addi14_result : std_logic_vector(5 downto 0);
  signal addi14_result_valid : std_logic;
  signal addi14_result_ready : std_logic;
  signal buffer45_outs : std_logic_vector(5 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(5 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(5 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal trunci11_outs : std_logic_vector(4 downto 0);
  signal trunci11_outs_valid : std_logic;
  signal trunci11_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer46_outs : std_logic_vector(0 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(0 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(0 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork21_outs_2 : std_logic_vector(0 downto 0);
  signal fork21_outs_2_valid : std_logic;
  signal fork21_outs_2_ready : std_logic;
  signal fork21_outs_3 : std_logic_vector(0 downto 0);
  signal fork21_outs_3_valid : std_logic;
  signal fork21_outs_3_ready : std_logic;
  signal fork21_outs_4 : std_logic_vector(0 downto 0);
  signal fork21_outs_4_valid : std_logic;
  signal fork21_outs_4_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(4 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(4 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal cond_br13_trueOut : std_logic_vector(31 downto 0);
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut : std_logic_vector(31 downto 0);
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal cond_br14_trueOut : std_logic_vector(31 downto 0);
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut : std_logic_vector(31 downto 0);
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal cond_br15_trueOut : std_logic_vector(4 downto 0);
  signal cond_br15_trueOut_valid : std_logic;
  signal cond_br15_trueOut_ready : std_logic;
  signal cond_br15_falseOut : std_logic_vector(4 downto 0);
  signal cond_br15_falseOut_valid : std_logic;
  signal cond_br15_falseOut_ready : std_logic;
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal extsi45_outs : std_logic_vector(5 downto 0);
  signal extsi45_outs_valid : std_logic;
  signal extsi45_outs_ready : std_logic;
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal constant35_outs : std_logic_vector(0 downto 0);
  signal constant35_outs_valid : std_logic;
  signal constant35_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant36_outs : std_logic_vector(4 downto 0);
  signal constant36_outs_valid : std_logic;
  signal constant36_outs_ready : std_logic;
  signal extsi46_outs : std_logic_vector(5 downto 0);
  signal extsi46_outs_valid : std_logic;
  signal extsi46_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant37_outs : std_logic_vector(1 downto 0);
  signal constant37_outs_valid : std_logic;
  signal constant37_outs_ready : std_logic;
  signal extsi47_outs : std_logic_vector(5 downto 0);
  signal extsi47_outs_valid : std_logic;
  signal extsi47_outs_ready : std_logic;
  signal buffer48_outs : std_logic_vector(5 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal addi15_result : std_logic_vector(5 downto 0);
  signal addi15_result_valid : std_logic;
  signal addi15_result_ready : std_logic;
  signal buffer49_outs : std_logic_vector(5 downto 0);
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(5 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(5 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal trunci12_outs : std_logic_vector(4 downto 0);
  signal trunci12_outs_valid : std_logic;
  signal trunci12_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer50_outs : std_logic_vector(0 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(0 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(0 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal fork24_outs_2 : std_logic_vector(0 downto 0);
  signal fork24_outs_2_valid : std_logic;
  signal fork24_outs_2_ready : std_logic;
  signal fork24_outs_3 : std_logic_vector(0 downto 0);
  signal fork24_outs_3_valid : std_logic;
  signal fork24_outs_3_ready : std_logic;
  signal fork24_outs_4 : std_logic_vector(0 downto 0);
  signal fork24_outs_4_valid : std_logic;
  signal fork24_outs_4_ready : std_logic;
  signal cond_br17_trueOut : std_logic_vector(4 downto 0);
  signal cond_br17_trueOut_valid : std_logic;
  signal cond_br17_trueOut_ready : std_logic;
  signal cond_br17_falseOut : std_logic_vector(4 downto 0);
  signal cond_br17_falseOut_valid : std_logic;
  signal cond_br17_falseOut_ready : std_logic;
  signal cond_br18_trueOut : std_logic_vector(31 downto 0);
  signal cond_br18_trueOut_valid : std_logic;
  signal cond_br18_trueOut_ready : std_logic;
  signal cond_br18_falseOut : std_logic_vector(31 downto 0);
  signal cond_br18_falseOut_valid : std_logic;
  signal cond_br18_falseOut_ready : std_logic;
  signal buffer47_outs : std_logic_vector(31 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal cond_br19_trueOut : std_logic_vector(31 downto 0);
  signal cond_br19_trueOut_valid : std_logic;
  signal cond_br19_trueOut_ready : std_logic;
  signal cond_br19_falseOut : std_logic_vector(31 downto 0);
  signal cond_br19_falseOut_valid : std_logic;
  signal cond_br19_falseOut_ready : std_logic;
  signal cond_br20_trueOut_valid : std_logic;
  signal cond_br20_trueOut_ready : std_logic;
  signal cond_br20_falseOut_valid : std_logic;
  signal cond_br20_falseOut_ready : std_logic;
  signal cond_br21_trueOut : std_logic_vector(0 downto 0);
  signal cond_br21_trueOut_valid : std_logic;
  signal cond_br21_trueOut_ready : std_logic;
  signal cond_br21_falseOut : std_logic_vector(0 downto 0);
  signal cond_br21_falseOut_valid : std_logic;
  signal cond_br21_falseOut_ready : std_logic;
  signal extsi27_outs : std_logic_vector(4 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal mux12_outs : std_logic_vector(4 downto 0);
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal mux13_outs : std_logic_vector(31 downto 0);
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal control_merge5_outs_valid : std_logic;
  signal control_merge5_outs_ready : std_logic;
  signal control_merge5_index : std_logic_vector(0 downto 0);
  signal control_merge5_index_valid : std_logic;
  signal control_merge5_index_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(0 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(0 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal constant38_outs : std_logic_vector(0 downto 0);
  signal constant38_outs_valid : std_logic;
  signal constant38_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(4 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal buffer53_outs : std_logic_vector(31 downto 0);
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal buffer52_outs : std_logic_vector(4 downto 0);
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal mux14_outs : std_logic_vector(4 downto 0);
  signal mux14_outs_valid : std_logic;
  signal mux14_outs_ready : std_logic;
  signal buffer54_outs : std_logic_vector(4 downto 0);
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(4 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(4 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal fork27_outs_2 : std_logic_vector(4 downto 0);
  signal fork27_outs_2_valid : std_logic;
  signal fork27_outs_2_ready : std_logic;
  signal extsi48_outs : std_logic_vector(6 downto 0);
  signal extsi48_outs_valid : std_logic;
  signal extsi48_outs_ready : std_logic;
  signal extsi49_outs : std_logic_vector(6 downto 0);
  signal extsi49_outs_valid : std_logic;
  signal extsi49_outs_ready : std_logic;
  signal mux15_outs : std_logic_vector(31 downto 0);
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
  signal buffer55_outs : std_logic_vector(31 downto 0);
  signal buffer55_outs_valid : std_logic;
  signal buffer55_outs_ready : std_logic;
  signal buffer56_outs : std_logic_vector(31 downto 0);
  signal buffer56_outs_valid : std_logic;
  signal buffer56_outs_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(31 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(31 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal mux16_outs : std_logic_vector(4 downto 0);
  signal mux16_outs_valid : std_logic;
  signal mux16_outs_ready : std_logic;
  signal buffer57_outs : std_logic_vector(4 downto 0);
  signal buffer57_outs_valid : std_logic;
  signal buffer57_outs_ready : std_logic;
  signal buffer58_outs : std_logic_vector(4 downto 0);
  signal buffer58_outs_valid : std_logic;
  signal buffer58_outs_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(4 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(4 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal extsi50_outs : std_logic_vector(31 downto 0);
  signal extsi50_outs_valid : std_logic;
  signal extsi50_outs_ready : std_logic;
  signal fork30_outs_0 : std_logic_vector(31 downto 0);
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1 : std_logic_vector(31 downto 0);
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal fork30_outs_2 : std_logic_vector(31 downto 0);
  signal fork30_outs_2_valid : std_logic;
  signal fork30_outs_2_ready : std_logic;
  signal fork30_outs_3 : std_logic_vector(31 downto 0);
  signal fork30_outs_3_valid : std_logic;
  signal fork30_outs_3_ready : std_logic;
  signal control_merge6_outs_valid : std_logic;
  signal control_merge6_outs_ready : std_logic;
  signal control_merge6_index : std_logic_vector(0 downto 0);
  signal control_merge6_index_valid : std_logic;
  signal control_merge6_index_ready : std_logic;
  signal fork31_outs_0 : std_logic_vector(0 downto 0);
  signal fork31_outs_0_valid : std_logic;
  signal fork31_outs_0_ready : std_logic;
  signal fork31_outs_1 : std_logic_vector(0 downto 0);
  signal fork31_outs_1_valid : std_logic;
  signal fork31_outs_1_ready : std_logic;
  signal fork31_outs_2 : std_logic_vector(0 downto 0);
  signal fork31_outs_2_valid : std_logic;
  signal fork31_outs_2_ready : std_logic;
  signal lazy_fork2_outs_0_valid : std_logic;
  signal lazy_fork2_outs_0_ready : std_logic;
  signal lazy_fork2_outs_1_valid : std_logic;
  signal lazy_fork2_outs_1_ready : std_logic;
  signal lazy_fork2_outs_2_valid : std_logic;
  signal lazy_fork2_outs_2_ready : std_logic;
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal constant39_outs : std_logic_vector(0 downto 0);
  signal constant39_outs_valid : std_logic;
  signal constant39_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant40_outs : std_logic_vector(1 downto 0);
  signal constant40_outs_valid : std_logic;
  signal constant40_outs_ready : std_logic;
  signal extsi16_outs : std_logic_vector(31 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal fork32_outs_0 : std_logic_vector(31 downto 0);
  signal fork32_outs_0_valid : std_logic;
  signal fork32_outs_0_ready : std_logic;
  signal fork32_outs_1 : std_logic_vector(31 downto 0);
  signal fork32_outs_1_valid : std_logic;
  signal fork32_outs_1_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal constant41_outs : std_logic_vector(2 downto 0);
  signal constant41_outs_valid : std_logic;
  signal constant41_outs_ready : std_logic;
  signal extsi17_outs : std_logic_vector(31 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal fork33_outs_0 : std_logic_vector(31 downto 0);
  signal fork33_outs_0_valid : std_logic;
  signal fork33_outs_0_ready : std_logic;
  signal fork33_outs_1 : std_logic_vector(31 downto 0);
  signal fork33_outs_1_valid : std_logic;
  signal fork33_outs_1_ready : std_logic;
  signal shli10_result : std_logic_vector(31 downto 0);
  signal shli10_result_valid : std_logic;
  signal shli10_result_ready : std_logic;
  signal buffer61_outs : std_logic_vector(31 downto 0);
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal trunci13_outs : std_logic_vector(6 downto 0);
  signal trunci13_outs_valid : std_logic;
  signal trunci13_outs_ready : std_logic;
  signal shli11_result : std_logic_vector(31 downto 0);
  signal shli11_result_valid : std_logic;
  signal shli11_result_ready : std_logic;
  signal buffer62_outs : std_logic_vector(31 downto 0);
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal trunci14_outs : std_logic_vector(6 downto 0);
  signal trunci14_outs_valid : std_logic;
  signal trunci14_outs_ready : std_logic;
  signal addi24_result : std_logic_vector(6 downto 0);
  signal addi24_result_valid : std_logic;
  signal addi24_result_ready : std_logic;
  signal buffer63_outs : std_logic_vector(6 downto 0);
  signal buffer63_outs_valid : std_logic;
  signal buffer63_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(6 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer64_outs : std_logic_vector(6 downto 0);
  signal buffer64_outs_valid : std_logic;
  signal buffer64_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(6 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal muli2_result : std_logic_vector(31 downto 0);
  signal muli2_result_valid : std_logic;
  signal muli2_result_ready : std_logic;
  signal shli12_result : std_logic_vector(31 downto 0);
  signal shli12_result_valid : std_logic;
  signal shli12_result_ready : std_logic;
  signal buffer66_outs : std_logic_vector(31 downto 0);
  signal buffer66_outs_valid : std_logic;
  signal buffer66_outs_ready : std_logic;
  signal trunci15_outs : std_logic_vector(6 downto 0);
  signal trunci15_outs_valid : std_logic;
  signal trunci15_outs_ready : std_logic;
  signal shli13_result : std_logic_vector(31 downto 0);
  signal shli13_result_valid : std_logic;
  signal shli13_result_ready : std_logic;
  signal buffer67_outs : std_logic_vector(31 downto 0);
  signal buffer67_outs_valid : std_logic;
  signal buffer67_outs_ready : std_logic;
  signal trunci16_outs : std_logic_vector(6 downto 0);
  signal trunci16_outs_valid : std_logic;
  signal trunci16_outs_ready : std_logic;
  signal addi25_result : std_logic_vector(6 downto 0);
  signal addi25_result_valid : std_logic;
  signal addi25_result_ready : std_logic;
  signal buffer68_outs : std_logic_vector(6 downto 0);
  signal buffer68_outs_valid : std_logic;
  signal buffer68_outs_ready : std_logic;
  signal addi8_result : std_logic_vector(6 downto 0);
  signal addi8_result_valid : std_logic;
  signal addi8_result_ready : std_logic;
  signal buffer65_outs : std_logic_vector(31 downto 0);
  signal buffer65_outs_valid : std_logic;
  signal buffer65_outs_ready : std_logic;
  signal buffer69_outs : std_logic_vector(6 downto 0);
  signal buffer69_outs_valid : std_logic;
  signal buffer69_outs_ready : std_logic;
  signal store2_addrOut : std_logic_vector(6 downto 0);
  signal store2_addrOut_valid : std_logic;
  signal store2_addrOut_ready : std_logic;
  signal store2_dataToMem : std_logic_vector(31 downto 0);
  signal store2_dataToMem_valid : std_logic;
  signal store2_dataToMem_ready : std_logic;
  signal extsi25_outs : std_logic_vector(4 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal mux17_outs : std_logic_vector(4 downto 0);
  signal mux17_outs_valid : std_logic;
  signal mux17_outs_ready : std_logic;
  signal buffer70_outs : std_logic_vector(4 downto 0);
  signal buffer70_outs_valid : std_logic;
  signal buffer70_outs_ready : std_logic;
  signal buffer71_outs : std_logic_vector(4 downto 0);
  signal buffer71_outs_valid : std_logic;
  signal buffer71_outs_ready : std_logic;
  signal fork34_outs_0 : std_logic_vector(4 downto 0);
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1 : std_logic_vector(4 downto 0);
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal fork34_outs_2 : std_logic_vector(4 downto 0);
  signal fork34_outs_2_valid : std_logic;
  signal fork34_outs_2_ready : std_logic;
  signal extsi51_outs : std_logic_vector(6 downto 0);
  signal extsi51_outs_valid : std_logic;
  signal extsi51_outs_ready : std_logic;
  signal extsi52_outs : std_logic_vector(5 downto 0);
  signal extsi52_outs_valid : std_logic;
  signal extsi52_outs_ready : std_logic;
  signal extsi53_outs : std_logic_vector(31 downto 0);
  signal extsi53_outs_valid : std_logic;
  signal extsi53_outs_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(31 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(31 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal mux18_outs : std_logic_vector(31 downto 0);
  signal mux18_outs_valid : std_logic;
  signal mux18_outs_ready : std_logic;
  signal mux19_outs : std_logic_vector(4 downto 0);
  signal mux19_outs_valid : std_logic;
  signal mux19_outs_ready : std_logic;
  signal buffer74_outs : std_logic_vector(4 downto 0);
  signal buffer74_outs_valid : std_logic;
  signal buffer74_outs_ready : std_logic;
  signal buffer75_outs : std_logic_vector(4 downto 0);
  signal buffer75_outs_valid : std_logic;
  signal buffer75_outs_ready : std_logic;
  signal fork36_outs_0 : std_logic_vector(4 downto 0);
  signal fork36_outs_0_valid : std_logic;
  signal fork36_outs_0_ready : std_logic;
  signal fork36_outs_1 : std_logic_vector(4 downto 0);
  signal fork36_outs_1_valid : std_logic;
  signal fork36_outs_1_ready : std_logic;
  signal extsi54_outs : std_logic_vector(31 downto 0);
  signal extsi54_outs_valid : std_logic;
  signal extsi54_outs_ready : std_logic;
  signal fork37_outs_0 : std_logic_vector(31 downto 0);
  signal fork37_outs_0_valid : std_logic;
  signal fork37_outs_0_ready : std_logic;
  signal fork37_outs_1 : std_logic_vector(31 downto 0);
  signal fork37_outs_1_valid : std_logic;
  signal fork37_outs_1_ready : std_logic;
  signal fork37_outs_2 : std_logic_vector(31 downto 0);
  signal fork37_outs_2_valid : std_logic;
  signal fork37_outs_2_ready : std_logic;
  signal fork37_outs_3 : std_logic_vector(31 downto 0);
  signal fork37_outs_3_valid : std_logic;
  signal fork37_outs_3_ready : std_logic;
  signal fork37_outs_4 : std_logic_vector(31 downto 0);
  signal fork37_outs_4_valid : std_logic;
  signal fork37_outs_4_ready : std_logic;
  signal fork37_outs_5 : std_logic_vector(31 downto 0);
  signal fork37_outs_5_valid : std_logic;
  signal fork37_outs_5_ready : std_logic;
  signal mux20_outs : std_logic_vector(4 downto 0);
  signal mux20_outs_valid : std_logic;
  signal mux20_outs_ready : std_logic;
  signal buffer76_outs : std_logic_vector(4 downto 0);
  signal buffer76_outs_valid : std_logic;
  signal buffer76_outs_ready : std_logic;
  signal buffer77_outs : std_logic_vector(4 downto 0);
  signal buffer77_outs_valid : std_logic;
  signal buffer77_outs_ready : std_logic;
  signal fork38_outs_0 : std_logic_vector(4 downto 0);
  signal fork38_outs_0_valid : std_logic;
  signal fork38_outs_0_ready : std_logic;
  signal fork38_outs_1 : std_logic_vector(4 downto 0);
  signal fork38_outs_1_valid : std_logic;
  signal fork38_outs_1_ready : std_logic;
  signal fork38_outs_2 : std_logic_vector(4 downto 0);
  signal fork38_outs_2_valid : std_logic;
  signal fork38_outs_2_ready : std_logic;
  signal fork38_outs_3 : std_logic_vector(4 downto 0);
  signal fork38_outs_3_valid : std_logic;
  signal fork38_outs_3_ready : std_logic;
  signal extsi55_outs : std_logic_vector(6 downto 0);
  signal extsi55_outs_valid : std_logic;
  signal extsi55_outs_ready : std_logic;
  signal extsi56_outs : std_logic_vector(6 downto 0);
  signal extsi56_outs_valid : std_logic;
  signal extsi56_outs_ready : std_logic;
  signal extsi57_outs : std_logic_vector(6 downto 0);
  signal extsi57_outs_valid : std_logic;
  signal extsi57_outs_ready : std_logic;
  signal control_merge7_outs_valid : std_logic;
  signal control_merge7_outs_ready : std_logic;
  signal control_merge7_index : std_logic_vector(0 downto 0);
  signal control_merge7_index_valid : std_logic;
  signal control_merge7_index_ready : std_logic;
  signal fork39_outs_0 : std_logic_vector(0 downto 0);
  signal fork39_outs_0_valid : std_logic;
  signal fork39_outs_0_ready : std_logic;
  signal fork39_outs_1 : std_logic_vector(0 downto 0);
  signal fork39_outs_1_valid : std_logic;
  signal fork39_outs_1_ready : std_logic;
  signal fork39_outs_2 : std_logic_vector(0 downto 0);
  signal fork39_outs_2_valid : std_logic;
  signal fork39_outs_2_ready : std_logic;
  signal fork39_outs_3 : std_logic_vector(0 downto 0);
  signal fork39_outs_3_valid : std_logic;
  signal fork39_outs_3_ready : std_logic;
  signal lazy_fork3_outs_0_valid : std_logic;
  signal lazy_fork3_outs_0_ready : std_logic;
  signal lazy_fork3_outs_1_valid : std_logic;
  signal lazy_fork3_outs_1_ready : std_logic;
  signal lazy_fork3_outs_2_valid : std_logic;
  signal lazy_fork3_outs_2_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal constant42_outs : std_logic_vector(4 downto 0);
  signal constant42_outs_valid : std_logic;
  signal constant42_outs_ready : std_logic;
  signal extsi58_outs : std_logic_vector(5 downto 0);
  signal extsi58_outs_valid : std_logic;
  signal extsi58_outs_ready : std_logic;
  signal source12_outs_valid : std_logic;
  signal source12_outs_ready : std_logic;
  signal constant43_outs : std_logic_vector(1 downto 0);
  signal constant43_outs_valid : std_logic;
  signal constant43_outs_ready : std_logic;
  signal fork40_outs_0 : std_logic_vector(1 downto 0);
  signal fork40_outs_0_valid : std_logic;
  signal fork40_outs_0_ready : std_logic;
  signal fork40_outs_1 : std_logic_vector(1 downto 0);
  signal fork40_outs_1_valid : std_logic;
  signal fork40_outs_1_ready : std_logic;
  signal extsi59_outs : std_logic_vector(5 downto 0);
  signal extsi59_outs_valid : std_logic;
  signal extsi59_outs_ready : std_logic;
  signal extsi19_outs : std_logic_vector(31 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal fork41_outs_0 : std_logic_vector(31 downto 0);
  signal fork41_outs_0_valid : std_logic;
  signal fork41_outs_0_ready : std_logic;
  signal fork41_outs_1 : std_logic_vector(31 downto 0);
  signal fork41_outs_1_valid : std_logic;
  signal fork41_outs_1_ready : std_logic;
  signal fork41_outs_2 : std_logic_vector(31 downto 0);
  signal fork41_outs_2_valid : std_logic;
  signal fork41_outs_2_ready : std_logic;
  signal fork41_outs_3 : std_logic_vector(31 downto 0);
  signal fork41_outs_3_valid : std_logic;
  signal fork41_outs_3_ready : std_logic;
  signal source13_outs_valid : std_logic;
  signal source13_outs_ready : std_logic;
  signal constant44_outs : std_logic_vector(2 downto 0);
  signal constant44_outs_valid : std_logic;
  signal constant44_outs_ready : std_logic;
  signal extsi20_outs : std_logic_vector(31 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal fork42_outs_0 : std_logic_vector(31 downto 0);
  signal fork42_outs_0_valid : std_logic;
  signal fork42_outs_0_ready : std_logic;
  signal fork42_outs_1 : std_logic_vector(31 downto 0);
  signal fork42_outs_1_valid : std_logic;
  signal fork42_outs_1_ready : std_logic;
  signal fork42_outs_2 : std_logic_vector(31 downto 0);
  signal fork42_outs_2_valid : std_logic;
  signal fork42_outs_2_ready : std_logic;
  signal fork42_outs_3 : std_logic_vector(31 downto 0);
  signal fork42_outs_3_valid : std_logic;
  signal fork42_outs_3_ready : std_logic;
  signal shli14_result : std_logic_vector(31 downto 0);
  signal shli14_result_valid : std_logic;
  signal shli14_result_ready : std_logic;
  signal buffer80_outs : std_logic_vector(31 downto 0);
  signal buffer80_outs_valid : std_logic;
  signal buffer80_outs_ready : std_logic;
  signal trunci17_outs : std_logic_vector(6 downto 0);
  signal trunci17_outs_valid : std_logic;
  signal trunci17_outs_ready : std_logic;
  signal shli15_result : std_logic_vector(31 downto 0);
  signal shli15_result_valid : std_logic;
  signal shli15_result_ready : std_logic;
  signal buffer81_outs : std_logic_vector(31 downto 0);
  signal buffer81_outs_valid : std_logic;
  signal buffer81_outs_ready : std_logic;
  signal trunci18_outs : std_logic_vector(6 downto 0);
  signal trunci18_outs_valid : std_logic;
  signal trunci18_outs_ready : std_logic;
  signal addi26_result : std_logic_vector(6 downto 0);
  signal addi26_result_valid : std_logic;
  signal addi26_result_ready : std_logic;
  signal buffer82_outs : std_logic_vector(6 downto 0);
  signal buffer82_outs_valid : std_logic;
  signal buffer82_outs_ready : std_logic;
  signal addi9_result : std_logic_vector(6 downto 0);
  signal addi9_result_valid : std_logic;
  signal addi9_result_ready : std_logic;
  signal buffer83_outs : std_logic_vector(6 downto 0);
  signal buffer83_outs_valid : std_logic;
  signal buffer83_outs_ready : std_logic;
  signal load4_addrOut : std_logic_vector(6 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal shli16_result : std_logic_vector(31 downto 0);
  signal shli16_result_valid : std_logic;
  signal shli16_result_ready : std_logic;
  signal buffer84_outs : std_logic_vector(31 downto 0);
  signal buffer84_outs_valid : std_logic;
  signal buffer84_outs_ready : std_logic;
  signal trunci19_outs : std_logic_vector(6 downto 0);
  signal trunci19_outs_valid : std_logic;
  signal trunci19_outs_ready : std_logic;
  signal shli17_result : std_logic_vector(31 downto 0);
  signal shli17_result_valid : std_logic;
  signal shli17_result_ready : std_logic;
  signal buffer85_outs : std_logic_vector(31 downto 0);
  signal buffer85_outs_valid : std_logic;
  signal buffer85_outs_ready : std_logic;
  signal trunci20_outs : std_logic_vector(6 downto 0);
  signal trunci20_outs_valid : std_logic;
  signal trunci20_outs_ready : std_logic;
  signal addi27_result : std_logic_vector(6 downto 0);
  signal addi27_result_valid : std_logic;
  signal addi27_result_ready : std_logic;
  signal buffer86_outs : std_logic_vector(6 downto 0);
  signal buffer86_outs_valid : std_logic;
  signal buffer86_outs_ready : std_logic;
  signal addi10_result : std_logic_vector(6 downto 0);
  signal addi10_result_valid : std_logic;
  signal addi10_result_ready : std_logic;
  signal load5_addrOut : std_logic_vector(6 downto 0);
  signal load5_addrOut_valid : std_logic;
  signal load5_addrOut_ready : std_logic;
  signal load5_dataOut : std_logic_vector(31 downto 0);
  signal load5_dataOut_valid : std_logic;
  signal load5_dataOut_ready : std_logic;
  signal muli3_result : std_logic_vector(31 downto 0);
  signal muli3_result_valid : std_logic;
  signal muli3_result_ready : std_logic;
  signal shli18_result : std_logic_vector(31 downto 0);
  signal shli18_result_valid : std_logic;
  signal shli18_result_ready : std_logic;
  signal buffer87_outs : std_logic_vector(31 downto 0);
  signal buffer87_outs_valid : std_logic;
  signal buffer87_outs_ready : std_logic;
  signal trunci21_outs : std_logic_vector(6 downto 0);
  signal trunci21_outs_valid : std_logic;
  signal trunci21_outs_ready : std_logic;
  signal shli19_result : std_logic_vector(31 downto 0);
  signal shli19_result_valid : std_logic;
  signal shli19_result_ready : std_logic;
  signal buffer88_outs : std_logic_vector(31 downto 0);
  signal buffer88_outs_valid : std_logic;
  signal buffer88_outs_ready : std_logic;
  signal trunci22_outs : std_logic_vector(6 downto 0);
  signal trunci22_outs_valid : std_logic;
  signal trunci22_outs_ready : std_logic;
  signal addi28_result : std_logic_vector(6 downto 0);
  signal addi28_result_valid : std_logic;
  signal addi28_result_ready : std_logic;
  signal buffer89_outs : std_logic_vector(6 downto 0);
  signal buffer89_outs_valid : std_logic;
  signal buffer89_outs_ready : std_logic;
  signal addi11_result : std_logic_vector(6 downto 0);
  signal addi11_result_valid : std_logic;
  signal addi11_result_ready : std_logic;
  signal buffer90_outs : std_logic_vector(6 downto 0);
  signal buffer90_outs_valid : std_logic;
  signal buffer90_outs_ready : std_logic;
  signal load6_addrOut : std_logic_vector(6 downto 0);
  signal load6_addrOut_valid : std_logic;
  signal load6_addrOut_ready : std_logic;
  signal load6_dataOut : std_logic_vector(31 downto 0);
  signal load6_dataOut_valid : std_logic;
  signal load6_dataOut_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal shli20_result : std_logic_vector(31 downto 0);
  signal shli20_result_valid : std_logic;
  signal shli20_result_ready : std_logic;
  signal buffer92_outs : std_logic_vector(31 downto 0);
  signal buffer92_outs_valid : std_logic;
  signal buffer92_outs_ready : std_logic;
  signal trunci23_outs : std_logic_vector(6 downto 0);
  signal trunci23_outs_valid : std_logic;
  signal trunci23_outs_ready : std_logic;
  signal shli21_result : std_logic_vector(31 downto 0);
  signal shli21_result_valid : std_logic;
  signal shli21_result_ready : std_logic;
  signal buffer93_outs : std_logic_vector(31 downto 0);
  signal buffer93_outs_valid : std_logic;
  signal buffer93_outs_ready : std_logic;
  signal trunci24_outs : std_logic_vector(6 downto 0);
  signal trunci24_outs_valid : std_logic;
  signal trunci24_outs_ready : std_logic;
  signal addi29_result : std_logic_vector(6 downto 0);
  signal addi29_result_valid : std_logic;
  signal addi29_result_ready : std_logic;
  signal buffer94_outs : std_logic_vector(6 downto 0);
  signal buffer94_outs_valid : std_logic;
  signal buffer94_outs_ready : std_logic;
  signal addi12_result : std_logic_vector(6 downto 0);
  signal addi12_result_valid : std_logic;
  signal addi12_result_ready : std_logic;
  signal buffer91_outs : std_logic_vector(31 downto 0);
  signal buffer91_outs_valid : std_logic;
  signal buffer91_outs_ready : std_logic;
  signal buffer95_outs : std_logic_vector(6 downto 0);
  signal buffer95_outs_valid : std_logic;
  signal buffer95_outs_ready : std_logic;
  signal store3_addrOut : std_logic_vector(6 downto 0);
  signal store3_addrOut_valid : std_logic;
  signal store3_addrOut_ready : std_logic;
  signal store3_dataToMem : std_logic_vector(31 downto 0);
  signal store3_dataToMem_valid : std_logic;
  signal store3_dataToMem_ready : std_logic;
  signal addi16_result : std_logic_vector(5 downto 0);
  signal addi16_result_valid : std_logic;
  signal addi16_result_ready : std_logic;
  signal buffer96_outs : std_logic_vector(5 downto 0);
  signal buffer96_outs_valid : std_logic;
  signal buffer96_outs_ready : std_logic;
  signal fork43_outs_0 : std_logic_vector(5 downto 0);
  signal fork43_outs_0_valid : std_logic;
  signal fork43_outs_0_ready : std_logic;
  signal fork43_outs_1 : std_logic_vector(5 downto 0);
  signal fork43_outs_1_valid : std_logic;
  signal fork43_outs_1_ready : std_logic;
  signal trunci25_outs : std_logic_vector(4 downto 0);
  signal trunci25_outs_valid : std_logic;
  signal trunci25_outs_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal buffer97_outs : std_logic_vector(0 downto 0);
  signal buffer97_outs_valid : std_logic;
  signal buffer97_outs_ready : std_logic;
  signal fork44_outs_0 : std_logic_vector(0 downto 0);
  signal fork44_outs_0_valid : std_logic;
  signal fork44_outs_0_ready : std_logic;
  signal fork44_outs_1 : std_logic_vector(0 downto 0);
  signal fork44_outs_1_valid : std_logic;
  signal fork44_outs_1_ready : std_logic;
  signal fork44_outs_2 : std_logic_vector(0 downto 0);
  signal fork44_outs_2_valid : std_logic;
  signal fork44_outs_2_ready : std_logic;
  signal fork44_outs_3 : std_logic_vector(0 downto 0);
  signal fork44_outs_3_valid : std_logic;
  signal fork44_outs_3_ready : std_logic;
  signal fork44_outs_4 : std_logic_vector(0 downto 0);
  signal fork44_outs_4_valid : std_logic;
  signal fork44_outs_4_ready : std_logic;
  signal cond_br22_trueOut : std_logic_vector(4 downto 0);
  signal cond_br22_trueOut_valid : std_logic;
  signal cond_br22_trueOut_ready : std_logic;
  signal cond_br22_falseOut : std_logic_vector(4 downto 0);
  signal cond_br22_falseOut_valid : std_logic;
  signal cond_br22_falseOut_ready : std_logic;
  signal buffer72_outs : std_logic_vector(31 downto 0);
  signal buffer72_outs_valid : std_logic;
  signal buffer72_outs_ready : std_logic;
  signal buffer73_outs : std_logic_vector(31 downto 0);
  signal buffer73_outs_valid : std_logic;
  signal buffer73_outs_ready : std_logic;
  signal cond_br23_trueOut : std_logic_vector(31 downto 0);
  signal cond_br23_trueOut_valid : std_logic;
  signal cond_br23_trueOut_ready : std_logic;
  signal cond_br23_falseOut : std_logic_vector(31 downto 0);
  signal cond_br23_falseOut_valid : std_logic;
  signal cond_br23_falseOut_ready : std_logic;
  signal cond_br24_trueOut : std_logic_vector(4 downto 0);
  signal cond_br24_trueOut_valid : std_logic;
  signal cond_br24_trueOut_ready : std_logic;
  signal cond_br24_falseOut : std_logic_vector(4 downto 0);
  signal cond_br24_falseOut_valid : std_logic;
  signal cond_br24_falseOut_ready : std_logic;
  signal cond_br25_trueOut : std_logic_vector(4 downto 0);
  signal cond_br25_trueOut_valid : std_logic;
  signal cond_br25_trueOut_ready : std_logic;
  signal cond_br25_falseOut : std_logic_vector(4 downto 0);
  signal cond_br25_falseOut_valid : std_logic;
  signal cond_br25_falseOut_ready : std_logic;
  signal buffer78_outs_valid : std_logic;
  signal buffer78_outs_ready : std_logic;
  signal buffer79_outs_valid : std_logic;
  signal buffer79_outs_ready : std_logic;
  signal cond_br26_trueOut_valid : std_logic;
  signal cond_br26_trueOut_ready : std_logic;
  signal cond_br26_falseOut_valid : std_logic;
  signal cond_br26_falseOut_ready : std_logic;
  signal extsi60_outs : std_logic_vector(5 downto 0);
  signal extsi60_outs_valid : std_logic;
  signal extsi60_outs_ready : std_logic;
  signal source14_outs_valid : std_logic;
  signal source14_outs_ready : std_logic;
  signal constant45_outs : std_logic_vector(4 downto 0);
  signal constant45_outs_valid : std_logic;
  signal constant45_outs_ready : std_logic;
  signal extsi61_outs : std_logic_vector(5 downto 0);
  signal extsi61_outs_valid : std_logic;
  signal extsi61_outs_ready : std_logic;
  signal source15_outs_valid : std_logic;
  signal source15_outs_ready : std_logic;
  signal constant46_outs : std_logic_vector(1 downto 0);
  signal constant46_outs_valid : std_logic;
  signal constant46_outs_ready : std_logic;
  signal extsi62_outs : std_logic_vector(5 downto 0);
  signal extsi62_outs_valid : std_logic;
  signal extsi62_outs_ready : std_logic;
  signal addi17_result : std_logic_vector(5 downto 0);
  signal addi17_result_valid : std_logic;
  signal addi17_result_ready : std_logic;
  signal buffer99_outs : std_logic_vector(5 downto 0);
  signal buffer99_outs_valid : std_logic;
  signal buffer99_outs_ready : std_logic;
  signal fork45_outs_0 : std_logic_vector(5 downto 0);
  signal fork45_outs_0_valid : std_logic;
  signal fork45_outs_0_ready : std_logic;
  signal fork45_outs_1 : std_logic_vector(5 downto 0);
  signal fork45_outs_1_valid : std_logic;
  signal fork45_outs_1_ready : std_logic;
  signal trunci26_outs : std_logic_vector(4 downto 0);
  signal trunci26_outs_valid : std_logic;
  signal trunci26_outs_ready : std_logic;
  signal cmpi4_result : std_logic_vector(0 downto 0);
  signal cmpi4_result_valid : std_logic;
  signal cmpi4_result_ready : std_logic;
  signal buffer100_outs : std_logic_vector(0 downto 0);
  signal buffer100_outs_valid : std_logic;
  signal buffer100_outs_ready : std_logic;
  signal fork46_outs_0 : std_logic_vector(0 downto 0);
  signal fork46_outs_0_valid : std_logic;
  signal fork46_outs_0_ready : std_logic;
  signal fork46_outs_1 : std_logic_vector(0 downto 0);
  signal fork46_outs_1_valid : std_logic;
  signal fork46_outs_1_ready : std_logic;
  signal fork46_outs_2 : std_logic_vector(0 downto 0);
  signal fork46_outs_2_valid : std_logic;
  signal fork46_outs_2_ready : std_logic;
  signal fork46_outs_3 : std_logic_vector(0 downto 0);
  signal fork46_outs_3_valid : std_logic;
  signal fork46_outs_3_ready : std_logic;
  signal cond_br27_trueOut : std_logic_vector(4 downto 0);
  signal cond_br27_trueOut_valid : std_logic;
  signal cond_br27_trueOut_ready : std_logic;
  signal cond_br27_falseOut : std_logic_vector(4 downto 0);
  signal cond_br27_falseOut_valid : std_logic;
  signal cond_br27_falseOut_ready : std_logic;
  signal cond_br28_trueOut : std_logic_vector(31 downto 0);
  signal cond_br28_trueOut_valid : std_logic;
  signal cond_br28_trueOut_ready : std_logic;
  signal cond_br28_falseOut : std_logic_vector(31 downto 0);
  signal cond_br28_falseOut_valid : std_logic;
  signal cond_br28_falseOut_ready : std_logic;
  signal buffer98_outs : std_logic_vector(4 downto 0);
  signal buffer98_outs_valid : std_logic;
  signal buffer98_outs_ready : std_logic;
  signal cond_br29_trueOut : std_logic_vector(4 downto 0);
  signal cond_br29_trueOut_valid : std_logic;
  signal cond_br29_trueOut_ready : std_logic;
  signal cond_br29_falseOut : std_logic_vector(4 downto 0);
  signal cond_br29_falseOut_valid : std_logic;
  signal cond_br29_falseOut_ready : std_logic;
  signal cond_br30_trueOut_valid : std_logic;
  signal cond_br30_trueOut_ready : std_logic;
  signal cond_br30_falseOut_valid : std_logic;
  signal cond_br30_falseOut_ready : std_logic;
  signal extsi63_outs : std_logic_vector(5 downto 0);
  signal extsi63_outs_valid : std_logic;
  signal extsi63_outs_ready : std_logic;
  signal source16_outs_valid : std_logic;
  signal source16_outs_ready : std_logic;
  signal constant47_outs : std_logic_vector(4 downto 0);
  signal constant47_outs_valid : std_logic;
  signal constant47_outs_ready : std_logic;
  signal extsi64_outs : std_logic_vector(5 downto 0);
  signal extsi64_outs_valid : std_logic;
  signal extsi64_outs_ready : std_logic;
  signal source17_outs_valid : std_logic;
  signal source17_outs_ready : std_logic;
  signal constant48_outs : std_logic_vector(1 downto 0);
  signal constant48_outs_valid : std_logic;
  signal constant48_outs_ready : std_logic;
  signal extsi65_outs : std_logic_vector(5 downto 0);
  signal extsi65_outs_valid : std_logic;
  signal extsi65_outs_ready : std_logic;
  signal addi18_result : std_logic_vector(5 downto 0);
  signal addi18_result_valid : std_logic;
  signal addi18_result_ready : std_logic;
  signal buffer102_outs : std_logic_vector(5 downto 0);
  signal buffer102_outs_valid : std_logic;
  signal buffer102_outs_ready : std_logic;
  signal fork47_outs_0 : std_logic_vector(5 downto 0);
  signal fork47_outs_0_valid : std_logic;
  signal fork47_outs_0_ready : std_logic;
  signal fork47_outs_1 : std_logic_vector(5 downto 0);
  signal fork47_outs_1_valid : std_logic;
  signal fork47_outs_1_ready : std_logic;
  signal trunci27_outs : std_logic_vector(4 downto 0);
  signal trunci27_outs_valid : std_logic;
  signal trunci27_outs_ready : std_logic;
  signal cmpi5_result : std_logic_vector(0 downto 0);
  signal cmpi5_result_valid : std_logic;
  signal cmpi5_result_ready : std_logic;
  signal buffer103_outs : std_logic_vector(0 downto 0);
  signal buffer103_outs_valid : std_logic;
  signal buffer103_outs_ready : std_logic;
  signal fork48_outs_0 : std_logic_vector(0 downto 0);
  signal fork48_outs_0_valid : std_logic;
  signal fork48_outs_0_ready : std_logic;
  signal fork48_outs_1 : std_logic_vector(0 downto 0);
  signal fork48_outs_1_valid : std_logic;
  signal fork48_outs_1_ready : std_logic;
  signal fork48_outs_2 : std_logic_vector(0 downto 0);
  signal fork48_outs_2_valid : std_logic;
  signal fork48_outs_2_ready : std_logic;
  signal cond_br31_trueOut : std_logic_vector(4 downto 0);
  signal cond_br31_trueOut_valid : std_logic;
  signal cond_br31_trueOut_ready : std_logic;
  signal cond_br31_falseOut : std_logic_vector(4 downto 0);
  signal cond_br31_falseOut_valid : std_logic;
  signal cond_br31_falseOut_ready : std_logic;
  signal buffer101_outs : std_logic_vector(31 downto 0);
  signal buffer101_outs_valid : std_logic;
  signal buffer101_outs_ready : std_logic;
  signal cond_br32_trueOut : std_logic_vector(31 downto 0);
  signal cond_br32_trueOut_valid : std_logic;
  signal cond_br32_trueOut_ready : std_logic;
  signal cond_br32_falseOut : std_logic_vector(31 downto 0);
  signal cond_br32_falseOut_valid : std_logic;
  signal cond_br32_falseOut_ready : std_logic;
  signal cond_br33_trueOut_valid : std_logic;
  signal cond_br33_trueOut_ready : std_logic;
  signal cond_br33_falseOut_valid : std_logic;
  signal cond_br33_falseOut_ready : std_logic;
  signal fork49_outs_0_valid : std_logic;
  signal fork49_outs_0_ready : std_logic;
  signal fork49_outs_1_valid : std_logic;
  signal fork49_outs_1_ready : std_logic;
  signal fork49_outs_2_valid : std_logic;
  signal fork49_outs_2_ready : std_logic;
  signal fork49_outs_3_valid : std_logic;
  signal fork49_outs_3_ready : std_logic;
  signal fork49_outs_4_valid : std_logic;
  signal fork49_outs_4_ready : std_logic;

begin

  tmp_end_valid <= lsq3_memEnd_valid;
  lsq3_memEnd_ready <= tmp_end_ready;
  A_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= A_end_ready;
  B_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= B_end_ready;
  C_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= C_end_ready;
  D_end_valid <= lsq2_memEnd_valid;
  lsq2_memEnd_ready <= D_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  tmp_loadEn <= lsq3_loadEn;
  tmp_loadAddr <= lsq3_loadAddr;
  tmp_storeEn <= lsq3_storeEn;
  tmp_storeAddr <= lsq3_storeAddr;
  tmp_storeData <= lsq3_storeData;
  A_loadEn <= mem_controller5_loadEn;
  A_loadAddr <= mem_controller5_loadAddr;
  A_storeEn <= mem_controller5_storeEn;
  A_storeAddr <= mem_controller5_storeAddr;
  A_storeData <= mem_controller5_storeData;
  B_loadEn <= mem_controller4_loadEn;
  B_loadAddr <= mem_controller4_loadAddr;
  B_storeEn <= mem_controller4_storeEn;
  B_storeAddr <= mem_controller4_storeAddr;
  B_storeData <= mem_controller4_storeData;
  C_loadEn <= mem_controller3_loadEn;
  C_loadAddr <= mem_controller3_loadAddr;
  C_storeEn <= mem_controller3_storeEn;
  C_storeAddr <= mem_controller3_storeAddr;
  C_storeData <= mem_controller3_storeData;
  D_loadEn <= lsq2_loadEn;
  D_loadAddr <= lsq2_loadAddr;
  D_storeEn <= lsq2_storeEn;
  D_storeAddr <= lsq2_storeAddr;
  D_storeData <= lsq2_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(3)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready
    );

  lsq2 : entity work.handshake_lsq_lsq2(arch)
    port map(
      io_loadData => D_loadData,
      io_memStart_valid => D_start_valid,
      io_memStart_ready => D_start_ready,
      io_ctrl_0_valid => lazy_fork2_outs_0_valid,
      io_ctrl_0_ready => lazy_fork2_outs_0_ready,
      io_ldAddr_0_bits => load3_addrOut,
      io_ldAddr_0_valid => load3_addrOut_valid,
      io_ldAddr_0_ready => load3_addrOut_ready,
      io_stAddr_0_bits => store2_addrOut,
      io_stAddr_0_valid => store2_addrOut_valid,
      io_stAddr_0_ready => store2_addrOut_ready,
      io_stData_0_bits => store2_dataToMem,
      io_stData_0_valid => store2_dataToMem_valid,
      io_stData_0_ready => store2_dataToMem_ready,
      io_ctrl_1_valid => lazy_fork3_outs_0_valid,
      io_ctrl_1_ready => lazy_fork3_outs_0_ready,
      io_ldAddr_1_bits => load6_addrOut,
      io_ldAddr_1_valid => load6_addrOut_valid,
      io_ldAddr_1_ready => load6_addrOut_ready,
      io_stAddr_1_bits => store3_addrOut,
      io_stAddr_1_valid => store3_addrOut_valid,
      io_stAddr_1_ready => store3_addrOut_ready,
      io_stData_1_bits => store3_dataToMem,
      io_stData_1_valid => store3_dataToMem_valid,
      io_stData_1_ready => store3_dataToMem_ready,
      io_ctrlEnd_valid => fork49_outs_4_valid,
      io_ctrlEnd_ready => fork49_outs_4_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq2_ldData_0,
      io_ldData_0_valid => lsq2_ldData_0_valid,
      io_ldData_0_ready => lsq2_ldData_0_ready,
      io_ldData_1_bits => lsq2_ldData_1,
      io_ldData_1_valid => lsq2_ldData_1_valid,
      io_ldData_1_ready => lsq2_ldData_1_ready,
      io_memEnd_valid => lsq2_memEnd_valid,
      io_memEnd_ready => lsq2_memEnd_ready,
      io_loadEn => lsq2_loadEn,
      io_loadAddr => lsq2_loadAddr,
      io_storeEn => lsq2_storeEn,
      io_storeAddr => lsq2_storeAddr,
      io_storeData => lsq2_storeData
    );

  mem_controller3 : entity work.mem_controller_storeless(arch) generic map(1, 32, 7)
    port map(
      loadData => C_loadData,
      memStart_valid => C_start_valid,
      memStart_ready => C_start_ready,
      ldAddr(0) => load5_addrOut,
      ldAddr_valid(0) => load5_addrOut_valid,
      ldAddr_ready(0) => load5_addrOut_ready,
      ctrlEnd_valid => fork49_outs_3_valid,
      ctrlEnd_ready => fork49_outs_3_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller3_ldData_0,
      ldData_valid(0) => mem_controller3_ldData_0_valid,
      ldData_ready(0) => mem_controller3_ldData_0_ready,
      memEnd_valid => mem_controller3_memEnd_valid,
      memEnd_ready => mem_controller3_memEnd_ready,
      loadEn => mem_controller3_loadEn,
      loadAddr => mem_controller3_loadAddr,
      storeEn => mem_controller3_storeEn,
      storeAddr => mem_controller3_storeAddr,
      storeData => mem_controller3_storeData
    );

  mem_controller4 : entity work.mem_controller_storeless(arch) generic map(1, 32, 7)
    port map(
      loadData => B_loadData,
      memStart_valid => B_start_valid,
      memStart_ready => B_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork49_outs_2_valid,
      ctrlEnd_ready => fork49_outs_2_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller4_ldData_0,
      ldData_valid(0) => mem_controller4_ldData_0_valid,
      ldData_ready(0) => mem_controller4_ldData_0_ready,
      memEnd_valid => mem_controller4_memEnd_valid,
      memEnd_ready => mem_controller4_memEnd_ready,
      loadEn => mem_controller4_loadEn,
      loadAddr => mem_controller4_loadAddr,
      storeEn => mem_controller4_storeEn,
      storeAddr => mem_controller4_storeAddr,
      storeData => mem_controller4_storeData
    );

  mem_controller5 : entity work.mem_controller_storeless(arch) generic map(1, 32, 7)
    port map(
      loadData => A_loadData,
      memStart_valid => A_start_valid,
      memStart_ready => A_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork49_outs_1_valid,
      ctrlEnd_ready => fork49_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller5_ldData_0,
      ldData_valid(0) => mem_controller5_ldData_0_valid,
      ldData_ready(0) => mem_controller5_ldData_0_ready,
      memEnd_valid => mem_controller5_memEnd_valid,
      memEnd_ready => mem_controller5_memEnd_ready,
      loadEn => mem_controller5_loadEn,
      loadAddr => mem_controller5_loadAddr,
      storeEn => mem_controller5_storeEn,
      storeAddr => mem_controller5_storeAddr,
      storeData => mem_controller5_storeData
    );

  lsq3 : entity work.handshake_lsq_lsq3(arch)
    port map(
      io_loadData => tmp_loadData,
      io_memStart_valid => tmp_start_valid,
      io_memStart_ready => tmp_start_ready,
      io_ctrl_0_valid => lazy_fork0_outs_0_valid,
      io_ctrl_0_ready => lazy_fork0_outs_0_ready,
      io_stAddr_0_bits => store0_addrOut,
      io_stAddr_0_valid => store0_addrOut_valid,
      io_stAddr_0_ready => store0_addrOut_ready,
      io_stData_0_bits => store0_dataToMem,
      io_stData_0_valid => store0_dataToMem_valid,
      io_stData_0_ready => store0_dataToMem_ready,
      io_ctrl_1_valid => lazy_fork1_outs_0_valid,
      io_ctrl_1_ready => lazy_fork1_outs_0_ready,
      io_ldAddr_0_bits => load2_addrOut,
      io_ldAddr_0_valid => load2_addrOut_valid,
      io_ldAddr_0_ready => load2_addrOut_ready,
      io_stAddr_1_bits => store1_addrOut,
      io_stAddr_1_valid => store1_addrOut_valid,
      io_stAddr_1_ready => store1_addrOut_ready,
      io_stData_1_bits => store1_dataToMem,
      io_stData_1_valid => store1_dataToMem_valid,
      io_stData_1_ready => store1_dataToMem_ready,
      io_ctrl_2_valid => lazy_fork3_outs_2_valid,
      io_ctrl_2_ready => lazy_fork3_outs_2_ready,
      io_ldAddr_1_bits => load4_addrOut,
      io_ldAddr_1_valid => load4_addrOut_valid,
      io_ldAddr_1_ready => load4_addrOut_ready,
      io_ctrlEnd_valid => fork49_outs_0_valid,
      io_ctrlEnd_ready => fork49_outs_0_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq3_ldData_0,
      io_ldData_0_valid => lsq3_ldData_0_valid,
      io_ldData_0_ready => lsq3_ldData_0_ready,
      io_ldData_1_bits => lsq3_ldData_1,
      io_ldData_1_valid => lsq3_ldData_1_valid,
      io_ldData_1_ready => lsq3_ldData_1_ready,
      io_memEnd_valid => lsq3_memEnd_valid,
      io_memEnd_ready => lsq3_memEnd_ready,
      io_loadEn => lsq3_loadEn,
      io_loadAddr => lsq3_loadAddr,
      io_storeEn => lsq3_storeEn,
      io_storeAddr => lsq3_storeAddr,
      io_storeData => lsq3_storeData
    );

  constant0 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  extsi30 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi30_outs,
      outs_valid => extsi30_outs_valid,
      outs_ready => extsi30_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork1_outs_0,
      index_valid => fork1_outs_0_valid,
      index_ready => fork1_outs_0_ready,
      ins(0) => extsi30_outs,
      ins(1) => cond_br17_trueOut,
      ins_valid(0) => extsi30_outs_valid,
      ins_valid(1) => cond_br17_trueOut_valid,
      ins_ready(0) => extsi30_outs_ready,
      ins_ready(1) => cond_br17_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer51 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br18_trueOut,
      ins_valid => cond_br18_trueOut_valid,
      ins_ready => cond_br18_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer51_outs,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork1_outs_1,
      index_valid => fork1_outs_1_valid,
      index_ready => fork1_outs_1_ready,
      ins(0) => alpha,
      ins(1) => buffer51_outs,
      ins_valid(0) => alpha_valid,
      ins_valid(1) => buffer51_outs_valid,
      ins_ready(0) => alpha_ready,
      ins_ready(1) => buffer51_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork1_outs_2,
      index_valid => fork1_outs_2_valid,
      index_ready => fork1_outs_2_ready,
      ins(0) => beta,
      ins(1) => cond_br19_trueOut,
      ins_valid(0) => beta_valid,
      ins_valid(1) => cond_br19_trueOut_valid,
      ins_ready(0) => beta_ready,
      ins_ready(1) => cond_br19_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br20_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br20_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready
    );

  fork2 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork2_outs_0_valid,
      ctrl_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi29 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi29_outs,
      outs_valid => extsi29_outs_valid,
      outs_ready => extsi29_outs_ready
    );

  buffer1 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  buffer2 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer0 : entity work.tehb(arch) generic map(5)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork6_outs_1,
      index_valid => fork6_outs_1_valid,
      index_ready => fork6_outs_1_ready,
      ins(0) => extsi29_outs,
      ins(1) => cond_br12_trueOut,
      ins_valid(0) => extsi29_outs_valid,
      ins_valid(1) => cond_br12_trueOut_valid,
      ins_ready(0) => extsi29_outs_ready,
      ins_ready(1) => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(5)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 5)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi31 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi31_outs,
      outs_valid => extsi31_outs_valid,
      outs_ready => extsi31_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork6_outs_2,
      index_valid => fork6_outs_2_valid,
      index_ready => fork6_outs_2_ready,
      ins(0) => buffer1_outs,
      ins(1) => cond_br13_trueOut,
      ins_valid(0) => buffer1_outs_valid,
      ins_valid(1) => cond_br13_trueOut_valid,
      ins_ready(0) => buffer1_outs_ready,
      ins_ready(1) => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork6_outs_3,
      index_valid => fork6_outs_3_valid,
      index_ready => fork6_outs_3_ready,
      ins(0) => buffer2_outs,
      ins(1) => cond_br14_trueOut,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => cond_br14_trueOut_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => cond_br14_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork6_outs_0,
      index_valid => fork6_outs_0_valid,
      index_ready => fork6_outs_0_ready,
      ins(0) => buffer0_outs,
      ins(1) => cond_br15_trueOut,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => cond_br15_trueOut_valid,
      ins_ready(0) => buffer0_outs_ready,
      ins_ready(1) => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer7 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer8 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer7_outs,
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(2, 5)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  extsi32 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork4_outs_1,
      ins_valid => fork4_outs_1_valid,
      ins_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi32_outs,
      outs_valid => extsi32_outs_valid,
      outs_ready => extsi32_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi32_outs,
      ins_valid => extsi32_outs_valid,
      ins_ready => extsi32_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => cond_br16_trueOut_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs(3) => fork6_outs_3,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_valid(3) => fork6_outs_3_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready,
      outs_ready(3) => fork6_outs_3_ready
    );

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_valid(2) => lazy_fork0_outs_2_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready,
      outs_ready(2) => lazy_fork0_outs_2_ready
    );

  buffer10 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => lazy_fork0_outs_2_valid,
      ins_ready => lazy_fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  constant2 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => buffer10_outs_valid,
      ctrl_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  extsi3 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant29 : entity work.handshake_constant_2(arch) generic map(3)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant29_outs,
      outs_valid => constant29_outs_valid,
      outs_ready => constant29_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant29_outs,
      ins_valid => constant29_outs_valid,
      ins_ready => constant29_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork5_outs_0,
      lhs_valid => fork5_outs_0_valid,
      lhs_ready => fork5_outs_0_ready,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer12 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  trunci0 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork5_outs_1,
      lhs_valid => fork5_outs_1_valid,
      lhs_ready => fork5_outs_1_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer13 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  addi19 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci0_outs,
      lhs_valid => trunci0_outs_valid,
      lhs_ready => trunci0_outs_ready,
      rhs => trunci1_outs,
      rhs_valid => trunci1_outs_valid,
      rhs_ready => trunci1_outs_ready,
      clk => clk,
      rst => rst,
      result => addi19_result,
      result_valid => addi19_result_valid,
      result_ready => addi19_result_ready
    );

  buffer14 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi19_result,
      ins_valid => addi19_result_valid,
      ins_ready => addi19_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi31_outs,
      lhs_valid => extsi31_outs_valid,
      lhs_ready => extsi31_outs_ready,
      rhs => buffer14_outs,
      rhs_valid => buffer14_outs_valid,
      rhs_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer11 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => extsi3_outs,
      ins_valid => extsi3_outs_valid,
      ins_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer15 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer15_outs,
      addrIn_valid => buffer15_outs_valid,
      addrIn_ready => buffer15_outs_ready,
      dataIn => buffer11_outs,
      dataIn_valid => buffer11_outs_valid,
      dataIn_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  extsi28 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => fork7_outs_0,
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi28_outs,
      outs_valid => extsi28_outs_valid,
      outs_ready => extsi28_outs_ready
    );

  buffer4 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer6 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer9 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_1_valid,
      ins_ready => lazy_fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork14_outs_2,
      index_valid => fork14_outs_2_valid,
      index_ready => fork14_outs_2_ready,
      ins(0) => extsi28_outs,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => extsi28_outs_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => extsi28_outs_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  buffer17 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer18 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(3, 5)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready
    );

  extsi33 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi33_outs,
      outs_valid => extsi33_outs_valid,
      outs_ready => extsi33_outs_ready
    );

  extsi34 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => fork8_outs_1,
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi34_outs,
      outs_valid => extsi34_outs_valid,
      outs_ready => extsi34_outs_ready
    );

  extsi35 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork8_outs_2,
      ins_valid => fork8_outs_2_valid,
      ins_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi35_outs,
      outs_valid => extsi35_outs_valid,
      outs_ready => extsi35_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi35_outs,
      ins_valid => extsi35_outs_valid,
      ins_ready => extsi35_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_3,
      index_valid => fork14_outs_3_valid,
      index_ready => fork14_outs_3_ready,
      ins(0) => buffer5_outs,
      ins(1) => cond_br7_trueOut,
      ins_valid(0) => buffer5_outs_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => buffer5_outs_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  buffer19 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  buffer16 : entity work.oehb(arch) generic map(32)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_4,
      index_valid => fork14_outs_4_valid,
      index_ready => fork14_outs_4_ready,
      ins(0) => buffer16_outs,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => buffer16_outs_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => buffer16_outs_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  mux10 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork14_outs_0,
      index_valid => fork14_outs_0_valid,
      index_ready => fork14_outs_0_ready,
      ins(0) => fork4_outs_0,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => fork4_outs_0_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => fork4_outs_0_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  buffer23 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux10_outs,
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  buffer24 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer23_outs,
      ins_valid => buffer23_outs_valid,
      ins_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 5)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  extsi36 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi36_outs,
      outs_valid => extsi36_outs_valid,
      outs_ready => extsi36_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(6, 32)
    port map(
      ins => extsi36_outs,
      ins_valid => extsi36_outs_valid,
      ins_ready => extsi36_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs(3) => fork12_outs_3,
      outs(4) => fork12_outs_4,
      outs(5) => fork12_outs_5,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_valid(3) => fork12_outs_3_valid,
      outs_valid(4) => fork12_outs_4_valid,
      outs_valid(5) => fork12_outs_5_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready,
      outs_ready(3) => fork12_outs_3_ready,
      outs_ready(4) => fork12_outs_4_ready,
      outs_ready(5) => fork12_outs_5_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork14_outs_1,
      index_valid => fork14_outs_1_valid,
      index_ready => fork14_outs_1_ready,
      ins(0) => fork3_outs_1,
      ins(1) => cond_br10_trueOut,
      ins_valid(0) => fork3_outs_1_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => fork3_outs_1_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  buffer25 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux11_outs,
      ins_valid => mux11_outs_valid,
      ins_ready => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  buffer26 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(4, 5)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs(3) => fork13_outs_3,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_valid(3) => fork13_outs_3_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready,
      outs_ready(3) => fork13_outs_3_ready
    );

  extsi37 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork13_outs_0,
      ins_valid => fork13_outs_0_valid,
      ins_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi37_outs,
      outs_valid => extsi37_outs_valid,
      outs_ready => extsi37_outs_ready
    );

  extsi38 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork13_outs_1,
      ins_valid => fork13_outs_1_valid,
      ins_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi38_outs,
      outs_valid => extsi38_outs_valid,
      outs_ready => extsi38_outs_ready
    );

  extsi39 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork13_outs_2,
      ins_valid => fork13_outs_2_valid,
      ins_ready => fork13_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi39_outs,
      outs_valid => extsi39_outs_valid,
      outs_ready => extsi39_outs_ready
    );

  control_merge2 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => buffer9_outs_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => buffer9_outs_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs(3) => fork14_outs_3,
      outs(4) => fork14_outs_4,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_valid(3) => fork14_outs_3_valid,
      outs_valid(4) => fork14_outs_4_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready,
      outs_ready(3) => fork14_outs_3_ready,
      outs_ready(4) => fork14_outs_4_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork1_outs_0_valid,
      outs_valid(1) => lazy_fork1_outs_1_valid,
      outs_ready(0) => lazy_fork1_outs_0_ready,
      outs_ready(1) => lazy_fork1_outs_1_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant30 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant30_outs,
      outs_valid => constant30_outs_valid,
      outs_ready => constant30_outs_ready
    );

  extsi40 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant30_outs,
      ins_valid => constant30_outs_valid,
      ins_ready => constant30_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi40_outs,
      outs_valid => extsi40_outs_valid,
      outs_ready => extsi40_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant31 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant31_outs,
      outs_valid => constant31_outs_valid,
      outs_ready => constant31_outs_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant31_outs,
      ins_valid => constant31_outs_valid,
      ins_ready => constant31_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  extsi41 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi41_outs,
      outs_valid => extsi41_outs_valid,
      outs_ready => extsi41_outs_ready
    );

  extsi7 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi7_outs,
      ins_valid => extsi7_outs_valid,
      ins_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs(2) => fork16_outs_2,
      outs(3) => fork16_outs_3,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_valid(2) => fork16_outs_2_valid,
      outs_valid(3) => fork16_outs_3_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready,
      outs_ready(2) => fork16_outs_2_ready,
      outs_ready(3) => fork16_outs_3_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant32 : entity work.handshake_constant_2(arch) generic map(3)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant32_outs,
      outs_valid => constant32_outs_valid,
      outs_ready => constant32_outs_ready
    );

  extsi8 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant32_outs,
      ins_valid => constant32_outs_valid,
      ins_ready => constant32_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi8_outs,
      ins_valid => extsi8_outs_valid,
      ins_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs(3) => fork17_outs_3,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_valid(3) => fork17_outs_3_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready,
      outs_ready(3) => fork17_outs_3_ready
    );

  shli2 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_0,
      lhs_valid => fork12_outs_0_valid,
      lhs_ready => fork12_outs_0_ready,
      rhs => fork16_outs_0,
      rhs_valid => fork16_outs_0_valid,
      rhs_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  buffer28 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer28_outs,
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => fork17_outs_0,
      rhs_valid => fork17_outs_0_valid,
      rhs_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer29 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer29_outs,
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  addi20 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci2_outs,
      lhs_valid => trunci2_outs_valid,
      lhs_ready => trunci2_outs_ready,
      rhs => trunci3_outs,
      rhs_valid => trunci3_outs_valid,
      rhs_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      result => addi20_result,
      result_valid => addi20_result_valid,
      result_ready => addi20_result_ready
    );

  buffer30 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi20_result,
      ins_valid => addi20_result_valid,
      ins_ready => addi20_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi33_outs,
      lhs_valid => extsi33_outs_valid,
      lhs_ready => extsi33_outs_ready,
      rhs => buffer30_outs,
      rhs_valid => buffer30_outs_valid,
      rhs_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  load0 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => addi3_result,
      addrIn_valid => addi3_result_valid,
      addrIn_ready => addi3_result_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => load0_dataOut,
      rhs_valid => load0_dataOut_valid,
      rhs_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  shli4 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork9_outs_0,
      lhs_valid => fork9_outs_0_valid,
      lhs_ready => fork9_outs_0_ready,
      rhs => fork16_outs_1,
      rhs_valid => fork16_outs_1_valid,
      rhs_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli4_result,
      result_valid => shli4_result_valid,
      result_ready => shli4_result_ready
    );

  buffer31 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli4_result,
      ins_valid => shli4_result_valid,
      ins_ready => shli4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  trunci4 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  shli5 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork9_outs_1,
      lhs_valid => fork9_outs_1_valid,
      lhs_ready => fork9_outs_1_ready,
      rhs => fork17_outs_1,
      rhs_valid => fork17_outs_1_valid,
      rhs_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli5_result,
      result_valid => shli5_result_valid,
      result_ready => shli5_result_ready
    );

  buffer32 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli5_result,
      ins_valid => shli5_result_valid,
      ins_ready => shli5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  trunci5 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer32_outs,
      ins_valid => buffer32_outs_valid,
      ins_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  addi21 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci4_outs,
      lhs_valid => trunci4_outs_valid,
      lhs_ready => trunci4_outs_ready,
      rhs => trunci5_outs,
      rhs_valid => trunci5_outs_valid,
      rhs_ready => trunci5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi21_result,
      result_valid => addi21_result_valid,
      result_ready => addi21_result_ready
    );

  buffer33 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi21_result,
      ins_valid => addi21_result_valid,
      ins_ready => addi21_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  addi4 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi37_outs,
      lhs_valid => extsi37_outs_valid,
      lhs_ready => extsi37_outs_ready,
      rhs => buffer33_outs,
      rhs_valid => buffer33_outs_valid,
      rhs_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  load1 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => addi4_result,
      addrIn_valid => addi4_result_valid,
      addrIn_ready => addi4_result_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  muli1 : entity work.muli(arch) generic map(32)
    port map(
      lhs => muli0_result,
      lhs_valid => muli0_result_valid,
      lhs_ready => muli0_result_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  shli6 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_2,
      lhs_valid => fork12_outs_2_valid,
      lhs_ready => fork12_outs_2_ready,
      rhs => fork16_outs_2,
      rhs_valid => fork16_outs_2_valid,
      rhs_ready => fork16_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli6_result,
      result_valid => shli6_result_valid,
      result_ready => shli6_result_ready
    );

  buffer34 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli6_result,
      ins_valid => shli6_result_valid,
      ins_ready => shli6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer34_outs,
      ins_valid => buffer34_outs_valid,
      ins_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  shli7 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_3,
      lhs_valid => fork12_outs_3_valid,
      lhs_ready => fork12_outs_3_ready,
      rhs => fork17_outs_2,
      rhs_valid => fork17_outs_2_valid,
      rhs_ready => fork17_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli7_result,
      result_valid => shli7_result_valid,
      result_ready => shli7_result_ready
    );

  buffer35 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli7_result,
      ins_valid => shli7_result_valid,
      ins_ready => shli7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  addi22 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci6_outs,
      lhs_valid => trunci6_outs_valid,
      lhs_ready => trunci6_outs_ready,
      rhs => trunci7_outs,
      rhs_valid => trunci7_outs_valid,
      rhs_ready => trunci7_outs_ready,
      clk => clk,
      rst => rst,
      result => addi22_result,
      result_valid => addi22_result_valid,
      result_ready => addi22_result_ready
    );

  buffer36 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi22_result,
      ins_valid => addi22_result_valid,
      ins_ready => addi22_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  addi5 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi38_outs,
      lhs_valid => extsi38_outs_valid,
      lhs_ready => extsi38_outs_ready,
      rhs => buffer36_outs,
      rhs_valid => buffer36_outs_valid,
      rhs_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer37 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer37_outs,
      addrIn_valid => buffer37_outs_valid,
      addrIn_ready => buffer37_outs_ready,
      dataFromMem => lsq3_ldData_0,
      dataFromMem_valid => lsq3_ldData_0_valid,
      dataFromMem_ready => lsq3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => muli1_result,
      rhs_valid => muli1_result_valid,
      rhs_ready => muli1_result_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  shli8 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_4,
      lhs_valid => fork12_outs_4_valid,
      lhs_ready => fork12_outs_4_ready,
      rhs => fork16_outs_3,
      rhs_valid => fork16_outs_3_valid,
      rhs_ready => fork16_outs_3_ready,
      clk => clk,
      rst => rst,
      result => shli8_result,
      result_valid => shli8_result_valid,
      result_ready => shli8_result_ready
    );

  buffer39 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli8_result,
      ins_valid => shli8_result_valid,
      ins_ready => shli8_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  trunci8 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer39_outs,
      ins_valid => buffer39_outs_valid,
      ins_ready => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  shli9 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_5,
      lhs_valid => fork12_outs_5_valid,
      lhs_ready => fork12_outs_5_ready,
      rhs => fork17_outs_3,
      rhs_valid => fork17_outs_3_valid,
      rhs_ready => fork17_outs_3_ready,
      clk => clk,
      rst => rst,
      result => shli9_result,
      result_valid => shli9_result_valid,
      result_ready => shli9_result_ready
    );

  buffer40 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli9_result,
      ins_valid => shli9_result_valid,
      ins_ready => shli9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  trunci9 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer40_outs,
      ins_valid => buffer40_outs_valid,
      ins_ready => buffer40_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  addi23 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci8_outs,
      lhs_valid => trunci8_outs_valid,
      lhs_ready => trunci8_outs_ready,
      rhs => trunci9_outs,
      rhs_valid => trunci9_outs_valid,
      rhs_ready => trunci9_outs_ready,
      clk => clk,
      rst => rst,
      result => addi23_result,
      result_valid => addi23_result_valid,
      result_ready => addi23_result_ready
    );

  buffer41 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi23_result,
      ins_valid => addi23_result_valid,
      ins_ready => addi23_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  addi6 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi39_outs,
      lhs_valid => extsi39_outs_valid,
      lhs_ready => extsi39_outs_ready,
      rhs => buffer41_outs,
      rhs_valid => buffer41_outs_valid,
      rhs_ready => buffer41_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer38 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  buffer42 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer42_outs,
      addrIn_valid => buffer42_outs_valid,
      addrIn_ready => buffer42_outs_ready,
      dataIn => buffer38_outs,
      dataIn_valid => buffer38_outs_valid,
      dataIn_ready => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  addi13 : entity work.addi(arch) generic map(6)
    port map(
      lhs => extsi34_outs,
      lhs_valid => extsi34_outs_valid,
      lhs_ready => extsi34_outs_ready,
      rhs => extsi41_outs,
      rhs_valid => extsi41_outs_valid,
      rhs_ready => extsi41_outs_ready,
      clk => clk,
      rst => rst,
      result => addi13_result,
      result_valid => addi13_result_valid,
      result_ready => addi13_result_ready
    );

  buffer43 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi13_result,
      ins_valid => addi13_result_valid,
      ins_ready => addi13_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer43_outs,
      ins_valid => buffer43_outs_valid,
      ins_ready => buffer43_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  trunci10 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci10_outs,
      outs_valid => trunci10_outs_valid,
      outs_ready => trunci10_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork18_outs_1,
      lhs_valid => fork18_outs_1_valid,
      lhs_ready => fork18_outs_1_ready,
      rhs => extsi40_outs,
      rhs_valid => extsi40_outs_valid,
      rhs_ready => extsi40_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer44 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => buffer44_outs,
      ins_valid => buffer44_outs_valid,
      ins_ready => buffer44_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs(2) => fork19_outs_2,
      outs(3) => fork19_outs_3,
      outs(4) => fork19_outs_4,
      outs(5) => fork19_outs_5,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_valid(2) => fork19_outs_2_valid,
      outs_valid(3) => fork19_outs_3_valid,
      outs_valid(4) => fork19_outs_4_valid,
      outs_valid(5) => fork19_outs_5_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready,
      outs_ready(2) => fork19_outs_2_ready,
      outs_ready(3) => fork19_outs_3_ready,
      outs_ready(4) => fork19_outs_4_ready,
      outs_ready(5) => fork19_outs_5_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork19_outs_0,
      condition_valid => fork19_outs_0_valid,
      condition_ready => fork19_outs_0_ready,
      data => trunci10_outs,
      data_valid => trunci10_outs_valid,
      data_ready => trunci10_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer20 : entity work.oehb(arch) generic map(32)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork19_outs_3,
      condition_valid => fork19_outs_3_valid,
      condition_ready => fork19_outs_3_ready,
      data => buffer20_outs,
      data_valid => buffer20_outs_valid,
      data_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  buffer21 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  buffer22 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork19_outs_4,
      condition_valid => fork19_outs_4_valid,
      condition_ready => fork19_outs_4_ready,
      data => buffer22_outs,
      data_valid => buffer22_outs_valid,
      data_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  cond_br9 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork19_outs_1,
      condition_valid => fork19_outs_1_valid,
      condition_ready => fork19_outs_1_ready,
      data => fork11_outs_0,
      data_valid => fork11_outs_0_valid,
      data_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork19_outs_2,
      condition_valid => fork19_outs_2_valid,
      condition_ready => fork19_outs_2_ready,
      data => fork13_outs_3,
      data_valid => fork13_outs_3_valid,
      data_ready => fork13_outs_3_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  buffer27 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_1_valid,
      ins_ready => lazy_fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  cond_br11 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork19_outs_5,
      condition_valid => fork19_outs_5_valid,
      condition_ready => fork19_outs_5_ready,
      data_valid => buffer27_outs_valid,
      data_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  extsi42 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi42_outs,
      outs_valid => extsi42_outs_valid,
      outs_ready => extsi42_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant33 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant33_outs,
      outs_valid => constant33_outs_valid,
      outs_ready => constant33_outs_ready
    );

  extsi43 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant33_outs,
      ins_valid => constant33_outs_valid,
      ins_ready => constant33_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi43_outs,
      outs_valid => extsi43_outs_valid,
      outs_ready => extsi43_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant34 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant34_outs,
      outs_valid => constant34_outs_valid,
      outs_ready => constant34_outs_ready
    );

  extsi44 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant34_outs,
      ins_valid => constant34_outs_valid,
      ins_ready => constant34_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi44_outs,
      outs_valid => extsi44_outs_valid,
      outs_ready => extsi44_outs_ready
    );

  addi14 : entity work.addi(arch) generic map(6)
    port map(
      lhs => extsi42_outs,
      lhs_valid => extsi42_outs_valid,
      lhs_ready => extsi42_outs_ready,
      rhs => extsi44_outs,
      rhs_valid => extsi44_outs_valid,
      rhs_ready => extsi44_outs_ready,
      clk => clk,
      rst => rst,
      result => addi14_result,
      result_valid => addi14_result_valid,
      result_ready => addi14_result_ready
    );

  buffer45 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi14_result,
      ins_valid => addi14_result_valid,
      ins_ready => addi14_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer45_outs,
      ins_valid => buffer45_outs_valid,
      ins_ready => buffer45_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  trunci11 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci11_outs,
      outs_valid => trunci11_outs_valid,
      outs_ready => trunci11_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork20_outs_1,
      lhs_valid => fork20_outs_1_valid,
      lhs_ready => fork20_outs_1_ready,
      rhs => extsi43_outs,
      rhs_valid => extsi43_outs_valid,
      rhs_ready => extsi43_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer46 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => buffer46_outs,
      ins_valid => buffer46_outs_valid,
      ins_ready => buffer46_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs(2) => fork21_outs_2,
      outs(3) => fork21_outs_3,
      outs(4) => fork21_outs_4,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_valid(2) => fork21_outs_2_valid,
      outs_valid(3) => fork21_outs_3_valid,
      outs_valid(4) => fork21_outs_4_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready,
      outs_ready(2) => fork21_outs_2_ready,
      outs_ready(3) => fork21_outs_3_ready,
      outs_ready(4) => fork21_outs_4_ready
    );

  cond_br12 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork21_outs_0,
      condition_valid => fork21_outs_0_valid,
      condition_ready => fork21_outs_0_ready,
      data => trunci11_outs,
      data_valid => trunci11_outs_valid,
      data_ready => trunci11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br12_trueOut,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut => cond_br12_falseOut,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br12_falseOut,
      ins_valid => cond_br12_falseOut_valid,
      ins_ready => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br13 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork21_outs_2,
      condition_valid => fork21_outs_2_valid,
      condition_ready => fork21_outs_2_ready,
      data => cond_br7_falseOut,
      data_valid => cond_br7_falseOut_valid,
      data_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  cond_br14 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork21_outs_3,
      condition_valid => fork21_outs_3_valid,
      condition_ready => fork21_outs_3_ready,
      data => cond_br8_falseOut,
      data_valid => cond_br8_falseOut_valid,
      data_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br14_trueOut,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut => cond_br14_falseOut,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  cond_br15 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork21_outs_1,
      condition_valid => fork21_outs_1_valid,
      condition_ready => fork21_outs_1_ready,
      data => cond_br9_falseOut,
      data_valid => cond_br9_falseOut_valid,
      data_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  cond_br16 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork21_outs_4,
      condition_valid => fork21_outs_4_valid,
      condition_ready => fork21_outs_4_ready,
      data_valid => cond_br11_falseOut_valid,
      data_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  extsi45 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => cond_br15_falseOut,
      ins_valid => cond_br15_falseOut_valid,
      ins_ready => cond_br15_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi45_outs,
      outs_valid => extsi45_outs_valid,
      outs_ready => extsi45_outs_ready
    );

  fork22 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br16_falseOut_valid,
      ins_ready => cond_br16_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  constant35 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork22_outs_0_valid,
      ctrl_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant35_outs,
      outs_valid => constant35_outs_valid,
      outs_ready => constant35_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant36 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant36_outs,
      outs_valid => constant36_outs_valid,
      outs_ready => constant36_outs_ready
    );

  extsi46 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant36_outs,
      ins_valid => constant36_outs_valid,
      ins_ready => constant36_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi46_outs,
      outs_valid => extsi46_outs_valid,
      outs_ready => extsi46_outs_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant37 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant37_outs,
      outs_valid => constant37_outs_valid,
      outs_ready => constant37_outs_ready
    );

  extsi47 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant37_outs,
      ins_valid => constant37_outs_valid,
      ins_ready => constant37_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi47_outs,
      outs_valid => extsi47_outs_valid,
      outs_ready => extsi47_outs_ready
    );

  buffer48 : entity work.oehb(arch) generic map(6)
    port map(
      ins => extsi45_outs,
      ins_valid => extsi45_outs_valid,
      ins_ready => extsi45_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  addi15 : entity work.addi(arch) generic map(6)
    port map(
      lhs => buffer48_outs,
      lhs_valid => buffer48_outs_valid,
      lhs_ready => buffer48_outs_ready,
      rhs => extsi47_outs,
      rhs_valid => extsi47_outs_valid,
      rhs_ready => extsi47_outs_ready,
      clk => clk,
      rst => rst,
      result => addi15_result,
      result_valid => addi15_result_valid,
      result_ready => addi15_result_ready
    );

  buffer49 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi15_result,
      ins_valid => addi15_result_valid,
      ins_ready => addi15_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer49_outs,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer49_outs,
      ins_valid => buffer49_outs_valid,
      ins_ready => buffer49_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  trunci12 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork23_outs_0,
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci12_outs,
      outs_valid => trunci12_outs_valid,
      outs_ready => trunci12_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork23_outs_1,
      lhs_valid => fork23_outs_1_valid,
      lhs_ready => fork23_outs_1_ready,
      rhs => extsi46_outs,
      rhs_valid => extsi46_outs_valid,
      rhs_ready => extsi46_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer50 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  fork24 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => buffer50_outs,
      ins_valid => buffer50_outs_valid,
      ins_ready => buffer50_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs(2) => fork24_outs_2,
      outs(3) => fork24_outs_3,
      outs(4) => fork24_outs_4,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_valid(2) => fork24_outs_2_valid,
      outs_valid(3) => fork24_outs_3_valid,
      outs_valid(4) => fork24_outs_4_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready,
      outs_ready(2) => fork24_outs_2_ready,
      outs_ready(3) => fork24_outs_3_ready,
      outs_ready(4) => fork24_outs_4_ready
    );

  cond_br17 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork24_outs_0,
      condition_valid => fork24_outs_0_valid,
      condition_ready => fork24_outs_0_ready,
      data => trunci12_outs,
      data_valid => trunci12_outs_valid,
      data_ready => trunci12_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br17_trueOut,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut => cond_br17_falseOut,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  sink4 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br17_falseOut,
      ins_valid => cond_br17_falseOut_valid,
      ins_ready => cond_br17_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br18 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork24_outs_1,
      condition_valid => fork24_outs_1_valid,
      condition_ready => fork24_outs_1_ready,
      data => cond_br13_falseOut,
      data_valid => cond_br13_falseOut_valid,
      data_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br18_trueOut,
      trueOut_valid => cond_br18_trueOut_valid,
      trueOut_ready => cond_br18_trueOut_ready,
      falseOut => cond_br18_falseOut,
      falseOut_valid => cond_br18_falseOut_valid,
      falseOut_ready => cond_br18_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br18_falseOut,
      ins_valid => cond_br18_falseOut_valid,
      ins_ready => cond_br18_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer47 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br14_falseOut,
      ins_valid => cond_br14_falseOut_valid,
      ins_ready => cond_br14_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  cond_br19 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork24_outs_2,
      condition_valid => fork24_outs_2_valid,
      condition_ready => fork24_outs_2_ready,
      data => buffer47_outs,
      data_valid => buffer47_outs_valid,
      data_ready => buffer47_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br19_trueOut,
      trueOut_valid => cond_br19_trueOut_valid,
      trueOut_ready => cond_br19_trueOut_ready,
      falseOut => cond_br19_falseOut,
      falseOut_valid => cond_br19_falseOut_valid,
      falseOut_ready => cond_br19_falseOut_ready
    );

  cond_br20 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork24_outs_3,
      condition_valid => fork24_outs_3_valid,
      condition_ready => fork24_outs_3_ready,
      data_valid => fork22_outs_1_valid,
      data_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br20_trueOut_valid,
      trueOut_ready => cond_br20_trueOut_ready,
      falseOut_valid => cond_br20_falseOut_valid,
      falseOut_ready => cond_br20_falseOut_ready
    );

  cond_br21 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork24_outs_4,
      condition_valid => fork24_outs_4_valid,
      condition_ready => fork24_outs_4_ready,
      data => constant35_outs,
      data_valid => constant35_outs_valid,
      data_ready => constant35_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br21_trueOut,
      trueOut_valid => cond_br21_trueOut_valid,
      trueOut_ready => cond_br21_trueOut_ready,
      falseOut => cond_br21_falseOut,
      falseOut_valid => cond_br21_falseOut_valid,
      falseOut_ready => cond_br21_falseOut_ready
    );

  sink6 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br21_trueOut,
      ins_valid => cond_br21_trueOut_valid,
      ins_ready => cond_br21_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi27 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => cond_br21_falseOut,
      ins_valid => cond_br21_falseOut_valid,
      ins_ready => cond_br21_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi27_outs,
      outs_valid => extsi27_outs_valid,
      outs_ready => extsi27_outs_ready
    );

  mux12 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork25_outs_0,
      index_valid => fork25_outs_0_valid,
      index_ready => fork25_outs_0_ready,
      ins(0) => extsi27_outs,
      ins(1) => cond_br31_trueOut,
      ins_valid(0) => extsi27_outs_valid,
      ins_valid(1) => cond_br31_trueOut_valid,
      ins_ready(0) => extsi27_outs_ready,
      ins_ready(1) => cond_br31_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux12_outs,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  mux13 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork25_outs_1,
      index_valid => fork25_outs_1_valid,
      index_ready => fork25_outs_1_ready,
      ins(0) => cond_br19_falseOut,
      ins(1) => cond_br32_trueOut,
      ins_valid(0) => cond_br19_falseOut_valid,
      ins_valid(1) => cond_br32_trueOut_valid,
      ins_ready(0) => cond_br19_falseOut_ready,
      ins_ready(1) => cond_br32_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux13_outs,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  control_merge5 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => cond_br20_falseOut_valid,
      ins_valid(1) => cond_br33_trueOut_valid,
      ins_ready(0) => cond_br20_falseOut_ready,
      ins_ready(1) => cond_br33_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge5_outs_valid,
      outs_ready => control_merge5_outs_ready,
      index => control_merge5_index,
      index_valid => control_merge5_index_valid,
      index_ready => control_merge5_index_ready
    );

  fork25 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge5_index,
      ins_valid => control_merge5_index_valid,
      ins_ready => control_merge5_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

  fork26 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge5_outs_valid,
      ins_ready => control_merge5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready
    );

  constant38 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork26_outs_0_valid,
      ctrl_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant38_outs,
      outs_valid => constant38_outs_valid,
      outs_ready => constant38_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => constant38_outs,
      ins_valid => constant38_outs_valid,
      ins_ready => constant38_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  buffer53 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux13_outs,
      ins_valid => mux13_outs_valid,
      ins_ready => mux13_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer53_outs,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  buffer52 : entity work.tehb(arch) generic map(5)
    port map(
      ins => mux12_outs,
      ins_valid => mux12_outs_valid,
      ins_ready => mux12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer52_outs,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  mux14 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork31_outs_1,
      index_valid => fork31_outs_1_valid,
      index_ready => fork31_outs_1_ready,
      ins(0) => extsi26_outs,
      ins(1) => cond_br27_trueOut,
      ins_valid(0) => extsi26_outs_valid,
      ins_valid(1) => cond_br27_trueOut_valid,
      ins_ready(0) => extsi26_outs_ready,
      ins_ready(1) => cond_br27_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux14_outs,
      outs_valid => mux14_outs_valid,
      outs_ready => mux14_outs_ready
    );

  buffer54 : entity work.tehb(arch) generic map(5)
    port map(
      ins => mux14_outs,
      ins_valid => mux14_outs_valid,
      ins_ready => mux14_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer54_outs,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  fork27 : entity work.handshake_fork(arch) generic map(3, 5)
    port map(
      ins => buffer54_outs,
      ins_valid => buffer54_outs_valid,
      ins_ready => buffer54_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork27_outs_0,
      outs(1) => fork27_outs_1,
      outs(2) => fork27_outs_2,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_valid(2) => fork27_outs_2_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready,
      outs_ready(2) => fork27_outs_2_ready
    );

  extsi48 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork27_outs_0,
      ins_valid => fork27_outs_0_valid,
      ins_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi48_outs,
      outs_valid => extsi48_outs_valid,
      outs_ready => extsi48_outs_ready
    );

  extsi49 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork27_outs_1,
      ins_valid => fork27_outs_1_valid,
      ins_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi49_outs,
      outs_valid => extsi49_outs_valid,
      outs_ready => extsi49_outs_ready
    );

  mux15 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork31_outs_2,
      index_valid => fork31_outs_2_valid,
      index_ready => fork31_outs_2_ready,
      ins(0) => buffer53_outs,
      ins(1) => cond_br28_trueOut,
      ins_valid(0) => buffer53_outs_valid,
      ins_valid(1) => cond_br28_trueOut_valid,
      ins_ready(0) => buffer53_outs_ready,
      ins_ready(1) => cond_br28_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux15_outs,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  buffer55 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux15_outs,
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer55_outs,
      outs_valid => buffer55_outs_valid,
      outs_ready => buffer55_outs_ready
    );

  buffer56 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer55_outs,
      ins_valid => buffer55_outs_valid,
      ins_ready => buffer55_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer56_outs,
      outs_valid => buffer56_outs_valid,
      outs_ready => buffer56_outs_ready
    );

  fork28 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer56_outs,
      ins_valid => buffer56_outs_valid,
      ins_ready => buffer56_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready
    );

  mux16 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork31_outs_0,
      index_valid => fork31_outs_0_valid,
      index_ready => fork31_outs_0_ready,
      ins(0) => buffer52_outs,
      ins(1) => cond_br29_trueOut,
      ins_valid(0) => buffer52_outs_valid,
      ins_valid(1) => cond_br29_trueOut_valid,
      ins_ready(0) => buffer52_outs_ready,
      ins_ready(1) => cond_br29_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux16_outs,
      outs_valid => mux16_outs_valid,
      outs_ready => mux16_outs_ready
    );

  buffer57 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux16_outs,
      ins_valid => mux16_outs_valid,
      ins_ready => mux16_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer57_outs,
      outs_valid => buffer57_outs_valid,
      outs_ready => buffer57_outs_ready
    );

  buffer58 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer57_outs,
      ins_valid => buffer57_outs_valid,
      ins_ready => buffer57_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer58_outs,
      outs_valid => buffer58_outs_valid,
      outs_ready => buffer58_outs_ready
    );

  fork29 : entity work.handshake_fork(arch) generic map(2, 5)
    port map(
      ins => buffer58_outs,
      ins_valid => buffer58_outs_valid,
      ins_ready => buffer58_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork29_outs_0,
      outs(1) => fork29_outs_1,
      outs_valid(0) => fork29_outs_0_valid,
      outs_valid(1) => fork29_outs_1_valid,
      outs_ready(0) => fork29_outs_0_ready,
      outs_ready(1) => fork29_outs_1_ready
    );

  extsi50 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork29_outs_1,
      ins_valid => fork29_outs_1_valid,
      ins_ready => fork29_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi50_outs,
      outs_valid => extsi50_outs_valid,
      outs_ready => extsi50_outs_ready
    );

  fork30 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi50_outs,
      ins_valid => extsi50_outs_valid,
      ins_ready => extsi50_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork30_outs_0,
      outs(1) => fork30_outs_1,
      outs(2) => fork30_outs_2,
      outs(3) => fork30_outs_3,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_valid(2) => fork30_outs_2_valid,
      outs_valid(3) => fork30_outs_3_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready,
      outs_ready(2) => fork30_outs_2_ready,
      outs_ready(3) => fork30_outs_3_ready
    );

  control_merge6 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork26_outs_1_valid,
      ins_valid(1) => cond_br30_trueOut_valid,
      ins_ready(0) => fork26_outs_1_ready,
      ins_ready(1) => cond_br30_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge6_outs_valid,
      outs_ready => control_merge6_outs_ready,
      index => control_merge6_index,
      index_valid => control_merge6_index_valid,
      index_ready => control_merge6_index_ready
    );

  fork31 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge6_index,
      ins_valid => control_merge6_index_valid,
      ins_ready => control_merge6_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork31_outs_0,
      outs(1) => fork31_outs_1,
      outs(2) => fork31_outs_2,
      outs_valid(0) => fork31_outs_0_valid,
      outs_valid(1) => fork31_outs_1_valid,
      outs_valid(2) => fork31_outs_2_valid,
      outs_ready(0) => fork31_outs_0_ready,
      outs_ready(1) => fork31_outs_1_ready,
      outs_ready(2) => fork31_outs_2_ready
    );

  lazy_fork2 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => control_merge6_outs_valid,
      ins_ready => control_merge6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork2_outs_0_valid,
      outs_valid(1) => lazy_fork2_outs_1_valid,
      outs_valid(2) => lazy_fork2_outs_2_valid,
      outs_ready(0) => lazy_fork2_outs_0_ready,
      outs_ready(1) => lazy_fork2_outs_1_ready,
      outs_ready(2) => lazy_fork2_outs_2_ready
    );

  buffer60 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => lazy_fork2_outs_2_valid,
      ins_ready => lazy_fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  constant39 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => buffer60_outs_valid,
      ctrl_ready => buffer60_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant39_outs,
      outs_valid => constant39_outs_valid,
      outs_ready => constant39_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant40 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant40_outs,
      outs_valid => constant40_outs_valid,
      outs_ready => constant40_outs_ready
    );

  extsi16 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant40_outs,
      ins_valid => constant40_outs_valid,
      ins_ready => constant40_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  fork32 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi16_outs,
      ins_valid => extsi16_outs_valid,
      ins_ready => extsi16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork32_outs_0,
      outs(1) => fork32_outs_1,
      outs_valid(0) => fork32_outs_0_valid,
      outs_valid(1) => fork32_outs_1_valid,
      outs_ready(0) => fork32_outs_0_ready,
      outs_ready(1) => fork32_outs_1_ready
    );

  source10 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  constant41 : entity work.handshake_constant_2(arch) generic map(3)
    port map(
      ctrl_valid => source10_outs_valid,
      ctrl_ready => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant41_outs,
      outs_valid => constant41_outs_valid,
      outs_ready => constant41_outs_ready
    );

  extsi17 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant41_outs,
      ins_valid => constant41_outs_valid,
      ins_ready => constant41_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  fork33 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi17_outs,
      ins_valid => extsi17_outs_valid,
      ins_ready => extsi17_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork33_outs_0,
      outs(1) => fork33_outs_1,
      outs_valid(0) => fork33_outs_0_valid,
      outs_valid(1) => fork33_outs_1_valid,
      outs_ready(0) => fork33_outs_0_ready,
      outs_ready(1) => fork33_outs_1_ready
    );

  shli10 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork30_outs_0,
      lhs_valid => fork30_outs_0_valid,
      lhs_ready => fork30_outs_0_ready,
      rhs => fork32_outs_0,
      rhs_valid => fork32_outs_0_valid,
      rhs_ready => fork32_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli10_result,
      result_valid => shli10_result_valid,
      result_ready => shli10_result_ready
    );

  buffer61 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli10_result,
      ins_valid => shli10_result_valid,
      ins_ready => shli10_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer61_outs,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  trunci13 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer61_outs,
      ins_valid => buffer61_outs_valid,
      ins_ready => buffer61_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci13_outs,
      outs_valid => trunci13_outs_valid,
      outs_ready => trunci13_outs_ready
    );

  shli11 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork30_outs_1,
      lhs_valid => fork30_outs_1_valid,
      lhs_ready => fork30_outs_1_ready,
      rhs => fork33_outs_0,
      rhs_valid => fork33_outs_0_valid,
      rhs_ready => fork33_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli11_result,
      result_valid => shli11_result_valid,
      result_ready => shli11_result_ready
    );

  buffer62 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli11_result,
      ins_valid => shli11_result_valid,
      ins_ready => shli11_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer62_outs,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  trunci14 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer62_outs,
      ins_valid => buffer62_outs_valid,
      ins_ready => buffer62_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci14_outs,
      outs_valid => trunci14_outs_valid,
      outs_ready => trunci14_outs_ready
    );

  addi24 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci13_outs,
      lhs_valid => trunci13_outs_valid,
      lhs_ready => trunci13_outs_ready,
      rhs => trunci14_outs,
      rhs_valid => trunci14_outs_valid,
      rhs_ready => trunci14_outs_ready,
      clk => clk,
      rst => rst,
      result => addi24_result,
      result_valid => addi24_result_valid,
      result_ready => addi24_result_ready
    );

  buffer63 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi24_result,
      ins_valid => addi24_result_valid,
      ins_ready => addi24_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer63_outs,
      outs_valid => buffer63_outs_valid,
      outs_ready => buffer63_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi48_outs,
      lhs_valid => extsi48_outs_valid,
      lhs_ready => extsi48_outs_ready,
      rhs => buffer63_outs,
      rhs_valid => buffer63_outs_valid,
      rhs_ready => buffer63_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer64 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer64_outs,
      outs_valid => buffer64_outs_valid,
      outs_ready => buffer64_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer64_outs,
      addrIn_valid => buffer64_outs_valid,
      addrIn_ready => buffer64_outs_ready,
      dataFromMem => lsq2_ldData_0,
      dataFromMem_valid => lsq2_ldData_0_valid,
      dataFromMem_ready => lsq2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  muli2 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load3_dataOut,
      lhs_valid => load3_dataOut_valid,
      lhs_ready => load3_dataOut_ready,
      rhs => fork28_outs_1,
      rhs_valid => fork28_outs_1_valid,
      rhs_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      result => muli2_result,
      result_valid => muli2_result_valid,
      result_ready => muli2_result_ready
    );

  shli12 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork30_outs_2,
      lhs_valid => fork30_outs_2_valid,
      lhs_ready => fork30_outs_2_ready,
      rhs => fork32_outs_1,
      rhs_valid => fork32_outs_1_valid,
      rhs_ready => fork32_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli12_result,
      result_valid => shli12_result_valid,
      result_ready => shli12_result_ready
    );

  buffer66 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli12_result,
      ins_valid => shli12_result_valid,
      ins_ready => shli12_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer66_outs,
      outs_valid => buffer66_outs_valid,
      outs_ready => buffer66_outs_ready
    );

  trunci15 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer66_outs,
      ins_valid => buffer66_outs_valid,
      ins_ready => buffer66_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci15_outs,
      outs_valid => trunci15_outs_valid,
      outs_ready => trunci15_outs_ready
    );

  shli13 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork30_outs_3,
      lhs_valid => fork30_outs_3_valid,
      lhs_ready => fork30_outs_3_ready,
      rhs => fork33_outs_1,
      rhs_valid => fork33_outs_1_valid,
      rhs_ready => fork33_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli13_result,
      result_valid => shli13_result_valid,
      result_ready => shli13_result_ready
    );

  buffer67 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli13_result,
      ins_valid => shli13_result_valid,
      ins_ready => shli13_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer67_outs,
      outs_valid => buffer67_outs_valid,
      outs_ready => buffer67_outs_ready
    );

  trunci16 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer67_outs,
      ins_valid => buffer67_outs_valid,
      ins_ready => buffer67_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci16_outs,
      outs_valid => trunci16_outs_valid,
      outs_ready => trunci16_outs_ready
    );

  addi25 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci15_outs,
      lhs_valid => trunci15_outs_valid,
      lhs_ready => trunci15_outs_ready,
      rhs => trunci16_outs,
      rhs_valid => trunci16_outs_valid,
      rhs_ready => trunci16_outs_ready,
      clk => clk,
      rst => rst,
      result => addi25_result,
      result_valid => addi25_result_valid,
      result_ready => addi25_result_ready
    );

  buffer68 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi25_result,
      ins_valid => addi25_result_valid,
      ins_ready => addi25_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer68_outs,
      outs_valid => buffer68_outs_valid,
      outs_ready => buffer68_outs_ready
    );

  addi8 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi49_outs,
      lhs_valid => extsi49_outs_valid,
      lhs_ready => extsi49_outs_ready,
      rhs => buffer68_outs,
      rhs_valid => buffer68_outs_valid,
      rhs_ready => buffer68_outs_ready,
      clk => clk,
      rst => rst,
      result => addi8_result,
      result_valid => addi8_result_valid,
      result_ready => addi8_result_ready
    );

  buffer65 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => muli2_result,
      ins_valid => muli2_result_valid,
      ins_ready => muli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer65_outs,
      outs_valid => buffer65_outs_valid,
      outs_ready => buffer65_outs_ready
    );

  buffer69 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi8_result,
      ins_valid => addi8_result_valid,
      ins_ready => addi8_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer69_outs,
      outs_valid => buffer69_outs_valid,
      outs_ready => buffer69_outs_ready
    );

  store2 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer69_outs,
      addrIn_valid => buffer69_outs_valid,
      addrIn_ready => buffer69_outs_ready,
      dataIn => buffer65_outs,
      dataIn_valid => buffer65_outs_valid,
      dataIn_ready => buffer65_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store2_addrOut,
      addrOut_valid => store2_addrOut_valid,
      addrOut_ready => store2_addrOut_ready,
      dataToMem => store2_dataToMem,
      dataToMem_valid => store2_dataToMem_valid,
      dataToMem_ready => store2_dataToMem_ready
    );

  extsi25 : entity work.extsi(arch) generic map(1, 5)
    port map(
      ins => constant39_outs,
      ins_valid => constant39_outs_valid,
      ins_ready => constant39_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  buffer59 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork2_outs_1_valid,
      ins_ready => lazy_fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  mux17 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork39_outs_2,
      index_valid => fork39_outs_2_valid,
      index_ready => fork39_outs_2_ready,
      ins(0) => extsi25_outs,
      ins(1) => cond_br22_trueOut,
      ins_valid(0) => extsi25_outs_valid,
      ins_valid(1) => cond_br22_trueOut_valid,
      ins_ready(0) => extsi25_outs_ready,
      ins_ready(1) => cond_br22_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux17_outs,
      outs_valid => mux17_outs_valid,
      outs_ready => mux17_outs_ready
    );

  buffer70 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux17_outs,
      ins_valid => mux17_outs_valid,
      ins_ready => mux17_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer70_outs,
      outs_valid => buffer70_outs_valid,
      outs_ready => buffer70_outs_ready
    );

  buffer71 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer70_outs,
      ins_valid => buffer70_outs_valid,
      ins_ready => buffer70_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer71_outs,
      outs_valid => buffer71_outs_valid,
      outs_ready => buffer71_outs_ready
    );

  fork34 : entity work.handshake_fork(arch) generic map(3, 5)
    port map(
      ins => buffer71_outs,
      ins_valid => buffer71_outs_valid,
      ins_ready => buffer71_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork34_outs_0,
      outs(1) => fork34_outs_1,
      outs(2) => fork34_outs_2,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_valid(2) => fork34_outs_2_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready,
      outs_ready(2) => fork34_outs_2_ready
    );

  extsi51 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork34_outs_0,
      ins_valid => fork34_outs_0_valid,
      ins_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi51_outs,
      outs_valid => extsi51_outs_valid,
      outs_ready => extsi51_outs_ready
    );

  extsi52 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => fork34_outs_1,
      ins_valid => fork34_outs_1_valid,
      ins_ready => fork34_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi52_outs,
      outs_valid => extsi52_outs_valid,
      outs_ready => extsi52_outs_ready
    );

  extsi53 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork34_outs_2,
      ins_valid => fork34_outs_2_valid,
      ins_ready => fork34_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi53_outs,
      outs_valid => extsi53_outs_valid,
      outs_ready => extsi53_outs_ready
    );

  fork35 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi53_outs,
      ins_valid => extsi53_outs_valid,
      ins_ready => extsi53_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready
    );

  mux18 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork39_outs_3,
      index_valid => fork39_outs_3_valid,
      index_ready => fork39_outs_3_ready,
      ins(0) => fork28_outs_0,
      ins(1) => cond_br23_trueOut,
      ins_valid(0) => fork28_outs_0_valid,
      ins_valid(1) => cond_br23_trueOut_valid,
      ins_ready(0) => fork28_outs_0_ready,
      ins_ready(1) => cond_br23_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux18_outs,
      outs_valid => mux18_outs_valid,
      outs_ready => mux18_outs_ready
    );

  mux19 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork39_outs_0,
      index_valid => fork39_outs_0_valid,
      index_ready => fork39_outs_0_ready,
      ins(0) => fork29_outs_0,
      ins(1) => cond_br24_trueOut,
      ins_valid(0) => fork29_outs_0_valid,
      ins_valid(1) => cond_br24_trueOut_valid,
      ins_ready(0) => fork29_outs_0_ready,
      ins_ready(1) => cond_br24_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux19_outs,
      outs_valid => mux19_outs_valid,
      outs_ready => mux19_outs_ready
    );

  buffer74 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux19_outs,
      ins_valid => mux19_outs_valid,
      ins_ready => mux19_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer74_outs,
      outs_valid => buffer74_outs_valid,
      outs_ready => buffer74_outs_ready
    );

  buffer75 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer74_outs,
      ins_valid => buffer74_outs_valid,
      ins_ready => buffer74_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer75_outs,
      outs_valid => buffer75_outs_valid,
      outs_ready => buffer75_outs_ready
    );

  fork36 : entity work.handshake_fork(arch) generic map(2, 5)
    port map(
      ins => buffer75_outs,
      ins_valid => buffer75_outs_valid,
      ins_ready => buffer75_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork36_outs_0,
      outs(1) => fork36_outs_1,
      outs_valid(0) => fork36_outs_0_valid,
      outs_valid(1) => fork36_outs_1_valid,
      outs_ready(0) => fork36_outs_0_ready,
      outs_ready(1) => fork36_outs_1_ready
    );

  extsi54 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => fork36_outs_1,
      ins_valid => fork36_outs_1_valid,
      ins_ready => fork36_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi54_outs,
      outs_valid => extsi54_outs_valid,
      outs_ready => extsi54_outs_ready
    );

  fork37 : entity work.handshake_fork(arch) generic map(6, 32)
    port map(
      ins => extsi54_outs,
      ins_valid => extsi54_outs_valid,
      ins_ready => extsi54_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork37_outs_0,
      outs(1) => fork37_outs_1,
      outs(2) => fork37_outs_2,
      outs(3) => fork37_outs_3,
      outs(4) => fork37_outs_4,
      outs(5) => fork37_outs_5,
      outs_valid(0) => fork37_outs_0_valid,
      outs_valid(1) => fork37_outs_1_valid,
      outs_valid(2) => fork37_outs_2_valid,
      outs_valid(3) => fork37_outs_3_valid,
      outs_valid(4) => fork37_outs_4_valid,
      outs_valid(5) => fork37_outs_5_valid,
      outs_ready(0) => fork37_outs_0_ready,
      outs_ready(1) => fork37_outs_1_ready,
      outs_ready(2) => fork37_outs_2_ready,
      outs_ready(3) => fork37_outs_3_ready,
      outs_ready(4) => fork37_outs_4_ready,
      outs_ready(5) => fork37_outs_5_ready
    );

  mux20 : entity work.mux(arch) generic map(2, 5, 1)
    port map(
      index => fork39_outs_1,
      index_valid => fork39_outs_1_valid,
      index_ready => fork39_outs_1_ready,
      ins(0) => fork27_outs_2,
      ins(1) => cond_br25_trueOut,
      ins_valid(0) => fork27_outs_2_valid,
      ins_valid(1) => cond_br25_trueOut_valid,
      ins_ready(0) => fork27_outs_2_ready,
      ins_ready(1) => cond_br25_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux20_outs,
      outs_valid => mux20_outs_valid,
      outs_ready => mux20_outs_ready
    );

  buffer76 : entity work.oehb(arch) generic map(5)
    port map(
      ins => mux20_outs,
      ins_valid => mux20_outs_valid,
      ins_ready => mux20_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer76_outs,
      outs_valid => buffer76_outs_valid,
      outs_ready => buffer76_outs_ready
    );

  buffer77 : entity work.tehb(arch) generic map(5)
    port map(
      ins => buffer76_outs,
      ins_valid => buffer76_outs_valid,
      ins_ready => buffer76_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer77_outs,
      outs_valid => buffer77_outs_valid,
      outs_ready => buffer77_outs_ready
    );

  fork38 : entity work.handshake_fork(arch) generic map(4, 5)
    port map(
      ins => buffer77_outs,
      ins_valid => buffer77_outs_valid,
      ins_ready => buffer77_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork38_outs_0,
      outs(1) => fork38_outs_1,
      outs(2) => fork38_outs_2,
      outs(3) => fork38_outs_3,
      outs_valid(0) => fork38_outs_0_valid,
      outs_valid(1) => fork38_outs_1_valid,
      outs_valid(2) => fork38_outs_2_valid,
      outs_valid(3) => fork38_outs_3_valid,
      outs_ready(0) => fork38_outs_0_ready,
      outs_ready(1) => fork38_outs_1_ready,
      outs_ready(2) => fork38_outs_2_ready,
      outs_ready(3) => fork38_outs_3_ready
    );

  extsi55 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork38_outs_0,
      ins_valid => fork38_outs_0_valid,
      ins_ready => fork38_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi55_outs,
      outs_valid => extsi55_outs_valid,
      outs_ready => extsi55_outs_ready
    );

  extsi56 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork38_outs_1,
      ins_valid => fork38_outs_1_valid,
      ins_ready => fork38_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi56_outs,
      outs_valid => extsi56_outs_valid,
      outs_ready => extsi56_outs_ready
    );

  extsi57 : entity work.extsi(arch) generic map(5, 7)
    port map(
      ins => fork38_outs_2,
      ins_valid => fork38_outs_2_valid,
      ins_ready => fork38_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi57_outs,
      outs_valid => extsi57_outs_valid,
      outs_ready => extsi57_outs_ready
    );

  control_merge7 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => buffer59_outs_valid,
      ins_valid(1) => cond_br26_trueOut_valid,
      ins_ready(0) => buffer59_outs_ready,
      ins_ready(1) => cond_br26_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge7_outs_valid,
      outs_ready => control_merge7_outs_ready,
      index => control_merge7_index,
      index_valid => control_merge7_index_valid,
      index_ready => control_merge7_index_ready
    );

  fork39 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => control_merge7_index,
      ins_valid => control_merge7_index_valid,
      ins_ready => control_merge7_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork39_outs_0,
      outs(1) => fork39_outs_1,
      outs(2) => fork39_outs_2,
      outs(3) => fork39_outs_3,
      outs_valid(0) => fork39_outs_0_valid,
      outs_valid(1) => fork39_outs_1_valid,
      outs_valid(2) => fork39_outs_2_valid,
      outs_valid(3) => fork39_outs_3_valid,
      outs_ready(0) => fork39_outs_0_ready,
      outs_ready(1) => fork39_outs_1_ready,
      outs_ready(2) => fork39_outs_2_ready,
      outs_ready(3) => fork39_outs_3_ready
    );

  lazy_fork3 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => control_merge7_outs_valid,
      ins_ready => control_merge7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork3_outs_0_valid,
      outs_valid(1) => lazy_fork3_outs_1_valid,
      outs_valid(2) => lazy_fork3_outs_2_valid,
      outs_ready(0) => lazy_fork3_outs_0_ready,
      outs_ready(1) => lazy_fork3_outs_1_ready,
      outs_ready(2) => lazy_fork3_outs_2_ready
    );

  source11 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source11_outs_valid,
      outs_ready => source11_outs_ready
    );

  constant42 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source11_outs_valid,
      ctrl_ready => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant42_outs,
      outs_valid => constant42_outs_valid,
      outs_ready => constant42_outs_ready
    );

  extsi58 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant42_outs,
      ins_valid => constant42_outs_valid,
      ins_ready => constant42_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi58_outs,
      outs_valid => extsi58_outs_valid,
      outs_ready => extsi58_outs_ready
    );

  source12 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source12_outs_valid,
      outs_ready => source12_outs_ready
    );

  constant43 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source12_outs_valid,
      ctrl_ready => source12_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant43_outs,
      outs_valid => constant43_outs_valid,
      outs_ready => constant43_outs_ready
    );

  fork40 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant43_outs,
      ins_valid => constant43_outs_valid,
      ins_ready => constant43_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork40_outs_0,
      outs(1) => fork40_outs_1,
      outs_valid(0) => fork40_outs_0_valid,
      outs_valid(1) => fork40_outs_1_valid,
      outs_ready(0) => fork40_outs_0_ready,
      outs_ready(1) => fork40_outs_1_ready
    );

  extsi59 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => fork40_outs_0,
      ins_valid => fork40_outs_0_valid,
      ins_ready => fork40_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi59_outs,
      outs_valid => extsi59_outs_valid,
      outs_ready => extsi59_outs_ready
    );

  extsi19 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => fork40_outs_1,
      ins_valid => fork40_outs_1_valid,
      ins_ready => fork40_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  fork41 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi19_outs,
      ins_valid => extsi19_outs_valid,
      ins_ready => extsi19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork41_outs_0,
      outs(1) => fork41_outs_1,
      outs(2) => fork41_outs_2,
      outs(3) => fork41_outs_3,
      outs_valid(0) => fork41_outs_0_valid,
      outs_valid(1) => fork41_outs_1_valid,
      outs_valid(2) => fork41_outs_2_valid,
      outs_valid(3) => fork41_outs_3_valid,
      outs_ready(0) => fork41_outs_0_ready,
      outs_ready(1) => fork41_outs_1_ready,
      outs_ready(2) => fork41_outs_2_ready,
      outs_ready(3) => fork41_outs_3_ready
    );

  source13 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source13_outs_valid,
      outs_ready => source13_outs_ready
    );

  constant44 : entity work.handshake_constant_2(arch) generic map(3)
    port map(
      ctrl_valid => source13_outs_valid,
      ctrl_ready => source13_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant44_outs,
      outs_valid => constant44_outs_valid,
      outs_ready => constant44_outs_ready
    );

  extsi20 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant44_outs,
      ins_valid => constant44_outs_valid,
      ins_ready => constant44_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  fork42 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi20_outs,
      ins_valid => extsi20_outs_valid,
      ins_ready => extsi20_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork42_outs_0,
      outs(1) => fork42_outs_1,
      outs(2) => fork42_outs_2,
      outs(3) => fork42_outs_3,
      outs_valid(0) => fork42_outs_0_valid,
      outs_valid(1) => fork42_outs_1_valid,
      outs_valid(2) => fork42_outs_2_valid,
      outs_valid(3) => fork42_outs_3_valid,
      outs_ready(0) => fork42_outs_0_ready,
      outs_ready(1) => fork42_outs_1_ready,
      outs_ready(2) => fork42_outs_2_ready,
      outs_ready(3) => fork42_outs_3_ready
    );

  shli14 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_0,
      lhs_valid => fork37_outs_0_valid,
      lhs_ready => fork37_outs_0_ready,
      rhs => fork41_outs_0,
      rhs_valid => fork41_outs_0_valid,
      rhs_ready => fork41_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli14_result,
      result_valid => shli14_result_valid,
      result_ready => shli14_result_ready
    );

  buffer80 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli14_result,
      ins_valid => shli14_result_valid,
      ins_ready => shli14_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer80_outs,
      outs_valid => buffer80_outs_valid,
      outs_ready => buffer80_outs_ready
    );

  trunci17 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer80_outs,
      ins_valid => buffer80_outs_valid,
      ins_ready => buffer80_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci17_outs,
      outs_valid => trunci17_outs_valid,
      outs_ready => trunci17_outs_ready
    );

  shli15 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_1,
      lhs_valid => fork37_outs_1_valid,
      lhs_ready => fork37_outs_1_ready,
      rhs => fork42_outs_0,
      rhs_valid => fork42_outs_0_valid,
      rhs_ready => fork42_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli15_result,
      result_valid => shli15_result_valid,
      result_ready => shli15_result_ready
    );

  buffer81 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli15_result,
      ins_valid => shli15_result_valid,
      ins_ready => shli15_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer81_outs,
      outs_valid => buffer81_outs_valid,
      outs_ready => buffer81_outs_ready
    );

  trunci18 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer81_outs,
      ins_valid => buffer81_outs_valid,
      ins_ready => buffer81_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci18_outs,
      outs_valid => trunci18_outs_valid,
      outs_ready => trunci18_outs_ready
    );

  addi26 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci17_outs,
      lhs_valid => trunci17_outs_valid,
      lhs_ready => trunci17_outs_ready,
      rhs => trunci18_outs,
      rhs_valid => trunci18_outs_valid,
      rhs_ready => trunci18_outs_ready,
      clk => clk,
      rst => rst,
      result => addi26_result,
      result_valid => addi26_result_valid,
      result_ready => addi26_result_ready
    );

  buffer82 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi26_result,
      ins_valid => addi26_result_valid,
      ins_ready => addi26_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer82_outs,
      outs_valid => buffer82_outs_valid,
      outs_ready => buffer82_outs_ready
    );

  addi9 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi51_outs,
      lhs_valid => extsi51_outs_valid,
      lhs_ready => extsi51_outs_ready,
      rhs => buffer82_outs,
      rhs_valid => buffer82_outs_valid,
      rhs_ready => buffer82_outs_ready,
      clk => clk,
      rst => rst,
      result => addi9_result,
      result_valid => addi9_result_valid,
      result_ready => addi9_result_ready
    );

  buffer83 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi9_result,
      ins_valid => addi9_result_valid,
      ins_ready => addi9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer83_outs,
      outs_valid => buffer83_outs_valid,
      outs_ready => buffer83_outs_ready
    );

  load4 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer83_outs,
      addrIn_valid => buffer83_outs_valid,
      addrIn_ready => buffer83_outs_ready,
      dataFromMem => lsq3_ldData_1,
      dataFromMem_valid => lsq3_ldData_1_valid,
      dataFromMem_ready => lsq3_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  shli16 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork35_outs_0,
      lhs_valid => fork35_outs_0_valid,
      lhs_ready => fork35_outs_0_ready,
      rhs => fork41_outs_1,
      rhs_valid => fork41_outs_1_valid,
      rhs_ready => fork41_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli16_result,
      result_valid => shli16_result_valid,
      result_ready => shli16_result_ready
    );

  buffer84 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli16_result,
      ins_valid => shli16_result_valid,
      ins_ready => shli16_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer84_outs,
      outs_valid => buffer84_outs_valid,
      outs_ready => buffer84_outs_ready
    );

  trunci19 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer84_outs,
      ins_valid => buffer84_outs_valid,
      ins_ready => buffer84_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci19_outs,
      outs_valid => trunci19_outs_valid,
      outs_ready => trunci19_outs_ready
    );

  shli17 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork35_outs_1,
      lhs_valid => fork35_outs_1_valid,
      lhs_ready => fork35_outs_1_ready,
      rhs => fork42_outs_1,
      rhs_valid => fork42_outs_1_valid,
      rhs_ready => fork42_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli17_result,
      result_valid => shli17_result_valid,
      result_ready => shli17_result_ready
    );

  buffer85 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli17_result,
      ins_valid => shli17_result_valid,
      ins_ready => shli17_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer85_outs,
      outs_valid => buffer85_outs_valid,
      outs_ready => buffer85_outs_ready
    );

  trunci20 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer85_outs,
      ins_valid => buffer85_outs_valid,
      ins_ready => buffer85_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci20_outs,
      outs_valid => trunci20_outs_valid,
      outs_ready => trunci20_outs_ready
    );

  addi27 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci19_outs,
      lhs_valid => trunci19_outs_valid,
      lhs_ready => trunci19_outs_ready,
      rhs => trunci20_outs,
      rhs_valid => trunci20_outs_valid,
      rhs_ready => trunci20_outs_ready,
      clk => clk,
      rst => rst,
      result => addi27_result,
      result_valid => addi27_result_valid,
      result_ready => addi27_result_ready
    );

  buffer86 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi27_result,
      ins_valid => addi27_result_valid,
      ins_ready => addi27_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer86_outs,
      outs_valid => buffer86_outs_valid,
      outs_ready => buffer86_outs_ready
    );

  addi10 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi55_outs,
      lhs_valid => extsi55_outs_valid,
      lhs_ready => extsi55_outs_ready,
      rhs => buffer86_outs,
      rhs_valid => buffer86_outs_valid,
      rhs_ready => buffer86_outs_ready,
      clk => clk,
      rst => rst,
      result => addi10_result,
      result_valid => addi10_result_valid,
      result_ready => addi10_result_ready
    );

  load5 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => addi10_result,
      addrIn_valid => addi10_result_valid,
      addrIn_ready => addi10_result_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load5_addrOut,
      addrOut_valid => load5_addrOut_valid,
      addrOut_ready => load5_addrOut_ready,
      dataOut => load5_dataOut,
      dataOut_valid => load5_dataOut_valid,
      dataOut_ready => load5_dataOut_ready
    );

  muli3 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load4_dataOut,
      lhs_valid => load4_dataOut_valid,
      lhs_ready => load4_dataOut_ready,
      rhs => load5_dataOut,
      rhs_valid => load5_dataOut_valid,
      rhs_ready => load5_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli3_result,
      result_valid => muli3_result_valid,
      result_ready => muli3_result_ready
    );

  shli18 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_2,
      lhs_valid => fork37_outs_2_valid,
      lhs_ready => fork37_outs_2_ready,
      rhs => fork41_outs_2,
      rhs_valid => fork41_outs_2_valid,
      rhs_ready => fork41_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli18_result,
      result_valid => shli18_result_valid,
      result_ready => shli18_result_ready
    );

  buffer87 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli18_result,
      ins_valid => shli18_result_valid,
      ins_ready => shli18_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer87_outs,
      outs_valid => buffer87_outs_valid,
      outs_ready => buffer87_outs_ready
    );

  trunci21 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer87_outs,
      ins_valid => buffer87_outs_valid,
      ins_ready => buffer87_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci21_outs,
      outs_valid => trunci21_outs_valid,
      outs_ready => trunci21_outs_ready
    );

  shli19 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_3,
      lhs_valid => fork37_outs_3_valid,
      lhs_ready => fork37_outs_3_ready,
      rhs => fork42_outs_2,
      rhs_valid => fork42_outs_2_valid,
      rhs_ready => fork42_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli19_result,
      result_valid => shli19_result_valid,
      result_ready => shli19_result_ready
    );

  buffer88 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli19_result,
      ins_valid => shli19_result_valid,
      ins_ready => shli19_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer88_outs,
      outs_valid => buffer88_outs_valid,
      outs_ready => buffer88_outs_ready
    );

  trunci22 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer88_outs,
      ins_valid => buffer88_outs_valid,
      ins_ready => buffer88_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci22_outs,
      outs_valid => trunci22_outs_valid,
      outs_ready => trunci22_outs_ready
    );

  addi28 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci21_outs,
      lhs_valid => trunci21_outs_valid,
      lhs_ready => trunci21_outs_ready,
      rhs => trunci22_outs,
      rhs_valid => trunci22_outs_valid,
      rhs_ready => trunci22_outs_ready,
      clk => clk,
      rst => rst,
      result => addi28_result,
      result_valid => addi28_result_valid,
      result_ready => addi28_result_ready
    );

  buffer89 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi28_result,
      ins_valid => addi28_result_valid,
      ins_ready => addi28_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer89_outs,
      outs_valid => buffer89_outs_valid,
      outs_ready => buffer89_outs_ready
    );

  addi11 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi56_outs,
      lhs_valid => extsi56_outs_valid,
      lhs_ready => extsi56_outs_ready,
      rhs => buffer89_outs,
      rhs_valid => buffer89_outs_valid,
      rhs_ready => buffer89_outs_ready,
      clk => clk,
      rst => rst,
      result => addi11_result,
      result_valid => addi11_result_valid,
      result_ready => addi11_result_ready
    );

  buffer90 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi11_result,
      ins_valid => addi11_result_valid,
      ins_ready => addi11_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer90_outs,
      outs_valid => buffer90_outs_valid,
      outs_ready => buffer90_outs_ready
    );

  load6 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer90_outs,
      addrIn_valid => buffer90_outs_valid,
      addrIn_ready => buffer90_outs_ready,
      dataFromMem => lsq2_ldData_1,
      dataFromMem_valid => lsq2_ldData_1_valid,
      dataFromMem_ready => lsq2_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load6_addrOut,
      addrOut_valid => load6_addrOut_valid,
      addrOut_ready => load6_addrOut_ready,
      dataOut => load6_dataOut,
      dataOut_valid => load6_dataOut_valid,
      dataOut_ready => load6_dataOut_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load6_dataOut,
      lhs_valid => load6_dataOut_valid,
      lhs_ready => load6_dataOut_ready,
      rhs => muli3_result,
      rhs_valid => muli3_result_valid,
      rhs_ready => muli3_result_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  shli20 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_4,
      lhs_valid => fork37_outs_4_valid,
      lhs_ready => fork37_outs_4_ready,
      rhs => fork41_outs_3,
      rhs_valid => fork41_outs_3_valid,
      rhs_ready => fork41_outs_3_ready,
      clk => clk,
      rst => rst,
      result => shli20_result,
      result_valid => shli20_result_valid,
      result_ready => shli20_result_ready
    );

  buffer92 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli20_result,
      ins_valid => shli20_result_valid,
      ins_ready => shli20_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer92_outs,
      outs_valid => buffer92_outs_valid,
      outs_ready => buffer92_outs_ready
    );

  trunci23 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer92_outs,
      ins_valid => buffer92_outs_valid,
      ins_ready => buffer92_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci23_outs,
      outs_valid => trunci23_outs_valid,
      outs_ready => trunci23_outs_ready
    );

  shli21 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork37_outs_5,
      lhs_valid => fork37_outs_5_valid,
      lhs_ready => fork37_outs_5_ready,
      rhs => fork42_outs_3,
      rhs_valid => fork42_outs_3_valid,
      rhs_ready => fork42_outs_3_ready,
      clk => clk,
      rst => rst,
      result => shli21_result,
      result_valid => shli21_result_valid,
      result_ready => shli21_result_ready
    );

  buffer93 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli21_result,
      ins_valid => shli21_result_valid,
      ins_ready => shli21_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer93_outs,
      outs_valid => buffer93_outs_valid,
      outs_ready => buffer93_outs_ready
    );

  trunci24 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer93_outs,
      ins_valid => buffer93_outs_valid,
      ins_ready => buffer93_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci24_outs,
      outs_valid => trunci24_outs_valid,
      outs_ready => trunci24_outs_ready
    );

  addi29 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci23_outs,
      lhs_valid => trunci23_outs_valid,
      lhs_ready => trunci23_outs_ready,
      rhs => trunci24_outs,
      rhs_valid => trunci24_outs_valid,
      rhs_ready => trunci24_outs_ready,
      clk => clk,
      rst => rst,
      result => addi29_result,
      result_valid => addi29_result_valid,
      result_ready => addi29_result_ready
    );

  buffer94 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi29_result,
      ins_valid => addi29_result_valid,
      ins_ready => addi29_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer94_outs,
      outs_valid => buffer94_outs_valid,
      outs_ready => buffer94_outs_ready
    );

  addi12 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi57_outs,
      lhs_valid => extsi57_outs_valid,
      lhs_ready => extsi57_outs_ready,
      rhs => buffer94_outs,
      rhs_valid => buffer94_outs_valid,
      rhs_ready => buffer94_outs_ready,
      clk => clk,
      rst => rst,
      result => addi12_result,
      result_valid => addi12_result_valid,
      result_ready => addi12_result_ready
    );

  buffer91 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer91_outs,
      outs_valid => buffer91_outs_valid,
      outs_ready => buffer91_outs_ready
    );

  buffer95 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi12_result,
      ins_valid => addi12_result_valid,
      ins_ready => addi12_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer95_outs,
      outs_valid => buffer95_outs_valid,
      outs_ready => buffer95_outs_ready
    );

  store3 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer95_outs,
      addrIn_valid => buffer95_outs_valid,
      addrIn_ready => buffer95_outs_ready,
      dataIn => buffer91_outs,
      dataIn_valid => buffer91_outs_valid,
      dataIn_ready => buffer91_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store3_addrOut,
      addrOut_valid => store3_addrOut_valid,
      addrOut_ready => store3_addrOut_ready,
      dataToMem => store3_dataToMem,
      dataToMem_valid => store3_dataToMem_valid,
      dataToMem_ready => store3_dataToMem_ready
    );

  addi16 : entity work.addi(arch) generic map(6)
    port map(
      lhs => extsi52_outs,
      lhs_valid => extsi52_outs_valid,
      lhs_ready => extsi52_outs_ready,
      rhs => extsi59_outs,
      rhs_valid => extsi59_outs_valid,
      rhs_ready => extsi59_outs_ready,
      clk => clk,
      rst => rst,
      result => addi16_result,
      result_valid => addi16_result_valid,
      result_ready => addi16_result_ready
    );

  buffer96 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi16_result,
      ins_valid => addi16_result_valid,
      ins_ready => addi16_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer96_outs,
      outs_valid => buffer96_outs_valid,
      outs_ready => buffer96_outs_ready
    );

  fork43 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer96_outs,
      ins_valid => buffer96_outs_valid,
      ins_ready => buffer96_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork43_outs_0,
      outs(1) => fork43_outs_1,
      outs_valid(0) => fork43_outs_0_valid,
      outs_valid(1) => fork43_outs_1_valid,
      outs_ready(0) => fork43_outs_0_ready,
      outs_ready(1) => fork43_outs_1_ready
    );

  trunci25 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork43_outs_0,
      ins_valid => fork43_outs_0_valid,
      ins_ready => fork43_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci25_outs,
      outs_valid => trunci25_outs_valid,
      outs_ready => trunci25_outs_ready
    );

  cmpi3 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork43_outs_1,
      lhs_valid => fork43_outs_1_valid,
      lhs_ready => fork43_outs_1_ready,
      rhs => extsi58_outs,
      rhs_valid => extsi58_outs_valid,
      rhs_ready => extsi58_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  buffer97 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer97_outs,
      outs_valid => buffer97_outs_valid,
      outs_ready => buffer97_outs_ready
    );

  fork44 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => buffer97_outs,
      ins_valid => buffer97_outs_valid,
      ins_ready => buffer97_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork44_outs_0,
      outs(1) => fork44_outs_1,
      outs(2) => fork44_outs_2,
      outs(3) => fork44_outs_3,
      outs(4) => fork44_outs_4,
      outs_valid(0) => fork44_outs_0_valid,
      outs_valid(1) => fork44_outs_1_valid,
      outs_valid(2) => fork44_outs_2_valid,
      outs_valid(3) => fork44_outs_3_valid,
      outs_valid(4) => fork44_outs_4_valid,
      outs_ready(0) => fork44_outs_0_ready,
      outs_ready(1) => fork44_outs_1_ready,
      outs_ready(2) => fork44_outs_2_ready,
      outs_ready(3) => fork44_outs_3_ready,
      outs_ready(4) => fork44_outs_4_ready
    );

  cond_br22 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork44_outs_0,
      condition_valid => fork44_outs_0_valid,
      condition_ready => fork44_outs_0_ready,
      data => trunci25_outs,
      data_valid => trunci25_outs_valid,
      data_ready => trunci25_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br22_trueOut,
      trueOut_valid => cond_br22_trueOut_valid,
      trueOut_ready => cond_br22_trueOut_ready,
      falseOut => cond_br22_falseOut,
      falseOut_valid => cond_br22_falseOut_valid,
      falseOut_ready => cond_br22_falseOut_ready
    );

  sink7 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br22_falseOut,
      ins_valid => cond_br22_falseOut_valid,
      ins_ready => cond_br22_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer72 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux18_outs,
      ins_valid => mux18_outs_valid,
      ins_ready => mux18_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer72_outs,
      outs_valid => buffer72_outs_valid,
      outs_ready => buffer72_outs_ready
    );

  buffer73 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer72_outs,
      ins_valid => buffer72_outs_valid,
      ins_ready => buffer72_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer73_outs,
      outs_valid => buffer73_outs_valid,
      outs_ready => buffer73_outs_ready
    );

  cond_br23 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork44_outs_3,
      condition_valid => fork44_outs_3_valid,
      condition_ready => fork44_outs_3_ready,
      data => buffer73_outs,
      data_valid => buffer73_outs_valid,
      data_ready => buffer73_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br23_trueOut,
      trueOut_valid => cond_br23_trueOut_valid,
      trueOut_ready => cond_br23_trueOut_ready,
      falseOut => cond_br23_falseOut,
      falseOut_valid => cond_br23_falseOut_valid,
      falseOut_ready => cond_br23_falseOut_ready
    );

  cond_br24 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork44_outs_1,
      condition_valid => fork44_outs_1_valid,
      condition_ready => fork44_outs_1_ready,
      data => fork36_outs_0,
      data_valid => fork36_outs_0_valid,
      data_ready => fork36_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br24_trueOut,
      trueOut_valid => cond_br24_trueOut_valid,
      trueOut_ready => cond_br24_trueOut_ready,
      falseOut => cond_br24_falseOut,
      falseOut_valid => cond_br24_falseOut_valid,
      falseOut_ready => cond_br24_falseOut_ready
    );

  cond_br25 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork44_outs_2,
      condition_valid => fork44_outs_2_valid,
      condition_ready => fork44_outs_2_ready,
      data => fork38_outs_3,
      data_valid => fork38_outs_3_valid,
      data_ready => fork38_outs_3_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br25_trueOut,
      trueOut_valid => cond_br25_trueOut_valid,
      trueOut_ready => cond_br25_trueOut_ready,
      falseOut => cond_br25_falseOut,
      falseOut_valid => cond_br25_falseOut_valid,
      falseOut_ready => cond_br25_falseOut_ready
    );

  buffer78 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork3_outs_1_valid,
      ins_ready => lazy_fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer78_outs_valid,
      outs_ready => buffer78_outs_ready
    );

  buffer79 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer78_outs_valid,
      ins_ready => buffer78_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer79_outs_valid,
      outs_ready => buffer79_outs_ready
    );

  cond_br26 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork44_outs_4,
      condition_valid => fork44_outs_4_valid,
      condition_ready => fork44_outs_4_ready,
      data_valid => buffer79_outs_valid,
      data_ready => buffer79_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br26_trueOut_valid,
      trueOut_ready => cond_br26_trueOut_ready,
      falseOut_valid => cond_br26_falseOut_valid,
      falseOut_ready => cond_br26_falseOut_ready
    );

  extsi60 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => cond_br25_falseOut,
      ins_valid => cond_br25_falseOut_valid,
      ins_ready => cond_br25_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi60_outs,
      outs_valid => extsi60_outs_valid,
      outs_ready => extsi60_outs_ready
    );

  source14 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source14_outs_valid,
      outs_ready => source14_outs_ready
    );

  constant45 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source14_outs_valid,
      ctrl_ready => source14_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant45_outs,
      outs_valid => constant45_outs_valid,
      outs_ready => constant45_outs_ready
    );

  extsi61 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant45_outs,
      ins_valid => constant45_outs_valid,
      ins_ready => constant45_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi61_outs,
      outs_valid => extsi61_outs_valid,
      outs_ready => extsi61_outs_ready
    );

  source15 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source15_outs_valid,
      outs_ready => source15_outs_ready
    );

  constant46 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source15_outs_valid,
      ctrl_ready => source15_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant46_outs,
      outs_valid => constant46_outs_valid,
      outs_ready => constant46_outs_ready
    );

  extsi62 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant46_outs,
      ins_valid => constant46_outs_valid,
      ins_ready => constant46_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi62_outs,
      outs_valid => extsi62_outs_valid,
      outs_ready => extsi62_outs_ready
    );

  addi17 : entity work.addi(arch) generic map(6)
    port map(
      lhs => extsi60_outs,
      lhs_valid => extsi60_outs_valid,
      lhs_ready => extsi60_outs_ready,
      rhs => extsi62_outs,
      rhs_valid => extsi62_outs_valid,
      rhs_ready => extsi62_outs_ready,
      clk => clk,
      rst => rst,
      result => addi17_result,
      result_valid => addi17_result_valid,
      result_ready => addi17_result_ready
    );

  buffer99 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi17_result,
      ins_valid => addi17_result_valid,
      ins_ready => addi17_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer99_outs,
      outs_valid => buffer99_outs_valid,
      outs_ready => buffer99_outs_ready
    );

  fork45 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer99_outs,
      ins_valid => buffer99_outs_valid,
      ins_ready => buffer99_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork45_outs_0,
      outs(1) => fork45_outs_1,
      outs_valid(0) => fork45_outs_0_valid,
      outs_valid(1) => fork45_outs_1_valid,
      outs_ready(0) => fork45_outs_0_ready,
      outs_ready(1) => fork45_outs_1_ready
    );

  trunci26 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork45_outs_0,
      ins_valid => fork45_outs_0_valid,
      ins_ready => fork45_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci26_outs,
      outs_valid => trunci26_outs_valid,
      outs_ready => trunci26_outs_ready
    );

  cmpi4 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork45_outs_1,
      lhs_valid => fork45_outs_1_valid,
      lhs_ready => fork45_outs_1_ready,
      rhs => extsi61_outs,
      rhs_valid => extsi61_outs_valid,
      rhs_ready => extsi61_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi4_result,
      result_valid => cmpi4_result_valid,
      result_ready => cmpi4_result_ready
    );

  buffer100 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi4_result,
      ins_valid => cmpi4_result_valid,
      ins_ready => cmpi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer100_outs,
      outs_valid => buffer100_outs_valid,
      outs_ready => buffer100_outs_ready
    );

  fork46 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer100_outs,
      ins_valid => buffer100_outs_valid,
      ins_ready => buffer100_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork46_outs_0,
      outs(1) => fork46_outs_1,
      outs(2) => fork46_outs_2,
      outs(3) => fork46_outs_3,
      outs_valid(0) => fork46_outs_0_valid,
      outs_valid(1) => fork46_outs_1_valid,
      outs_valid(2) => fork46_outs_2_valid,
      outs_valid(3) => fork46_outs_3_valid,
      outs_ready(0) => fork46_outs_0_ready,
      outs_ready(1) => fork46_outs_1_ready,
      outs_ready(2) => fork46_outs_2_ready,
      outs_ready(3) => fork46_outs_3_ready
    );

  cond_br27 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork46_outs_0,
      condition_valid => fork46_outs_0_valid,
      condition_ready => fork46_outs_0_ready,
      data => trunci26_outs,
      data_valid => trunci26_outs_valid,
      data_ready => trunci26_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br27_trueOut,
      trueOut_valid => cond_br27_trueOut_valid,
      trueOut_ready => cond_br27_trueOut_ready,
      falseOut => cond_br27_falseOut,
      falseOut_valid => cond_br27_falseOut_valid,
      falseOut_ready => cond_br27_falseOut_ready
    );

  sink9 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br27_falseOut,
      ins_valid => cond_br27_falseOut_valid,
      ins_ready => cond_br27_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br28 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork46_outs_2,
      condition_valid => fork46_outs_2_valid,
      condition_ready => fork46_outs_2_ready,
      data => cond_br23_falseOut,
      data_valid => cond_br23_falseOut_valid,
      data_ready => cond_br23_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br28_trueOut,
      trueOut_valid => cond_br28_trueOut_valid,
      trueOut_ready => cond_br28_trueOut_ready,
      falseOut => cond_br28_falseOut,
      falseOut_valid => cond_br28_falseOut_valid,
      falseOut_ready => cond_br28_falseOut_ready
    );

  buffer98 : entity work.oehb(arch) generic map(5)
    port map(
      ins => cond_br24_falseOut,
      ins_valid => cond_br24_falseOut_valid,
      ins_ready => cond_br24_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer98_outs,
      outs_valid => buffer98_outs_valid,
      outs_ready => buffer98_outs_ready
    );

  cond_br29 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork46_outs_1,
      condition_valid => fork46_outs_1_valid,
      condition_ready => fork46_outs_1_ready,
      data => buffer98_outs,
      data_valid => buffer98_outs_valid,
      data_ready => buffer98_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br29_trueOut,
      trueOut_valid => cond_br29_trueOut_valid,
      trueOut_ready => cond_br29_trueOut_ready,
      falseOut => cond_br29_falseOut,
      falseOut_valid => cond_br29_falseOut_valid,
      falseOut_ready => cond_br29_falseOut_ready
    );

  cond_br30 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork46_outs_3,
      condition_valid => fork46_outs_3_valid,
      condition_ready => fork46_outs_3_ready,
      data_valid => cond_br26_falseOut_valid,
      data_ready => cond_br26_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br30_trueOut_valid,
      trueOut_ready => cond_br30_trueOut_ready,
      falseOut_valid => cond_br30_falseOut_valid,
      falseOut_ready => cond_br30_falseOut_ready
    );

  extsi63 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => cond_br29_falseOut,
      ins_valid => cond_br29_falseOut_valid,
      ins_ready => cond_br29_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi63_outs,
      outs_valid => extsi63_outs_valid,
      outs_ready => extsi63_outs_ready
    );

  source16 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source16_outs_valid,
      outs_ready => source16_outs_ready
    );

  constant47 : entity work.handshake_constant_3(arch) generic map(5)
    port map(
      ctrl_valid => source16_outs_valid,
      ctrl_ready => source16_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant47_outs,
      outs_valid => constant47_outs_valid,
      outs_ready => constant47_outs_ready
    );

  extsi64 : entity work.extsi(arch) generic map(5, 6)
    port map(
      ins => constant47_outs,
      ins_valid => constant47_outs_valid,
      ins_ready => constant47_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi64_outs,
      outs_valid => extsi64_outs_valid,
      outs_ready => extsi64_outs_ready
    );

  source17 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source17_outs_valid,
      outs_ready => source17_outs_ready
    );

  constant48 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source17_outs_valid,
      ctrl_ready => source17_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant48_outs,
      outs_valid => constant48_outs_valid,
      outs_ready => constant48_outs_ready
    );

  extsi65 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant48_outs,
      ins_valid => constant48_outs_valid,
      ins_ready => constant48_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi65_outs,
      outs_valid => extsi65_outs_valid,
      outs_ready => extsi65_outs_ready
    );

  addi18 : entity work.addi(arch) generic map(6)
    port map(
      lhs => extsi63_outs,
      lhs_valid => extsi63_outs_valid,
      lhs_ready => extsi63_outs_ready,
      rhs => extsi65_outs,
      rhs_valid => extsi65_outs_valid,
      rhs_ready => extsi65_outs_ready,
      clk => clk,
      rst => rst,
      result => addi18_result,
      result_valid => addi18_result_valid,
      result_ready => addi18_result_ready
    );

  buffer102 : entity work.oehb(arch) generic map(6)
    port map(
      ins => addi18_result,
      ins_valid => addi18_result_valid,
      ins_ready => addi18_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer102_outs,
      outs_valid => buffer102_outs_valid,
      outs_ready => buffer102_outs_ready
    );

  fork47 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer102_outs,
      ins_valid => buffer102_outs_valid,
      ins_ready => buffer102_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork47_outs_0,
      outs(1) => fork47_outs_1,
      outs_valid(0) => fork47_outs_0_valid,
      outs_valid(1) => fork47_outs_1_valid,
      outs_ready(0) => fork47_outs_0_ready,
      outs_ready(1) => fork47_outs_1_ready
    );

  trunci27 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork47_outs_0,
      ins_valid => fork47_outs_0_valid,
      ins_ready => fork47_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci27_outs,
      outs_valid => trunci27_outs_valid,
      outs_ready => trunci27_outs_ready
    );

  cmpi5 : entity work.handshake_cmpi_0(arch) generic map(6)
    port map(
      lhs => fork47_outs_1,
      lhs_valid => fork47_outs_1_valid,
      lhs_ready => fork47_outs_1_ready,
      rhs => extsi64_outs,
      rhs_valid => extsi64_outs_valid,
      rhs_ready => extsi64_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi5_result,
      result_valid => cmpi5_result_valid,
      result_ready => cmpi5_result_ready
    );

  buffer103 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi5_result,
      ins_valid => cmpi5_result_valid,
      ins_ready => cmpi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer103_outs,
      outs_valid => buffer103_outs_valid,
      outs_ready => buffer103_outs_ready
    );

  fork48 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer103_outs,
      ins_valid => buffer103_outs_valid,
      ins_ready => buffer103_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork48_outs_0,
      outs(1) => fork48_outs_1,
      outs(2) => fork48_outs_2,
      outs_valid(0) => fork48_outs_0_valid,
      outs_valid(1) => fork48_outs_1_valid,
      outs_valid(2) => fork48_outs_2_valid,
      outs_ready(0) => fork48_outs_0_ready,
      outs_ready(1) => fork48_outs_1_ready,
      outs_ready(2) => fork48_outs_2_ready
    );

  cond_br31 : entity work.cond_br(arch) generic map(5)
    port map(
      condition => fork48_outs_0,
      condition_valid => fork48_outs_0_valid,
      condition_ready => fork48_outs_0_ready,
      data => trunci27_outs,
      data_valid => trunci27_outs_valid,
      data_ready => trunci27_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br31_trueOut,
      trueOut_valid => cond_br31_trueOut_valid,
      trueOut_ready => cond_br31_trueOut_ready,
      falseOut => cond_br31_falseOut,
      falseOut_valid => cond_br31_falseOut_valid,
      falseOut_ready => cond_br31_falseOut_ready
    );

  sink11 : entity work.sink(arch) generic map(5)
    port map(
      ins => cond_br31_falseOut,
      ins_valid => cond_br31_falseOut_valid,
      ins_ready => cond_br31_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer101 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br28_falseOut,
      ins_valid => cond_br28_falseOut_valid,
      ins_ready => cond_br28_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer101_outs,
      outs_valid => buffer101_outs_valid,
      outs_ready => buffer101_outs_ready
    );

  cond_br32 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork48_outs_1,
      condition_valid => fork48_outs_1_valid,
      condition_ready => fork48_outs_1_ready,
      data => buffer101_outs,
      data_valid => buffer101_outs_valid,
      data_ready => buffer101_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br32_trueOut,
      trueOut_valid => cond_br32_trueOut_valid,
      trueOut_ready => cond_br32_trueOut_ready,
      falseOut => cond_br32_falseOut,
      falseOut_valid => cond_br32_falseOut_valid,
      falseOut_ready => cond_br32_falseOut_ready
    );

  sink12 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br32_falseOut,
      ins_valid => cond_br32_falseOut_valid,
      ins_ready => cond_br32_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br33 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork48_outs_2,
      condition_valid => fork48_outs_2_valid,
      condition_ready => fork48_outs_2_ready,
      data_valid => cond_br30_falseOut_valid,
      data_ready => cond_br30_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br33_trueOut_valid,
      trueOut_ready => cond_br33_trueOut_ready,
      falseOut_valid => cond_br33_falseOut_valid,
      falseOut_ready => cond_br33_falseOut_ready
    );

  fork49 : entity work.fork_dataless(arch) generic map(5)
    port map(
      ins_valid => cond_br33_falseOut_valid,
      ins_ready => cond_br33_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork49_outs_0_valid,
      outs_valid(1) => fork49_outs_1_valid,
      outs_valid(2) => fork49_outs_2_valid,
      outs_valid(3) => fork49_outs_3_valid,
      outs_valid(4) => fork49_outs_4_valid,
      outs_ready(0) => fork49_outs_0_ready,
      outs_ready(1) => fork49_outs_1_ready,
      outs_ready(2) => fork49_outs_2_ready,
      outs_ready(3) => fork49_outs_3_ready,
      outs_ready(4) => fork49_outs_4_ready
    );

end architecture;
