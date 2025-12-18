library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity atax is
  port (
    A_loadData : in std_logic_vector(31 downto 0);
    x_loadData : in std_logic_vector(31 downto 0);
    y_loadData : in std_logic_vector(31 downto 0);
    tmp_loadData : in std_logic_vector(31 downto 0);
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
    A_loadEn : out std_logic;
    A_loadAddr : out std_logic_vector(8 downto 0);
    A_storeEn : out std_logic;
    A_storeAddr : out std_logic_vector(8 downto 0);
    A_storeData : out std_logic_vector(31 downto 0);
    x_loadEn : out std_logic;
    x_loadAddr : out std_logic_vector(4 downto 0);
    x_storeEn : out std_logic;
    x_storeAddr : out std_logic_vector(4 downto 0);
    x_storeData : out std_logic_vector(31 downto 0);
    y_loadEn : out std_logic;
    y_loadAddr : out std_logic_vector(4 downto 0);
    y_storeEn : out std_logic;
    y_storeAddr : out std_logic_vector(4 downto 0);
    y_storeData : out std_logic_vector(31 downto 0);
    tmp_loadEn : out std_logic;
    tmp_loadAddr : out std_logic_vector(4 downto 0);
    tmp_storeEn : out std_logic;
    tmp_storeAddr : out std_logic_vector(4 downto 0);
    tmp_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of atax is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal fork0_outs_3_valid : std_logic;
  signal fork0_outs_3_ready : std_logic;
  signal fork0_outs_4_valid : std_logic;
  signal fork0_outs_4_ready : std_logic;
  signal fork0_outs_5_valid : std_logic;
  signal fork0_outs_5_ready : std_logic;
  signal fork0_outs_6_valid : std_logic;
  signal fork0_outs_6_ready : std_logic;
  signal fork0_outs_7_valid : std_logic;
  signal fork0_outs_7_ready : std_logic;
  signal mem_controller2_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0_valid : std_logic;
  signal mem_controller2_ldData_0_ready : std_logic;
  signal mem_controller2_stDone_0_valid : std_logic;
  signal mem_controller2_stDone_0_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_stDone_0_valid : std_logic;
  signal mem_controller3_stDone_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0_valid : std_logic;
  signal mem_controller4_ldData_0_ready : std_logic;
  signal mem_controller4_memEnd_valid : std_logic;
  signal mem_controller4_memEnd_ready : std_logic;
  signal mem_controller4_loadEn : std_logic;
  signal mem_controller4_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller4_storeEn : std_logic;
  signal mem_controller4_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller4_storeData : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0_valid : std_logic;
  signal mem_controller5_ldData_0_ready : std_logic;
  signal mem_controller5_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_1_valid : std_logic;
  signal mem_controller5_ldData_1_ready : std_logic;
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(8 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(8 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(10 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(10 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(10 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(10 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi20_outs : std_logic_vector(5 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(31 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal mux13_outs : std_logic_vector(10 downto 0);
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
  signal mux17_outs_valid : std_logic;
  signal mux17_outs_ready : std_logic;
  signal mux18_outs_valid : std_logic;
  signal mux18_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2 : std_logic_vector(0 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_3 : std_logic_vector(0 downto 0);
  signal fork2_outs_3_valid : std_logic;
  signal fork2_outs_3_ready : std_logic;
  signal fork2_outs_4 : std_logic_vector(0 downto 0);
  signal fork2_outs_4_valid : std_logic;
  signal fork2_outs_4_ready : std_logic;
  signal fork2_outs_5 : std_logic_vector(0 downto 0);
  signal fork2_outs_5_valid : std_logic;
  signal fork2_outs_5_ready : std_logic;
  signal fork2_outs_6 : std_logic_vector(0 downto 0);
  signal fork2_outs_6_valid : std_logic;
  signal fork2_outs_6_ready : std_logic;
  signal unbundle1_outs_0_valid : std_logic;
  signal unbundle1_outs_0_ready : std_logic;
  signal unbundle1_outs_1 : std_logic_vector(31 downto 0);
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(5 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(5 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(5 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(5 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(5 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal trunci0_outs : std_logic_vector(4 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal constant17_outs : std_logic_vector(0 downto 0);
  signal constant17_outs_valid : std_logic;
  signal constant17_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(5 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal extsi21_outs : std_logic_vector(31 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal init14_outs : std_logic_vector(31 downto 0);
  signal init14_outs_valid : std_logic;
  signal init14_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal init15_outs : std_logic_vector(31 downto 0);
  signal init15_outs_valid : std_logic;
  signal init15_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal init16_outs_valid : std_logic;
  signal init16_outs_ready : std_logic;
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal init17_outs_valid : std_logic;
  signal init17_outs_ready : std_logic;
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal init18_outs_valid : std_logic;
  signal init18_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(4 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal extsi19_outs : std_logic_vector(5 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(5 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(5 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(5 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(5 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(5 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal extsi22_outs : std_logic_vector(8 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal extsi23_outs : std_logic_vector(6 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(4 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(5 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(5 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(5 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(5 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(5 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal extsi24_outs : std_logic_vector(31 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(0 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(0 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(0 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal constant21_outs : std_logic_vector(0 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant22_outs : std_logic_vector(5 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal extsi25_outs : std_logic_vector(6 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant23_outs : std_logic_vector(1 downto 0);
  signal constant23_outs_valid : std_logic;
  signal constant23_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(6 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant24_outs : std_logic_vector(3 downto 0);
  signal constant24_outs_valid : std_logic;
  signal constant24_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant25_outs : std_logic_vector(2 downto 0);
  signal constant25_outs_valid : std_logic;
  signal constant25_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(31 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(8 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(8 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal addi5_result : std_logic_vector(8 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(8 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(8 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal load1_addrOut : std_logic_vector(8 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal load2_addrOut : std_logic_vector(4 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer21_outs : std_logic_vector(6 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(6 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(6 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal trunci4_outs : std_logic_vector(5 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer22_outs : std_logic_vector(0 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(0 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(0 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_2 : std_logic_vector(0 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal fork17_outs_3 : std_logic_vector(0 downto 0);
  signal fork17_outs_3_valid : std_logic;
  signal fork17_outs_3_ready : std_logic;
  signal fork17_outs_4 : std_logic_vector(0 downto 0);
  signal fork17_outs_4_valid : std_logic;
  signal fork17_outs_4_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(5 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(5 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(31 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(31 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(5 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(5 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer36_outs : std_logic_vector(5 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(0 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(0 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal extsi18_outs : std_logic_vector(5 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal cond_br45_trueOut : std_logic_vector(31 downto 0);
  signal cond_br45_trueOut_valid : std_logic;
  signal cond_br45_trueOut_ready : std_logic;
  signal cond_br45_falseOut : std_logic_vector(31 downto 0);
  signal cond_br45_falseOut_valid : std_logic;
  signal cond_br45_falseOut_ready : std_logic;
  signal buffer39_outs : std_logic_vector(0 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal cond_br46_trueOut_valid : std_logic;
  signal cond_br46_trueOut_ready : std_logic;
  signal cond_br46_falseOut_valid : std_logic;
  signal cond_br46_falseOut_ready : std_logic;
  signal buffer40_outs : std_logic_vector(0 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal cond_br47_trueOut : std_logic_vector(31 downto 0);
  signal cond_br47_trueOut_valid : std_logic;
  signal cond_br47_trueOut_ready : std_logic;
  signal cond_br47_falseOut : std_logic_vector(31 downto 0);
  signal cond_br47_falseOut_valid : std_logic;
  signal cond_br47_falseOut_ready : std_logic;
  signal buffer41_outs : std_logic_vector(0 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal buffer42_outs : std_logic_vector(31 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal cond_br48_trueOut : std_logic_vector(5 downto 0);
  signal cond_br48_trueOut_valid : std_logic;
  signal cond_br48_trueOut_ready : std_logic;
  signal cond_br48_falseOut : std_logic_vector(5 downto 0);
  signal cond_br48_falseOut_valid : std_logic;
  signal cond_br48_falseOut_ready : std_logic;
  signal buffer43_outs : std_logic_vector(0 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal buffer44_outs : std_logic_vector(5 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal extsi27_outs : std_logic_vector(10 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal cond_br49_trueOut_valid : std_logic;
  signal cond_br49_trueOut_ready : std_logic;
  signal cond_br49_falseOut_valid : std_logic;
  signal cond_br49_falseOut_ready : std_logic;
  signal buffer45_outs : std_logic_vector(0 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal cond_br50_trueOut_valid : std_logic;
  signal cond_br50_trueOut_ready : std_logic;
  signal cond_br50_falseOut_valid : std_logic;
  signal cond_br50_falseOut_ready : std_logic;
  signal buffer46_outs : std_logic_vector(0 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal cond_br51_trueOut_valid : std_logic;
  signal cond_br51_trueOut_ready : std_logic;
  signal cond_br51_falseOut_valid : std_logic;
  signal cond_br51_falseOut_ready : std_logic;
  signal buffer47_outs : std_logic_vector(0 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal mux21_outs : std_logic_vector(31 downto 0);
  signal mux21_outs_valid : std_logic;
  signal mux21_outs_ready : std_logic;
  signal buffer48_outs : std_logic_vector(0 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal mux22_outs_valid : std_logic;
  signal mux22_outs_ready : std_logic;
  signal buffer49_outs : std_logic_vector(0 downto 0);
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal mux23_outs : std_logic_vector(31 downto 0);
  signal mux23_outs_valid : std_logic;
  signal mux23_outs_ready : std_logic;
  signal buffer50_outs : std_logic_vector(0 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal mux24_outs : std_logic_vector(10 downto 0);
  signal mux24_outs_valid : std_logic;
  signal mux24_outs_ready : std_logic;
  signal buffer51_outs : std_logic_vector(0 downto 0);
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(10 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal extsi28_outs : std_logic_vector(31 downto 0);
  signal extsi28_outs_valid : std_logic;
  signal extsi28_outs_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal mux25_outs_valid : std_logic;
  signal mux25_outs_ready : std_logic;
  signal buffer52_outs : std_logic_vector(0 downto 0);
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal mux26_outs_valid : std_logic;
  signal mux26_outs_ready : std_logic;
  signal buffer53_outs : std_logic_vector(0 downto 0);
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal mux27_outs_valid : std_logic;
  signal mux27_outs_ready : std_logic;
  signal buffer54_outs : std_logic_vector(0 downto 0);
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal init19_outs : std_logic_vector(0 downto 0);
  signal init19_outs_valid : std_logic;
  signal init19_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(0 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(0 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(0 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal fork18_outs_3 : std_logic_vector(0 downto 0);
  signal fork18_outs_3_valid : std_logic;
  signal fork18_outs_3_ready : std_logic;
  signal fork18_outs_4 : std_logic_vector(0 downto 0);
  signal fork18_outs_4_valid : std_logic;
  signal fork18_outs_4_ready : std_logic;
  signal fork18_outs_5 : std_logic_vector(0 downto 0);
  signal fork18_outs_5_valid : std_logic;
  signal fork18_outs_5_ready : std_logic;
  signal fork18_outs_6 : std_logic_vector(0 downto 0);
  signal fork18_outs_6_valid : std_logic;
  signal fork18_outs_6_ready : std_logic;
  signal mux4_outs : std_logic_vector(5 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer31_outs : std_logic_vector(5 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer32_outs : std_logic_vector(5 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(5 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(5 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(5 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal fork19_outs_3 : std_logic_vector(5 downto 0);
  signal fork19_outs_3_valid : std_logic;
  signal fork19_outs_3_ready : std_logic;
  signal fork19_outs_4 : std_logic_vector(5 downto 0);
  signal fork19_outs_4_valid : std_logic;
  signal fork19_outs_4_ready : std_logic;
  signal extsi29_outs : std_logic_vector(8 downto 0);
  signal extsi29_outs_valid : std_logic;
  signal extsi29_outs_ready : std_logic;
  signal buffer57_outs : std_logic_vector(5 downto 0);
  signal buffer57_outs_valid : std_logic;
  signal buffer57_outs_ready : std_logic;
  signal extsi30_outs : std_logic_vector(6 downto 0);
  signal extsi30_outs_valid : std_logic;
  signal extsi30_outs_ready : std_logic;
  signal extsi31_outs : std_logic_vector(31 downto 0);
  signal extsi31_outs_valid : std_logic;
  signal extsi31_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal trunci5_outs : std_logic_vector(4 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal buffer60_outs : std_logic_vector(5 downto 0);
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(5 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer33_outs : std_logic_vector(5 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal buffer34_outs : std_logic_vector(5 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(5 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(5 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal extsi32_outs : std_logic_vector(31 downto 0);
  signal extsi32_outs_valid : std_logic;
  signal extsi32_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(31 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(31 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal buffer66_outs : std_logic_vector(31 downto 0);
  signal buffer66_outs_valid : std_logic;
  signal buffer66_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer63_outs : std_logic_vector(0 downto 0);
  signal buffer63_outs_valid : std_logic;
  signal buffer63_outs_ready : std_logic;
  signal buffer35_outs : std_logic_vector(31 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(31 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(31 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(0 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(0 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal fork24_outs_2 : std_logic_vector(0 downto 0);
  signal fork24_outs_2_valid : std_logic;
  signal fork24_outs_2_ready : std_logic;
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal constant26_outs : std_logic_vector(1 downto 0);
  signal constant26_outs_valid : std_logic;
  signal constant26_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(31 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant27_outs : std_logic_vector(5 downto 0);
  signal constant27_outs_valid : std_logic;
  signal constant27_outs_ready : std_logic;
  signal extsi33_outs : std_logic_vector(6 downto 0);
  signal extsi33_outs_valid : std_logic;
  signal extsi33_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant28_outs : std_logic_vector(1 downto 0);
  signal constant28_outs_valid : std_logic;
  signal constant28_outs_ready : std_logic;
  signal extsi34_outs : std_logic_vector(6 downto 0);
  signal extsi34_outs_valid : std_logic;
  signal extsi34_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant29_outs : std_logic_vector(3 downto 0);
  signal constant29_outs_valid : std_logic;
  signal constant29_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(31 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant30_outs : std_logic_vector(2 downto 0);
  signal constant30_outs_valid : std_logic;
  signal constant30_outs_ready : std_logic;
  signal extsi14_outs : std_logic_vector(31 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal buffer64_outs : std_logic_vector(31 downto 0);
  signal buffer64_outs_valid : std_logic;
  signal buffer64_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(31 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(31 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal fork26_outs_2 : std_logic_vector(31 downto 0);
  signal fork26_outs_2_valid : std_logic;
  signal fork26_outs_2_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(0 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(0 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal buffer26_outs : std_logic_vector(31 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal cmpi4_result : std_logic_vector(0 downto 0);
  signal cmpi4_result_valid : std_logic;
  signal cmpi4_result_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(0 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(0 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal cmpi5_result : std_logic_vector(0 downto 0);
  signal cmpi5_result_valid : std_logic;
  signal cmpi5_result_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(0 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(0 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal cond_br25_trueOut_valid : std_logic;
  signal cond_br25_trueOut_ready : std_logic;
  signal cond_br25_falseOut_valid : std_logic;
  signal cond_br25_falseOut_ready : std_logic;
  signal buffer68_outs : std_logic_vector(0 downto 0);
  signal buffer68_outs_valid : std_logic;
  signal buffer68_outs_ready : std_logic;
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal cond_br26_trueOut_valid : std_logic;
  signal cond_br26_trueOut_ready : std_logic;
  signal cond_br26_falseOut_valid : std_logic;
  signal cond_br26_falseOut_ready : std_logic;
  signal buffer69_outs : std_logic_vector(0 downto 0);
  signal buffer69_outs_valid : std_logic;
  signal buffer69_outs_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal cond_br27_trueOut_valid : std_logic;
  signal cond_br27_trueOut_ready : std_logic;
  signal cond_br27_falseOut_valid : std_logic;
  signal cond_br27_falseOut_ready : std_logic;
  signal buffer70_outs : std_logic_vector(0 downto 0);
  signal buffer70_outs_valid : std_logic;
  signal buffer70_outs_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal mux28_outs_valid : std_logic;
  signal mux28_outs_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal mux29_outs_valid : std_logic;
  signal mux29_outs_ready : std_logic;
  signal source12_outs_valid : std_logic;
  signal source12_outs_ready : std_logic;
  signal mux30_outs_valid : std_logic;
  signal mux30_outs_ready : std_logic;
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal buffer55_outs_valid : std_logic;
  signal buffer55_outs_ready : std_logic;
  signal buffer56_outs_valid : std_logic;
  signal buffer56_outs_ready : std_logic;
  signal join0_outs_valid : std_logic;
  signal join0_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal buffer74_outs : std_logic_vector(31 downto 0);
  signal buffer74_outs_valid : std_logic;
  signal buffer74_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(4 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(4 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal shli2_result : std_logic_vector(31 downto 0);
  signal shli2_result_valid : std_logic;
  signal shli2_result_ready : std_logic;
  signal buffer75_outs : std_logic_vector(31 downto 0);
  signal buffer75_outs_valid : std_logic;
  signal buffer75_outs_ready : std_logic;
  signal buffer58_outs : std_logic_vector(31 downto 0);
  signal buffer58_outs_valid : std_logic;
  signal buffer58_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(8 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer76_outs : std_logic_vector(31 downto 0);
  signal buffer76_outs_valid : std_logic;
  signal buffer76_outs_ready : std_logic;
  signal buffer59_outs : std_logic_vector(31 downto 0);
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal trunci8_outs : std_logic_vector(8 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal addi6_result : std_logic_vector(8 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer61_outs : std_logic_vector(8 downto 0);
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(8 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal load4_addrOut : std_logic_vector(8 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal buffer77_outs : std_logic_vector(31 downto 0);
  signal buffer77_outs_valid : std_logic;
  signal buffer77_outs_ready : std_logic;
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal buffer2_outs : std_logic_vector(5 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork30_outs_0 : std_logic_vector(5 downto 0);
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1 : std_logic_vector(5 downto 0);
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal extsi35_outs : std_logic_vector(31 downto 0);
  signal extsi35_outs_valid : std_logic;
  signal extsi35_outs_ready : std_logic;
  signal init26_outs : std_logic_vector(31 downto 0);
  signal init26_outs_valid : std_logic;
  signal init26_outs_ready : std_logic;
  signal fork31_outs_0 : std_logic_vector(31 downto 0);
  signal fork31_outs_0_valid : std_logic;
  signal fork31_outs_0_ready : std_logic;
  signal fork31_outs_1 : std_logic_vector(31 downto 0);
  signal fork31_outs_1_valid : std_logic;
  signal fork31_outs_1_ready : std_logic;
  signal init27_outs : std_logic_vector(31 downto 0);
  signal init27_outs_valid : std_logic;
  signal init27_outs_ready : std_logic;
  signal buffer80_outs : std_logic_vector(31 downto 0);
  signal buffer80_outs_valid : std_logic;
  signal buffer80_outs_ready : std_logic;
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork32_outs_0_valid : std_logic;
  signal fork32_outs_0_ready : std_logic;
  signal fork32_outs_1_valid : std_logic;
  signal fork32_outs_1_ready : std_logic;
  signal init28_outs_valid : std_logic;
  signal init28_outs_ready : std_logic;
  signal fork33_outs_0_valid : std_logic;
  signal fork33_outs_0_ready : std_logic;
  signal fork33_outs_1_valid : std_logic;
  signal fork33_outs_1_ready : std_logic;
  signal init29_outs_valid : std_logic;
  signal init29_outs_ready : std_logic;
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal init30_outs_valid : std_logic;
  signal init30_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(4 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal addi3_result : std_logic_vector(6 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer62_outs : std_logic_vector(6 downto 0);
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(6 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(6 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal trunci9_outs : std_logic_vector(5 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer65_outs : std_logic_vector(0 downto 0);
  signal buffer65_outs_valid : std_logic;
  signal buffer65_outs_ready : std_logic;
  signal fork36_outs_0 : std_logic_vector(0 downto 0);
  signal fork36_outs_0_valid : std_logic;
  signal fork36_outs_0_ready : std_logic;
  signal fork36_outs_1 : std_logic_vector(0 downto 0);
  signal fork36_outs_1_valid : std_logic;
  signal fork36_outs_1_ready : std_logic;
  signal fork36_outs_2 : std_logic_vector(0 downto 0);
  signal fork36_outs_2_valid : std_logic;
  signal fork36_outs_2_ready : std_logic;
  signal fork36_outs_3 : std_logic_vector(0 downto 0);
  signal fork36_outs_3_valid : std_logic;
  signal fork36_outs_3_ready : std_logic;
  signal fork36_outs_4 : std_logic_vector(0 downto 0);
  signal fork36_outs_4_valid : std_logic;
  signal fork36_outs_4_ready : std_logic;
  signal fork36_outs_5 : std_logic_vector(0 downto 0);
  signal fork36_outs_5_valid : std_logic;
  signal fork36_outs_5_ready : std_logic;
  signal fork36_outs_6 : std_logic_vector(0 downto 0);
  signal fork36_outs_6_valid : std_logic;
  signal fork36_outs_6_ready : std_logic;
  signal fork36_outs_7 : std_logic_vector(0 downto 0);
  signal fork36_outs_7_valid : std_logic;
  signal fork36_outs_7_ready : std_logic;
  signal fork36_outs_8 : std_logic_vector(0 downto 0);
  signal fork36_outs_8_valid : std_logic;
  signal fork36_outs_8_ready : std_logic;
  signal fork36_outs_9 : std_logic_vector(0 downto 0);
  signal fork36_outs_9_valid : std_logic;
  signal fork36_outs_9_ready : std_logic;
  signal fork36_outs_10 : std_logic_vector(0 downto 0);
  signal fork36_outs_10_valid : std_logic;
  signal fork36_outs_10_ready : std_logic;
  signal fork36_outs_11 : std_logic_vector(0 downto 0);
  signal fork36_outs_11_valid : std_logic;
  signal fork36_outs_11_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(5 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(5 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(5 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(5 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(31 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(31 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal buffer86_outs : std_logic_vector(0 downto 0);
  signal buffer86_outs_valid : std_logic;
  signal buffer86_outs_ready : std_logic;
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal buffer88_outs : std_logic_vector(0 downto 0);
  signal buffer88_outs_valid : std_logic;
  signal buffer88_outs_ready : std_logic;
  signal cond_br52_trueOut : std_logic_vector(31 downto 0);
  signal cond_br52_trueOut_valid : std_logic;
  signal cond_br52_trueOut_ready : std_logic;
  signal cond_br52_falseOut : std_logic_vector(31 downto 0);
  signal cond_br52_falseOut_valid : std_logic;
  signal cond_br52_falseOut_ready : std_logic;
  signal buffer89_outs : std_logic_vector(0 downto 0);
  signal buffer89_outs_valid : std_logic;
  signal buffer89_outs_ready : std_logic;
  signal cond_br53_trueOut_valid : std_logic;
  signal cond_br53_trueOut_ready : std_logic;
  signal cond_br53_falseOut_valid : std_logic;
  signal cond_br53_falseOut_ready : std_logic;
  signal buffer90_outs : std_logic_vector(0 downto 0);
  signal buffer90_outs_valid : std_logic;
  signal buffer90_outs_ready : std_logic;
  signal cond_br54_trueOut_valid : std_logic;
  signal cond_br54_trueOut_ready : std_logic;
  signal cond_br54_falseOut_valid : std_logic;
  signal cond_br54_falseOut_ready : std_logic;
  signal buffer91_outs : std_logic_vector(0 downto 0);
  signal buffer91_outs_valid : std_logic;
  signal buffer91_outs_ready : std_logic;
  signal cond_br55_trueOut_valid : std_logic;
  signal cond_br55_trueOut_ready : std_logic;
  signal cond_br55_falseOut_valid : std_logic;
  signal cond_br55_falseOut_ready : std_logic;
  signal buffer92_outs : std_logic_vector(0 downto 0);
  signal buffer92_outs_valid : std_logic;
  signal buffer92_outs_ready : std_logic;
  signal cond_br56_trueOut : std_logic_vector(31 downto 0);
  signal cond_br56_trueOut_valid : std_logic;
  signal cond_br56_trueOut_ready : std_logic;
  signal cond_br56_falseOut : std_logic_vector(31 downto 0);
  signal cond_br56_falseOut_valid : std_logic;
  signal cond_br56_falseOut_ready : std_logic;
  signal buffer93_outs : std_logic_vector(0 downto 0);
  signal buffer93_outs_valid : std_logic;
  signal buffer93_outs_ready : std_logic;
  signal cond_br57_trueOut : std_logic_vector(5 downto 0);
  signal cond_br57_trueOut_valid : std_logic;
  signal cond_br57_trueOut_ready : std_logic;
  signal cond_br57_falseOut : std_logic_vector(5 downto 0);
  signal cond_br57_falseOut_valid : std_logic;
  signal cond_br57_falseOut_ready : std_logic;
  signal buffer94_outs : std_logic_vector(0 downto 0);
  signal buffer94_outs_valid : std_logic;
  signal buffer94_outs_ready : std_logic;
  signal extsi36_outs : std_logic_vector(10 downto 0);
  signal extsi36_outs_valid : std_logic;
  signal extsi36_outs_ready : std_logic;
  signal cond_br58_trueOut_valid : std_logic;
  signal cond_br58_trueOut_ready : std_logic;
  signal cond_br58_falseOut_valid : std_logic;
  signal cond_br58_falseOut_ready : std_logic;
  signal buffer95_outs : std_logic_vector(0 downto 0);
  signal buffer95_outs_valid : std_logic;
  signal buffer95_outs_ready : std_logic;
  signal buffer67_outs : std_logic_vector(5 downto 0);
  signal buffer67_outs_valid : std_logic;
  signal buffer67_outs_ready : std_logic;
  signal fork37_outs_0 : std_logic_vector(5 downto 0);
  signal fork37_outs_0_valid : std_logic;
  signal fork37_outs_0_ready : std_logic;
  signal fork37_outs_1 : std_logic_vector(5 downto 0);
  signal fork37_outs_1_valid : std_logic;
  signal fork37_outs_1_ready : std_logic;
  signal extsi37_outs : std_logic_vector(6 downto 0);
  signal extsi37_outs_valid : std_logic;
  signal extsi37_outs_ready : std_logic;
  signal extsi38_outs : std_logic_vector(31 downto 0);
  signal extsi38_outs_valid : std_logic;
  signal extsi38_outs_ready : std_logic;
  signal fork38_outs_0 : std_logic_vector(31 downto 0);
  signal fork38_outs_0_valid : std_logic;
  signal fork38_outs_0_ready : std_logic;
  signal fork38_outs_1 : std_logic_vector(31 downto 0);
  signal fork38_outs_1_valid : std_logic;
  signal fork38_outs_1_ready : std_logic;
  signal fork39_outs_0_valid : std_logic;
  signal fork39_outs_0_ready : std_logic;
  signal fork39_outs_1_valid : std_logic;
  signal fork39_outs_1_ready : std_logic;
  signal constant31_outs : std_logic_vector(1 downto 0);
  signal constant31_outs_valid : std_logic;
  signal constant31_outs_ready : std_logic;
  signal extsi15_outs : std_logic_vector(31 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant32_outs : std_logic_vector(5 downto 0);
  signal constant32_outs_valid : std_logic;
  signal constant32_outs_ready : std_logic;
  signal extsi39_outs : std_logic_vector(6 downto 0);
  signal extsi39_outs_valid : std_logic;
  signal extsi39_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant33_outs : std_logic_vector(1 downto 0);
  signal constant33_outs_valid : std_logic;
  signal constant33_outs_ready : std_logic;
  signal extsi40_outs : std_logic_vector(6 downto 0);
  signal extsi40_outs_valid : std_logic;
  signal extsi40_outs_ready : std_logic;
  signal gate2_outs : std_logic_vector(31 downto 0);
  signal gate2_outs_valid : std_logic;
  signal gate2_outs_ready : std_logic;
  signal fork40_outs_0 : std_logic_vector(31 downto 0);
  signal fork40_outs_0_valid : std_logic;
  signal fork40_outs_0_ready : std_logic;
  signal fork40_outs_1 : std_logic_vector(31 downto 0);
  signal fork40_outs_1_valid : std_logic;
  signal fork40_outs_1_ready : std_logic;
  signal fork40_outs_2 : std_logic_vector(31 downto 0);
  signal fork40_outs_2_valid : std_logic;
  signal fork40_outs_2_ready : std_logic;
  signal cmpi6_result : std_logic_vector(0 downto 0);
  signal cmpi6_result_valid : std_logic;
  signal cmpi6_result_ready : std_logic;
  signal fork41_outs_0 : std_logic_vector(0 downto 0);
  signal fork41_outs_0_valid : std_logic;
  signal fork41_outs_0_ready : std_logic;
  signal fork41_outs_1 : std_logic_vector(0 downto 0);
  signal fork41_outs_1_valid : std_logic;
  signal fork41_outs_1_ready : std_logic;
  signal cmpi7_result : std_logic_vector(0 downto 0);
  signal cmpi7_result_valid : std_logic;
  signal cmpi7_result_ready : std_logic;
  signal fork42_outs_0 : std_logic_vector(0 downto 0);
  signal fork42_outs_0_valid : std_logic;
  signal fork42_outs_0_ready : std_logic;
  signal fork42_outs_1 : std_logic_vector(0 downto 0);
  signal fork42_outs_1_valid : std_logic;
  signal fork42_outs_1_ready : std_logic;
  signal cmpi8_result : std_logic_vector(0 downto 0);
  signal cmpi8_result_valid : std_logic;
  signal cmpi8_result_ready : std_logic;
  signal fork43_outs_0 : std_logic_vector(0 downto 0);
  signal fork43_outs_0_valid : std_logic;
  signal fork43_outs_0_ready : std_logic;
  signal fork43_outs_1 : std_logic_vector(0 downto 0);
  signal fork43_outs_1_valid : std_logic;
  signal fork43_outs_1_ready : std_logic;
  signal cond_br42_trueOut_valid : std_logic;
  signal cond_br42_trueOut_ready : std_logic;
  signal cond_br42_falseOut_valid : std_logic;
  signal cond_br42_falseOut_ready : std_logic;
  signal cond_br43_trueOut_valid : std_logic;
  signal cond_br43_trueOut_ready : std_logic;
  signal cond_br43_falseOut_valid : std_logic;
  signal cond_br43_falseOut_ready : std_logic;
  signal cond_br44_trueOut_valid : std_logic;
  signal cond_br44_trueOut_ready : std_logic;
  signal cond_br44_falseOut_valid : std_logic;
  signal cond_br44_falseOut_ready : std_logic;
  signal source13_outs_valid : std_logic;
  signal source13_outs_ready : std_logic;
  signal mux31_outs_valid : std_logic;
  signal mux31_outs_ready : std_logic;
  signal source14_outs_valid : std_logic;
  signal source14_outs_ready : std_logic;
  signal mux32_outs_valid : std_logic;
  signal mux32_outs_ready : std_logic;
  signal source15_outs_valid : std_logic;
  signal source15_outs_ready : std_logic;
  signal mux33_outs_valid : std_logic;
  signal mux33_outs_ready : std_logic;
  signal buffer109_outs : std_logic_vector(0 downto 0);
  signal buffer109_outs_valid : std_logic;
  signal buffer109_outs_ready : std_logic;
  signal join1_outs_valid : std_logic;
  signal join1_outs_ready : std_logic;
  signal buffer71_outs_valid : std_logic;
  signal buffer71_outs_ready : std_logic;
  signal gate3_outs : std_logic_vector(31 downto 0);
  signal gate3_outs_valid : std_logic;
  signal gate3_outs_ready : std_logic;
  signal buffer110_outs : std_logic_vector(31 downto 0);
  signal buffer110_outs_valid : std_logic;
  signal buffer110_outs_ready : std_logic;
  signal trunci10_outs : std_logic_vector(4 downto 0);
  signal trunci10_outs_valid : std_logic;
  signal trunci10_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(4 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal store1_doneOut_valid : std_logic;
  signal store1_doneOut_ready : std_logic;
  signal addi4_result : std_logic_vector(6 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer72_outs : std_logic_vector(6 downto 0);
  signal buffer72_outs_valid : std_logic;
  signal buffer72_outs_ready : std_logic;
  signal fork44_outs_0 : std_logic_vector(6 downto 0);
  signal fork44_outs_0_valid : std_logic;
  signal fork44_outs_0_ready : std_logic;
  signal fork44_outs_1 : std_logic_vector(6 downto 0);
  signal fork44_outs_1_valid : std_logic;
  signal fork44_outs_1_ready : std_logic;
  signal trunci11_outs : std_logic_vector(5 downto 0);
  signal trunci11_outs_valid : std_logic;
  signal trunci11_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer73_outs : std_logic_vector(0 downto 0);
  signal buffer73_outs_valid : std_logic;
  signal buffer73_outs_ready : std_logic;
  signal fork45_outs_0 : std_logic_vector(0 downto 0);
  signal fork45_outs_0_valid : std_logic;
  signal fork45_outs_0_ready : std_logic;
  signal fork45_outs_1 : std_logic_vector(0 downto 0);
  signal fork45_outs_1_valid : std_logic;
  signal fork45_outs_1_ready : std_logic;
  signal fork45_outs_2 : std_logic_vector(0 downto 0);
  signal fork45_outs_2_valid : std_logic;
  signal fork45_outs_2_ready : std_logic;
  signal fork45_outs_3 : std_logic_vector(0 downto 0);
  signal fork45_outs_3_valid : std_logic;
  signal fork45_outs_3_ready : std_logic;
  signal fork45_outs_4 : std_logic_vector(0 downto 0);
  signal fork45_outs_4_valid : std_logic;
  signal fork45_outs_4_ready : std_logic;
  signal fork45_outs_5 : std_logic_vector(0 downto 0);
  signal fork45_outs_5_valid : std_logic;
  signal fork45_outs_5_ready : std_logic;
  signal fork45_outs_6 : std_logic_vector(0 downto 0);
  signal fork45_outs_6_valid : std_logic;
  signal fork45_outs_6_ready : std_logic;
  signal fork45_outs_7 : std_logic_vector(0 downto 0);
  signal fork45_outs_7_valid : std_logic;
  signal fork45_outs_7_ready : std_logic;
  signal fork45_outs_8 : std_logic_vector(0 downto 0);
  signal fork45_outs_8_valid : std_logic;
  signal fork45_outs_8_ready : std_logic;
  signal fork45_outs_9 : std_logic_vector(0 downto 0);
  signal fork45_outs_9_valid : std_logic;
  signal fork45_outs_9_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(5 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(5 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal fork46_outs_0_valid : std_logic;
  signal fork46_outs_0_ready : std_logic;
  signal fork46_outs_1_valid : std_logic;
  signal fork46_outs_1_ready : std_logic;
  signal fork46_outs_2_valid : std_logic;
  signal fork46_outs_2_ready : std_logic;
  signal fork46_outs_3_valid : std_logic;
  signal fork46_outs_3_ready : std_logic;

begin

  A_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= A_end_ready;
  x_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= x_end_ready;
  y_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= y_end_ready;
  tmp_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= tmp_end_ready;
  end_valid <= fork0_outs_2_valid;
  fork0_outs_2_ready <= end_ready;
  A_loadEn <= mem_controller5_loadEn;
  A_loadAddr <= mem_controller5_loadAddr;
  A_storeEn <= mem_controller5_storeEn;
  A_storeAddr <= mem_controller5_storeAddr;
  A_storeData <= mem_controller5_storeData;
  x_loadEn <= mem_controller4_loadEn;
  x_loadAddr <= mem_controller4_loadAddr;
  x_storeEn <= mem_controller4_storeEn;
  x_storeAddr <= mem_controller4_storeAddr;
  x_storeData <= mem_controller4_storeData;
  y_loadEn <= mem_controller3_loadEn;
  y_loadAddr <= mem_controller3_loadAddr;
  y_storeEn <= mem_controller3_storeEn;
  y_storeAddr <= mem_controller3_storeAddr;
  y_storeData <= mem_controller3_storeData;
  tmp_loadEn <= mem_controller2_loadEn;
  tmp_loadAddr <= mem_controller2_loadAddr;
  tmp_storeEn <= mem_controller2_storeEn;
  tmp_storeAddr <= mem_controller2_storeAddr;
  tmp_storeData <= mem_controller2_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(8)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_valid(3) => fork0_outs_3_valid,
      outs_valid(4) => fork0_outs_4_valid,
      outs_valid(5) => fork0_outs_5_valid,
      outs_valid(6) => fork0_outs_6_valid,
      outs_valid(7) => fork0_outs_7_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready,
      outs_ready(3) => fork0_outs_3_ready,
      outs_ready(4) => fork0_outs_4_ready,
      outs_ready(5) => fork0_outs_5_ready,
      outs_ready(6) => fork0_outs_6_ready,
      outs_ready(7) => fork0_outs_7_ready
    );

  mem_controller2 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 5)
    port map(
      loadData => tmp_loadData,
      memStart_valid => tmp_start_valid,
      memStart_ready => tmp_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrl(0) => extsi15_outs,
      ctrl_valid(0) => extsi15_outs_valid,
      ctrl_ready(0) => extsi15_outs_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork46_outs_3_valid,
      ctrlEnd_ready => fork46_outs_3_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller2_ldData_0,
      ldData_valid(0) => mem_controller2_ldData_0_valid,
      ldData_ready(0) => mem_controller2_ldData_0_ready,
      stDone_valid(0) => mem_controller2_stDone_0_valid,
      stDone_ready(0) => mem_controller2_stDone_0_ready,
      memEnd_valid => mem_controller2_memEnd_valid,
      memEnd_ready => mem_controller2_memEnd_ready,
      loadEn => mem_controller2_loadEn,
      loadAddr => mem_controller2_loadAddr,
      storeEn => mem_controller2_storeEn,
      storeAddr => mem_controller2_storeAddr,
      storeData => mem_controller2_storeData
    );

  mem_controller3 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 5)
    port map(
      loadData => y_loadData,
      memStart_valid => y_start_valid,
      memStart_ready => y_start_ready,
      ctrl(0) => extsi10_outs,
      ctrl_valid(0) => extsi10_outs_valid,
      ctrl_ready(0) => extsi10_outs_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => fork46_outs_2_valid,
      ctrlEnd_ready => fork46_outs_2_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller3_ldData_0,
      ldData_valid(0) => mem_controller3_ldData_0_valid,
      ldData_ready(0) => mem_controller3_ldData_0_ready,
      stDone_valid(0) => mem_controller3_stDone_0_valid,
      stDone_ready(0) => mem_controller3_stDone_0_ready,
      memEnd_valid => mem_controller3_memEnd_valid,
      memEnd_ready => mem_controller3_memEnd_ready,
      loadEn => mem_controller3_loadEn,
      loadAddr => mem_controller3_loadAddr,
      storeEn => mem_controller3_storeEn,
      storeAddr => mem_controller3_storeAddr,
      storeData => mem_controller3_storeData
    );

  mem_controller4 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => x_loadData,
      memStart_valid => x_start_valid,
      memStart_ready => x_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork46_outs_1_valid,
      ctrlEnd_ready => fork46_outs_1_ready,
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

  mem_controller5 : entity work.mem_controller_storeless(arch) generic map(2, 32, 9)
    port map(
      loadData => A_loadData,
      memStart_valid => A_start_valid,
      memStart_ready => A_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr(1) => load4_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_valid(1) => load4_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ldAddr_ready(1) => load4_addrOut_ready,
      ctrlEnd_valid => fork46_outs_0_valid,
      ctrlEnd_ready => fork46_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller5_ldData_0,
      ldData(1) => mem_controller5_ldData_1,
      ldData_valid(0) => mem_controller5_ldData_0_valid,
      ldData_valid(1) => mem_controller5_ldData_1_valid,
      ldData_ready(0) => mem_controller5_ldData_0_ready,
      ldData_ready(1) => mem_controller5_ldData_1_ready,
      memEnd_valid => mem_controller5_memEnd_valid,
      memEnd_ready => mem_controller5_memEnd_ready,
      loadEn => mem_controller5_loadEn,
      loadAddr => mem_controller5_loadAddr,
      storeEn => mem_controller5_storeEn,
      storeAddr => mem_controller5_storeAddr,
      storeData => mem_controller5_storeData
    );

  constant0 : entity work.handshake_constant_0(arch) generic map(11)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(3, 11)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
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

  extsi0 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi0_outs,
      outs_valid => extsi0_outs_valid,
      outs_ready => extsi0_outs_ready
    );

  extsi1 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_2,
      ins_valid => fork1_outs_2_valid,
      ins_ready => fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  constant2 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi20 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => extsi0_outs,
      ins(1) => cond_br52_trueOut,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => cond_br52_trueOut_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => cond_br52_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux10 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_2,
      index_valid => fork2_outs_2_valid,
      index_ready => fork2_outs_2_ready,
      ins_valid(0) => fork0_outs_6_valid,
      ins_valid(1) => cond_br58_trueOut_valid,
      ins_ready(0) => fork0_outs_6_ready,
      ins_ready(1) => cond_br58_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_3,
      index_valid => fork2_outs_3_valid,
      index_ready => fork2_outs_3_ready,
      ins(0) => extsi1_outs,
      ins(1) => cond_br56_trueOut,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => cond_br56_trueOut_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => cond_br56_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  mux13 : entity work.mux(arch) generic map(2, 11, 1)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => fork1_outs_0,
      ins(1) => extsi36_outs,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => extsi36_outs_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => extsi36_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux13_outs,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  mux15 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_4,
      index_valid => fork2_outs_4_valid,
      index_ready => fork2_outs_4_ready,
      ins_valid(0) => fork0_outs_5_valid,
      ins_valid(1) => cond_br54_trueOut_valid,
      ins_ready(0) => fork0_outs_5_ready,
      ins_ready(1) => cond_br54_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  mux17 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_5,
      index_valid => fork2_outs_5_valid,
      index_ready => fork2_outs_5_ready,
      ins_valid(0) => fork0_outs_4_valid,
      ins_valid(1) => cond_br53_trueOut_valid,
      ins_ready(0) => fork0_outs_4_ready,
      ins_ready(1) => cond_br53_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux17_outs_valid,
      outs_ready => mux17_outs_ready
    );

  mux18 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_6,
      index_valid => fork2_outs_6_valid,
      index_ready => fork2_outs_6_ready,
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => cond_br55_trueOut_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => cond_br55_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux18_outs_valid,
      outs_ready => mux18_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork45_outs_7,
      ins_valid => fork45_outs_7_valid,
      ins_ready => fork45_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(7, 1)
    port map(
      ins => init0_outs,
      ins_valid => init0_outs_valid,
      ins_ready => init0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs(3) => fork2_outs_3,
      outs(4) => fork2_outs_4,
      outs(5) => fork2_outs_5,
      outs(6) => fork2_outs_6,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_valid(5) => fork2_outs_5_valid,
      outs_valid(6) => fork2_outs_6_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready,
      outs_ready(5) => fork2_outs_5_ready,
      outs_ready(6) => fork2_outs_6_ready
    );

  unbundle1 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle1_outs_0_valid,
      outs_ready => unbundle1_outs_0_ready,
      outs => unbundle1_outs_1
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi20_outs,
      ins(1) => cond_br12_trueOut,
      ins_valid(0) => extsi20_outs_valid,
      ins_valid(1) => cond_br12_trueOut_valid,
      ins_ready(0) => extsi20_outs_ready,
      ins_ready(1) => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer11 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs(2) => fork3_outs_2,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_valid(2) => fork3_outs_2_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready,
      outs_ready(2) => fork3_outs_2_ready
    );

  trunci0 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_7_valid,
      ins_valid(1) => cond_br13_trueOut_valid,
      ins_ready(0) => fork0_outs_7_ready,
      ins_ready(1) => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork4 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  constant17 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork4_outs_0_valid,
      ctrl_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant17_outs,
      outs_valid => constant17_outs_valid,
      outs_ready => constant17_outs_ready
    );

  buffer0 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork3_outs_2,
      ins_valid => fork3_outs_2_valid,
      ins_ready => fork3_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  extsi21 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi21_outs,
      ins_valid => extsi21_outs_valid,
      ins_ready => extsi21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  init14 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => fork5_outs_0,
      ins_valid => fork5_outs_0_valid,
      ins_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => init14_outs,
      outs_valid => init14_outs_valid,
      outs_ready => init14_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init14_outs,
      ins_valid => init14_outs_valid,
      ins_ready => init14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  init15 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => fork6_outs_1,
      ins_valid => fork6_outs_1_valid,
      ins_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => init15_outs,
      outs_valid => init15_outs_valid,
      outs_ready => init15_outs_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle1_outs_0_valid,
      ins_ready => unbundle1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork7 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  init16 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => init16_outs_valid,
      outs_ready => init16_outs_ready
    );

  fork8 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init16_outs_valid,
      ins_ready => init16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  init17 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => init17_outs_valid,
      outs_ready => init17_outs_ready
    );

  fork9 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init17_outs_valid,
      ins_ready => init17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  init18 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork9_outs_1_valid,
      ins_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => init18_outs_valid,
      outs_ready => init18_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller2_ldData_0,
      dataFromMem_valid => mem_controller2_ldData_0_valid,
      dataFromMem_ready => mem_controller2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  extsi19 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant17_outs,
      ins_valid => constant17_outs_valid,
      ins_ready => constant17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork14_outs_1,
      index_valid => fork14_outs_1_valid,
      index_ready => fork14_outs_1_ready,
      ins(0) => extsi19_outs,
      ins(1) => cond_br3_trueOut,
      ins_valid(0) => extsi19_outs_valid,
      ins_valid(1) => cond_br3_trueOut_valid,
      ins_ready(0) => extsi19_outs_ready,
      ins_ready(1) => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs(2) => fork11_outs_2,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready
    );

  extsi22 : entity work.extsi(arch) generic map(6, 9)
    port map(
      ins => fork11_outs_0,
      ins_valid => fork11_outs_0_valid,
      ins_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  extsi23 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork11_outs_2,
      ins_valid => fork11_outs_2_valid,
      ins_ready => fork11_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_2,
      index_valid => fork14_outs_2_valid,
      index_ready => fork14_outs_2_ready,
      ins(0) => fork10_outs_1,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => fork10_outs_1_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => fork10_outs_1_ready,
      ins_ready(1) => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork14_outs_0,
      index_valid => fork14_outs_0_valid,
      index_ready => fork14_outs_0_ready,
      ins(0) => fork3_outs_1,
      ins(1) => cond_br5_trueOut,
      ins_valid(0) => fork3_outs_1_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => fork3_outs_1_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer15 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  buffer16 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  extsi24 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork12_outs_1,
      ins_valid => fork12_outs_1_valid,
      ins_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi24_outs,
      outs_valid => extsi24_outs_valid,
      outs_ready => extsi24_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi24_outs,
      ins_valid => extsi24_outs_valid,
      ins_ready => extsi24_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork4_outs_1_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => fork4_outs_1_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready
    );

  buffer17 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  fork15 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  constant21 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork15_outs_0_valid,
      ctrl_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant22 : entity work.handshake_constant_2(arch) generic map(6)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  extsi25 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant23 : entity work.handshake_constant_3(arch) generic map(2)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant23_outs,
      outs_valid => constant23_outs_valid,
      outs_ready => constant23_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant23_outs,
      ins_valid => constant23_outs_valid,
      ins_ready => constant23_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant24 : entity work.handshake_constant_4(arch) generic map(4)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant24_outs,
      outs_valid => constant24_outs_valid,
      outs_ready => constant24_outs_ready
    );

  extsi8 : entity work.extsi(arch) generic map(4, 32)
    port map(
      ins => constant24_outs,
      ins_valid => constant24_outs_valid,
      ins_ready => constant24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant25 : entity work.handshake_constant_5(arch) generic map(3)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant25_outs,
      outs_valid => constant25_outs_valid,
      outs_ready => constant25_outs_ready
    );

  extsi9 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant25_outs,
      ins_valid => constant25_outs_valid,
      ins_ready => constant25_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork13_outs_0,
      lhs_valid => fork13_outs_0_valid,
      lhs_ready => fork13_outs_0_ready,
      rhs => extsi9_outs,
      rhs_valid => extsi9_outs_valid,
      rhs_ready => extsi9_outs_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer18 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork13_outs_1,
      lhs_valid => fork13_outs_1_valid,
      lhs_ready => fork13_outs_1_ready,
      rhs => extsi8_outs,
      rhs_valid => extsi8_outs_valid,
      rhs_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer19 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  addi5 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci2_outs,
      lhs_valid => trunci2_outs_valid,
      lhs_ready => trunci2_outs_ready,
      rhs => trunci3_outs,
      rhs_valid => trunci3_outs_valid,
      rhs_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer20 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi22_outs,
      lhs_valid => extsi22_outs_valid,
      lhs_ready => extsi22_outs_ready,
      rhs => buffer20_outs,
      rhs_valid => buffer20_outs_valid,
      rhs_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  load1 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => addi0_result,
      addrIn_valid => addi0_result_valid,
      addrIn_ready => addi0_result_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  load2 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  mulf0 : entity work.mulf(arch_32_2_875333) generic map(32)
    port map(
      lhs => load1_dataOut,
      lhs_valid => load1_dataOut_valid,
      lhs_ready => load1_dataOut_ready,
      rhs => load2_dataOut,
      rhs_valid => load2_dataOut_valid,
      rhs_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  buffer14 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  addf0 : entity work.addf(arch_32_2_922000) generic map(32)
    port map(
      lhs => buffer14_outs,
      lhs_valid => buffer14_outs_valid,
      lhs_ready => buffer14_outs_ready,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi23_outs,
      lhs_valid => extsi23_outs_valid,
      lhs_ready => extsi23_outs_ready,
      rhs => extsi26_outs,
      rhs_valid => extsi26_outs_valid,
      rhs_ready => extsi26_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer21 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  trunci4 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork16_outs_1,
      lhs_valid => fork16_outs_1_valid,
      lhs_ready => fork16_outs_1_ready,
      rhs => extsi25_outs,
      rhs_valid => extsi25_outs_valid,
      rhs_ready => extsi25_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer22 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => buffer22_outs,
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs(3) => fork17_outs_3,
      outs(4) => fork17_outs_4,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_valid(3) => fork17_outs_3_valid,
      outs_valid(4) => fork17_outs_4_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready,
      outs_ready(3) => fork17_outs_3_ready,
      outs_ready(4) => fork17_outs_4_ready
    );

  cond_br3 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork17_outs_0,
      condition_valid => fork17_outs_0_valid,
      condition_ready => fork17_outs_0_ready,
      data => trunci4_outs,
      data_valid => trunci4_outs_valid,
      data_ready => trunci4_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br4 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork17_outs_2,
      condition_valid => fork17_outs_2_valid,
      condition_ready => fork17_outs_2_ready,
      data => addf0_result,
      data_valid => addf0_result_valid,
      data_ready => addf0_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  cond_br5 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork17_outs_1,
      condition_valid => fork17_outs_1_valid,
      condition_ready => fork17_outs_1_ready,
      data => buffer36_outs,
      data_valid => buffer36_outs_valid,
      data_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer36 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  cond_br6 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork17_outs_3,
      condition_valid => fork17_outs_3_valid,
      condition_ready => fork17_outs_3_ready,
      data_valid => fork15_outs_1_valid,
      data_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork17_outs_4,
      condition_valid => fork17_outs_4_valid,
      condition_ready => fork17_outs_4_ready,
      data => constant21_outs,
      data_valid => constant21_outs_valid,
      data_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br7_trueOut,
      ins_valid => cond_br7_trueOut_valid,
      ins_ready => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi18 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  cond_br45 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer39_outs,
      condition_valid => buffer39_outs_valid,
      condition_ready => buffer39_outs_ready,
      data => init27_outs,
      data_valid => init27_outs_valid,
      data_ready => init27_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br45_trueOut,
      trueOut_valid => cond_br45_trueOut_valid,
      trueOut_ready => cond_br45_trueOut_ready,
      falseOut => cond_br45_falseOut,
      falseOut_valid => cond_br45_falseOut_valid,
      falseOut_ready => cond_br45_falseOut_ready
    );

  buffer39 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork36_outs_9,
      ins_valid => fork36_outs_9_valid,
      ins_ready => fork36_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  cond_br46 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer40_outs,
      condition_valid => buffer40_outs_valid,
      condition_ready => buffer40_outs_ready,
      data_valid => fork33_outs_1_valid,
      data_ready => fork33_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br46_trueOut_valid,
      trueOut_ready => cond_br46_trueOut_ready,
      falseOut_valid => cond_br46_falseOut_valid,
      falseOut_ready => cond_br46_falseOut_ready
    );

  buffer40 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork36_outs_8,
      ins_valid => fork36_outs_8_valid,
      ins_ready => fork36_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  cond_br47 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer41_outs,
      condition_valid => buffer41_outs_valid,
      condition_ready => buffer41_outs_ready,
      data => buffer42_outs,
      data_valid => buffer42_outs_valid,
      data_ready => buffer42_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br47_trueOut,
      trueOut_valid => cond_br47_trueOut_valid,
      trueOut_ready => cond_br47_trueOut_ready,
      falseOut => cond_br47_falseOut,
      falseOut_valid => cond_br47_falseOut_valid,
      falseOut_ready => cond_br47_falseOut_ready
    );

  buffer41 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork36_outs_7,
      ins_valid => fork36_outs_7_valid,
      ins_ready => fork36_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  buffer42 : entity work.tfifo(arch) generic map(13, 32)
    port map(
      ins => fork31_outs_1,
      ins_valid => fork31_outs_1_valid,
      ins_ready => fork31_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  cond_br48 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => buffer43_outs,
      condition_valid => buffer43_outs_valid,
      condition_ready => buffer43_outs_ready,
      data => buffer44_outs,
      data_valid => buffer44_outs_valid,
      data_ready => buffer44_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br48_trueOut,
      trueOut_valid => cond_br48_trueOut_valid,
      trueOut_ready => cond_br48_trueOut_ready,
      falseOut => cond_br48_falseOut,
      falseOut_valid => cond_br48_falseOut_valid,
      falseOut_ready => cond_br48_falseOut_ready
    );

  buffer43 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork36_outs_2,
      ins_valid => fork36_outs_2_valid,
      ins_ready => fork36_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  buffer44 : entity work.tfifo(arch) generic map(12, 6)
    port map(
      ins => fork30_outs_0,
      ins_valid => fork30_outs_0_valid,
      ins_ready => fork30_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  extsi27 : entity work.extsi(arch) generic map(6, 11)
    port map(
      ins => cond_br48_trueOut,
      ins_valid => cond_br48_trueOut_valid,
      ins_ready => cond_br48_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi27_outs,
      outs_valid => extsi27_outs_valid,
      outs_ready => extsi27_outs_ready
    );

  cond_br49 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer45_outs,
      condition_valid => buffer45_outs_valid,
      condition_ready => buffer45_outs_ready,
      data_valid => fork32_outs_1_valid,
      data_ready => fork32_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br49_trueOut_valid,
      trueOut_ready => cond_br49_trueOut_ready,
      falseOut_valid => cond_br49_falseOut_valid,
      falseOut_ready => cond_br49_falseOut_ready
    );

  buffer45 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork36_outs_6,
      ins_valid => fork36_outs_6_valid,
      ins_ready => fork36_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  cond_br50 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer46_outs,
      condition_valid => buffer46_outs_valid,
      condition_ready => buffer46_outs_ready,
      data_valid => fork34_outs_1_valid,
      data_ready => fork34_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br50_trueOut_valid,
      trueOut_ready => cond_br50_trueOut_ready,
      falseOut_valid => cond_br50_falseOut_valid,
      falseOut_ready => cond_br50_falseOut_ready
    );

  buffer46 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork36_outs_5,
      ins_valid => fork36_outs_5_valid,
      ins_ready => fork36_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  cond_br51 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer47_outs,
      condition_valid => buffer47_outs_valid,
      condition_ready => buffer47_outs_ready,
      data_valid => init30_outs_valid,
      data_ready => init30_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br51_trueOut_valid,
      trueOut_ready => cond_br51_trueOut_ready,
      falseOut_valid => cond_br51_falseOut_valid,
      falseOut_ready => cond_br51_falseOut_ready
    );

  buffer47 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork36_outs_4,
      ins_valid => fork36_outs_4_valid,
      ins_ready => fork36_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  mux21 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer48_outs,
      index_valid => buffer48_outs_valid,
      index_ready => buffer48_outs_ready,
      ins(0) => mux7_outs,
      ins(1) => cond_br45_trueOut,
      ins_valid(0) => mux7_outs_valid,
      ins_valid(1) => cond_br45_trueOut_valid,
      ins_ready(0) => mux7_outs_ready,
      ins_ready(1) => cond_br45_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux21_outs,
      outs_valid => mux21_outs_valid,
      outs_ready => mux21_outs_ready
    );

  buffer48 : entity work.tfifo(arch) generic map(12, 1)
    port map(
      ins => fork18_outs_1,
      ins_valid => fork18_outs_1_valid,
      ins_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  buffer4 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  mux22 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer49_outs,
      index_valid => buffer49_outs_valid,
      index_ready => buffer49_outs_ready,
      ins_valid(0) => buffer4_outs_valid,
      ins_valid(1) => cond_br51_trueOut_valid,
      ins_ready(0) => buffer4_outs_ready,
      ins_ready(1) => cond_br51_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux22_outs_valid,
      outs_ready => mux22_outs_ready
    );

  buffer49 : entity work.tfifo(arch) generic map(12, 1)
    port map(
      ins => fork18_outs_2,
      ins_valid => fork18_outs_2_valid,
      ins_ready => fork18_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer49_outs,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  mux23 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer50_outs,
      index_valid => buffer50_outs_valid,
      index_ready => buffer50_outs_ready,
      ins(0) => mux11_outs,
      ins(1) => cond_br47_trueOut,
      ins_valid(0) => mux11_outs_valid,
      ins_valid(1) => cond_br47_trueOut_valid,
      ins_ready(0) => mux11_outs_ready,
      ins_ready(1) => cond_br47_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux23_outs,
      outs_valid => mux23_outs_valid,
      outs_ready => mux23_outs_ready
    );

  buffer50 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork18_outs_3,
      ins_valid => fork18_outs_3_valid,
      ins_ready => fork18_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  mux24 : entity work.mux(arch) generic map(2, 11, 1)
    port map(
      index => buffer51_outs,
      index_valid => buffer51_outs_valid,
      index_ready => buffer51_outs_ready,
      ins(0) => mux13_outs,
      ins(1) => extsi27_outs,
      ins_valid(0) => mux13_outs_valid,
      ins_valid(1) => extsi27_outs_valid,
      ins_ready(0) => mux13_outs_ready,
      ins_ready(1) => extsi27_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux24_outs,
      outs_valid => mux24_outs_valid,
      outs_ready => mux24_outs_ready
    );

  buffer51 : entity work.tfifo(arch) generic map(12, 1)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer51_outs,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  buffer27 : entity work.oehb(arch) generic map(11)
    port map(
      ins => mux24_outs,
      ins_valid => mux24_outs_valid,
      ins_ready => mux24_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  extsi28 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi28_outs,
      outs_valid => extsi28_outs_valid,
      outs_ready => extsi28_outs_ready
    );

  buffer5 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer6 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  mux25 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer52_outs,
      index_valid => buffer52_outs_valid,
      index_ready => buffer52_outs_ready,
      ins_valid(0) => buffer6_outs_valid,
      ins_valid(1) => cond_br46_trueOut_valid,
      ins_ready(0) => buffer6_outs_ready,
      ins_ready(1) => cond_br46_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux25_outs_valid,
      outs_ready => mux25_outs_ready
    );

  buffer52 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork18_outs_4,
      ins_valid => fork18_outs_4_valid,
      ins_ready => fork18_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer52_outs,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  buffer7 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux17_outs_valid,
      ins_ready => mux17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer8 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  mux26 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer53_outs,
      index_valid => buffer53_outs_valid,
      index_ready => buffer53_outs_ready,
      ins_valid(0) => buffer8_outs_valid,
      ins_valid(1) => cond_br49_trueOut_valid,
      ins_ready(0) => buffer8_outs_ready,
      ins_ready(1) => cond_br49_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux26_outs_valid,
      outs_ready => mux26_outs_ready
    );

  buffer53 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork18_outs_5,
      ins_valid => fork18_outs_5_valid,
      ins_ready => fork18_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer53_outs,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  buffer9 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux18_outs_valid,
      ins_ready => mux18_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer10 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  mux27 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer54_outs,
      index_valid => buffer54_outs_valid,
      index_ready => buffer54_outs_ready,
      ins_valid(0) => buffer10_outs_valid,
      ins_valid(1) => cond_br50_trueOut_valid,
      ins_ready(0) => buffer10_outs_ready,
      ins_ready(1) => cond_br50_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux27_outs_valid,
      outs_ready => mux27_outs_ready
    );

  buffer54 : entity work.tfifo(arch) generic map(13, 1)
    port map(
      ins => fork18_outs_6,
      ins_valid => fork18_outs_6_valid,
      ins_ready => fork18_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer54_outs,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  init19 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork36_outs_3,
      ins_valid => fork36_outs_3_valid,
      ins_ready => fork36_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => init19_outs,
      outs_valid => init19_outs_valid,
      outs_ready => init19_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(7, 1)
    port map(
      ins => init19_outs,
      ins_valid => init19_outs_valid,
      ins_ready => init19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs(3) => fork18_outs_3,
      outs(4) => fork18_outs_4,
      outs(5) => fork18_outs_5,
      outs(6) => fork18_outs_6,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_valid(3) => fork18_outs_3_valid,
      outs_valid(4) => fork18_outs_4_valid,
      outs_valid(5) => fork18_outs_5_valid,
      outs_valid(6) => fork18_outs_6_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready,
      outs_ready(3) => fork18_outs_3_ready,
      outs_ready(4) => fork18_outs_4_ready,
      outs_ready(5) => fork18_outs_5_ready,
      outs_ready(6) => fork18_outs_6_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork24_outs_1,
      index_valid => fork24_outs_1_valid,
      index_ready => fork24_outs_1_ready,
      ins(0) => extsi18_outs,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => extsi18_outs_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => extsi18_outs_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer31 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  buffer32 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(5, 6)
    port map(
      ins => buffer32_outs,
      ins_valid => buffer32_outs_valid,
      ins_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs(2) => fork19_outs_2,
      outs(3) => fork19_outs_3,
      outs(4) => fork19_outs_4,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_valid(2) => fork19_outs_2_valid,
      outs_valid(3) => fork19_outs_3_valid,
      outs_valid(4) => fork19_outs_4_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready,
      outs_ready(2) => fork19_outs_2_ready,
      outs_ready(3) => fork19_outs_3_ready,
      outs_ready(4) => fork19_outs_4_ready
    );

  extsi29 : entity work.extsi(arch) generic map(6, 9)
    port map(
      ins => buffer57_outs,
      ins_valid => buffer57_outs_valid,
      ins_ready => buffer57_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi29_outs,
      outs_valid => extsi29_outs_valid,
      outs_ready => extsi29_outs_ready
    );

  buffer57 : entity work.tfifo(arch) generic map(10, 6)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer57_outs,
      outs_valid => buffer57_outs_valid,
      outs_ready => buffer57_outs_ready
    );

  extsi30 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork19_outs_2,
      ins_valid => fork19_outs_2_valid,
      ins_ready => fork19_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi30_outs,
      outs_valid => extsi30_outs_valid,
      outs_ready => extsi30_outs_ready
    );

  extsi31 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork19_outs_4,
      ins_valid => fork19_outs_4_valid,
      ins_ready => fork19_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => extsi31_outs,
      outs_valid => extsi31_outs_valid,
      outs_ready => extsi31_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi31_outs,
      ins_valid => extsi31_outs_valid,
      ins_ready => extsi31_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  trunci5 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => buffer60_outs,
      ins_valid => buffer60_outs_valid,
      ins_ready => buffer60_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  buffer60 : entity work.tfifo(arch) generic map(15, 6)
    port map(
      ins => fork19_outs_1,
      ins_valid => fork19_outs_1_valid,
      ins_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer60_outs,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork24_outs_0,
      index_valid => fork24_outs_0_valid,
      index_ready => fork24_outs_0_ready,
      ins(0) => cond_br5_falseOut,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => cond_br5_falseOut_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => cond_br5_falseOut_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer33 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  buffer34 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer33_outs,
      ins_valid => buffer33_outs_valid,
      ins_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer34_outs,
      ins_valid => buffer34_outs_valid,
      ins_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  extsi32 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork21_outs_1,
      ins_valid => fork21_outs_1_valid,
      ins_ready => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi32_outs,
      outs_valid => extsi32_outs_valid,
      outs_ready => extsi32_outs_ready
    );

  fork22 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi32_outs,
      ins_valid => extsi32_outs_valid,
      ins_ready => extsi32_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  buffer66 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br10_trueOut,
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer66_outs,
      outs_valid => buffer66_outs_valid,
      outs_ready => buffer66_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer63_outs,
      index_valid => buffer63_outs_valid,
      index_ready => buffer63_outs_ready,
      ins(0) => cond_br4_falseOut,
      ins(1) => buffer66_outs,
      ins_valid(0) => cond_br4_falseOut_valid,
      ins_valid(1) => buffer66_outs_valid,
      ins_ready(0) => cond_br4_falseOut_ready,
      ins_ready(1) => buffer66_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer63 : entity work.tfifo(arch) generic map(9, 1)
    port map(
      ins => fork24_outs_2,
      ins_valid => fork24_outs_2_valid,
      ins_ready => fork24_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer63_outs,
      outs_valid => buffer63_outs_valid,
      outs_ready => buffer63_outs_ready
    );

  buffer35 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  control_merge2 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => cond_br6_falseOut_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => cond_br6_falseOut_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork24 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs(2) => fork24_outs_2,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_valid(2) => fork24_outs_2_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready,
      outs_ready(2) => fork24_outs_2_ready
    );

  fork25 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

  constant26 : entity work.handshake_constant_3(arch) generic map(2)
    port map(
      ctrl_valid => fork25_outs_0_valid,
      ctrl_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant26_outs,
      outs_valid => constant26_outs_valid,
      outs_ready => constant26_outs_ready
    );

  extsi10 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant26_outs,
      ins_valid => constant26_outs_valid,
      ins_ready => constant26_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant27 : entity work.handshake_constant_2(arch) generic map(6)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant27_outs,
      outs_valid => constant27_outs_valid,
      outs_ready => constant27_outs_ready
    );

  extsi33 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant27_outs,
      ins_valid => constant27_outs_valid,
      ins_ready => constant27_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi33_outs,
      outs_valid => extsi33_outs_valid,
      outs_ready => extsi33_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant28 : entity work.handshake_constant_3(arch) generic map(2)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant28_outs,
      outs_valid => constant28_outs_valid,
      outs_ready => constant28_outs_ready
    );

  extsi34 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant28_outs,
      ins_valid => constant28_outs_valid,
      ins_ready => constant28_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi34_outs,
      outs_valid => extsi34_outs_valid,
      outs_ready => extsi34_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant29 : entity work.handshake_constant_4(arch) generic map(4)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant29_outs,
      outs_valid => constant29_outs_valid,
      outs_ready => constant29_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(4, 32)
    port map(
      ins => constant29_outs,
      ins_valid => constant29_outs_valid,
      ins_ready => constant29_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant30 : entity work.handshake_constant_5(arch) generic map(3)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant30_outs,
      outs_valid => constant30_outs_valid,
      outs_ready => constant30_outs_ready
    );

  extsi14 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant30_outs,
      ins_valid => constant30_outs_valid,
      ins_ready => constant30_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  buffer24 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux22_outs_valid,
      ins_ready => mux22_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  buffer25 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => buffer64_outs,
      ins_valid(0) => buffer64_outs_valid,
      ins_valid(1) => buffer25_outs_valid,
      ins_ready(0) => buffer64_outs_ready,
      ins_ready(1) => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  buffer64 : entity work.tfifo(arch) generic map(13, 32)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer64_outs,
      outs_valid => buffer64_outs_valid,
      outs_ready => buffer64_outs_ready
    );

  fork26 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork26_outs_0,
      outs(1) => fork26_outs_1,
      outs(2) => fork26_outs_2,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_valid(2) => fork26_outs_2_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready,
      outs_ready(2) => fork26_outs_2_ready
    );

  cmpi3 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork26_outs_2,
      lhs_valid => fork26_outs_2_valid,
      lhs_ready => fork26_outs_2_ready,
      rhs => extsi28_outs,
      rhs_valid => extsi28_outs_valid,
      rhs_ready => extsi28_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork27 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork27_outs_0,
      outs(1) => fork27_outs_1,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready
    );

  buffer26 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux23_outs,
      ins_valid => mux23_outs_valid,
      ins_ready => mux23_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  cmpi4 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork26_outs_1,
      lhs_valid => fork26_outs_1_valid,
      lhs_ready => fork26_outs_1_ready,
      rhs => buffer26_outs,
      rhs_valid => buffer26_outs_valid,
      rhs_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi4_result,
      result_valid => cmpi4_result_valid,
      result_ready => cmpi4_result_ready
    );

  fork28 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi4_result,
      ins_valid => cmpi4_result_valid,
      ins_ready => cmpi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready
    );

  buffer23 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux21_outs,
      ins_valid => mux21_outs_valid,
      ins_ready => mux21_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  cmpi5 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork26_outs_0,
      lhs_valid => fork26_outs_0_valid,
      lhs_ready => fork26_outs_0_ready,
      rhs => buffer23_outs,
      rhs_valid => buffer23_outs_valid,
      rhs_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi5_result,
      result_valid => cmpi5_result_valid,
      result_ready => cmpi5_result_ready
    );

  fork29 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi5_result,
      ins_valid => cmpi5_result_valid,
      ins_ready => cmpi5_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork29_outs_0,
      outs(1) => fork29_outs_1,
      outs_valid(0) => fork29_outs_0_valid,
      outs_valid(1) => fork29_outs_1_valid,
      outs_ready(0) => fork29_outs_0_ready,
      outs_ready(1) => fork29_outs_1_ready
    );

  buffer29 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux26_outs_valid,
      ins_ready => mux26_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  cond_br25 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer68_outs,
      condition_valid => buffer68_outs_valid,
      condition_ready => buffer68_outs_ready,
      data_valid => buffer29_outs_valid,
      data_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br25_trueOut_valid,
      trueOut_ready => cond_br25_trueOut_ready,
      falseOut_valid => cond_br25_falseOut_valid,
      falseOut_ready => cond_br25_falseOut_ready
    );

  buffer68 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork27_outs_1,
      ins_valid => fork27_outs_1_valid,
      ins_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer68_outs,
      outs_valid => buffer68_outs_valid,
      outs_ready => buffer68_outs_ready
    );

  sink2 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br25_trueOut_valid,
      ins_ready => cond_br25_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer28 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux25_outs_valid,
      ins_ready => mux25_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  cond_br26 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer69_outs,
      condition_valid => buffer69_outs_valid,
      condition_ready => buffer69_outs_ready,
      data_valid => buffer28_outs_valid,
      data_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br26_trueOut_valid,
      trueOut_ready => cond_br26_trueOut_ready,
      falseOut_valid => cond_br26_falseOut_valid,
      falseOut_ready => cond_br26_falseOut_ready
    );

  buffer69 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork28_outs_1,
      ins_valid => fork28_outs_1_valid,
      ins_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer69_outs,
      outs_valid => buffer69_outs_valid,
      outs_ready => buffer69_outs_ready
    );

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br26_trueOut_valid,
      ins_ready => cond_br26_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer30 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux27_outs_valid,
      ins_ready => mux27_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  cond_br27 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer70_outs,
      condition_valid => buffer70_outs_valid,
      condition_ready => buffer70_outs_ready,
      data_valid => buffer30_outs_valid,
      data_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br27_trueOut_valid,
      trueOut_ready => cond_br27_trueOut_ready,
      falseOut_valid => cond_br27_falseOut_valid,
      falseOut_ready => cond_br27_falseOut_ready
    );

  buffer70 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork29_outs_1,
      ins_valid => fork29_outs_1_valid,
      ins_ready => fork29_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer70_outs,
      outs_valid => buffer70_outs_valid,
      outs_ready => buffer70_outs_ready
    );

  sink4 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br27_trueOut_valid,
      ins_ready => cond_br27_trueOut_ready,
      clk => clk,
      rst => rst
    );

  source10 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  mux28 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork27_outs_0,
      index_valid => fork27_outs_0_valid,
      index_ready => fork27_outs_0_ready,
      ins_valid(0) => cond_br25_falseOut_valid,
      ins_valid(1) => source10_outs_valid,
      ins_ready(0) => cond_br25_falseOut_ready,
      ins_ready(1) => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux28_outs_valid,
      outs_ready => mux28_outs_ready
    );

  source11 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source11_outs_valid,
      outs_ready => source11_outs_ready
    );

  mux29 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork28_outs_0,
      index_valid => fork28_outs_0_valid,
      index_ready => fork28_outs_0_ready,
      ins_valid(0) => cond_br26_falseOut_valid,
      ins_valid(1) => source11_outs_valid,
      ins_ready(0) => cond_br26_falseOut_ready,
      ins_ready(1) => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux29_outs_valid,
      outs_ready => mux29_outs_ready
    );

  source12 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source12_outs_valid,
      outs_ready => source12_outs_ready
    );

  mux30 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork29_outs_0,
      index_valid => fork29_outs_0_valid,
      index_ready => fork29_outs_0_ready,
      ins_valid(0) => cond_br27_falseOut_valid,
      ins_valid(1) => source12_outs_valid,
      ins_ready(0) => cond_br27_falseOut_ready,
      ins_ready(1) => source12_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux30_outs_valid,
      outs_ready => mux30_outs_ready
    );

  buffer38 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux28_outs_valid,
      ins_ready => mux28_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  buffer55 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux29_outs_valid,
      ins_ready => mux29_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer55_outs_valid,
      outs_ready => buffer55_outs_ready
    );

  buffer56 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux30_outs_valid,
      ins_ready => mux30_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer56_outs_valid,
      outs_ready => buffer56_outs_ready
    );

  join0 : entity work.join_handshake(arch) generic map(3)
    port map(
      ins_valid(0) => buffer38_outs_valid,
      ins_valid(1) => buffer55_outs_valid,
      ins_valid(2) => buffer56_outs_valid,
      ins_ready(0) => buffer38_outs_ready,
      ins_ready(1) => buffer55_outs_ready,
      ins_ready(2) => buffer56_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join0_outs_valid,
      outs_ready => join0_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => buffer74_outs,
      ins_valid(0) => buffer74_outs_valid,
      ins_valid(1) => join0_outs_valid,
      ins_ready(0) => buffer74_outs_ready,
      ins_ready(1) => join0_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  buffer74 : entity work.tfifo(arch) generic map(12, 32)
    port map(
      ins => fork20_outs_1,
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer74_outs,
      outs_valid => buffer74_outs_valid,
      outs_ready => buffer74_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(32, 5)
    port map(
      ins => gate1_outs,
      ins_valid => gate1_outs_valid,
      ins_ready => gate1_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci6_outs,
      addrIn_valid => trunci6_outs_valid,
      addrIn_ready => trunci6_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  shli2 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer75_outs,
      lhs_valid => buffer75_outs_valid,
      lhs_ready => buffer75_outs_ready,
      rhs => extsi14_outs,
      rhs_valid => extsi14_outs_valid,
      rhs_ready => extsi14_outs_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  buffer75 : entity work.tfifo(arch) generic map(9, 32)
    port map(
      ins => fork22_outs_0,
      ins_valid => fork22_outs_0_valid,
      ins_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer75_outs,
      outs_valid => buffer75_outs_valid,
      outs_ready => buffer75_outs_ready
    );

  buffer58 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer58_outs,
      outs_valid => buffer58_outs_valid,
      outs_ready => buffer58_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer58_outs,
      ins_valid => buffer58_outs_valid,
      ins_ready => buffer58_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer76_outs,
      lhs_valid => buffer76_outs_valid,
      lhs_ready => buffer76_outs_ready,
      rhs => extsi13_outs,
      rhs_valid => extsi13_outs_valid,
      rhs_ready => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer76 : entity work.tfifo(arch) generic map(9, 32)
    port map(
      ins => fork22_outs_1,
      ins_valid => fork22_outs_1_valid,
      ins_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer76_outs,
      outs_valid => buffer76_outs_valid,
      outs_ready => buffer76_outs_ready
    );

  buffer59 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer59_outs,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  trunci8 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer59_outs,
      ins_valid => buffer59_outs_valid,
      ins_ready => buffer59_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  addi6 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci7_outs,
      lhs_valid => trunci7_outs_valid,
      lhs_ready => trunci7_outs_ready,
      rhs => trunci8_outs,
      rhs_valid => trunci8_outs_valid,
      rhs_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer61 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer61_outs,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi29_outs,
      lhs_valid => extsi29_outs_valid,
      lhs_ready => extsi29_outs_ready,
      rhs => buffer61_outs,
      rhs_valid => buffer61_outs_valid,
      rhs_ready => buffer61_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  load4 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => addi1_result,
      addrIn_valid => addi1_result_valid,
      addrIn_ready => addi1_result_ready,
      dataFromMem => mem_controller5_ldData_1,
      dataFromMem_valid => mem_controller5_ldData_1_valid,
      dataFromMem_ready => mem_controller5_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  mulf1 : entity work.mulf(arch_32_2_875333) generic map(32)
    port map(
      lhs => load4_dataOut,
      lhs_valid => load4_dataOut_valid,
      lhs_ready => load4_dataOut_ready,
      rhs => buffer77_outs,
      rhs_valid => buffer77_outs_valid,
      rhs_ready => buffer77_outs_ready,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready
    );

  buffer77 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer77_outs,
      outs_valid => buffer77_outs_valid,
      outs_ready => buffer77_outs_ready
    );

  addf1 : entity work.addf(arch_32_2_922000) generic map(32)
    port map(
      lhs => load3_dataOut,
      lhs_valid => load3_dataOut_valid,
      lhs_ready => load3_dataOut_ready,
      rhs => mulf1_result,
      rhs_valid => mulf1_result_valid,
      rhs_ready => mulf1_result_ready,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready
    );

  buffer2 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork19_outs_3,
      ins_valid => fork19_outs_3_valid,
      ins_ready => fork19_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  fork30 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork30_outs_0,
      outs(1) => fork30_outs_1,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready
    );

  extsi35 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork30_outs_1,
      ins_valid => fork30_outs_1_valid,
      ins_ready => fork30_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi35_outs,
      outs_valid => extsi35_outs_valid,
      outs_ready => extsi35_outs_ready
    );

  init26 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => extsi35_outs,
      ins_valid => extsi35_outs_valid,
      ins_ready => extsi35_outs_ready,
      clk => clk,
      rst => rst,
      outs => init26_outs,
      outs_valid => init26_outs_valid,
      outs_ready => init26_outs_ready
    );

  fork31 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init26_outs,
      ins_valid => init26_outs_valid,
      ins_ready => init26_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork31_outs_0,
      outs(1) => fork31_outs_1,
      outs_valid(0) => fork31_outs_0_valid,
      outs_valid(1) => fork31_outs_1_valid,
      outs_ready(0) => fork31_outs_0_ready,
      outs_ready(1) => fork31_outs_1_ready
    );

  init27 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer80_outs,
      ins_valid => buffer80_outs_valid,
      ins_ready => buffer80_outs_ready,
      clk => clk,
      rst => rst,
      outs => init27_outs,
      outs_valid => init27_outs_valid,
      outs_ready => init27_outs_ready
    );

  buffer80 : entity work.tfifo(arch) generic map(15, 32)
    port map(
      ins => fork31_outs_0,
      ins_valid => fork31_outs_0_valid,
      ins_ready => fork31_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer80_outs,
      outs_valid => buffer80_outs_valid,
      outs_ready => buffer80_outs_ready
    );

  buffer3 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => store0_doneOut_valid,
      ins_ready => store0_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork32 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork32_outs_0_valid,
      outs_valid(1) => fork32_outs_1_valid,
      outs_ready(0) => fork32_outs_0_ready,
      outs_ready(1) => fork32_outs_1_ready
    );

  init28 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork32_outs_0_valid,
      ins_ready => fork32_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init28_outs_valid,
      outs_ready => init28_outs_ready
    );

  fork33 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init28_outs_valid,
      ins_ready => init28_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork33_outs_0_valid,
      outs_valid(1) => fork33_outs_1_valid,
      outs_ready(0) => fork33_outs_0_ready,
      outs_ready(1) => fork33_outs_1_ready
    );

  init29 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork33_outs_0_valid,
      ins_ready => fork33_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init29_outs_valid,
      outs_ready => init29_outs_ready
    );

  fork34 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init29_outs_valid,
      ins_ready => init29_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready
    );

  init30 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork34_outs_0_valid,
      ins_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init30_outs_valid,
      outs_ready => init30_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => trunci5_outs,
      addrIn_valid => trunci5_outs_valid,
      addrIn_ready => trunci5_outs_ready,
      dataIn => addf1_result,
      dataIn_valid => addf1_result_valid,
      dataIn_ready => addf1_result_ready,
      doneFromMem_valid => mem_controller3_stDone_0_valid,
      doneFromMem_ready => mem_controller3_stDone_0_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready,
      doneOut_valid => store0_doneOut_valid,
      doneOut_ready => store0_doneOut_ready
    );

  addi3 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi30_outs,
      lhs_valid => extsi30_outs_valid,
      lhs_ready => extsi30_outs_ready,
      rhs => extsi34_outs,
      rhs_valid => extsi34_outs_valid,
      rhs_ready => extsi34_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer62 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer62_outs,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  fork35 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer62_outs,
      ins_valid => buffer62_outs_valid,
      ins_ready => buffer62_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready
    );

  trunci9 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork35_outs_0,
      ins_valid => fork35_outs_0_valid,
      ins_ready => fork35_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork35_outs_1,
      lhs_valid => fork35_outs_1_valid,
      lhs_ready => fork35_outs_1_ready,
      rhs => extsi33_outs,
      rhs_valid => extsi33_outs_valid,
      rhs_ready => extsi33_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer65 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer65_outs,
      outs_valid => buffer65_outs_valid,
      outs_ready => buffer65_outs_ready
    );

  fork36 : entity work.handshake_fork(arch) generic map(12, 1)
    port map(
      ins => buffer65_outs,
      ins_valid => buffer65_outs_valid,
      ins_ready => buffer65_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork36_outs_0,
      outs(1) => fork36_outs_1,
      outs(2) => fork36_outs_2,
      outs(3) => fork36_outs_3,
      outs(4) => fork36_outs_4,
      outs(5) => fork36_outs_5,
      outs(6) => fork36_outs_6,
      outs(7) => fork36_outs_7,
      outs(8) => fork36_outs_8,
      outs(9) => fork36_outs_9,
      outs(10) => fork36_outs_10,
      outs(11) => fork36_outs_11,
      outs_valid(0) => fork36_outs_0_valid,
      outs_valid(1) => fork36_outs_1_valid,
      outs_valid(2) => fork36_outs_2_valid,
      outs_valid(3) => fork36_outs_3_valid,
      outs_valid(4) => fork36_outs_4_valid,
      outs_valid(5) => fork36_outs_5_valid,
      outs_valid(6) => fork36_outs_6_valid,
      outs_valid(7) => fork36_outs_7_valid,
      outs_valid(8) => fork36_outs_8_valid,
      outs_valid(9) => fork36_outs_9_valid,
      outs_valid(10) => fork36_outs_10_valid,
      outs_valid(11) => fork36_outs_11_valid,
      outs_ready(0) => fork36_outs_0_ready,
      outs_ready(1) => fork36_outs_1_ready,
      outs_ready(2) => fork36_outs_2_ready,
      outs_ready(3) => fork36_outs_3_ready,
      outs_ready(4) => fork36_outs_4_ready,
      outs_ready(5) => fork36_outs_5_ready,
      outs_ready(6) => fork36_outs_6_ready,
      outs_ready(7) => fork36_outs_7_ready,
      outs_ready(8) => fork36_outs_8_ready,
      outs_ready(9) => fork36_outs_9_ready,
      outs_ready(10) => fork36_outs_10_ready,
      outs_ready(11) => fork36_outs_11_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork36_outs_0,
      condition_valid => fork36_outs_0_valid,
      condition_ready => fork36_outs_0_ready,
      data => trunci9_outs,
      data_valid => trunci9_outs_valid,
      data_ready => trunci9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br9 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork36_outs_1,
      condition_valid => fork36_outs_1_valid,
      condition_ready => fork36_outs_1_ready,
      data => fork21_outs_0,
      data_valid => fork21_outs_0_valid,
      data_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer86_outs,
      condition_valid => buffer86_outs_valid,
      condition_ready => buffer86_outs_ready,
      data => fork23_outs_0,
      data_valid => fork23_outs_0_valid,
      data_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  buffer86 : entity work.tfifo(arch) generic map(9, 1)
    port map(
      ins => fork36_outs_10,
      ins_valid => fork36_outs_10_valid,
      ins_ready => fork36_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => buffer86_outs,
      outs_valid => buffer86_outs_valid,
      outs_ready => buffer86_outs_ready
    );

  buffer37 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => fork25_outs_1_valid,
      ins_ready => fork25_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  cond_br11 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer88_outs,
      condition_valid => buffer88_outs_valid,
      condition_ready => buffer88_outs_ready,
      data_valid => buffer37_outs_valid,
      data_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  buffer88 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork36_outs_11,
      ins_valid => fork36_outs_11_valid,
      ins_ready => fork36_outs_11_ready,
      clk => clk,
      rst => rst,
      outs => buffer88_outs,
      outs_valid => buffer88_outs_valid,
      outs_ready => buffer88_outs_ready
    );

  cond_br52 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer89_outs,
      condition_valid => buffer89_outs_valid,
      condition_ready => buffer89_outs_ready,
      data => cond_br45_falseOut,
      data_valid => cond_br45_falseOut_valid,
      data_ready => cond_br45_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br52_trueOut,
      trueOut_valid => cond_br52_trueOut_valid,
      trueOut_ready => cond_br52_trueOut_ready,
      falseOut => cond_br52_falseOut,
      falseOut_valid => cond_br52_falseOut_valid,
      falseOut_ready => cond_br52_falseOut_ready
    );

  buffer89 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_6,
      ins_valid => fork45_outs_6_valid,
      ins_ready => fork45_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer89_outs,
      outs_valid => buffer89_outs_valid,
      outs_ready => buffer89_outs_ready
    );

  sink6 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br52_falseOut,
      ins_valid => cond_br52_falseOut_valid,
      ins_ready => cond_br52_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br53 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer90_outs,
      condition_valid => buffer90_outs_valid,
      condition_ready => buffer90_outs_ready,
      data_valid => cond_br49_falseOut_valid,
      data_ready => cond_br49_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br53_trueOut_valid,
      trueOut_ready => cond_br53_trueOut_ready,
      falseOut_valid => cond_br53_falseOut_valid,
      falseOut_ready => cond_br53_falseOut_ready
    );

  buffer90 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_5,
      ins_valid => fork45_outs_5_valid,
      ins_ready => fork45_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer90_outs,
      outs_valid => buffer90_outs_valid,
      outs_ready => buffer90_outs_ready
    );

  sink7 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br53_falseOut_valid,
      ins_ready => cond_br53_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br54 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer91_outs,
      condition_valid => buffer91_outs_valid,
      condition_ready => buffer91_outs_ready,
      data_valid => cond_br46_falseOut_valid,
      data_ready => cond_br46_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br54_trueOut_valid,
      trueOut_ready => cond_br54_trueOut_ready,
      falseOut_valid => cond_br54_falseOut_valid,
      falseOut_ready => cond_br54_falseOut_ready
    );

  buffer91 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_4,
      ins_valid => fork45_outs_4_valid,
      ins_ready => fork45_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer91_outs,
      outs_valid => buffer91_outs_valid,
      outs_ready => buffer91_outs_ready
    );

  sink8 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br54_falseOut_valid,
      ins_ready => cond_br54_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br55 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer92_outs,
      condition_valid => buffer92_outs_valid,
      condition_ready => buffer92_outs_ready,
      data_valid => cond_br50_falseOut_valid,
      data_ready => cond_br50_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br55_trueOut_valid,
      trueOut_ready => cond_br55_trueOut_ready,
      falseOut_valid => cond_br55_falseOut_valid,
      falseOut_ready => cond_br55_falseOut_ready
    );

  buffer92 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_3,
      ins_valid => fork45_outs_3_valid,
      ins_ready => fork45_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer92_outs,
      outs_valid => buffer92_outs_valid,
      outs_ready => buffer92_outs_ready
    );

  sink9 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br55_falseOut_valid,
      ins_ready => cond_br55_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br56 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer93_outs,
      condition_valid => buffer93_outs_valid,
      condition_ready => buffer93_outs_ready,
      data => cond_br47_falseOut,
      data_valid => cond_br47_falseOut_valid,
      data_ready => cond_br47_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br56_trueOut,
      trueOut_valid => cond_br56_trueOut_valid,
      trueOut_ready => cond_br56_trueOut_ready,
      falseOut => cond_br56_falseOut,
      falseOut_valid => cond_br56_falseOut_valid,
      falseOut_ready => cond_br56_falseOut_ready
    );

  buffer93 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_2,
      ins_valid => fork45_outs_2_valid,
      ins_ready => fork45_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer93_outs,
      outs_valid => buffer93_outs_valid,
      outs_ready => buffer93_outs_ready
    );

  sink10 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br56_falseOut,
      ins_valid => cond_br56_falseOut_valid,
      ins_ready => cond_br56_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br57 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => buffer94_outs,
      condition_valid => buffer94_outs_valid,
      condition_ready => buffer94_outs_ready,
      data => cond_br48_falseOut,
      data_valid => cond_br48_falseOut_valid,
      data_ready => cond_br48_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br57_trueOut,
      trueOut_valid => cond_br57_trueOut_valid,
      trueOut_ready => cond_br57_trueOut_ready,
      falseOut => cond_br57_falseOut,
      falseOut_valid => cond_br57_falseOut_valid,
      falseOut_ready => cond_br57_falseOut_ready
    );

  buffer94 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_9,
      ins_valid => fork45_outs_9_valid,
      ins_ready => fork45_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => buffer94_outs,
      outs_valid => buffer94_outs_valid,
      outs_ready => buffer94_outs_ready
    );

  sink11 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br57_falseOut,
      ins_valid => cond_br57_falseOut_valid,
      ins_ready => cond_br57_falseOut_ready,
      clk => clk,
      rst => rst
    );

  extsi36 : entity work.extsi(arch) generic map(6, 11)
    port map(
      ins => cond_br57_trueOut,
      ins_valid => cond_br57_trueOut_valid,
      ins_ready => cond_br57_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi36_outs,
      outs_valid => extsi36_outs_valid,
      outs_ready => extsi36_outs_ready
    );

  cond_br58 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer95_outs,
      condition_valid => buffer95_outs_valid,
      condition_ready => buffer95_outs_ready,
      data_valid => cond_br51_falseOut_valid,
      data_ready => cond_br51_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br58_trueOut_valid,
      trueOut_ready => cond_br58_trueOut_ready,
      falseOut_valid => cond_br58_falseOut_valid,
      falseOut_ready => cond_br58_falseOut_ready
    );

  buffer95 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_1,
      ins_valid => fork45_outs_1_valid,
      ins_ready => fork45_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer95_outs,
      outs_valid => buffer95_outs_valid,
      outs_ready => buffer95_outs_ready
    );

  sink12 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br58_falseOut_valid,
      ins_ready => cond_br58_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer67 : entity work.oehb(arch) generic map(6)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer67_outs,
      outs_valid => buffer67_outs_valid,
      outs_ready => buffer67_outs_ready
    );

  fork37 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer67_outs,
      ins_valid => buffer67_outs_valid,
      ins_ready => buffer67_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork37_outs_0,
      outs(1) => fork37_outs_1,
      outs_valid(0) => fork37_outs_0_valid,
      outs_valid(1) => fork37_outs_1_valid,
      outs_ready(0) => fork37_outs_0_ready,
      outs_ready(1) => fork37_outs_1_ready
    );

  extsi37 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork37_outs_0,
      ins_valid => fork37_outs_0_valid,
      ins_ready => fork37_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi37_outs,
      outs_valid => extsi37_outs_valid,
      outs_ready => extsi37_outs_ready
    );

  extsi38 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork37_outs_1,
      ins_valid => fork37_outs_1_valid,
      ins_ready => fork37_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi38_outs,
      outs_valid => extsi38_outs_valid,
      outs_ready => extsi38_outs_ready
    );

  fork38 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi38_outs,
      ins_valid => extsi38_outs_valid,
      ins_ready => extsi38_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork38_outs_0,
      outs(1) => fork38_outs_1,
      outs_valid(0) => fork38_outs_0_valid,
      outs_valid(1) => fork38_outs_1_valid,
      outs_ready(0) => fork38_outs_0_ready,
      outs_ready(1) => fork38_outs_1_ready
    );

  fork39 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br11_falseOut_valid,
      ins_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork39_outs_0_valid,
      outs_valid(1) => fork39_outs_1_valid,
      outs_ready(0) => fork39_outs_0_ready,
      outs_ready(1) => fork39_outs_1_ready
    );

  constant31 : entity work.handshake_constant_3(arch) generic map(2)
    port map(
      ctrl_valid => fork39_outs_0_valid,
      ctrl_ready => fork39_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant31_outs,
      outs_valid => constant31_outs_valid,
      outs_ready => constant31_outs_ready
    );

  extsi15 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant31_outs,
      ins_valid => constant31_outs_valid,
      ins_ready => constant31_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant32 : entity work.handshake_constant_2(arch) generic map(6)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant32_outs,
      outs_valid => constant32_outs_valid,
      outs_ready => constant32_outs_ready
    );

  extsi39 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant32_outs,
      ins_valid => constant32_outs_valid,
      ins_ready => constant32_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi39_outs,
      outs_valid => extsi39_outs_valid,
      outs_ready => extsi39_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant33 : entity work.handshake_constant_3(arch) generic map(2)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant33_outs,
      outs_valid => constant33_outs_valid,
      outs_ready => constant33_outs_ready
    );

  extsi40 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant33_outs,
      ins_valid => constant33_outs_valid,
      ins_ready => constant33_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi40_outs,
      outs_valid => extsi40_outs_valid,
      outs_ready => extsi40_outs_ready
    );

  gate2 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork38_outs_0,
      ins_valid(0) => fork38_outs_0_valid,
      ins_valid(1) => init18_outs_valid,
      ins_ready(0) => fork38_outs_0_ready,
      ins_ready(1) => init18_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate2_outs,
      outs_valid => gate2_outs_valid,
      outs_ready => gate2_outs_ready
    );

  fork40 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => gate2_outs,
      ins_valid => gate2_outs_valid,
      ins_ready => gate2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork40_outs_0,
      outs(1) => fork40_outs_1,
      outs(2) => fork40_outs_2,
      outs_valid(0) => fork40_outs_0_valid,
      outs_valid(1) => fork40_outs_1_valid,
      outs_valid(2) => fork40_outs_2_valid,
      outs_ready(0) => fork40_outs_0_ready,
      outs_ready(1) => fork40_outs_1_ready,
      outs_ready(2) => fork40_outs_2_ready
    );

  cmpi6 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork40_outs_2,
      lhs_valid => fork40_outs_2_valid,
      lhs_ready => fork40_outs_2_ready,
      rhs => fork5_outs_1,
      rhs_valid => fork5_outs_1_valid,
      rhs_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi6_result,
      result_valid => cmpi6_result_valid,
      result_ready => cmpi6_result_ready
    );

  fork41 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi6_result,
      ins_valid => cmpi6_result_valid,
      ins_ready => cmpi6_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork41_outs_0,
      outs(1) => fork41_outs_1,
      outs_valid(0) => fork41_outs_0_valid,
      outs_valid(1) => fork41_outs_1_valid,
      outs_ready(0) => fork41_outs_0_ready,
      outs_ready(1) => fork41_outs_1_ready
    );

  cmpi7 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork40_outs_1,
      lhs_valid => fork40_outs_1_valid,
      lhs_ready => fork40_outs_1_ready,
      rhs => fork6_outs_0,
      rhs_valid => fork6_outs_0_valid,
      rhs_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi7_result,
      result_valid => cmpi7_result_valid,
      result_ready => cmpi7_result_ready
    );

  fork42 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi7_result,
      ins_valid => cmpi7_result_valid,
      ins_ready => cmpi7_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork42_outs_0,
      outs(1) => fork42_outs_1,
      outs_valid(0) => fork42_outs_0_valid,
      outs_valid(1) => fork42_outs_1_valid,
      outs_ready(0) => fork42_outs_0_ready,
      outs_ready(1) => fork42_outs_1_ready
    );

  cmpi8 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork40_outs_0,
      lhs_valid => fork40_outs_0_valid,
      lhs_ready => fork40_outs_0_ready,
      rhs => init15_outs,
      rhs_valid => init15_outs_valid,
      rhs_ready => init15_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi8_result,
      result_valid => cmpi8_result_valid,
      result_ready => cmpi8_result_ready
    );

  fork43 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi8_result,
      ins_valid => cmpi8_result_valid,
      ins_ready => cmpi8_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork43_outs_0,
      outs(1) => fork43_outs_1,
      outs_valid(0) => fork43_outs_0_valid,
      outs_valid(1) => fork43_outs_1_valid,
      outs_ready(0) => fork43_outs_0_ready,
      outs_ready(1) => fork43_outs_1_ready
    );

  cond_br42 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork41_outs_1,
      condition_valid => fork41_outs_1_valid,
      condition_ready => fork41_outs_1_ready,
      data_valid => fork7_outs_0_valid,
      data_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br42_trueOut_valid,
      trueOut_ready => cond_br42_trueOut_ready,
      falseOut_valid => cond_br42_falseOut_valid,
      falseOut_ready => cond_br42_falseOut_ready
    );

  sink14 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br42_trueOut_valid,
      ins_ready => cond_br42_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br43 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork42_outs_1,
      condition_valid => fork42_outs_1_valid,
      condition_ready => fork42_outs_1_ready,
      data_valid => fork8_outs_0_valid,
      data_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br43_trueOut_valid,
      trueOut_ready => cond_br43_trueOut_ready,
      falseOut_valid => cond_br43_falseOut_valid,
      falseOut_ready => cond_br43_falseOut_ready
    );

  sink15 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br43_trueOut_valid,
      ins_ready => cond_br43_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br44 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork43_outs_1,
      condition_valid => fork43_outs_1_valid,
      condition_ready => fork43_outs_1_ready,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br44_trueOut_valid,
      trueOut_ready => cond_br44_trueOut_ready,
      falseOut_valid => cond_br44_falseOut_valid,
      falseOut_ready => cond_br44_falseOut_ready
    );

  sink16 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br44_trueOut_valid,
      ins_ready => cond_br44_trueOut_ready,
      clk => clk,
      rst => rst
    );

  source13 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source13_outs_valid,
      outs_ready => source13_outs_ready
    );

  mux31 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork41_outs_0,
      index_valid => fork41_outs_0_valid,
      index_ready => fork41_outs_0_ready,
      ins_valid(0) => cond_br42_falseOut_valid,
      ins_valid(1) => source13_outs_valid,
      ins_ready(0) => cond_br42_falseOut_ready,
      ins_ready(1) => source13_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux31_outs_valid,
      outs_ready => mux31_outs_ready
    );

  source14 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source14_outs_valid,
      outs_ready => source14_outs_ready
    );

  mux32 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork42_outs_0,
      index_valid => fork42_outs_0_valid,
      index_ready => fork42_outs_0_ready,
      ins_valid(0) => cond_br43_falseOut_valid,
      ins_valid(1) => source14_outs_valid,
      ins_ready(0) => cond_br43_falseOut_ready,
      ins_ready(1) => source14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux32_outs_valid,
      outs_ready => mux32_outs_ready
    );

  source15 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source15_outs_valid,
      outs_ready => source15_outs_ready
    );

  mux33 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer109_outs,
      index_valid => buffer109_outs_valid,
      index_ready => buffer109_outs_ready,
      ins_valid(0) => cond_br44_falseOut_valid,
      ins_valid(1) => source15_outs_valid,
      ins_ready(0) => cond_br44_falseOut_ready,
      ins_ready(1) => source15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux33_outs_valid,
      outs_ready => mux33_outs_ready
    );

  buffer109 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork43_outs_0,
      ins_valid => fork43_outs_0_valid,
      ins_ready => fork43_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer109_outs,
      outs_valid => buffer109_outs_valid,
      outs_ready => buffer109_outs_ready
    );

  join1 : entity work.join_handshake(arch) generic map(3)
    port map(
      ins_valid(0) => mux31_outs_valid,
      ins_valid(1) => mux32_outs_valid,
      ins_valid(2) => mux33_outs_valid,
      ins_ready(0) => mux31_outs_ready,
      ins_ready(1) => mux32_outs_ready,
      ins_ready(2) => mux33_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join1_outs_valid,
      outs_ready => join1_outs_ready
    );

  buffer71 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => join1_outs_valid,
      ins_ready => join1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer71_outs_valid,
      outs_ready => buffer71_outs_ready
    );

  gate3 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => buffer110_outs,
      ins_valid(0) => buffer110_outs_valid,
      ins_valid(1) => buffer71_outs_valid,
      ins_ready(0) => buffer110_outs_ready,
      ins_ready(1) => buffer71_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate3_outs,
      outs_valid => gate3_outs_valid,
      outs_ready => gate3_outs_ready
    );

  buffer110 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork38_outs_1,
      ins_valid => fork38_outs_1_valid,
      ins_ready => fork38_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer110_outs,
      outs_valid => buffer110_outs_valid,
      outs_ready => buffer110_outs_ready
    );

  trunci10 : entity work.trunci(arch) generic map(32, 5)
    port map(
      ins => gate3_outs,
      ins_valid => gate3_outs_valid,
      ins_ready => gate3_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci10_outs,
      outs_valid => trunci10_outs_valid,
      outs_ready => trunci10_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => trunci10_outs,
      addrIn_valid => trunci10_outs_valid,
      addrIn_ready => trunci10_outs_ready,
      dataIn => cond_br10_falseOut,
      dataIn_valid => cond_br10_falseOut_valid,
      dataIn_ready => cond_br10_falseOut_ready,
      doneFromMem_valid => mem_controller2_stDone_0_valid,
      doneFromMem_ready => mem_controller2_stDone_0_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready,
      doneOut_valid => store1_doneOut_valid,
      doneOut_ready => store1_doneOut_ready
    );

  sink17 : entity work.sink_dataless(arch)
    port map(
      ins_valid => store1_doneOut_valid,
      ins_ready => store1_doneOut_ready,
      clk => clk,
      rst => rst
    );

  addi4 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi37_outs,
      lhs_valid => extsi37_outs_valid,
      lhs_ready => extsi37_outs_ready,
      rhs => extsi40_outs,
      rhs_valid => extsi40_outs_valid,
      rhs_ready => extsi40_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer72 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer72_outs,
      outs_valid => buffer72_outs_valid,
      outs_ready => buffer72_outs_ready
    );

  fork44 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer72_outs,
      ins_valid => buffer72_outs_valid,
      ins_ready => buffer72_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork44_outs_0,
      outs(1) => fork44_outs_1,
      outs_valid(0) => fork44_outs_0_valid,
      outs_valid(1) => fork44_outs_1_valid,
      outs_ready(0) => fork44_outs_0_ready,
      outs_ready(1) => fork44_outs_1_ready
    );

  trunci11 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork44_outs_0,
      ins_valid => fork44_outs_0_valid,
      ins_ready => fork44_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci11_outs,
      outs_valid => trunci11_outs_valid,
      outs_ready => trunci11_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork44_outs_1,
      lhs_valid => fork44_outs_1_valid,
      lhs_ready => fork44_outs_1_ready,
      rhs => extsi39_outs,
      rhs_valid => extsi39_outs_valid,
      rhs_ready => extsi39_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer73 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer73_outs,
      outs_valid => buffer73_outs_valid,
      outs_ready => buffer73_outs_ready
    );

  fork45 : entity work.handshake_fork(arch) generic map(10, 1)
    port map(
      ins => buffer73_outs,
      ins_valid => buffer73_outs_valid,
      ins_ready => buffer73_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork45_outs_0,
      outs(1) => fork45_outs_1,
      outs(2) => fork45_outs_2,
      outs(3) => fork45_outs_3,
      outs(4) => fork45_outs_4,
      outs(5) => fork45_outs_5,
      outs(6) => fork45_outs_6,
      outs(7) => fork45_outs_7,
      outs(8) => fork45_outs_8,
      outs(9) => fork45_outs_9,
      outs_valid(0) => fork45_outs_0_valid,
      outs_valid(1) => fork45_outs_1_valid,
      outs_valid(2) => fork45_outs_2_valid,
      outs_valid(3) => fork45_outs_3_valid,
      outs_valid(4) => fork45_outs_4_valid,
      outs_valid(5) => fork45_outs_5_valid,
      outs_valid(6) => fork45_outs_6_valid,
      outs_valid(7) => fork45_outs_7_valid,
      outs_valid(8) => fork45_outs_8_valid,
      outs_valid(9) => fork45_outs_9_valid,
      outs_ready(0) => fork45_outs_0_ready,
      outs_ready(1) => fork45_outs_1_ready,
      outs_ready(2) => fork45_outs_2_ready,
      outs_ready(3) => fork45_outs_3_ready,
      outs_ready(4) => fork45_outs_4_ready,
      outs_ready(5) => fork45_outs_5_ready,
      outs_ready(6) => fork45_outs_6_ready,
      outs_ready(7) => fork45_outs_7_ready,
      outs_ready(8) => fork45_outs_8_ready,
      outs_ready(9) => fork45_outs_9_ready
    );

  cond_br12 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork45_outs_0,
      condition_valid => fork45_outs_0_valid,
      condition_ready => fork45_outs_0_ready,
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

  sink18 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br12_falseOut,
      ins_valid => cond_br12_falseOut_valid,
      ins_ready => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br13 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork45_outs_8,
      condition_valid => fork45_outs_8_valid,
      condition_ready => fork45_outs_8_ready,
      data_valid => fork39_outs_1_valid,
      data_ready => fork39_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  fork46 : entity work.fork_dataless(arch) generic map(4)
    port map(
      ins_valid => cond_br13_falseOut_valid,
      ins_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork46_outs_0_valid,
      outs_valid(1) => fork46_outs_1_valid,
      outs_valid(2) => fork46_outs_2_valid,
      outs_valid(3) => fork46_outs_3_valid,
      outs_ready(0) => fork46_outs_0_ready,
      outs_ready(1) => fork46_outs_1_ready,
      outs_ready(2) => fork46_outs_2_ready,
      outs_ready(3) => fork46_outs_3_ready
    );

end architecture;
