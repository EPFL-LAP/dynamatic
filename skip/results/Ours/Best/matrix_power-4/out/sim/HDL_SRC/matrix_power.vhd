library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity matrix_power is
  port (
    mat_loadData : in std_logic_vector(31 downto 0);
    row_loadData : in std_logic_vector(31 downto 0);
    col_loadData : in std_logic_vector(31 downto 0);
    a_loadData : in std_logic_vector(31 downto 0);
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
    mat_loadEn : out std_logic;
    mat_loadAddr : out std_logic_vector(8 downto 0);
    mat_storeEn : out std_logic;
    mat_storeAddr : out std_logic_vector(8 downto 0);
    mat_storeData : out std_logic_vector(31 downto 0);
    row_loadEn : out std_logic;
    row_loadAddr : out std_logic_vector(4 downto 0);
    row_storeEn : out std_logic;
    row_storeAddr : out std_logic_vector(4 downto 0);
    row_storeData : out std_logic_vector(31 downto 0);
    col_loadEn : out std_logic;
    col_loadAddr : out std_logic_vector(4 downto 0);
    col_storeEn : out std_logic;
    col_storeAddr : out std_logic_vector(4 downto 0);
    col_storeData : out std_logic_vector(31 downto 0);
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(4 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(4 downto 0);
    a_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of matrix_power is

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
  signal fork0_outs_8_valid : std_logic;
  signal fork0_outs_8_ready : std_logic;
  signal fork0_outs_9_valid : std_logic;
  signal fork0_outs_9_ready : std_logic;
  signal fork0_outs_10_valid : std_logic;
  signal fork0_outs_10_ready : std_logic;
  signal fork0_outs_11_valid : std_logic;
  signal fork0_outs_11_ready : std_logic;
  signal fork0_outs_12_valid : std_logic;
  signal fork0_outs_12_ready : std_logic;
  signal fork0_outs_13_valid : std_logic;
  signal fork0_outs_13_ready : std_logic;
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
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
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal mem_controller6_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller6_ldData_0_valid : std_logic;
  signal mem_controller6_ldData_0_ready : std_logic;
  signal mem_controller6_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller6_ldData_1_valid : std_logic;
  signal mem_controller6_ldData_1_ready : std_logic;
  signal mem_controller6_stDone_0_valid : std_logic;
  signal mem_controller6_stDone_0_ready : std_logic;
  signal mem_controller6_memEnd_valid : std_logic;
  signal mem_controller6_memEnd_ready : std_logic;
  signal mem_controller6_loadEn : std_logic;
  signal mem_controller6_loadAddr : std_logic_vector(8 downto 0);
  signal mem_controller6_storeEn : std_logic;
  signal mem_controller6_storeAddr : std_logic_vector(8 downto 0);
  signal mem_controller6_storeData : std_logic_vector(31 downto 0);
  signal constant18_outs : std_logic_vector(10 downto 0);
  signal constant18_outs_valid : std_logic;
  signal constant18_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(10 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(10 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal fork1_outs_2 : std_logic_vector(10 downto 0);
  signal fork1_outs_2_valid : std_logic;
  signal fork1_outs_2_ready : std_logic;
  signal fork1_outs_3 : std_logic_vector(10 downto 0);
  signal fork1_outs_3_valid : std_logic;
  signal fork1_outs_3_ready : std_logic;
  signal fork1_outs_4 : std_logic_vector(10 downto 0);
  signal fork1_outs_4_valid : std_logic;
  signal fork1_outs_4_ready : std_logic;
  signal fork1_outs_5 : std_logic_vector(10 downto 0);
  signal fork1_outs_5_valid : std_logic;
  signal fork1_outs_5_ready : std_logic;
  signal fork1_outs_6 : std_logic_vector(10 downto 0);
  signal fork1_outs_6_valid : std_logic;
  signal fork1_outs_6_ready : std_logic;
  signal fork1_outs_7 : std_logic_vector(10 downto 0);
  signal fork1_outs_7_valid : std_logic;
  signal fork1_outs_7_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal constant19_outs : std_logic_vector(1 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal extsi18_outs : std_logic_vector(5 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(31 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(31 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(31 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal mux10_outs : std_logic_vector(31 downto 0);
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal mux12_outs : std_logic_vector(31 downto 0);
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal mux13_outs : std_logic_vector(31 downto 0);
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal mux14_outs_valid : std_logic;
  signal mux14_outs_ready : std_logic;
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
  signal mux16_outs_valid : std_logic;
  signal mux16_outs_ready : std_logic;
  signal mux17_outs_valid : std_logic;
  signal mux17_outs_ready : std_logic;
  signal mux18_outs_valid : std_logic;
  signal mux18_outs_ready : std_logic;
  signal mux19_outs_valid : std_logic;
  signal mux19_outs_ready : std_logic;
  signal mux20_outs_valid : std_logic;
  signal mux20_outs_ready : std_logic;
  signal mux21_outs_valid : std_logic;
  signal mux21_outs_ready : std_logic;
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
  signal fork2_outs_7 : std_logic_vector(0 downto 0);
  signal fork2_outs_7_valid : std_logic;
  signal fork2_outs_7_ready : std_logic;
  signal fork2_outs_8 : std_logic_vector(0 downto 0);
  signal fork2_outs_8_valid : std_logic;
  signal fork2_outs_8_ready : std_logic;
  signal fork2_outs_9 : std_logic_vector(0 downto 0);
  signal fork2_outs_9_valid : std_logic;
  signal fork2_outs_9_ready : std_logic;
  signal fork2_outs_10 : std_logic_vector(0 downto 0);
  signal fork2_outs_10_valid : std_logic;
  signal fork2_outs_10_ready : std_logic;
  signal fork2_outs_11 : std_logic_vector(0 downto 0);
  signal fork2_outs_11_valid : std_logic;
  signal fork2_outs_11_ready : std_logic;
  signal fork2_outs_12 : std_logic_vector(0 downto 0);
  signal fork2_outs_12_valid : std_logic;
  signal fork2_outs_12_ready : std_logic;
  signal fork2_outs_13 : std_logic_vector(0 downto 0);
  signal fork2_outs_13_valid : std_logic;
  signal fork2_outs_13_ready : std_logic;
  signal fork2_outs_14 : std_logic_vector(0 downto 0);
  signal fork2_outs_14_valid : std_logic;
  signal fork2_outs_14_ready : std_logic;
  signal fork2_outs_15 : std_logic_vector(0 downto 0);
  signal fork2_outs_15_valid : std_logic;
  signal fork2_outs_15_ready : std_logic;
  signal fork2_outs_16 : std_logic_vector(0 downto 0);
  signal fork2_outs_16_valid : std_logic;
  signal fork2_outs_16_ready : std_logic;
  signal fork2_outs_17 : std_logic_vector(0 downto 0);
  signal fork2_outs_17_valid : std_logic;
  signal fork2_outs_17_ready : std_logic;
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(5 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(5 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(5 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi19_outs : std_logic_vector(31 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal constant20_outs : std_logic_vector(0 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal extsi17_outs : std_logic_vector(5 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal cond_br68_trueOut_valid : std_logic;
  signal cond_br68_trueOut_ready : std_logic;
  signal cond_br68_falseOut_valid : std_logic;
  signal cond_br68_falseOut_ready : std_logic;
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal cond_br69_trueOut_valid : std_logic;
  signal cond_br69_trueOut_ready : std_logic;
  signal cond_br69_falseOut_valid : std_logic;
  signal cond_br69_falseOut_ready : std_logic;
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal cond_br70_trueOut : std_logic_vector(31 downto 0);
  signal cond_br70_trueOut_valid : std_logic;
  signal cond_br70_trueOut_ready : std_logic;
  signal cond_br70_falseOut : std_logic_vector(31 downto 0);
  signal cond_br70_falseOut_valid : std_logic;
  signal cond_br70_falseOut_ready : std_logic;
  signal buffer42_outs : std_logic_vector(31 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal cond_br71_trueOut_valid : std_logic;
  signal cond_br71_trueOut_ready : std_logic;
  signal cond_br71_falseOut_valid : std_logic;
  signal cond_br71_falseOut_ready : std_logic;
  signal buffer43_outs : std_logic_vector(0 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal cond_br72_trueOut : std_logic_vector(31 downto 0);
  signal cond_br72_trueOut_valid : std_logic;
  signal cond_br72_trueOut_ready : std_logic;
  signal cond_br72_falseOut : std_logic_vector(31 downto 0);
  signal cond_br72_falseOut_valid : std_logic;
  signal cond_br72_falseOut_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal cond_br73_trueOut_valid : std_logic;
  signal cond_br73_trueOut_ready : std_logic;
  signal cond_br73_falseOut_valid : std_logic;
  signal cond_br73_falseOut_ready : std_logic;
  signal buffer45_outs : std_logic_vector(0 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal cond_br74_trueOut_valid : std_logic;
  signal cond_br74_trueOut_ready : std_logic;
  signal cond_br74_falseOut_valid : std_logic;
  signal cond_br74_falseOut_ready : std_logic;
  signal buffer46_outs : std_logic_vector(0 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal cond_br75_trueOut : std_logic_vector(31 downto 0);
  signal cond_br75_trueOut_valid : std_logic;
  signal cond_br75_trueOut_ready : std_logic;
  signal cond_br75_falseOut : std_logic_vector(31 downto 0);
  signal cond_br75_falseOut_valid : std_logic;
  signal cond_br75_falseOut_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(31 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(31 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal cond_br76_trueOut : std_logic_vector(31 downto 0);
  signal cond_br76_trueOut_valid : std_logic;
  signal cond_br76_trueOut_ready : std_logic;
  signal cond_br76_falseOut : std_logic_vector(31 downto 0);
  signal cond_br76_falseOut_valid : std_logic;
  signal cond_br76_falseOut_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal mux22_outs : std_logic_vector(31 downto 0);
  signal mux22_outs_valid : std_logic;
  signal mux22_outs_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux23_outs_valid : std_logic;
  signal mux23_outs_ready : std_logic;
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal mux24_outs_valid : std_logic;
  signal mux24_outs_ready : std_logic;
  signal mux25_outs : std_logic_vector(31 downto 0);
  signal mux25_outs_valid : std_logic;
  signal mux25_outs_ready : std_logic;
  signal mux26_outs : std_logic_vector(31 downto 0);
  signal mux26_outs_valid : std_logic;
  signal mux26_outs_ready : std_logic;
  signal mux27_outs : std_logic_vector(31 downto 0);
  signal mux27_outs_valid : std_logic;
  signal mux27_outs_ready : std_logic;
  signal mux28_outs : std_logic_vector(31 downto 0);
  signal mux28_outs_valid : std_logic;
  signal mux28_outs_ready : std_logic;
  signal mux29_outs : std_logic_vector(31 downto 0);
  signal mux29_outs_valid : std_logic;
  signal mux29_outs_ready : std_logic;
  signal mux30_outs : std_logic_vector(31 downto 0);
  signal mux30_outs_valid : std_logic;
  signal mux30_outs_ready : std_logic;
  signal mux31_outs : std_logic_vector(31 downto 0);
  signal mux31_outs_valid : std_logic;
  signal mux31_outs_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal mux32_outs_valid : std_logic;
  signal mux32_outs_ready : std_logic;
  signal buffer69_outs : std_logic_vector(0 downto 0);
  signal buffer69_outs_valid : std_logic;
  signal buffer69_outs_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal mux33_outs_valid : std_logic;
  signal mux33_outs_ready : std_logic;
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal mux34_outs_valid : std_logic;
  signal mux34_outs_ready : std_logic;
  signal buffer71_outs : std_logic_vector(0 downto 0);
  signal buffer71_outs_valid : std_logic;
  signal buffer71_outs_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal mux35_outs_valid : std_logic;
  signal mux35_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal mux36_outs_valid : std_logic;
  signal mux36_outs_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal mux37_outs_valid : std_logic;
  signal mux37_outs_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal mux38_outs_valid : std_logic;
  signal mux38_outs_ready : std_logic;
  signal buffer75_outs : std_logic_vector(0 downto 0);
  signal buffer75_outs_valid : std_logic;
  signal buffer75_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal mux39_outs_valid : std_logic;
  signal mux39_outs_ready : std_logic;
  signal buffer76_outs : std_logic_vector(0 downto 0);
  signal buffer76_outs_valid : std_logic;
  signal buffer76_outs_ready : std_logic;
  signal init18_outs : std_logic_vector(0 downto 0);
  signal init18_outs_valid : std_logic;
  signal init18_outs_ready : std_logic;
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
  signal fork14_outs_5 : std_logic_vector(0 downto 0);
  signal fork14_outs_5_valid : std_logic;
  signal fork14_outs_5_ready : std_logic;
  signal fork14_outs_6 : std_logic_vector(0 downto 0);
  signal fork14_outs_6_valid : std_logic;
  signal fork14_outs_6_ready : std_logic;
  signal fork14_outs_7 : std_logic_vector(0 downto 0);
  signal fork14_outs_7_valid : std_logic;
  signal fork14_outs_7_ready : std_logic;
  signal fork14_outs_8 : std_logic_vector(0 downto 0);
  signal fork14_outs_8_valid : std_logic;
  signal fork14_outs_8_ready : std_logic;
  signal fork14_outs_9 : std_logic_vector(0 downto 0);
  signal fork14_outs_9_valid : std_logic;
  signal fork14_outs_9_ready : std_logic;
  signal fork14_outs_10 : std_logic_vector(0 downto 0);
  signal fork14_outs_10_valid : std_logic;
  signal fork14_outs_10_ready : std_logic;
  signal fork14_outs_11 : std_logic_vector(0 downto 0);
  signal fork14_outs_11_valid : std_logic;
  signal fork14_outs_11_ready : std_logic;
  signal fork14_outs_12 : std_logic_vector(0 downto 0);
  signal fork14_outs_12_valid : std_logic;
  signal fork14_outs_12_ready : std_logic;
  signal fork14_outs_13 : std_logic_vector(0 downto 0);
  signal fork14_outs_13_valid : std_logic;
  signal fork14_outs_13_ready : std_logic;
  signal fork14_outs_14 : std_logic_vector(0 downto 0);
  signal fork14_outs_14_valid : std_logic;
  signal fork14_outs_14_ready : std_logic;
  signal fork14_outs_15 : std_logic_vector(0 downto 0);
  signal fork14_outs_15_valid : std_logic;
  signal fork14_outs_15_ready : std_logic;
  signal fork14_outs_16 : std_logic_vector(0 downto 0);
  signal fork14_outs_16_valid : std_logic;
  signal fork14_outs_16_ready : std_logic;
  signal fork14_outs_17 : std_logic_vector(0 downto 0);
  signal fork14_outs_17_valid : std_logic;
  signal fork14_outs_17_ready : std_logic;
  signal mux1_outs : std_logic_vector(5 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer37_outs : std_logic_vector(5 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(5 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(5 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork15_outs_2 : std_logic_vector(5 downto 0);
  signal fork15_outs_2_valid : std_logic;
  signal fork15_outs_2_ready : std_logic;
  signal fork15_outs_3 : std_logic_vector(5 downto 0);
  signal fork15_outs_3_valid : std_logic;
  signal fork15_outs_3_ready : std_logic;
  signal extsi20_outs : std_logic_vector(6 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal trunci0_outs : std_logic_vector(4 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(4 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(4 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(5 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer38_outs : std_logic_vector(5 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal buffer39_outs : std_logic_vector(5 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(5 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(5 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal extsi21_outs : std_logic_vector(31 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
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
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer40_outs : std_logic_vector(31 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal buffer41_outs : std_logic_vector(31 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(31 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(31 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(31 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(0 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(0 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(0 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal constant21_outs : std_logic_vector(1 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(31 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant22_outs : std_logic_vector(1 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal extsi22_outs : std_logic_vector(6 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant23_outs : std_logic_vector(5 downto 0);
  signal constant23_outs_valid : std_logic;
  signal constant23_outs_ready : std_logic;
  signal extsi23_outs : std_logic_vector(6 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant24_outs : std_logic_vector(3 downto 0);
  signal constant24_outs_valid : std_logic;
  signal constant24_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(31 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(31 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(31 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork21_outs_2 : std_logic_vector(31 downto 0);
  signal fork21_outs_2_valid : std_logic;
  signal fork21_outs_2_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant25_outs : std_logic_vector(2 downto 0);
  signal constant25_outs_valid : std_logic;
  signal constant25_outs_ready : std_logic;
  signal extsi14_outs : std_logic_vector(31 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(31 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(31 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal fork22_outs_2 : std_logic_vector(31 downto 0);
  signal fork22_outs_2_valid : std_logic;
  signal fork22_outs_2_ready : std_logic;
  signal load0_addrOut : std_logic_vector(4 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(31 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(31 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal load1_addrOut : std_logic_vector(4 downto 0);
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
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer47_outs : std_logic_vector(31 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal buffer48_outs : std_logic_vector(31 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(31 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer49_outs : std_logic_vector(31 downto 0);
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(31 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(31 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal buffer50_outs : std_logic_vector(31 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(31 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(31 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork25_outs_2 : std_logic_vector(31 downto 0);
  signal fork25_outs_2_valid : std_logic;
  signal fork25_outs_2_ready : std_logic;
  signal fork25_outs_3 : std_logic_vector(31 downto 0);
  signal fork25_outs_3_valid : std_logic;
  signal fork25_outs_3_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(0 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(0 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(0 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(0 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
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
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal cond_br42_trueOut_valid : std_logic;
  signal cond_br42_trueOut_ready : std_logic;
  signal cond_br42_falseOut_valid : std_logic;
  signal cond_br42_falseOut_ready : std_logic;
  signal buffer95_outs : std_logic_vector(0 downto 0);
  signal buffer95_outs_valid : std_logic;
  signal buffer95_outs_ready : std_logic;
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal cond_br43_trueOut_valid : std_logic;
  signal cond_br43_trueOut_ready : std_logic;
  signal cond_br43_falseOut_valid : std_logic;
  signal cond_br43_falseOut_ready : std_logic;
  signal buffer96_outs : std_logic_vector(0 downto 0);
  signal buffer96_outs_valid : std_logic;
  signal buffer96_outs_ready : std_logic;
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal cond_br44_trueOut_valid : std_logic;
  signal cond_br44_trueOut_ready : std_logic;
  signal cond_br44_falseOut_valid : std_logic;
  signal cond_br44_falseOut_ready : std_logic;
  signal buffer97_outs : std_logic_vector(0 downto 0);
  signal buffer97_outs_valid : std_logic;
  signal buffer97_outs_ready : std_logic;
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal cond_br45_trueOut_valid : std_logic;
  signal cond_br45_trueOut_ready : std_logic;
  signal cond_br45_falseOut_valid : std_logic;
  signal cond_br45_falseOut_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal mux40_outs_valid : std_logic;
  signal mux40_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal mux41_outs_valid : std_logic;
  signal mux41_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal mux42_outs_valid : std_logic;
  signal mux42_outs_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal mux43_outs_valid : std_logic;
  signal mux43_outs_ready : std_logic;
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal buffer55_outs_valid : std_logic;
  signal buffer55_outs_ready : std_logic;
  signal join0_outs_valid : std_logic;
  signal join0_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(8 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(8 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal shli2_result : std_logic_vector(31 downto 0);
  signal shli2_result_valid : std_logic;
  signal shli2_result_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer56_outs : std_logic_vector(31 downto 0);
  signal buffer56_outs_valid : std_logic;
  signal buffer56_outs_ready : std_logic;
  signal buffer57_outs : std_logic_vector(31 downto 0);
  signal buffer57_outs_valid : std_logic;
  signal buffer57_outs_ready : std_logic;
  signal addi8_result : std_logic_vector(31 downto 0);
  signal addi8_result_valid : std_logic;
  signal addi8_result_ready : std_logic;
  signal buffer58_outs : std_logic_vector(31 downto 0);
  signal buffer58_outs_valid : std_logic;
  signal buffer58_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(31 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal fork30_outs_0 : std_logic_vector(31 downto 0);
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1 : std_logic_vector(31 downto 0);
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal gate2_outs : std_logic_vector(31 downto 0);
  signal gate2_outs_valid : std_logic;
  signal gate2_outs_ready : std_logic;
  signal buffer59_outs : std_logic_vector(31 downto 0);
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal fork31_outs_0 : std_logic_vector(31 downto 0);
  signal fork31_outs_0_valid : std_logic;
  signal fork31_outs_0_ready : std_logic;
  signal fork31_outs_1 : std_logic_vector(31 downto 0);
  signal fork31_outs_1_valid : std_logic;
  signal fork31_outs_1_ready : std_logic;
  signal fork31_outs_2 : std_logic_vector(31 downto 0);
  signal fork31_outs_2_valid : std_logic;
  signal fork31_outs_2_ready : std_logic;
  signal fork31_outs_3 : std_logic_vector(31 downto 0);
  signal fork31_outs_3_valid : std_logic;
  signal fork31_outs_3_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal cmpi6_result : std_logic_vector(0 downto 0);
  signal cmpi6_result_valid : std_logic;
  signal cmpi6_result_ready : std_logic;
  signal fork32_outs_0 : std_logic_vector(0 downto 0);
  signal fork32_outs_0_valid : std_logic;
  signal fork32_outs_0_ready : std_logic;
  signal fork32_outs_1 : std_logic_vector(0 downto 0);
  signal fork32_outs_1_valid : std_logic;
  signal fork32_outs_1_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal cmpi7_result : std_logic_vector(0 downto 0);
  signal cmpi7_result_valid : std_logic;
  signal cmpi7_result_ready : std_logic;
  signal fork33_outs_0 : std_logic_vector(0 downto 0);
  signal fork33_outs_0_valid : std_logic;
  signal fork33_outs_0_ready : std_logic;
  signal fork33_outs_1 : std_logic_vector(0 downto 0);
  signal fork33_outs_1_valid : std_logic;
  signal fork33_outs_1_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal cmpi8_result : std_logic_vector(0 downto 0);
  signal cmpi8_result_valid : std_logic;
  signal cmpi8_result_ready : std_logic;
  signal fork34_outs_0 : std_logic_vector(0 downto 0);
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1 : std_logic_vector(0 downto 0);
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal buffer24_outs : std_logic_vector(31 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal cmpi9_result : std_logic_vector(0 downto 0);
  signal cmpi9_result_valid : std_logic;
  signal cmpi9_result_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(0 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(0 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal cond_br46_trueOut_valid : std_logic;
  signal cond_br46_trueOut_ready : std_logic;
  signal cond_br46_falseOut_valid : std_logic;
  signal cond_br46_falseOut_ready : std_logic;
  signal buffer114_outs : std_logic_vector(0 downto 0);
  signal buffer114_outs_valid : std_logic;
  signal buffer114_outs_ready : std_logic;
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal cond_br47_trueOut_valid : std_logic;
  signal cond_br47_trueOut_ready : std_logic;
  signal cond_br47_falseOut_valid : std_logic;
  signal cond_br47_falseOut_ready : std_logic;
  signal buffer115_outs : std_logic_vector(0 downto 0);
  signal buffer115_outs_valid : std_logic;
  signal buffer115_outs_ready : std_logic;
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal cond_br48_trueOut_valid : std_logic;
  signal cond_br48_trueOut_ready : std_logic;
  signal cond_br48_falseOut_valid : std_logic;
  signal cond_br48_falseOut_ready : std_logic;
  signal buffer116_outs : std_logic_vector(0 downto 0);
  signal buffer116_outs_valid : std_logic;
  signal buffer116_outs_ready : std_logic;
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal cond_br49_trueOut_valid : std_logic;
  signal cond_br49_trueOut_ready : std_logic;
  signal cond_br49_falseOut_valid : std_logic;
  signal cond_br49_falseOut_ready : std_logic;
  signal buffer117_outs : std_logic_vector(0 downto 0);
  signal buffer117_outs_valid : std_logic;
  signal buffer117_outs_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal mux44_outs_valid : std_logic;
  signal mux44_outs_ready : std_logic;
  signal buffer118_outs : std_logic_vector(0 downto 0);
  signal buffer118_outs_valid : std_logic;
  signal buffer118_outs_ready : std_logic;
  signal source12_outs_valid : std_logic;
  signal source12_outs_ready : std_logic;
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal mux45_outs_valid : std_logic;
  signal mux45_outs_ready : std_logic;
  signal buffer119_outs : std_logic_vector(0 downto 0);
  signal buffer119_outs_valid : std_logic;
  signal buffer119_outs_ready : std_logic;
  signal source13_outs_valid : std_logic;
  signal source13_outs_ready : std_logic;
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal mux46_outs_valid : std_logic;
  signal mux46_outs_ready : std_logic;
  signal buffer120_outs : std_logic_vector(0 downto 0);
  signal buffer120_outs_valid : std_logic;
  signal buffer120_outs_ready : std_logic;
  signal source14_outs_valid : std_logic;
  signal source14_outs_ready : std_logic;
  signal mux47_outs_valid : std_logic;
  signal mux47_outs_ready : std_logic;
  signal buffer121_outs : std_logic_vector(0 downto 0);
  signal buffer121_outs_valid : std_logic;
  signal buffer121_outs_ready : std_logic;
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal buffer63_outs_valid : std_logic;
  signal buffer63_outs_ready : std_logic;
  signal buffer64_outs_valid : std_logic;
  signal buffer64_outs_ready : std_logic;
  signal buffer65_outs_valid : std_logic;
  signal buffer65_outs_ready : std_logic;
  signal join1_outs_valid : std_logic;
  signal join1_outs_ready : std_logic;
  signal gate3_outs : std_logic_vector(31 downto 0);
  signal gate3_outs_valid : std_logic;
  signal gate3_outs_ready : std_logic;
  signal buffer122_outs : std_logic_vector(31 downto 0);
  signal buffer122_outs_valid : std_logic;
  signal buffer122_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(8 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal load4_addrOut : std_logic_vector(8 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal shli4_result : std_logic_vector(31 downto 0);
  signal shli4_result_valid : std_logic;
  signal shli4_result_ready : std_logic;
  signal shli5_result : std_logic_vector(31 downto 0);
  signal shli5_result_valid : std_logic;
  signal shli5_result_ready : std_logic;
  signal buffer126_outs : std_logic_vector(31 downto 0);
  signal buffer126_outs_valid : std_logic;
  signal buffer126_outs_ready : std_logic;
  signal buffer66_outs : std_logic_vector(31 downto 0);
  signal buffer66_outs_valid : std_logic;
  signal buffer66_outs_ready : std_logic;
  signal buffer67_outs : std_logic_vector(31 downto 0);
  signal buffer67_outs_valid : std_logic;
  signal buffer67_outs_ready : std_logic;
  signal addi9_result : std_logic_vector(31 downto 0);
  signal addi9_result_valid : std_logic;
  signal addi9_result_ready : std_logic;
  signal buffer68_outs : std_logic_vector(31 downto 0);
  signal buffer68_outs_valid : std_logic;
  signal buffer68_outs_ready : std_logic;
  signal addi4_result : std_logic_vector(31 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer127_outs : std_logic_vector(31 downto 0);
  signal buffer127_outs_valid : std_logic;
  signal buffer127_outs_ready : std_logic;
  signal buffer70_outs : std_logic_vector(31 downto 0);
  signal buffer70_outs_valid : std_logic;
  signal buffer70_outs_ready : std_logic;
  signal fork36_outs_0 : std_logic_vector(31 downto 0);
  signal fork36_outs_0_valid : std_logic;
  signal fork36_outs_0_ready : std_logic;
  signal fork36_outs_1 : std_logic_vector(31 downto 0);
  signal fork36_outs_1_valid : std_logic;
  signal fork36_outs_1_ready : std_logic;
  signal trunci5_outs : std_logic_vector(8 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal buffer128_outs : std_logic_vector(31 downto 0);
  signal buffer128_outs_valid : std_logic;
  signal buffer128_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork37_outs_0 : std_logic_vector(31 downto 0);
  signal fork37_outs_0_valid : std_logic;
  signal fork37_outs_0_ready : std_logic;
  signal fork37_outs_1 : std_logic_vector(31 downto 0);
  signal fork37_outs_1_valid : std_logic;
  signal fork37_outs_1_ready : std_logic;
  signal init36_outs : std_logic_vector(31 downto 0);
  signal init36_outs_valid : std_logic;
  signal init36_outs_ready : std_logic;
  signal fork38_outs_0 : std_logic_vector(31 downto 0);
  signal fork38_outs_0_valid : std_logic;
  signal fork38_outs_0_ready : std_logic;
  signal fork38_outs_1 : std_logic_vector(31 downto 0);
  signal fork38_outs_1_valid : std_logic;
  signal fork38_outs_1_ready : std_logic;
  signal init37_outs : std_logic_vector(31 downto 0);
  signal init37_outs_valid : std_logic;
  signal init37_outs_ready : std_logic;
  signal buffer131_outs : std_logic_vector(31 downto 0);
  signal buffer131_outs_valid : std_logic;
  signal buffer131_outs_ready : std_logic;
  signal fork39_outs_0 : std_logic_vector(31 downto 0);
  signal fork39_outs_0_valid : std_logic;
  signal fork39_outs_0_ready : std_logic;
  signal fork39_outs_1 : std_logic_vector(31 downto 0);
  signal fork39_outs_1_valid : std_logic;
  signal fork39_outs_1_ready : std_logic;
  signal init38_outs : std_logic_vector(31 downto 0);
  signal init38_outs_valid : std_logic;
  signal init38_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer72_outs_valid : std_logic;
  signal buffer72_outs_ready : std_logic;
  signal fork40_outs_0_valid : std_logic;
  signal fork40_outs_0_ready : std_logic;
  signal fork40_outs_1_valid : std_logic;
  signal fork40_outs_1_ready : std_logic;
  signal init39_outs_valid : std_logic;
  signal init39_outs_ready : std_logic;
  signal fork41_outs_0_valid : std_logic;
  signal fork41_outs_0_ready : std_logic;
  signal fork41_outs_1_valid : std_logic;
  signal fork41_outs_1_ready : std_logic;
  signal init40_outs_valid : std_logic;
  signal init40_outs_ready : std_logic;
  signal fork42_outs_0_valid : std_logic;
  signal fork42_outs_0_ready : std_logic;
  signal fork42_outs_1_valid : std_logic;
  signal fork42_outs_1_ready : std_logic;
  signal init41_outs_valid : std_logic;
  signal init41_outs_ready : std_logic;
  signal fork43_outs_0_valid : std_logic;
  signal fork43_outs_0_ready : std_logic;
  signal fork43_outs_1_valid : std_logic;
  signal fork43_outs_1_ready : std_logic;
  signal init42_outs_valid : std_logic;
  signal init42_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(8 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal addi5_result : std_logic_vector(6 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer73_outs : std_logic_vector(6 downto 0);
  signal buffer73_outs_valid : std_logic;
  signal buffer73_outs_ready : std_logic;
  signal fork44_outs_0 : std_logic_vector(6 downto 0);
  signal fork44_outs_0_valid : std_logic;
  signal fork44_outs_0_ready : std_logic;
  signal fork44_outs_1 : std_logic_vector(6 downto 0);
  signal fork44_outs_1_valid : std_logic;
  signal fork44_outs_1_ready : std_logic;
  signal trunci6_outs : std_logic_vector(5 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer74_outs : std_logic_vector(0 downto 0);
  signal buffer74_outs_valid : std_logic;
  signal buffer74_outs_ready : std_logic;
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
  signal fork45_outs_10 : std_logic_vector(0 downto 0);
  signal fork45_outs_10_valid : std_logic;
  signal fork45_outs_10_ready : std_logic;
  signal fork45_outs_11 : std_logic_vector(0 downto 0);
  signal fork45_outs_11_valid : std_logic;
  signal fork45_outs_11_ready : std_logic;
  signal fork45_outs_12 : std_logic_vector(0 downto 0);
  signal fork45_outs_12_valid : std_logic;
  signal fork45_outs_12_ready : std_logic;
  signal fork45_outs_13 : std_logic_vector(0 downto 0);
  signal fork45_outs_13_valid : std_logic;
  signal fork45_outs_13_ready : std_logic;
  signal cond_br2_trueOut : std_logic_vector(5 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(5 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(5 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(5 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer137_outs : std_logic_vector(5 downto 0);
  signal buffer137_outs_valid : std_logic;
  signal buffer137_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(31 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(31 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal buffer139_outs : std_logic_vector(31 downto 0);
  signal buffer139_outs_valid : std_logic;
  signal buffer139_outs_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal cond_br77_trueOut_valid : std_logic;
  signal cond_br77_trueOut_ready : std_logic;
  signal cond_br77_falseOut_valid : std_logic;
  signal cond_br77_falseOut_ready : std_logic;
  signal fork46_outs_0_valid : std_logic;
  signal fork46_outs_0_ready : std_logic;
  signal fork46_outs_1_valid : std_logic;
  signal fork46_outs_1_ready : std_logic;
  signal cond_br78_trueOut_valid : std_logic;
  signal cond_br78_trueOut_ready : std_logic;
  signal cond_br78_falseOut_valid : std_logic;
  signal cond_br78_falseOut_ready : std_logic;
  signal fork47_outs_0_valid : std_logic;
  signal fork47_outs_0_ready : std_logic;
  signal fork47_outs_1_valid : std_logic;
  signal fork47_outs_1_ready : std_logic;
  signal cond_br79_trueOut_valid : std_logic;
  signal cond_br79_trueOut_ready : std_logic;
  signal cond_br79_falseOut_valid : std_logic;
  signal cond_br79_falseOut_ready : std_logic;
  signal fork48_outs_0_valid : std_logic;
  signal fork48_outs_0_ready : std_logic;
  signal fork48_outs_1_valid : std_logic;
  signal fork48_outs_1_ready : std_logic;
  signal cond_br80_trueOut_valid : std_logic;
  signal cond_br80_trueOut_ready : std_logic;
  signal cond_br80_falseOut_valid : std_logic;
  signal cond_br80_falseOut_ready : std_logic;
  signal buffer144_outs : std_logic_vector(0 downto 0);
  signal buffer144_outs_valid : std_logic;
  signal buffer144_outs_ready : std_logic;
  signal fork49_outs_0_valid : std_logic;
  signal fork49_outs_0_ready : std_logic;
  signal fork49_outs_1_valid : std_logic;
  signal fork49_outs_1_ready : std_logic;
  signal cond_br81_trueOut : std_logic_vector(31 downto 0);
  signal cond_br81_trueOut_valid : std_logic;
  signal cond_br81_trueOut_ready : std_logic;
  signal cond_br81_falseOut : std_logic_vector(31 downto 0);
  signal cond_br81_falseOut_valid : std_logic;
  signal cond_br81_falseOut_ready : std_logic;
  signal fork50_outs_0 : std_logic_vector(31 downto 0);
  signal fork50_outs_0_valid : std_logic;
  signal fork50_outs_0_ready : std_logic;
  signal fork50_outs_1 : std_logic_vector(31 downto 0);
  signal fork50_outs_1_valid : std_logic;
  signal fork50_outs_1_ready : std_logic;
  signal cond_br82_trueOut : std_logic_vector(31 downto 0);
  signal cond_br82_trueOut_valid : std_logic;
  signal cond_br82_trueOut_ready : std_logic;
  signal cond_br82_falseOut : std_logic_vector(31 downto 0);
  signal cond_br82_falseOut_valid : std_logic;
  signal cond_br82_falseOut_ready : std_logic;
  signal fork51_outs_0 : std_logic_vector(31 downto 0);
  signal fork51_outs_0_valid : std_logic;
  signal fork51_outs_0_ready : std_logic;
  signal fork51_outs_1 : std_logic_vector(31 downto 0);
  signal fork51_outs_1_valid : std_logic;
  signal fork51_outs_1_ready : std_logic;
  signal cond_br83_trueOut : std_logic_vector(31 downto 0);
  signal cond_br83_trueOut_valid : std_logic;
  signal cond_br83_trueOut_ready : std_logic;
  signal cond_br83_falseOut : std_logic_vector(31 downto 0);
  signal cond_br83_falseOut_valid : std_logic;
  signal cond_br83_falseOut_ready : std_logic;
  signal fork52_outs_0 : std_logic_vector(31 downto 0);
  signal fork52_outs_0_valid : std_logic;
  signal fork52_outs_0_ready : std_logic;
  signal fork52_outs_1 : std_logic_vector(31 downto 0);
  signal fork52_outs_1_valid : std_logic;
  signal fork52_outs_1_ready : std_logic;
  signal cond_br84_trueOut_valid : std_logic;
  signal cond_br84_trueOut_ready : std_logic;
  signal cond_br84_falseOut_valid : std_logic;
  signal cond_br84_falseOut_ready : std_logic;
  signal fork53_outs_0_valid : std_logic;
  signal fork53_outs_0_ready : std_logic;
  signal fork53_outs_1_valid : std_logic;
  signal fork53_outs_1_ready : std_logic;
  signal cond_br85_trueOut : std_logic_vector(31 downto 0);
  signal cond_br85_trueOut_valid : std_logic;
  signal cond_br85_trueOut_ready : std_logic;
  signal cond_br85_falseOut : std_logic_vector(31 downto 0);
  signal cond_br85_falseOut_valid : std_logic;
  signal cond_br85_falseOut_ready : std_logic;
  signal fork54_outs_0 : std_logic_vector(31 downto 0);
  signal fork54_outs_0_valid : std_logic;
  signal fork54_outs_0_ready : std_logic;
  signal fork54_outs_1 : std_logic_vector(31 downto 0);
  signal fork54_outs_1_valid : std_logic;
  signal fork54_outs_1_ready : std_logic;
  signal extsi24_outs : std_logic_vector(6 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant26_outs : std_logic_vector(1 downto 0);
  signal constant26_outs_valid : std_logic;
  signal constant26_outs_ready : std_logic;
  signal extsi25_outs : std_logic_vector(6 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant27_outs : std_logic_vector(5 downto 0);
  signal constant27_outs_valid : std_logic;
  signal constant27_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(6 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal addi6_result : std_logic_vector(6 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer77_outs : std_logic_vector(6 downto 0);
  signal buffer77_outs_valid : std_logic;
  signal buffer77_outs_ready : std_logic;
  signal fork55_outs_0 : std_logic_vector(6 downto 0);
  signal fork55_outs_0_valid : std_logic;
  signal fork55_outs_0_ready : std_logic;
  signal fork55_outs_1 : std_logic_vector(6 downto 0);
  signal fork55_outs_1_valid : std_logic;
  signal fork55_outs_1_ready : std_logic;
  signal trunci7_outs : std_logic_vector(5 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer78_outs : std_logic_vector(0 downto 0);
  signal buffer78_outs_valid : std_logic;
  signal buffer78_outs_ready : std_logic;
  signal fork56_outs_0 : std_logic_vector(0 downto 0);
  signal fork56_outs_0_valid : std_logic;
  signal fork56_outs_0_ready : std_logic;
  signal fork56_outs_1 : std_logic_vector(0 downto 0);
  signal fork56_outs_1_valid : std_logic;
  signal fork56_outs_1_ready : std_logic;
  signal fork56_outs_2 : std_logic_vector(0 downto 0);
  signal fork56_outs_2_valid : std_logic;
  signal fork56_outs_2_ready : std_logic;
  signal fork56_outs_3 : std_logic_vector(0 downto 0);
  signal fork56_outs_3_valid : std_logic;
  signal fork56_outs_3_ready : std_logic;
  signal fork56_outs_4 : std_logic_vector(0 downto 0);
  signal fork56_outs_4_valid : std_logic;
  signal fork56_outs_4_ready : std_logic;
  signal fork56_outs_5 : std_logic_vector(0 downto 0);
  signal fork56_outs_5_valid : std_logic;
  signal fork56_outs_5_ready : std_logic;
  signal fork56_outs_6 : std_logic_vector(0 downto 0);
  signal fork56_outs_6_valid : std_logic;
  signal fork56_outs_6_ready : std_logic;
  signal fork56_outs_7 : std_logic_vector(0 downto 0);
  signal fork56_outs_7_valid : std_logic;
  signal fork56_outs_7_ready : std_logic;
  signal fork56_outs_8 : std_logic_vector(0 downto 0);
  signal fork56_outs_8_valid : std_logic;
  signal fork56_outs_8_ready : std_logic;
  signal fork56_outs_9 : std_logic_vector(0 downto 0);
  signal fork56_outs_9_valid : std_logic;
  signal fork56_outs_9_ready : std_logic;
  signal fork56_outs_10 : std_logic_vector(0 downto 0);
  signal fork56_outs_10_valid : std_logic;
  signal fork56_outs_10_ready : std_logic;
  signal fork56_outs_11 : std_logic_vector(0 downto 0);
  signal fork56_outs_11_valid : std_logic;
  signal fork56_outs_11_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(5 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(5 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal fork57_outs_0_valid : std_logic;
  signal fork57_outs_0_ready : std_logic;
  signal fork57_outs_1_valid : std_logic;
  signal fork57_outs_1_ready : std_logic;
  signal fork57_outs_2_valid : std_logic;
  signal fork57_outs_2_ready : std_logic;
  signal fork57_outs_3_valid : std_logic;
  signal fork57_outs_3_ready : std_logic;

begin

  mat_end_valid <= mem_controller6_memEnd_valid;
  mem_controller6_memEnd_ready <= mat_end_ready;
  row_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= row_end_ready;
  col_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= col_end_ready;
  a_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= a_end_ready;
  end_valid <= fork0_outs_2_valid;
  fork0_outs_2_ready <= end_ready;
  mat_loadEn <= mem_controller6_loadEn;
  mat_loadAddr <= mem_controller6_loadAddr;
  mat_storeEn <= mem_controller6_storeEn;
  mat_storeAddr <= mem_controller6_storeAddr;
  mat_storeData <= mem_controller6_storeData;
  row_loadEn <= mem_controller5_loadEn;
  row_loadAddr <= mem_controller5_loadAddr;
  row_storeEn <= mem_controller5_storeEn;
  row_storeAddr <= mem_controller5_storeAddr;
  row_storeData <= mem_controller5_storeData;
  col_loadEn <= mem_controller4_loadEn;
  col_loadAddr <= mem_controller4_loadAddr;
  col_storeEn <= mem_controller4_storeEn;
  col_storeAddr <= mem_controller4_storeAddr;
  col_storeData <= mem_controller4_storeData;
  a_loadEn <= mem_controller3_loadEn;
  a_loadAddr <= mem_controller3_loadAddr;
  a_storeEn <= mem_controller3_storeEn;
  a_storeAddr <= mem_controller3_storeAddr;
  a_storeData <= mem_controller3_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(14)
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
      outs_valid(8) => fork0_outs_8_valid,
      outs_valid(9) => fork0_outs_9_valid,
      outs_valid(10) => fork0_outs_10_valid,
      outs_valid(11) => fork0_outs_11_valid,
      outs_valid(12) => fork0_outs_12_valid,
      outs_valid(13) => fork0_outs_13_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready,
      outs_ready(3) => fork0_outs_3_ready,
      outs_ready(4) => fork0_outs_4_ready,
      outs_ready(5) => fork0_outs_5_ready,
      outs_ready(6) => fork0_outs_6_ready,
      outs_ready(7) => fork0_outs_7_ready,
      outs_ready(8) => fork0_outs_8_ready,
      outs_ready(9) => fork0_outs_9_ready,
      outs_ready(10) => fork0_outs_10_ready,
      outs_ready(11) => fork0_outs_11_ready,
      outs_ready(12) => fork0_outs_12_ready,
      outs_ready(13) => fork0_outs_13_ready
    );

  mem_controller3 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork57_outs_3_valid,
      ctrlEnd_ready => fork57_outs_3_ready,
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

  mem_controller4 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => col_loadData,
      memStart_valid => col_start_valid,
      memStart_ready => col_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork57_outs_2_valid,
      ctrlEnd_ready => fork57_outs_2_ready,
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

  mem_controller5 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => row_loadData,
      memStart_valid => row_start_valid,
      memStart_ready => row_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork57_outs_1_valid,
      ctrlEnd_ready => fork57_outs_1_ready,
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

  mem_controller6 : entity work.mem_controller(arch) generic map(1, 2, 1, 32, 9)
    port map(
      loadData => mat_loadData,
      memStart_valid => mat_start_valid,
      memStart_ready => mat_start_ready,
      ctrl(0) => extsi10_outs,
      ctrl_valid(0) => extsi10_outs_valid,
      ctrl_ready(0) => extsi10_outs_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr(1) => load4_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_valid(1) => load4_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      ldAddr_ready(1) => load4_addrOut_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => fork57_outs_0_valid,
      ctrlEnd_ready => fork57_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller6_ldData_0,
      ldData(1) => mem_controller6_ldData_1,
      ldData_valid(0) => mem_controller6_ldData_0_valid,
      ldData_valid(1) => mem_controller6_ldData_1_valid,
      ldData_ready(0) => mem_controller6_ldData_0_ready,
      ldData_ready(1) => mem_controller6_ldData_1_ready,
      stDone_valid(0) => mem_controller6_stDone_0_valid,
      stDone_ready(0) => mem_controller6_stDone_0_ready,
      memEnd_valid => mem_controller6_memEnd_valid,
      memEnd_ready => mem_controller6_memEnd_ready,
      loadEn => mem_controller6_loadEn,
      loadAddr => mem_controller6_loadAddr,
      storeEn => mem_controller6_storeEn,
      storeAddr => mem_controller6_storeAddr,
      storeData => mem_controller6_storeData
    );

  constant18 : entity work.handshake_constant_0(arch) generic map(11)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant18_outs,
      outs_valid => constant18_outs_valid,
      outs_ready => constant18_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(8, 11)
    port map(
      ins => constant18_outs,
      ins_valid => constant18_outs_valid,
      ins_ready => constant18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs(3) => fork1_outs_3,
      outs(4) => fork1_outs_4,
      outs(5) => fork1_outs_5,
      outs(6) => fork1_outs_6,
      outs(7) => fork1_outs_7,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_valid(3) => fork1_outs_3_valid,
      outs_valid(4) => fork1_outs_4_valid,
      outs_valid(5) => fork1_outs_5_valid,
      outs_valid(6) => fork1_outs_6_valid,
      outs_valid(7) => fork1_outs_7_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready,
      outs_ready(3) => fork1_outs_3_ready,
      outs_ready(4) => fork1_outs_4_ready,
      outs_ready(5) => fork1_outs_5_ready,
      outs_ready(6) => fork1_outs_6_ready,
      outs_ready(7) => fork1_outs_7_ready
    );

  extsi0 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi0_outs,
      outs_valid => extsi0_outs_valid,
      outs_ready => extsi0_outs_ready
    );

  extsi1 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_2,
      ins_valid => fork1_outs_2_valid,
      ins_ready => fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_3,
      ins_valid => fork1_outs_3_valid,
      ins_ready => fork1_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_4,
      ins_valid => fork1_outs_4_valid,
      ins_ready => fork1_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_5,
      ins_valid => fork1_outs_5_valid,
      ins_ready => fork1_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  extsi6 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_6,
      ins_valid => fork1_outs_6_valid,
      ins_ready => fork1_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  extsi7 : entity work.extsi(arch) generic map(11, 32)
    port map(
      ins => fork1_outs_7,
      ins_valid => fork1_outs_7_valid,
      ins_ready => fork1_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  constant19 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant19_outs,
      outs_valid => constant19_outs_valid,
      outs_ready => constant19_outs_ready
    );

  extsi18 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant19_outs,
      ins_valid => constant19_outs_valid,
      ins_ready => constant19_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => extsi0_outs,
      ins(1) => fork52_outs_0,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => fork52_outs_0_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => fork52_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins_valid(0) => fork0_outs_12_valid,
      ins_valid(1) => fork46_outs_0_valid,
      ins_ready(0) => fork0_outs_12_ready,
      ins_ready(1) => fork46_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_2,
      index_valid => fork2_outs_2_valid,
      index_ready => fork2_outs_2_ready,
      ins_valid(0) => fork0_outs_11_valid,
      ins_valid(1) => fork46_outs_1_valid,
      ins_ready(0) => fork0_outs_11_ready,
      ins_ready(1) => fork46_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_3,
      index_valid => fork2_outs_3_valid,
      index_ready => fork2_outs_3_ready,
      ins(0) => extsi1_outs,
      ins(1) => fork50_outs_0,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => fork50_outs_0_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => fork50_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_4,
      index_valid => fork2_outs_4_valid,
      index_ready => fork2_outs_4_ready,
      ins(0) => extsi2_outs,
      ins(1) => fork54_outs_0,
      ins_valid(0) => extsi2_outs_valid,
      ins_valid(1) => fork54_outs_0_valid,
      ins_ready(0) => extsi2_outs_ready,
      ins_ready(1) => fork54_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_5,
      index_valid => fork2_outs_5_valid,
      index_ready => fork2_outs_5_ready,
      ins(0) => extsi3_outs,
      ins(1) => fork54_outs_1,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => fork54_outs_1_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => fork54_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  mux10 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_6,
      index_valid => fork2_outs_6_valid,
      index_ready => fork2_outs_6_ready,
      ins(0) => extsi4_outs,
      ins(1) => fork50_outs_1,
      ins_valid(0) => extsi4_outs_valid,
      ins_valid(1) => fork50_outs_1_valid,
      ins_ready(0) => extsi4_outs_ready,
      ins_ready(1) => fork50_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux10_outs,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_7,
      index_valid => fork2_outs_7_valid,
      index_ready => fork2_outs_7_ready,
      ins(0) => extsi5_outs,
      ins(1) => fork52_outs_1,
      ins_valid(0) => extsi5_outs_valid,
      ins_valid(1) => fork52_outs_1_valid,
      ins_ready(0) => extsi5_outs_ready,
      ins_ready(1) => fork52_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  mux12 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_8,
      index_valid => fork2_outs_8_valid,
      index_ready => fork2_outs_8_ready,
      ins(0) => extsi6_outs,
      ins(1) => fork51_outs_0,
      ins_valid(0) => extsi6_outs_valid,
      ins_valid(1) => fork51_outs_0_valid,
      ins_ready(0) => extsi6_outs_ready,
      ins_ready(1) => fork51_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux12_outs,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  mux13 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_9,
      index_valid => fork2_outs_9_valid,
      index_ready => fork2_outs_9_ready,
      ins(0) => extsi7_outs,
      ins(1) => fork51_outs_1,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => fork51_outs_1_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => fork51_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux13_outs,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  mux14 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_10,
      index_valid => fork2_outs_10_valid,
      index_ready => fork2_outs_10_ready,
      ins_valid(0) => fork0_outs_10_valid,
      ins_valid(1) => fork53_outs_0_valid,
      ins_ready(0) => fork0_outs_10_ready,
      ins_ready(1) => fork53_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux14_outs_valid,
      outs_ready => mux14_outs_ready
    );

  mux15 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_11,
      index_valid => fork2_outs_11_valid,
      index_ready => fork2_outs_11_ready,
      ins_valid(0) => fork0_outs_9_valid,
      ins_valid(1) => fork48_outs_0_valid,
      ins_ready(0) => fork0_outs_9_ready,
      ins_ready(1) => fork48_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  mux16 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_12,
      index_valid => fork2_outs_12_valid,
      index_ready => fork2_outs_12_ready,
      ins_valid(0) => fork0_outs_8_valid,
      ins_valid(1) => fork49_outs_0_valid,
      ins_ready(0) => fork0_outs_8_ready,
      ins_ready(1) => fork49_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux16_outs_valid,
      outs_ready => mux16_outs_ready
    );

  mux17 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_13,
      index_valid => fork2_outs_13_valid,
      index_ready => fork2_outs_13_ready,
      ins_valid(0) => fork0_outs_7_valid,
      ins_valid(1) => fork47_outs_0_valid,
      ins_ready(0) => fork0_outs_7_ready,
      ins_ready(1) => fork47_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux17_outs_valid,
      outs_ready => mux17_outs_ready
    );

  mux18 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_14,
      index_valid => fork2_outs_14_valid,
      index_ready => fork2_outs_14_ready,
      ins_valid(0) => fork0_outs_6_valid,
      ins_valid(1) => fork48_outs_1_valid,
      ins_ready(0) => fork0_outs_6_ready,
      ins_ready(1) => fork48_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux18_outs_valid,
      outs_ready => mux18_outs_ready
    );

  mux19 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_15,
      index_valid => fork2_outs_15_valid,
      index_ready => fork2_outs_15_ready,
      ins_valid(0) => fork0_outs_5_valid,
      ins_valid(1) => fork47_outs_1_valid,
      ins_ready(0) => fork0_outs_5_ready,
      ins_ready(1) => fork47_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux19_outs_valid,
      outs_ready => mux19_outs_ready
    );

  mux20 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_16,
      index_valid => fork2_outs_16_valid,
      index_ready => fork2_outs_16_ready,
      ins_valid(0) => fork0_outs_4_valid,
      ins_valid(1) => fork53_outs_1_valid,
      ins_ready(0) => fork0_outs_4_ready,
      ins_ready(1) => fork53_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux20_outs_valid,
      outs_ready => mux20_outs_ready
    );

  mux21 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_17,
      index_valid => fork2_outs_17_valid,
      index_ready => fork2_outs_17_ready,
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => fork49_outs_1_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => fork49_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux21_outs_valid,
      outs_ready => mux21_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork56_outs_10,
      ins_valid => fork56_outs_10_valid,
      ins_ready => fork56_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(18, 1)
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
      outs(7) => fork2_outs_7,
      outs(8) => fork2_outs_8,
      outs(9) => fork2_outs_9,
      outs(10) => fork2_outs_10,
      outs(11) => fork2_outs_11,
      outs(12) => fork2_outs_12,
      outs(13) => fork2_outs_13,
      outs(14) => fork2_outs_14,
      outs(15) => fork2_outs_15,
      outs(16) => fork2_outs_16,
      outs(17) => fork2_outs_17,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_valid(5) => fork2_outs_5_valid,
      outs_valid(6) => fork2_outs_6_valid,
      outs_valid(7) => fork2_outs_7_valid,
      outs_valid(8) => fork2_outs_8_valid,
      outs_valid(9) => fork2_outs_9_valid,
      outs_valid(10) => fork2_outs_10_valid,
      outs_valid(11) => fork2_outs_11_valid,
      outs_valid(12) => fork2_outs_12_valid,
      outs_valid(13) => fork2_outs_13_valid,
      outs_valid(14) => fork2_outs_14_valid,
      outs_valid(15) => fork2_outs_15_valid,
      outs_valid(16) => fork2_outs_16_valid,
      outs_valid(17) => fork2_outs_17_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready,
      outs_ready(5) => fork2_outs_5_ready,
      outs_ready(6) => fork2_outs_6_ready,
      outs_ready(7) => fork2_outs_7_ready,
      outs_ready(8) => fork2_outs_8_ready,
      outs_ready(9) => fork2_outs_9_ready,
      outs_ready(10) => fork2_outs_10_ready,
      outs_ready(11) => fork2_outs_11_ready,
      outs_ready(12) => fork2_outs_12_ready,
      outs_ready(13) => fork2_outs_13_ready,
      outs_ready(14) => fork2_outs_14_ready,
      outs_ready(15) => fork2_outs_15_ready,
      outs_ready(16) => fork2_outs_16_ready,
      outs_ready(17) => fork2_outs_17_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi18_outs,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => extsi18_outs_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => extsi18_outs_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi19 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_13_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => fork0_outs_13_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
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

  constant20 : entity work.handshake_constant_2(arch) generic map(1)
    port map(
      ctrl_valid => fork4_outs_0_valid,
      ctrl_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant7 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => extsi19_outs,
      lhs_valid => extsi19_outs_valid,
      lhs_ready => extsi19_outs_ready,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  extsi17 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant20_outs,
      ins_valid => constant20_outs_valid,
      ins_ready => constant20_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  cond_br68 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork45_outs_11,
      condition_valid => fork45_outs_11_valid,
      condition_ready => fork45_outs_11_ready,
      data_valid => fork43_outs_1_valid,
      data_ready => fork43_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br68_trueOut_valid,
      trueOut_ready => cond_br68_trueOut_ready,
      falseOut_valid => cond_br68_falseOut_valid,
      falseOut_ready => cond_br68_falseOut_ready
    );

  fork5 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br68_trueOut_valid,
      ins_ready => cond_br68_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  cond_br69 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork45_outs_10,
      condition_valid => fork45_outs_10_valid,
      condition_ready => fork45_outs_10_ready,
      data_valid => init42_outs_valid,
      data_ready => init42_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br69_trueOut_valid,
      trueOut_ready => cond_br69_trueOut_ready,
      falseOut_valid => cond_br69_falseOut_valid,
      falseOut_ready => cond_br69_falseOut_ready
    );

  fork6 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br69_trueOut_valid,
      ins_ready => cond_br69_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  cond_br70 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork45_outs_9,
      condition_valid => fork45_outs_9_valid,
      condition_ready => fork45_outs_9_ready,
      data => buffer42_outs,
      data_valid => buffer42_outs_valid,
      data_ready => buffer42_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br70_trueOut,
      trueOut_valid => cond_br70_trueOut_valid,
      trueOut_ready => cond_br70_trueOut_ready,
      falseOut => cond_br70_falseOut,
      falseOut_valid => cond_br70_falseOut_valid,
      falseOut_ready => cond_br70_falseOut_ready
    );

  buffer42 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork37_outs_1,
      ins_valid => fork37_outs_1_valid,
      ins_ready => fork37_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br70_trueOut,
      ins_valid => cond_br70_trueOut_valid,
      ins_ready => cond_br70_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  cond_br71 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer43_outs,
      condition_valid => buffer43_outs_valid,
      condition_ready => buffer43_outs_ready,
      data_valid => fork42_outs_1_valid,
      data_ready => fork42_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br71_trueOut_valid,
      trueOut_ready => cond_br71_trueOut_ready,
      falseOut_valid => cond_br71_falseOut_valid,
      falseOut_ready => cond_br71_falseOut_ready
    );

  buffer43 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork45_outs_8,
      ins_valid => fork45_outs_8_valid,
      ins_ready => fork45_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  fork8 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br71_trueOut_valid,
      ins_ready => cond_br71_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  cond_br72 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork45_outs_7,
      condition_valid => fork45_outs_7_valid,
      condition_ready => fork45_outs_7_ready,
      data => init38_outs,
      data_valid => init38_outs_valid,
      data_ready => init38_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br72_trueOut,
      trueOut_valid => cond_br72_trueOut_valid,
      trueOut_ready => cond_br72_trueOut_ready,
      falseOut => cond_br72_falseOut,
      falseOut_valid => cond_br72_falseOut_valid,
      falseOut_ready => cond_br72_falseOut_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br72_trueOut,
      ins_valid => cond_br72_trueOut_valid,
      ins_ready => cond_br72_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  cond_br73 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer45_outs,
      condition_valid => buffer45_outs_valid,
      condition_ready => buffer45_outs_ready,
      data_valid => fork40_outs_1_valid,
      data_ready => fork40_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br73_trueOut_valid,
      trueOut_ready => cond_br73_trueOut_ready,
      falseOut_valid => cond_br73_falseOut_valid,
      falseOut_ready => cond_br73_falseOut_ready
    );

  buffer45 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork45_outs_6,
      ins_valid => fork45_outs_6_valid,
      ins_ready => fork45_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  fork10 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br73_trueOut_valid,
      ins_ready => cond_br73_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  cond_br74 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer46_outs,
      condition_valid => buffer46_outs_valid,
      condition_ready => buffer46_outs_ready,
      data_valid => fork41_outs_1_valid,
      data_ready => fork41_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br74_trueOut_valid,
      trueOut_ready => cond_br74_trueOut_ready,
      falseOut_valid => cond_br74_falseOut_valid,
      falseOut_ready => cond_br74_falseOut_ready
    );

  buffer46 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork45_outs_5,
      ins_valid => fork45_outs_5_valid,
      ins_ready => fork45_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  fork11 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br74_trueOut_valid,
      ins_ready => cond_br74_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  cond_br75 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork45_outs_4,
      condition_valid => fork45_outs_4_valid,
      condition_ready => fork45_outs_4_ready,
      data => fork39_outs_1,
      data_valid => fork39_outs_1_valid,
      data_ready => fork39_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br75_trueOut,
      trueOut_valid => cond_br75_trueOut_valid,
      trueOut_ready => cond_br75_trueOut_ready,
      falseOut => cond_br75_falseOut,
      falseOut_valid => cond_br75_falseOut_valid,
      falseOut_ready => cond_br75_falseOut_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br75_trueOut,
      ins_valid => cond_br75_trueOut_valid,
      ins_ready => cond_br75_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  cond_br76 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork45_outs_3,
      condition_valid => fork45_outs_3_valid,
      condition_ready => fork45_outs_3_ready,
      data => fork38_outs_1,
      data_valid => fork38_outs_1_valid,
      data_ready => fork38_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br76_trueOut,
      trueOut_valid => cond_br76_trueOut_valid,
      trueOut_ready => cond_br76_trueOut_ready,
      falseOut => cond_br76_falseOut,
      falseOut_valid => cond_br76_falseOut_valid,
      falseOut_ready => cond_br76_falseOut_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br76_trueOut,
      ins_valid => cond_br76_trueOut_valid,
      ins_ready => cond_br76_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  mux22 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_0,
      index_valid => fork14_outs_0_valid,
      index_ready => fork14_outs_0_ready,
      ins(0) => mux4_outs,
      ins(1) => fork7_outs_1,
      ins_valid(0) => mux4_outs_valid,
      ins_valid(1) => fork7_outs_1_valid,
      ins_ready(0) => mux4_outs_ready,
      ins_ready(1) => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux22_outs,
      outs_valid => mux22_outs_valid,
      outs_ready => mux22_outs_ready
    );

  buffer2 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  mux23 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_1,
      index_valid => fork14_outs_1_valid,
      index_ready => fork14_outs_1_ready,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => fork6_outs_1_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux23_outs_valid,
      outs_ready => mux23_outs_ready
    );

  buffer3 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  mux24 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_2,
      index_valid => fork14_outs_2_valid,
      index_ready => fork14_outs_2_ready,
      ins_valid(0) => buffer3_outs_valid,
      ins_valid(1) => fork6_outs_0_valid,
      ins_ready(0) => buffer3_outs_ready,
      ins_ready(1) => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux24_outs_valid,
      outs_ready => mux24_outs_ready
    );

  mux25 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_3,
      index_valid => fork14_outs_3_valid,
      index_ready => fork14_outs_3_ready,
      ins(0) => mux7_outs,
      ins(1) => fork12_outs_1,
      ins_valid(0) => mux7_outs_valid,
      ins_valid(1) => fork12_outs_1_valid,
      ins_ready(0) => mux7_outs_ready,
      ins_ready(1) => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux25_outs,
      outs_valid => mux25_outs_valid,
      outs_ready => mux25_outs_ready
    );

  mux26 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_4,
      index_valid => fork14_outs_4_valid,
      index_ready => fork14_outs_4_ready,
      ins(0) => mux8_outs,
      ins(1) => fork13_outs_1,
      ins_valid(0) => mux8_outs_valid,
      ins_valid(1) => fork13_outs_1_valid,
      ins_ready(0) => mux8_outs_ready,
      ins_ready(1) => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux26_outs,
      outs_valid => mux26_outs_valid,
      outs_ready => mux26_outs_ready
    );

  mux27 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_5,
      index_valid => fork14_outs_5_valid,
      index_ready => fork14_outs_5_ready,
      ins(0) => mux9_outs,
      ins(1) => fork13_outs_0,
      ins_valid(0) => mux9_outs_valid,
      ins_valid(1) => fork13_outs_0_valid,
      ins_ready(0) => mux9_outs_ready,
      ins_ready(1) => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux27_outs,
      outs_valid => mux27_outs_valid,
      outs_ready => mux27_outs_ready
    );

  mux28 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_6,
      index_valid => fork14_outs_6_valid,
      index_ready => fork14_outs_6_ready,
      ins(0) => mux10_outs,
      ins(1) => fork12_outs_0,
      ins_valid(0) => mux10_outs_valid,
      ins_valid(1) => fork12_outs_0_valid,
      ins_ready(0) => mux10_outs_ready,
      ins_ready(1) => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux28_outs,
      outs_valid => mux28_outs_valid,
      outs_ready => mux28_outs_ready
    );

  mux29 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_7,
      index_valid => fork14_outs_7_valid,
      index_ready => fork14_outs_7_ready,
      ins(0) => mux11_outs,
      ins(1) => fork7_outs_0,
      ins_valid(0) => mux11_outs_valid,
      ins_valid(1) => fork7_outs_0_valid,
      ins_ready(0) => mux11_outs_ready,
      ins_ready(1) => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux29_outs,
      outs_valid => mux29_outs_valid,
      outs_ready => mux29_outs_ready
    );

  mux30 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_8,
      index_valid => fork14_outs_8_valid,
      index_ready => fork14_outs_8_ready,
      ins(0) => mux12_outs,
      ins(1) => fork9_outs_1,
      ins_valid(0) => mux12_outs_valid,
      ins_valid(1) => fork9_outs_1_valid,
      ins_ready(0) => mux12_outs_ready,
      ins_ready(1) => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux30_outs,
      outs_valid => mux30_outs_valid,
      outs_ready => mux30_outs_ready
    );

  mux31 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork14_outs_9,
      index_valid => fork14_outs_9_valid,
      index_ready => fork14_outs_9_ready,
      ins(0) => mux13_outs,
      ins(1) => fork9_outs_0,
      ins_valid(0) => mux13_outs_valid,
      ins_valid(1) => fork9_outs_0_valid,
      ins_ready(0) => mux13_outs_ready,
      ins_ready(1) => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux31_outs,
      outs_valid => mux31_outs_valid,
      outs_ready => mux31_outs_ready
    );

  buffer4 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux14_outs_valid,
      ins_ready => mux14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  mux32 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer69_outs,
      index_valid => buffer69_outs_valid,
      index_ready => buffer69_outs_ready,
      ins_valid(0) => buffer4_outs_valid,
      ins_valid(1) => fork11_outs_1_valid,
      ins_ready(0) => buffer4_outs_ready,
      ins_ready(1) => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux32_outs_valid,
      outs_ready => mux32_outs_ready
    );

  buffer69 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_10,
      ins_valid => fork14_outs_10_valid,
      ins_ready => fork14_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => buffer69_outs,
      outs_valid => buffer69_outs_valid,
      outs_ready => buffer69_outs_ready
    );

  buffer5 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  mux33 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_11,
      index_valid => fork14_outs_11_valid,
      index_ready => fork14_outs_11_ready,
      ins_valid(0) => buffer5_outs_valid,
      ins_valid(1) => fork5_outs_1_valid,
      ins_ready(0) => buffer5_outs_ready,
      ins_ready(1) => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux33_outs_valid,
      outs_ready => mux33_outs_ready
    );

  buffer6 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux16_outs_valid,
      ins_ready => mux16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer7 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  mux34 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer71_outs,
      index_valid => buffer71_outs_valid,
      index_ready => buffer71_outs_ready,
      ins_valid(0) => buffer7_outs_valid,
      ins_valid(1) => fork10_outs_1_valid,
      ins_ready(0) => buffer7_outs_ready,
      ins_ready(1) => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux34_outs_valid,
      outs_ready => mux34_outs_ready
    );

  buffer71 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork14_outs_12,
      ins_valid => fork14_outs_12_valid,
      ins_ready => fork14_outs_12_ready,
      clk => clk,
      rst => rst,
      outs => buffer71_outs,
      outs_valid => buffer71_outs_valid,
      outs_ready => buffer71_outs_ready
    );

  buffer8 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux17_outs_valid,
      ins_ready => mux17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  mux35 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_13,
      index_valid => fork14_outs_13_valid,
      index_ready => fork14_outs_13_ready,
      ins_valid(0) => buffer8_outs_valid,
      ins_valid(1) => fork8_outs_1_valid,
      ins_ready(0) => buffer8_outs_ready,
      ins_ready(1) => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux35_outs_valid,
      outs_ready => mux35_outs_ready
    );

  buffer9 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux18_outs_valid,
      ins_ready => mux18_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  mux36 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_14,
      index_valid => fork14_outs_14_valid,
      index_ready => fork14_outs_14_ready,
      ins_valid(0) => buffer9_outs_valid,
      ins_valid(1) => fork5_outs_0_valid,
      ins_ready(0) => buffer9_outs_ready,
      ins_ready(1) => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux36_outs_valid,
      outs_ready => mux36_outs_ready
    );

  buffer10 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux19_outs_valid,
      ins_ready => mux19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  mux37 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_15,
      index_valid => fork14_outs_15_valid,
      index_ready => fork14_outs_15_ready,
      ins_valid(0) => buffer10_outs_valid,
      ins_valid(1) => fork8_outs_0_valid,
      ins_ready(0) => buffer10_outs_ready,
      ins_ready(1) => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux37_outs_valid,
      outs_ready => mux37_outs_ready
    );

  buffer11 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux20_outs_valid,
      ins_ready => mux20_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  mux38 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer75_outs,
      index_valid => buffer75_outs_valid,
      index_ready => buffer75_outs_ready,
      ins_valid(0) => buffer11_outs_valid,
      ins_valid(1) => fork11_outs_0_valid,
      ins_ready(0) => buffer11_outs_ready,
      ins_ready(1) => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux38_outs_valid,
      outs_ready => mux38_outs_ready
    );

  buffer75 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_16,
      ins_valid => fork14_outs_16_valid,
      ins_ready => fork14_outs_16_ready,
      clk => clk,
      rst => rst,
      outs => buffer75_outs,
      outs_valid => buffer75_outs_valid,
      outs_ready => buffer75_outs_ready
    );

  buffer12 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux21_outs_valid,
      ins_ready => mux21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  mux39 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer76_outs,
      index_valid => buffer76_outs_valid,
      index_ready => buffer76_outs_ready,
      ins_valid(0) => buffer12_outs_valid,
      ins_valid(1) => fork10_outs_0_valid,
      ins_ready(0) => buffer12_outs_ready,
      ins_ready(1) => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux39_outs_valid,
      outs_ready => mux39_outs_ready
    );

  buffer76 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork14_outs_17,
      ins_valid => fork14_outs_17_valid,
      ins_ready => fork14_outs_17_ready,
      clk => clk,
      rst => rst,
      outs => buffer76_outs,
      outs_valid => buffer76_outs_valid,
      outs_ready => buffer76_outs_ready
    );

  init18 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork45_outs_2,
      ins_valid => fork45_outs_2_valid,
      ins_ready => fork45_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => init18_outs,
      outs_valid => init18_outs_valid,
      outs_ready => init18_outs_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(18, 1)
    port map(
      ins => init18_outs,
      ins_valid => init18_outs_valid,
      ins_ready => init18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs(3) => fork14_outs_3,
      outs(4) => fork14_outs_4,
      outs(5) => fork14_outs_5,
      outs(6) => fork14_outs_6,
      outs(7) => fork14_outs_7,
      outs(8) => fork14_outs_8,
      outs(9) => fork14_outs_9,
      outs(10) => fork14_outs_10,
      outs(11) => fork14_outs_11,
      outs(12) => fork14_outs_12,
      outs(13) => fork14_outs_13,
      outs(14) => fork14_outs_14,
      outs(15) => fork14_outs_15,
      outs(16) => fork14_outs_16,
      outs(17) => fork14_outs_17,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_valid(3) => fork14_outs_3_valid,
      outs_valid(4) => fork14_outs_4_valid,
      outs_valid(5) => fork14_outs_5_valid,
      outs_valid(6) => fork14_outs_6_valid,
      outs_valid(7) => fork14_outs_7_valid,
      outs_valid(8) => fork14_outs_8_valid,
      outs_valid(9) => fork14_outs_9_valid,
      outs_valid(10) => fork14_outs_10_valid,
      outs_valid(11) => fork14_outs_11_valid,
      outs_valid(12) => fork14_outs_12_valid,
      outs_valid(13) => fork14_outs_13_valid,
      outs_valid(14) => fork14_outs_14_valid,
      outs_valid(15) => fork14_outs_15_valid,
      outs_valid(16) => fork14_outs_16_valid,
      outs_valid(17) => fork14_outs_17_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready,
      outs_ready(3) => fork14_outs_3_ready,
      outs_ready(4) => fork14_outs_4_ready,
      outs_ready(5) => fork14_outs_5_ready,
      outs_ready(6) => fork14_outs_6_ready,
      outs_ready(7) => fork14_outs_7_ready,
      outs_ready(8) => fork14_outs_8_ready,
      outs_ready(9) => fork14_outs_9_ready,
      outs_ready(10) => fork14_outs_10_ready,
      outs_ready(11) => fork14_outs_11_ready,
      outs_ready(12) => fork14_outs_12_ready,
      outs_ready(13) => fork14_outs_13_ready,
      outs_ready(14) => fork14_outs_14_ready,
      outs_ready(15) => fork14_outs_15_ready,
      outs_ready(16) => fork14_outs_16_ready,
      outs_ready(17) => fork14_outs_17_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork19_outs_1,
      index_valid => fork19_outs_1_valid,
      index_ready => fork19_outs_1_ready,
      ins(0) => extsi17_outs,
      ins(1) => cond_br2_trueOut,
      ins_valid(0) => extsi17_outs_valid,
      ins_valid(1) => cond_br2_trueOut_valid,
      ins_ready(0) => extsi17_outs_ready,
      ins_ready(1) => cond_br2_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer37 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(4, 6)
    port map(
      ins => buffer37_outs,
      ins_valid => buffer37_outs_valid,
      ins_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs(2) => fork15_outs_2,
      outs(3) => fork15_outs_3,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_valid(2) => fork15_outs_2_valid,
      outs_valid(3) => fork15_outs_3_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready,
      outs_ready(2) => fork15_outs_2_ready,
      outs_ready(3) => fork15_outs_3_ready
    );

  extsi20 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork15_outs_3,
      ins_valid => fork15_outs_3_valid,
      ins_ready => fork15_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  trunci0 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork15_outs_2,
      ins_valid => fork15_outs_2_valid,
      ins_ready => fork15_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork19_outs_0,
      index_valid => fork19_outs_0_valid,
      index_ready => fork19_outs_0_ready,
      ins(0) => fork3_outs_0,
      ins(1) => cond_br3_trueOut,
      ins_valid(0) => fork3_outs_0_valid,
      ins_valid(1) => cond_br3_trueOut_valid,
      ins_ready(0) => fork3_outs_0_ready,
      ins_ready(1) => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer38 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  buffer39 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer38_outs,
      ins_valid => buffer38_outs_valid,
      ins_ready => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer39_outs,
      ins_valid => buffer39_outs_valid,
      ins_ready => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  extsi21 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork16_outs_1,
      ins_valid => fork16_outs_1_valid,
      ins_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi21_outs,
      ins_valid => extsi21_outs_valid,
      ins_ready => extsi21_outs_ready,
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

  buffer14 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork19_outs_2,
      index_valid => fork19_outs_2_valid,
      index_ready => fork19_outs_2_ready,
      ins(0) => buffer14_outs,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => buffer14_outs_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => buffer14_outs_ready,
      ins_ready(1) => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer40 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  buffer41 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer40_outs,
      ins_valid => buffer40_outs_valid,
      ins_ready => buffer40_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => buffer41_outs,
      ins_valid => buffer41_outs_valid,
      ins_ready => buffer41_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork4_outs_1_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => fork4_outs_1_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs(2) => fork19_outs_2,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_valid(2) => fork19_outs_2_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready,
      outs_ready(2) => fork19_outs_2_ready
    );

  buffer44 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  fork20 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer44_outs_valid,
      ins_ready => buffer44_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  constant21 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork20_outs_0_valid,
      ctrl_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  extsi10 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant21_outs,
      ins_valid => constant21_outs_valid,
      ins_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant22 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  extsi22 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant23 : entity work.handshake_constant_4(arch) generic map(6)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant23_outs,
      outs_valid => constant23_outs_valid,
      outs_ready => constant23_outs_ready
    );

  extsi23 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant23_outs,
      ins_valid => constant23_outs_valid,
      ins_ready => constant23_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant24 : entity work.handshake_constant_5(arch) generic map(4)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant24_outs,
      outs_valid => constant24_outs_valid,
      outs_ready => constant24_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(4, 32)
    port map(
      ins => constant24_outs,
      ins_valid => constant24_outs_valid,
      ins_ready => constant24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi13_outs,
      ins_valid => extsi13_outs_valid,
      ins_ready => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs(2) => fork21_outs_2,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_valid(2) => fork21_outs_2_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready,
      outs_ready(2) => fork21_outs_2_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant25 : entity work.handshake_constant_6(arch) generic map(3)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant25_outs,
      outs_valid => constant25_outs_valid,
      outs_ready => constant25_outs_ready
    );

  extsi14 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant25_outs,
      ins_valid => constant25_outs_valid,
      ins_ready => constant25_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  fork22 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi14_outs,
      ins_valid => extsi14_outs_valid,
      ins_ready => extsi14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs(2) => fork22_outs_2,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_valid(2) => fork22_outs_2_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready,
      outs_ready(2) => fork22_outs_2_ready
    );

  load0 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
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

  fork23 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  load1 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
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
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
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

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork18_outs_2,
      lhs_valid => fork18_outs_2_valid,
      lhs_ready => fork18_outs_2_ready,
      rhs => fork22_outs_0,
      rhs_valid => fork22_outs_0_valid,
      rhs_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork18_outs_1,
      lhs_valid => fork18_outs_1_valid,
      lhs_ready => fork18_outs_1_ready,
      rhs => fork21_outs_0,
      rhs_valid => fork21_outs_0_valid,
      rhs_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer47 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  buffer48 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer47_outs,
      lhs_valid => buffer47_outs_valid,
      lhs_ready => buffer47_outs_ready,
      rhs => buffer48_outs,
      rhs_valid => buffer48_outs_valid,
      rhs_ready => buffer48_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer49 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer49_outs,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => buffer49_outs,
      rhs_valid => buffer49_outs_valid,
      rhs_ready => buffer49_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  fork24 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  buffer17 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux24_outs_valid,
      ins_ready => mux24_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork24_outs_1,
      ins_valid(0) => fork24_outs_1_valid,
      ins_valid(1) => buffer17_outs_valid,
      ins_ready(0) => fork24_outs_1_ready,
      ins_ready(1) => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  buffer50 : entity work.oehb(arch) generic map(32)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  fork25 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => buffer50_outs,
      ins_valid => buffer50_outs_valid,
      ins_ready => buffer50_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs(2) => fork25_outs_2,
      outs(3) => fork25_outs_3,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_valid(2) => fork25_outs_2_valid,
      outs_valid(3) => fork25_outs_3_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready,
      outs_ready(2) => fork25_outs_2_ready,
      outs_ready(3) => fork25_outs_3_ready
    );

  buffer15 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux22_outs,
      ins_valid => mux22_outs_valid,
      ins_ready => mux22_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork25_outs_3,
      lhs_valid => fork25_outs_3_valid,
      lhs_ready => fork25_outs_3_ready,
      rhs => buffer15_outs,
      rhs_valid => buffer15_outs_valid,
      rhs_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  fork26 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork26_outs_0,
      outs(1) => fork26_outs_1,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready
    );

  buffer20 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux27_outs,
      ins_valid => mux27_outs_valid,
      ins_ready => mux27_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cmpi3 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork25_outs_2,
      lhs_valid => fork25_outs_2_valid,
      lhs_ready => fork25_outs_2_ready,
      rhs => buffer20_outs,
      rhs_valid => buffer20_outs_valid,
      rhs_ready => buffer20_outs_ready,
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

  buffer18 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux25_outs,
      ins_valid => mux25_outs_valid,
      ins_ready => mux25_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  cmpi4 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork25_outs_1,
      lhs_valid => fork25_outs_1_valid,
      lhs_ready => fork25_outs_1_ready,
      rhs => buffer18_outs,
      rhs_valid => buffer18_outs_valid,
      rhs_ready => buffer18_outs_ready,
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
      ins => mux30_outs,
      ins_valid => mux30_outs_valid,
      ins_ready => mux30_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  cmpi5 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork25_outs_0,
      lhs_valid => fork25_outs_0_valid,
      lhs_ready => fork25_outs_0_ready,
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

  buffer36 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux39_outs_valid,
      ins_ready => mux39_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  cond_br42 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer95_outs,
      condition_valid => buffer95_outs_valid,
      condition_ready => buffer95_outs_ready,
      data_valid => buffer36_outs_valid,
      data_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br42_trueOut_valid,
      trueOut_ready => cond_br42_trueOut_ready,
      falseOut_valid => cond_br42_falseOut_valid,
      falseOut_ready => cond_br42_falseOut_ready
    );

  buffer95 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork26_outs_1,
      ins_valid => fork26_outs_1_valid,
      ins_ready => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer95_outs,
      outs_valid => buffer95_outs_valid,
      outs_ready => buffer95_outs_ready
    );

  sink0 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br42_trueOut_valid,
      ins_ready => cond_br42_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer34 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux38_outs_valid,
      ins_ready => mux38_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  buffer35 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer34_outs_valid,
      ins_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  cond_br43 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer96_outs,
      condition_valid => buffer96_outs_valid,
      condition_ready => buffer96_outs_ready,
      data_valid => buffer35_outs_valid,
      data_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br43_trueOut_valid,
      trueOut_ready => cond_br43_trueOut_ready,
      falseOut_valid => cond_br43_falseOut_valid,
      falseOut_ready => cond_br43_falseOut_ready
    );

  buffer96 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork27_outs_1,
      ins_valid => fork27_outs_1_valid,
      ins_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer96_outs,
      outs_valid => buffer96_outs_valid,
      outs_ready => buffer96_outs_ready
    );

  sink1 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br43_trueOut_valid,
      ins_ready => cond_br43_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer29 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux35_outs_valid,
      ins_ready => mux35_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  buffer30 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  cond_br44 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer97_outs,
      condition_valid => buffer97_outs_valid,
      condition_ready => buffer97_outs_ready,
      data_valid => buffer30_outs_valid,
      data_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br44_trueOut_valid,
      trueOut_ready => cond_br44_trueOut_ready,
      falseOut_valid => cond_br44_falseOut_valid,
      falseOut_ready => cond_br44_falseOut_ready
    );

  buffer97 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork28_outs_1,
      ins_valid => fork28_outs_1_valid,
      ins_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer97_outs,
      outs_valid => buffer97_outs_valid,
      outs_ready => buffer97_outs_ready
    );

  sink2 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br44_trueOut_valid,
      ins_ready => cond_br44_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer26 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux33_outs_valid,
      ins_ready => mux33_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  buffer27 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  cond_br45 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork29_outs_1,
      condition_valid => fork29_outs_1_valid,
      condition_ready => fork29_outs_1_ready,
      data_valid => buffer27_outs_valid,
      data_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br45_trueOut_valid,
      trueOut_ready => cond_br45_trueOut_ready,
      falseOut_valid => cond_br45_falseOut_valid,
      falseOut_ready => cond_br45_falseOut_ready
    );

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br45_trueOut_valid,
      ins_ready => cond_br45_trueOut_ready,
      clk => clk,
      rst => rst
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  buffer51 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br42_falseOut_valid,
      ins_ready => cond_br42_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  mux40 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork26_outs_0,
      index_valid => fork26_outs_0_valid,
      index_ready => fork26_outs_0_ready,
      ins_valid(0) => buffer51_outs_valid,
      ins_valid(1) => source7_outs_valid,
      ins_ready(0) => buffer51_outs_ready,
      ins_ready(1) => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux40_outs_valid,
      outs_ready => mux40_outs_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  mux41 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork27_outs_0,
      index_valid => fork27_outs_0_valid,
      index_ready => fork27_outs_0_ready,
      ins_valid(0) => cond_br43_falseOut_valid,
      ins_valid(1) => source8_outs_valid,
      ins_ready(0) => cond_br43_falseOut_ready,
      ins_ready(1) => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux41_outs_valid,
      outs_ready => mux41_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  mux42 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork28_outs_0,
      index_valid => fork28_outs_0_valid,
      index_ready => fork28_outs_0_ready,
      ins_valid(0) => cond_br44_falseOut_valid,
      ins_valid(1) => source9_outs_valid,
      ins_ready(0) => cond_br44_falseOut_ready,
      ins_ready(1) => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux42_outs_valid,
      outs_ready => mux42_outs_ready
    );

  source10 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  mux43 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork29_outs_0,
      index_valid => fork29_outs_0_valid,
      index_ready => fork29_outs_0_ready,
      ins_valid(0) => cond_br45_falseOut_valid,
      ins_valid(1) => source10_outs_valid,
      ins_ready(0) => cond_br45_falseOut_ready,
      ins_ready(1) => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux43_outs_valid,
      outs_ready => mux43_outs_ready
    );

  buffer52 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux40_outs_valid,
      ins_ready => mux40_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  buffer53 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux41_outs_valid,
      ins_ready => mux41_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  buffer54 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux42_outs_valid,
      ins_ready => mux42_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  buffer55 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux43_outs_valid,
      ins_ready => mux43_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer55_outs_valid,
      outs_ready => buffer55_outs_ready
    );

  join0 : entity work.join_handshake(arch) generic map(4)
    port map(
      ins_valid(0) => buffer52_outs_valid,
      ins_valid(1) => buffer53_outs_valid,
      ins_valid(2) => buffer54_outs_valid,
      ins_valid(3) => buffer55_outs_valid,
      ins_ready(0) => buffer52_outs_ready,
      ins_ready(1) => buffer53_outs_ready,
      ins_ready(2) => buffer54_outs_ready,
      ins_ready(3) => buffer55_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join0_outs_valid,
      outs_ready => join0_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork24_outs_0,
      ins_valid(0) => fork24_outs_0_valid,
      ins_valid(1) => join0_outs_valid,
      ins_ready(0) => fork24_outs_0_ready,
      ins_ready(1) => join0_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => gate1_outs,
      ins_valid => gate1_outs_valid,
      ins_ready => gate1_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => trunci3_outs,
      addrIn_valid => trunci3_outs_valid,
      addrIn_ready => trunci3_outs_ready,
      dataFromMem => mem_controller6_ldData_0,
      dataFromMem_valid => mem_controller6_ldData_0_valid,
      dataFromMem_ready => mem_controller6_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load1_dataOut,
      lhs_valid => load1_dataOut_valid,
      lhs_ready => load1_dataOut_ready,
      rhs => load3_dataOut,
      rhs_valid => load3_dataOut_valid,
      rhs_ready => load3_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  shli2 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork17_outs_0,
      lhs_valid => fork17_outs_0_valid,
      lhs_ready => fork17_outs_0_ready,
      rhs => fork22_outs_1,
      rhs_valid => fork22_outs_1_valid,
      rhs_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork17_outs_1,
      lhs_valid => fork17_outs_1_valid,
      lhs_ready => fork17_outs_1_ready,
      rhs => fork21_outs_1,
      rhs_valid => fork21_outs_1_valid,
      rhs_ready => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer56 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer56_outs,
      outs_valid => buffer56_outs_valid,
      outs_ready => buffer56_outs_ready
    );

  buffer57 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer57_outs,
      outs_valid => buffer57_outs_valid,
      outs_ready => buffer57_outs_ready
    );

  addi8 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer56_outs,
      lhs_valid => buffer56_outs_valid,
      lhs_ready => buffer56_outs_ready,
      rhs => buffer57_outs,
      rhs_valid => buffer57_outs_valid,
      rhs_ready => buffer57_outs_ready,
      clk => clk,
      rst => rst,
      result => addi8_result,
      result_valid => addi8_result_valid,
      result_ready => addi8_result_ready
    );

  buffer58 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi8_result,
      ins_valid => addi8_result_valid,
      ins_ready => addi8_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer58_outs,
      outs_valid => buffer58_outs_valid,
      outs_ready => buffer58_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork23_outs_0,
      lhs_valid => fork23_outs_0_valid,
      lhs_ready => fork23_outs_0_ready,
      rhs => buffer58_outs,
      rhs_valid => buffer58_outs_valid,
      rhs_ready => buffer58_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  fork30 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork30_outs_0,
      outs(1) => fork30_outs_1,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready
    );

  buffer16 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux23_outs_valid,
      ins_ready => mux23_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  gate2 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork30_outs_1,
      ins_valid(0) => fork30_outs_1_valid,
      ins_valid(1) => buffer16_outs_valid,
      ins_ready(0) => fork30_outs_1_ready,
      ins_ready(1) => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate2_outs,
      outs_valid => gate2_outs_valid,
      outs_ready => gate2_outs_ready
    );

  buffer59 : entity work.oehb(arch) generic map(32)
    port map(
      ins => gate2_outs,
      ins_valid => gate2_outs_valid,
      ins_ready => gate2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer59_outs,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  fork31 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => buffer59_outs,
      ins_valid => buffer59_outs_valid,
      ins_ready => buffer59_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork31_outs_0,
      outs(1) => fork31_outs_1,
      outs(2) => fork31_outs_2,
      outs(3) => fork31_outs_3,
      outs_valid(0) => fork31_outs_0_valid,
      outs_valid(1) => fork31_outs_1_valid,
      outs_valid(2) => fork31_outs_2_valid,
      outs_valid(3) => fork31_outs_3_valid,
      outs_ready(0) => fork31_outs_0_ready,
      outs_ready(1) => fork31_outs_1_ready,
      outs_ready(2) => fork31_outs_2_ready,
      outs_ready(3) => fork31_outs_3_ready
    );

  buffer22 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux29_outs,
      ins_valid => mux29_outs_valid,
      ins_ready => mux29_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  cmpi6 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork31_outs_3,
      lhs_valid => fork31_outs_3_valid,
      lhs_ready => fork31_outs_3_ready,
      rhs => buffer22_outs,
      rhs_valid => buffer22_outs_valid,
      rhs_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi6_result,
      result_valid => cmpi6_result_valid,
      result_ready => cmpi6_result_ready
    );

  fork32 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi6_result,
      ins_valid => cmpi6_result_valid,
      ins_ready => cmpi6_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork32_outs_0,
      outs(1) => fork32_outs_1,
      outs_valid(0) => fork32_outs_0_valid,
      outs_valid(1) => fork32_outs_1_valid,
      outs_ready(0) => fork32_outs_0_ready,
      outs_ready(1) => fork32_outs_1_ready
    );

  buffer19 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux26_outs,
      ins_valid => mux26_outs_valid,
      ins_ready => mux26_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  cmpi7 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork31_outs_2,
      lhs_valid => fork31_outs_2_valid,
      lhs_ready => fork31_outs_2_ready,
      rhs => buffer19_outs,
      rhs_valid => buffer19_outs_valid,
      rhs_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi7_result,
      result_valid => cmpi7_result_valid,
      result_ready => cmpi7_result_ready
    );

  fork33 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi7_result,
      ins_valid => cmpi7_result_valid,
      ins_ready => cmpi7_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork33_outs_0,
      outs(1) => fork33_outs_1,
      outs_valid(0) => fork33_outs_0_valid,
      outs_valid(1) => fork33_outs_1_valid,
      outs_ready(0) => fork33_outs_0_ready,
      outs_ready(1) => fork33_outs_1_ready
    );

  buffer21 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux28_outs,
      ins_valid => mux28_outs_valid,
      ins_ready => mux28_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  cmpi8 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork31_outs_1,
      lhs_valid => fork31_outs_1_valid,
      lhs_ready => fork31_outs_1_ready,
      rhs => buffer21_outs,
      rhs_valid => buffer21_outs_valid,
      rhs_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi8_result,
      result_valid => cmpi8_result_valid,
      result_ready => cmpi8_result_ready
    );

  fork34 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi8_result,
      ins_valid => cmpi8_result_valid,
      ins_ready => cmpi8_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork34_outs_0,
      outs(1) => fork34_outs_1,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready
    );

  buffer24 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux31_outs,
      ins_valid => mux31_outs_valid,
      ins_ready => mux31_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  cmpi9 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork31_outs_0,
      lhs_valid => fork31_outs_0_valid,
      lhs_ready => fork31_outs_0_ready,
      rhs => buffer24_outs,
      rhs_valid => buffer24_outs_valid,
      rhs_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi9_result,
      result_valid => cmpi9_result_valid,
      result_ready => cmpi9_result_ready
    );

  fork35 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi9_result,
      ins_valid => cmpi9_result_valid,
      ins_ready => cmpi9_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready
    );

  buffer28 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux34_outs_valid,
      ins_ready => mux34_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  cond_br46 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer114_outs,
      condition_valid => buffer114_outs_valid,
      condition_ready => buffer114_outs_ready,
      data_valid => buffer28_outs_valid,
      data_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br46_trueOut_valid,
      trueOut_ready => cond_br46_trueOut_ready,
      falseOut_valid => cond_br46_falseOut_valid,
      falseOut_ready => cond_br46_falseOut_ready
    );

  buffer114 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork32_outs_1,
      ins_valid => fork32_outs_1_valid,
      ins_ready => fork32_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer114_outs,
      outs_valid => buffer114_outs_valid,
      outs_ready => buffer114_outs_ready
    );

  sink4 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br46_trueOut_valid,
      ins_ready => cond_br46_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer25 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux32_outs_valid,
      ins_ready => mux32_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  cond_br47 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer115_outs,
      condition_valid => buffer115_outs_valid,
      condition_ready => buffer115_outs_ready,
      data_valid => buffer25_outs_valid,
      data_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br47_trueOut_valid,
      trueOut_ready => cond_br47_trueOut_ready,
      falseOut_valid => cond_br47_falseOut_valid,
      falseOut_ready => cond_br47_falseOut_ready
    );

  buffer115 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork33_outs_1,
      ins_valid => fork33_outs_1_valid,
      ins_ready => fork33_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer115_outs,
      outs_valid => buffer115_outs_valid,
      outs_ready => buffer115_outs_ready
    );

  sink5 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br47_trueOut_valid,
      ins_ready => cond_br47_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer33 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux37_outs_valid,
      ins_ready => mux37_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  cond_br48 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer116_outs,
      condition_valid => buffer116_outs_valid,
      condition_ready => buffer116_outs_ready,
      data_valid => buffer33_outs_valid,
      data_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br48_trueOut_valid,
      trueOut_ready => cond_br48_trueOut_ready,
      falseOut_valid => cond_br48_falseOut_valid,
      falseOut_ready => cond_br48_falseOut_ready
    );

  buffer116 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork34_outs_1,
      ins_valid => fork34_outs_1_valid,
      ins_ready => fork34_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer116_outs,
      outs_valid => buffer116_outs_valid,
      outs_ready => buffer116_outs_ready
    );

  sink6 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br48_trueOut_valid,
      ins_ready => cond_br48_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer31 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux36_outs_valid,
      ins_ready => mux36_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  buffer32 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  cond_br49 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer117_outs,
      condition_valid => buffer117_outs_valid,
      condition_ready => buffer117_outs_ready,
      data_valid => buffer32_outs_valid,
      data_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br49_trueOut_valid,
      trueOut_ready => cond_br49_trueOut_ready,
      falseOut_valid => cond_br49_falseOut_valid,
      falseOut_ready => cond_br49_falseOut_ready
    );

  buffer117 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork35_outs_1,
      ins_valid => fork35_outs_1_valid,
      ins_ready => fork35_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer117_outs,
      outs_valid => buffer117_outs_valid,
      outs_ready => buffer117_outs_ready
    );

  sink7 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br49_trueOut_valid,
      ins_ready => cond_br49_trueOut_ready,
      clk => clk,
      rst => rst
    );

  source11 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source11_outs_valid,
      outs_ready => source11_outs_ready
    );

  mux44 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer118_outs,
      index_valid => buffer118_outs_valid,
      index_ready => buffer118_outs_ready,
      ins_valid(0) => cond_br46_falseOut_valid,
      ins_valid(1) => source11_outs_valid,
      ins_ready(0) => cond_br46_falseOut_ready,
      ins_ready(1) => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux44_outs_valid,
      outs_ready => mux44_outs_ready
    );

  buffer118 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork32_outs_0,
      ins_valid => fork32_outs_0_valid,
      ins_ready => fork32_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer118_outs,
      outs_valid => buffer118_outs_valid,
      outs_ready => buffer118_outs_ready
    );

  source12 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source12_outs_valid,
      outs_ready => source12_outs_ready
    );

  buffer60 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br47_falseOut_valid,
      ins_ready => cond_br47_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  mux45 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer119_outs,
      index_valid => buffer119_outs_valid,
      index_ready => buffer119_outs_ready,
      ins_valid(0) => buffer60_outs_valid,
      ins_valid(1) => source12_outs_valid,
      ins_ready(0) => buffer60_outs_ready,
      ins_ready(1) => source12_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux45_outs_valid,
      outs_ready => mux45_outs_ready
    );

  buffer119 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork33_outs_0,
      ins_valid => fork33_outs_0_valid,
      ins_ready => fork33_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer119_outs,
      outs_valid => buffer119_outs_valid,
      outs_ready => buffer119_outs_ready
    );

  source13 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source13_outs_valid,
      outs_ready => source13_outs_ready
    );

  buffer61 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br48_falseOut_valid,
      ins_ready => cond_br48_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  mux46 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer120_outs,
      index_valid => buffer120_outs_valid,
      index_ready => buffer120_outs_ready,
      ins_valid(0) => buffer61_outs_valid,
      ins_valid(1) => source13_outs_valid,
      ins_ready(0) => buffer61_outs_ready,
      ins_ready(1) => source13_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux46_outs_valid,
      outs_ready => mux46_outs_ready
    );

  buffer120 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork34_outs_0,
      ins_valid => fork34_outs_0_valid,
      ins_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer120_outs,
      outs_valid => buffer120_outs_valid,
      outs_ready => buffer120_outs_ready
    );

  source14 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source14_outs_valid,
      outs_ready => source14_outs_ready
    );

  mux47 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer121_outs,
      index_valid => buffer121_outs_valid,
      index_ready => buffer121_outs_ready,
      ins_valid(0) => cond_br49_falseOut_valid,
      ins_valid(1) => source14_outs_valid,
      ins_ready(0) => cond_br49_falseOut_ready,
      ins_ready(1) => source14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux47_outs_valid,
      outs_ready => mux47_outs_ready
    );

  buffer121 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork35_outs_0,
      ins_valid => fork35_outs_0_valid,
      ins_ready => fork35_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer121_outs,
      outs_valid => buffer121_outs_valid,
      outs_ready => buffer121_outs_ready
    );

  buffer62 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux44_outs_valid,
      ins_ready => mux44_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  buffer63 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux45_outs_valid,
      ins_ready => mux45_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer63_outs_valid,
      outs_ready => buffer63_outs_ready
    );

  buffer64 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux46_outs_valid,
      ins_ready => mux46_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer64_outs_valid,
      outs_ready => buffer64_outs_ready
    );

  buffer65 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux47_outs_valid,
      ins_ready => mux47_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer65_outs_valid,
      outs_ready => buffer65_outs_ready
    );

  join1 : entity work.join_handshake(arch) generic map(4)
    port map(
      ins_valid(0) => buffer62_outs_valid,
      ins_valid(1) => buffer63_outs_valid,
      ins_valid(2) => buffer64_outs_valid,
      ins_valid(3) => buffer65_outs_valid,
      ins_ready(0) => buffer62_outs_ready,
      ins_ready(1) => buffer63_outs_ready,
      ins_ready(2) => buffer64_outs_ready,
      ins_ready(3) => buffer65_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join1_outs_valid,
      outs_ready => join1_outs_ready
    );

  gate3 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => buffer122_outs,
      ins_valid(0) => buffer122_outs_valid,
      ins_valid(1) => join1_outs_valid,
      ins_ready(0) => buffer122_outs_ready,
      ins_ready(1) => join1_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate3_outs,
      outs_valid => gate3_outs_valid,
      outs_ready => gate3_outs_ready
    );

  buffer122 : entity work.tfifo(arch) generic map(2, 32)
    port map(
      ins => fork30_outs_0,
      ins_valid => fork30_outs_0_valid,
      ins_ready => fork30_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer122_outs,
      outs_valid => buffer122_outs_valid,
      outs_ready => buffer122_outs_ready
    );

  trunci4 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => gate3_outs,
      ins_valid => gate3_outs_valid,
      ins_ready => gate3_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  load4 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => trunci4_outs,
      addrIn_valid => trunci4_outs_valid,
      addrIn_ready => trunci4_outs_ready,
      dataFromMem => mem_controller6_ldData_1,
      dataFromMem_valid => mem_controller6_ldData_1_valid,
      dataFromMem_ready => mem_controller6_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load4_dataOut,
      lhs_valid => load4_dataOut_valid,
      lhs_ready => load4_dataOut_ready,
      rhs => muli0_result,
      rhs_valid => muli0_result_valid,
      rhs_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  shli4 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork17_outs_2,
      lhs_valid => fork17_outs_2_valid,
      lhs_ready => fork17_outs_2_ready,
      rhs => fork22_outs_2,
      rhs_valid => fork22_outs_2_valid,
      rhs_ready => fork22_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli4_result,
      result_valid => shli4_result_valid,
      result_ready => shli4_result_ready
    );

  shli5 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer126_outs,
      lhs_valid => buffer126_outs_valid,
      lhs_ready => buffer126_outs_ready,
      rhs => fork21_outs_2,
      rhs_valid => fork21_outs_2_valid,
      rhs_ready => fork21_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli5_result,
      result_valid => shli5_result_valid,
      result_ready => shli5_result_ready
    );

  buffer126 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork17_outs_3,
      ins_valid => fork17_outs_3_valid,
      ins_ready => fork17_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer126_outs,
      outs_valid => buffer126_outs_valid,
      outs_ready => buffer126_outs_ready
    );

  buffer66 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli4_result,
      ins_valid => shli4_result_valid,
      ins_ready => shli4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer66_outs,
      outs_valid => buffer66_outs_valid,
      outs_ready => buffer66_outs_ready
    );

  buffer67 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli5_result,
      ins_valid => shli5_result_valid,
      ins_ready => shli5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer67_outs,
      outs_valid => buffer67_outs_valid,
      outs_ready => buffer67_outs_ready
    );

  addi9 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer66_outs,
      lhs_valid => buffer66_outs_valid,
      lhs_ready => buffer66_outs_ready,
      rhs => buffer67_outs,
      rhs_valid => buffer67_outs_valid,
      rhs_ready => buffer67_outs_ready,
      clk => clk,
      rst => rst,
      result => addi9_result,
      result_valid => addi9_result_valid,
      result_ready => addi9_result_ready
    );

  buffer68 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi9_result,
      ins_valid => addi9_result_valid,
      ins_ready => addi9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer68_outs,
      outs_valid => buffer68_outs_valid,
      outs_ready => buffer68_outs_ready
    );

  addi4 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer127_outs,
      lhs_valid => buffer127_outs_valid,
      lhs_ready => buffer127_outs_ready,
      rhs => buffer68_outs,
      rhs_valid => buffer68_outs_valid,
      rhs_ready => buffer68_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer127 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer127_outs,
      outs_valid => buffer127_outs_valid,
      outs_ready => buffer127_outs_ready
    );

  buffer70 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer70_outs,
      outs_valid => buffer70_outs_valid,
      outs_ready => buffer70_outs_ready
    );

  fork36 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer70_outs,
      ins_valid => buffer70_outs_valid,
      ins_ready => buffer70_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork36_outs_0,
      outs(1) => fork36_outs_1,
      outs_valid(0) => fork36_outs_0_valid,
      outs_valid(1) => fork36_outs_1_valid,
      outs_ready(0) => fork36_outs_0_ready,
      outs_ready(1) => fork36_outs_1_ready
    );

  trunci5 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer128_outs,
      ins_valid => buffer128_outs_valid,
      ins_ready => buffer128_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  buffer128 : entity work.tfifo(arch) generic map(3, 32)
    port map(
      ins => fork36_outs_0,
      ins_valid => fork36_outs_0_valid,
      ins_ready => fork36_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer128_outs,
      outs_valid => buffer128_outs_valid,
      outs_ready => buffer128_outs_ready
    );

  buffer0 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork36_outs_1,
      ins_valid => fork36_outs_1_valid,
      ins_ready => fork36_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  fork37 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork37_outs_0,
      outs(1) => fork37_outs_1,
      outs_valid(0) => fork37_outs_0_valid,
      outs_valid(1) => fork37_outs_1_valid,
      outs_ready(0) => fork37_outs_0_ready,
      outs_ready(1) => fork37_outs_1_ready
    );

  init36 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => fork37_outs_0,
      ins_valid => fork37_outs_0_valid,
      ins_ready => fork37_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => init36_outs,
      outs_valid => init36_outs_valid,
      outs_ready => init36_outs_ready
    );

  fork38 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init36_outs,
      ins_valid => init36_outs_valid,
      ins_ready => init36_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork38_outs_0,
      outs(1) => fork38_outs_1,
      outs_valid(0) => fork38_outs_0_valid,
      outs_valid(1) => fork38_outs_1_valid,
      outs_ready(0) => fork38_outs_0_ready,
      outs_ready(1) => fork38_outs_1_ready
    );

  init37 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer131_outs,
      ins_valid => buffer131_outs_valid,
      ins_ready => buffer131_outs_ready,
      clk => clk,
      rst => rst,
      outs => init37_outs,
      outs_valid => init37_outs_valid,
      outs_ready => init37_outs_ready
    );

  buffer131 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork38_outs_0,
      ins_valid => fork38_outs_0_valid,
      ins_ready => fork38_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer131_outs,
      outs_valid => buffer131_outs_valid,
      outs_ready => buffer131_outs_ready
    );

  fork39 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init37_outs,
      ins_valid => init37_outs_valid,
      ins_ready => init37_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork39_outs_0,
      outs(1) => fork39_outs_1,
      outs_valid(0) => fork39_outs_0_valid,
      outs_valid(1) => fork39_outs_1_valid,
      outs_ready(0) => fork39_outs_0_ready,
      outs_ready(1) => fork39_outs_1_ready
    );

  init38 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => fork39_outs_0,
      ins_valid => fork39_outs_0_valid,
      ins_ready => fork39_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => init38_outs,
      outs_valid => init38_outs_valid,
      outs_ready => init38_outs_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => store0_doneOut_valid,
      ins_ready => store0_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  buffer72 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer72_outs_valid,
      outs_ready => buffer72_outs_ready
    );

  fork40 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer72_outs_valid,
      ins_ready => buffer72_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork40_outs_0_valid,
      outs_valid(1) => fork40_outs_1_valid,
      outs_ready(0) => fork40_outs_0_ready,
      outs_ready(1) => fork40_outs_1_ready
    );

  init39 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork40_outs_0_valid,
      ins_ready => fork40_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init39_outs_valid,
      outs_ready => init39_outs_ready
    );

  fork41 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init39_outs_valid,
      ins_ready => init39_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork41_outs_0_valid,
      outs_valid(1) => fork41_outs_1_valid,
      outs_ready(0) => fork41_outs_0_ready,
      outs_ready(1) => fork41_outs_1_ready
    );

  init40 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork41_outs_0_valid,
      ins_ready => fork41_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init40_outs_valid,
      outs_ready => init40_outs_ready
    );

  fork42 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init40_outs_valid,
      ins_ready => init40_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork42_outs_0_valid,
      outs_valid(1) => fork42_outs_1_valid,
      outs_ready(0) => fork42_outs_0_ready,
      outs_ready(1) => fork42_outs_1_ready
    );

  init41 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork42_outs_0_valid,
      ins_ready => fork42_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init41_outs_valid,
      outs_ready => init41_outs_ready
    );

  fork43 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init41_outs_valid,
      ins_ready => init41_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork43_outs_0_valid,
      outs_valid(1) => fork43_outs_1_valid,
      outs_ready(0) => fork43_outs_0_ready,
      outs_ready(1) => fork43_outs_1_ready
    );

  init42 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork43_outs_0_valid,
      ins_ready => fork43_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init42_outs_valid,
      outs_ready => init42_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 9)
    port map(
      addrIn => trunci5_outs,
      addrIn_valid => trunci5_outs_valid,
      addrIn_ready => trunci5_outs_ready,
      dataIn => addi1_result,
      dataIn_valid => addi1_result_valid,
      dataIn_ready => addi1_result_ready,
      doneFromMem_valid => mem_controller6_stDone_0_valid,
      doneFromMem_ready => mem_controller6_stDone_0_ready,
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

  addi5 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi20_outs,
      lhs_valid => extsi20_outs_valid,
      lhs_ready => extsi20_outs_ready,
      rhs => extsi22_outs,
      rhs_valid => extsi22_outs_valid,
      rhs_ready => extsi22_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer73 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer73_outs,
      outs_valid => buffer73_outs_valid,
      outs_ready => buffer73_outs_ready
    );

  fork44 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer73_outs,
      ins_valid => buffer73_outs_valid,
      ins_ready => buffer73_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork44_outs_0,
      outs(1) => fork44_outs_1,
      outs_valid(0) => fork44_outs_0_valid,
      outs_valid(1) => fork44_outs_1_valid,
      outs_ready(0) => fork44_outs_0_ready,
      outs_ready(1) => fork44_outs_1_ready
    );

  trunci6 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork44_outs_0,
      ins_valid => fork44_outs_0_valid,
      ins_ready => fork44_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_1(arch) generic map(7)
    port map(
      lhs => fork44_outs_1,
      lhs_valid => fork44_outs_1_valid,
      lhs_ready => fork44_outs_1_ready,
      rhs => extsi23_outs,
      rhs_valid => extsi23_outs_valid,
      rhs_ready => extsi23_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer74 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer74_outs,
      outs_valid => buffer74_outs_valid,
      outs_ready => buffer74_outs_ready
    );

  fork45 : entity work.handshake_fork(arch) generic map(14, 1)
    port map(
      ins => buffer74_outs,
      ins_valid => buffer74_outs_valid,
      ins_ready => buffer74_outs_ready,
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
      outs(10) => fork45_outs_10,
      outs(11) => fork45_outs_11,
      outs(12) => fork45_outs_12,
      outs(13) => fork45_outs_13,
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
      outs_valid(10) => fork45_outs_10_valid,
      outs_valid(11) => fork45_outs_11_valid,
      outs_valid(12) => fork45_outs_12_valid,
      outs_valid(13) => fork45_outs_13_valid,
      outs_ready(0) => fork45_outs_0_ready,
      outs_ready(1) => fork45_outs_1_ready,
      outs_ready(2) => fork45_outs_2_ready,
      outs_ready(3) => fork45_outs_3_ready,
      outs_ready(4) => fork45_outs_4_ready,
      outs_ready(5) => fork45_outs_5_ready,
      outs_ready(6) => fork45_outs_6_ready,
      outs_ready(7) => fork45_outs_7_ready,
      outs_ready(8) => fork45_outs_8_ready,
      outs_ready(9) => fork45_outs_9_ready,
      outs_ready(10) => fork45_outs_10_ready,
      outs_ready(11) => fork45_outs_11_ready,
      outs_ready(12) => fork45_outs_12_ready,
      outs_ready(13) => fork45_outs_13_ready
    );

  cond_br2 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork45_outs_0,
      condition_valid => fork45_outs_0_valid,
      condition_ready => fork45_outs_0_ready,
      data => trunci6_outs,
      data_valid => trunci6_outs_valid,
      data_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready
    );

  sink8 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br3 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork45_outs_1,
      condition_valid => fork45_outs_1_valid,
      condition_ready => fork45_outs_1_ready,
      data => buffer137_outs,
      data_valid => buffer137_outs_valid,
      data_ready => buffer137_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer137 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer137_outs,
      outs_valid => buffer137_outs_valid,
      outs_ready => buffer137_outs_ready
    );

  cond_br4 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork45_outs_12,
      condition_valid => fork45_outs_12_valid,
      condition_ready => fork45_outs_12_ready,
      data => buffer139_outs,
      data_valid => buffer139_outs_valid,
      data_ready => buffer139_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  buffer139 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer139_outs,
      outs_valid => buffer139_outs_valid,
      outs_ready => buffer139_outs_ready
    );

  sink9 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br5 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork45_outs_13,
      condition_valid => fork45_outs_13_valid,
      condition_ready => fork45_outs_13_ready,
      data_valid => fork20_outs_1_valid,
      data_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  cond_br77 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork56_outs_9,
      condition_valid => fork56_outs_9_valid,
      condition_ready => fork56_outs_9_ready,
      data_valid => cond_br69_falseOut_valid,
      data_ready => cond_br69_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br77_trueOut_valid,
      trueOut_ready => cond_br77_trueOut_ready,
      falseOut_valid => cond_br77_falseOut_valid,
      falseOut_ready => cond_br77_falseOut_ready
    );

  sink10 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br77_falseOut_valid,
      ins_ready => cond_br77_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork46 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br77_trueOut_valid,
      ins_ready => cond_br77_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork46_outs_0_valid,
      outs_valid(1) => fork46_outs_1_valid,
      outs_ready(0) => fork46_outs_0_ready,
      outs_ready(1) => fork46_outs_1_ready
    );

  cond_br78 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork56_outs_8,
      condition_valid => fork56_outs_8_valid,
      condition_ready => fork56_outs_8_ready,
      data_valid => cond_br71_falseOut_valid,
      data_ready => cond_br71_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br78_trueOut_valid,
      trueOut_ready => cond_br78_trueOut_ready,
      falseOut_valid => cond_br78_falseOut_valid,
      falseOut_ready => cond_br78_falseOut_ready
    );

  sink11 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br78_falseOut_valid,
      ins_ready => cond_br78_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork47 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br78_trueOut_valid,
      ins_ready => cond_br78_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork47_outs_0_valid,
      outs_valid(1) => fork47_outs_1_valid,
      outs_ready(0) => fork47_outs_0_ready,
      outs_ready(1) => fork47_outs_1_ready
    );

  cond_br79 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork56_outs_7,
      condition_valid => fork56_outs_7_valid,
      condition_ready => fork56_outs_7_ready,
      data_valid => cond_br68_falseOut_valid,
      data_ready => cond_br68_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br79_trueOut_valid,
      trueOut_ready => cond_br79_trueOut_ready,
      falseOut_valid => cond_br79_falseOut_valid,
      falseOut_ready => cond_br79_falseOut_ready
    );

  sink12 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br79_falseOut_valid,
      ins_ready => cond_br79_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork48 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br79_trueOut_valid,
      ins_ready => cond_br79_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork48_outs_0_valid,
      outs_valid(1) => fork48_outs_1_valid,
      outs_ready(0) => fork48_outs_0_ready,
      outs_ready(1) => fork48_outs_1_ready
    );

  cond_br80 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer144_outs,
      condition_valid => buffer144_outs_valid,
      condition_ready => buffer144_outs_ready,
      data_valid => cond_br73_falseOut_valid,
      data_ready => cond_br73_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br80_trueOut_valid,
      trueOut_ready => cond_br80_trueOut_ready,
      falseOut_valid => cond_br80_falseOut_valid,
      falseOut_ready => cond_br80_falseOut_ready
    );

  buffer144 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork56_outs_6,
      ins_valid => fork56_outs_6_valid,
      ins_ready => fork56_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer144_outs,
      outs_valid => buffer144_outs_valid,
      outs_ready => buffer144_outs_ready
    );

  sink13 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br80_falseOut_valid,
      ins_ready => cond_br80_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork49 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br80_trueOut_valid,
      ins_ready => cond_br80_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork49_outs_0_valid,
      outs_valid(1) => fork49_outs_1_valid,
      outs_ready(0) => fork49_outs_0_ready,
      outs_ready(1) => fork49_outs_1_ready
    );

  cond_br81 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork56_outs_5,
      condition_valid => fork56_outs_5_valid,
      condition_ready => fork56_outs_5_ready,
      data => cond_br75_falseOut,
      data_valid => cond_br75_falseOut_valid,
      data_ready => cond_br75_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br81_trueOut,
      trueOut_valid => cond_br81_trueOut_valid,
      trueOut_ready => cond_br81_trueOut_ready,
      falseOut => cond_br81_falseOut,
      falseOut_valid => cond_br81_falseOut_valid,
      falseOut_ready => cond_br81_falseOut_ready
    );

  sink14 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br81_falseOut,
      ins_valid => cond_br81_falseOut_valid,
      ins_ready => cond_br81_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork50 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br81_trueOut,
      ins_valid => cond_br81_trueOut_valid,
      ins_ready => cond_br81_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork50_outs_0,
      outs(1) => fork50_outs_1,
      outs_valid(0) => fork50_outs_0_valid,
      outs_valid(1) => fork50_outs_1_valid,
      outs_ready(0) => fork50_outs_0_ready,
      outs_ready(1) => fork50_outs_1_ready
    );

  cond_br82 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork56_outs_4,
      condition_valid => fork56_outs_4_valid,
      condition_ready => fork56_outs_4_ready,
      data => cond_br72_falseOut,
      data_valid => cond_br72_falseOut_valid,
      data_ready => cond_br72_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br82_trueOut,
      trueOut_valid => cond_br82_trueOut_valid,
      trueOut_ready => cond_br82_trueOut_ready,
      falseOut => cond_br82_falseOut,
      falseOut_valid => cond_br82_falseOut_valid,
      falseOut_ready => cond_br82_falseOut_ready
    );

  sink15 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br82_falseOut,
      ins_valid => cond_br82_falseOut_valid,
      ins_ready => cond_br82_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork51 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br82_trueOut,
      ins_valid => cond_br82_trueOut_valid,
      ins_ready => cond_br82_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork51_outs_0,
      outs(1) => fork51_outs_1,
      outs_valid(0) => fork51_outs_0_valid,
      outs_valid(1) => fork51_outs_1_valid,
      outs_ready(0) => fork51_outs_0_ready,
      outs_ready(1) => fork51_outs_1_ready
    );

  cond_br83 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork56_outs_3,
      condition_valid => fork56_outs_3_valid,
      condition_ready => fork56_outs_3_ready,
      data => cond_br70_falseOut,
      data_valid => cond_br70_falseOut_valid,
      data_ready => cond_br70_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br83_trueOut,
      trueOut_valid => cond_br83_trueOut_valid,
      trueOut_ready => cond_br83_trueOut_ready,
      falseOut => cond_br83_falseOut,
      falseOut_valid => cond_br83_falseOut_valid,
      falseOut_ready => cond_br83_falseOut_ready
    );

  sink16 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br83_falseOut,
      ins_valid => cond_br83_falseOut_valid,
      ins_ready => cond_br83_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork52 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br83_trueOut,
      ins_valid => cond_br83_trueOut_valid,
      ins_ready => cond_br83_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork52_outs_0,
      outs(1) => fork52_outs_1,
      outs_valid(0) => fork52_outs_0_valid,
      outs_valid(1) => fork52_outs_1_valid,
      outs_ready(0) => fork52_outs_0_ready,
      outs_ready(1) => fork52_outs_1_ready
    );

  cond_br84 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork56_outs_2,
      condition_valid => fork56_outs_2_valid,
      condition_ready => fork56_outs_2_ready,
      data_valid => cond_br74_falseOut_valid,
      data_ready => cond_br74_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br84_trueOut_valid,
      trueOut_ready => cond_br84_trueOut_ready,
      falseOut_valid => cond_br84_falseOut_valid,
      falseOut_ready => cond_br84_falseOut_ready
    );

  sink17 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br84_falseOut_valid,
      ins_ready => cond_br84_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork53 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br84_trueOut_valid,
      ins_ready => cond_br84_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork53_outs_0_valid,
      outs_valid(1) => fork53_outs_1_valid,
      outs_ready(0) => fork53_outs_0_ready,
      outs_ready(1) => fork53_outs_1_ready
    );

  cond_br85 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork56_outs_1,
      condition_valid => fork56_outs_1_valid,
      condition_ready => fork56_outs_1_ready,
      data => cond_br76_falseOut,
      data_valid => cond_br76_falseOut_valid,
      data_ready => cond_br76_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br85_trueOut,
      trueOut_valid => cond_br85_trueOut_valid,
      trueOut_ready => cond_br85_trueOut_ready,
      falseOut => cond_br85_falseOut,
      falseOut_valid => cond_br85_falseOut_valid,
      falseOut_ready => cond_br85_falseOut_ready
    );

  sink18 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br85_falseOut,
      ins_valid => cond_br85_falseOut_valid,
      ins_ready => cond_br85_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork54 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br85_trueOut,
      ins_valid => cond_br85_trueOut_valid,
      ins_ready => cond_br85_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork54_outs_0,
      outs(1) => fork54_outs_1,
      outs_valid(0) => fork54_outs_0_valid,
      outs_valid(1) => fork54_outs_1_valid,
      outs_ready(0) => fork54_outs_0_ready,
      outs_ready(1) => fork54_outs_1_ready
    );

  extsi24 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi24_outs,
      outs_valid => extsi24_outs_valid,
      outs_ready => extsi24_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant26 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant26_outs,
      outs_valid => constant26_outs_valid,
      outs_ready => constant26_outs_ready
    );

  extsi25 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant26_outs,
      ins_valid => constant26_outs_valid,
      ins_ready => constant26_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant27 : entity work.handshake_constant_4(arch) generic map(6)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant27_outs,
      outs_valid => constant27_outs_valid,
      outs_ready => constant27_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant27_outs,
      ins_valid => constant27_outs_valid,
      ins_ready => constant27_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  addi6 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi24_outs,
      lhs_valid => extsi24_outs_valid,
      lhs_ready => extsi24_outs_ready,
      rhs => extsi25_outs,
      rhs_valid => extsi25_outs_valid,
      rhs_ready => extsi25_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer77 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer77_outs,
      outs_valid => buffer77_outs_valid,
      outs_ready => buffer77_outs_ready
    );

  fork55 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer77_outs,
      ins_valid => buffer77_outs_valid,
      ins_ready => buffer77_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork55_outs_0,
      outs(1) => fork55_outs_1,
      outs_valid(0) => fork55_outs_0_valid,
      outs_valid(1) => fork55_outs_1_valid,
      outs_ready(0) => fork55_outs_0_ready,
      outs_ready(1) => fork55_outs_1_ready
    );

  trunci7 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork55_outs_0,
      ins_valid => fork55_outs_0_valid,
      ins_ready => fork55_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(7)
    port map(
      lhs => fork55_outs_1,
      lhs_valid => fork55_outs_1_valid,
      lhs_ready => fork55_outs_1_ready,
      rhs => extsi26_outs,
      rhs_valid => extsi26_outs_valid,
      rhs_ready => extsi26_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer78 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer78_outs,
      outs_valid => buffer78_outs_valid,
      outs_ready => buffer78_outs_ready
    );

  fork56 : entity work.handshake_fork(arch) generic map(12, 1)
    port map(
      ins => buffer78_outs,
      ins_valid => buffer78_outs_valid,
      ins_ready => buffer78_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork56_outs_0,
      outs(1) => fork56_outs_1,
      outs(2) => fork56_outs_2,
      outs(3) => fork56_outs_3,
      outs(4) => fork56_outs_4,
      outs(5) => fork56_outs_5,
      outs(6) => fork56_outs_6,
      outs(7) => fork56_outs_7,
      outs(8) => fork56_outs_8,
      outs(9) => fork56_outs_9,
      outs(10) => fork56_outs_10,
      outs(11) => fork56_outs_11,
      outs_valid(0) => fork56_outs_0_valid,
      outs_valid(1) => fork56_outs_1_valid,
      outs_valid(2) => fork56_outs_2_valid,
      outs_valid(3) => fork56_outs_3_valid,
      outs_valid(4) => fork56_outs_4_valid,
      outs_valid(5) => fork56_outs_5_valid,
      outs_valid(6) => fork56_outs_6_valid,
      outs_valid(7) => fork56_outs_7_valid,
      outs_valid(8) => fork56_outs_8_valid,
      outs_valid(9) => fork56_outs_9_valid,
      outs_valid(10) => fork56_outs_10_valid,
      outs_valid(11) => fork56_outs_11_valid,
      outs_ready(0) => fork56_outs_0_ready,
      outs_ready(1) => fork56_outs_1_ready,
      outs_ready(2) => fork56_outs_2_ready,
      outs_ready(3) => fork56_outs_3_ready,
      outs_ready(4) => fork56_outs_4_ready,
      outs_ready(5) => fork56_outs_5_ready,
      outs_ready(6) => fork56_outs_6_ready,
      outs_ready(7) => fork56_outs_7_ready,
      outs_ready(8) => fork56_outs_8_ready,
      outs_ready(9) => fork56_outs_9_ready,
      outs_ready(10) => fork56_outs_10_ready,
      outs_ready(11) => fork56_outs_11_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork56_outs_0,
      condition_valid => fork56_outs_0_valid,
      condition_ready => fork56_outs_0_ready,
      data => trunci7_outs,
      data_valid => trunci7_outs_valid,
      data_ready => trunci7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink20 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br7 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork56_outs_11,
      condition_valid => fork56_outs_11_valid,
      condition_ready => fork56_outs_11_ready,
      data_valid => cond_br5_falseOut_valid,
      data_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  fork57 : entity work.fork_dataless(arch) generic map(4)
    port map(
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork57_outs_0_valid,
      outs_valid(1) => fork57_outs_1_valid,
      outs_valid(2) => fork57_outs_2_valid,
      outs_valid(3) => fork57_outs_3_valid,
      outs_ready(0) => fork57_outs_0_ready,
      outs_ready(1) => fork57_outs_1_ready,
      outs_ready(2) => fork57_outs_2_ready,
      outs_ready(3) => fork57_outs_3_ready
    );

end architecture;
