library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity gaussian is
  port (
    c_loadData : in std_logic_vector(31 downto 0);
    a_loadData : in std_logic_vector(31 downto 0);
    c_start_valid : in std_logic;
    a_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    c_end_ready : in std_logic;
    a_end_ready : in std_logic;
    end_ready : in std_logic;
    c_start_ready : out std_logic;
    a_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    c_end_valid : out std_logic;
    a_end_valid : out std_logic;
    end_valid : out std_logic;
    c_loadEn : out std_logic;
    c_loadAddr : out std_logic_vector(4 downto 0);
    c_storeEn : out std_logic;
    c_storeAddr : out std_logic_vector(4 downto 0);
    c_storeData : out std_logic_vector(31 downto 0);
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(8 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(8 downto 0);
    a_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of gaussian is

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
  signal mem_controller1_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0_valid : std_logic;
  signal mem_controller1_ldData_0_ready : std_logic;
  signal mem_controller1_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_1_valid : std_logic;
  signal mem_controller1_ldData_1_ready : std_logic;
  signal mem_controller1_stDone_0_valid : std_logic;
  signal mem_controller1_stDone_0_ready : std_logic;
  signal mem_controller1_memEnd_valid : std_logic;
  signal mem_controller1_memEnd_ready : std_logic;
  signal mem_controller1_loadEn : std_logic;
  signal mem_controller1_loadAddr : std_logic_vector(8 downto 0);
  signal mem_controller1_storeEn : std_logic;
  signal mem_controller1_storeAddr : std_logic_vector(8 downto 0);
  signal mem_controller1_storeData : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0_valid : std_logic;
  signal mem_controller2_ldData_0_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal constant5_outs : std_logic_vector(10 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(10 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(10 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(0 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal constant19_outs : std_logic_vector(1 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal extsi18_outs : std_logic_vector(5 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal extsi19_outs : std_logic_vector(31 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal mux11_outs : std_logic_vector(31 downto 0);
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal mux12_outs : std_logic_vector(31 downto 0);
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal mux14_outs_valid : std_logic;
  signal mux14_outs_ready : std_logic;
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
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
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(5 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(5 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(5 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi20_outs : std_logic_vector(6 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal buffer73_outs : std_logic_vector(31 downto 0);
  signal buffer73_outs_valid : std_logic;
  signal buffer73_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant20_outs : std_logic_vector(1 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal extsi21_outs : std_logic_vector(6 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer10_outs : std_logic_vector(6 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux16_outs_valid : std_logic;
  signal mux16_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal mux17_outs : std_logic_vector(31 downto 0);
  signal mux17_outs_valid : std_logic;
  signal mux17_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal mux18_outs : std_logic_vector(31 downto 0);
  signal mux18_outs_valid : std_logic;
  signal mux18_outs_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal mux19_outs_valid : std_logic;
  signal mux19_outs_ready : std_logic;
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal mux20_outs_valid : std_logic;
  signal mux20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal mux21_outs_valid : std_logic;
  signal mux21_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(0 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal init6_outs : std_logic_vector(0 downto 0);
  signal init6_outs_valid : std_logic;
  signal init6_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal fork5_outs_2 : std_logic_vector(0 downto 0);
  signal fork5_outs_2_valid : std_logic;
  signal fork5_outs_2_ready : std_logic;
  signal fork5_outs_3 : std_logic_vector(0 downto 0);
  signal fork5_outs_3_valid : std_logic;
  signal fork5_outs_3_ready : std_logic;
  signal fork5_outs_4 : std_logic_vector(0 downto 0);
  signal fork5_outs_4_valid : std_logic;
  signal fork5_outs_4_ready : std_logic;
  signal fork5_outs_5 : std_logic_vector(0 downto 0);
  signal fork5_outs_5_valid : std_logic;
  signal fork5_outs_5_ready : std_logic;
  signal mux2_outs : std_logic_vector(6 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer25_outs : std_logic_vector(6 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(6 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(6 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal trunci0_outs : std_logic_vector(5 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(5 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant21_outs : std_logic_vector(5 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi22_outs : std_logic_vector(6 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal constant22_outs : std_logic_vector(1 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(1 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(1 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer31_outs : std_logic_vector(0 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(0 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(0 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal fork10_outs_2 : std_logic_vector(0 downto 0);
  signal fork10_outs_2_valid : std_logic;
  signal fork10_outs_2_ready : std_logic;
  signal fork10_outs_3 : std_logic_vector(0 downto 0);
  signal fork10_outs_3_valid : std_logic;
  signal fork10_outs_3_ready : std_logic;
  signal fork10_outs_4 : std_logic_vector(0 downto 0);
  signal fork10_outs_4_valid : std_logic;
  signal fork10_outs_4_ready : std_logic;
  signal fork10_outs_5 : std_logic_vector(0 downto 0);
  signal fork10_outs_5_valid : std_logic;
  signal fork10_outs_5_ready : std_logic;
  signal fork10_outs_6 : std_logic_vector(0 downto 0);
  signal fork10_outs_6_valid : std_logic;
  signal fork10_outs_6_ready : std_logic;
  signal fork10_outs_7 : std_logic_vector(0 downto 0);
  signal fork10_outs_7_valid : std_logic;
  signal fork10_outs_7_ready : std_logic;
  signal fork10_outs_8 : std_logic_vector(0 downto 0);
  signal fork10_outs_8_valid : std_logic;
  signal fork10_outs_8_ready : std_logic;
  signal fork10_outs_9 : std_logic_vector(0 downto 0);
  signal fork10_outs_9_valid : std_logic;
  signal fork10_outs_9_ready : std_logic;
  signal fork10_outs_10 : std_logic_vector(0 downto 0);
  signal fork10_outs_10_valid : std_logic;
  signal fork10_outs_10_ready : std_logic;
  signal fork10_outs_11 : std_logic_vector(0 downto 0);
  signal fork10_outs_11_valid : std_logic;
  signal fork10_outs_11_ready : std_logic;
  signal fork10_outs_12 : std_logic_vector(0 downto 0);
  signal fork10_outs_12_valid : std_logic;
  signal fork10_outs_12_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(1 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(1 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal extsi17_outs : std_logic_vector(5 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(1 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(1 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal extsi23_outs : std_logic_vector(31 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(31 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(31 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer28_outs : std_logic_vector(5 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer29_outs : std_logic_vector(5 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(5 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(5 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(5 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(5 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal cond_br47_trueOut : std_logic_vector(31 downto 0);
  signal cond_br47_trueOut_valid : std_logic;
  signal cond_br47_trueOut_ready : std_logic;
  signal cond_br47_falseOut : std_logic_vector(31 downto 0);
  signal cond_br47_falseOut_valid : std_logic;
  signal cond_br47_falseOut_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal cond_br48_trueOut_valid : std_logic;
  signal cond_br48_trueOut_ready : std_logic;
  signal cond_br48_falseOut_valid : std_logic;
  signal cond_br48_falseOut_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal cond_br49_trueOut : std_logic_vector(31 downto 0);
  signal cond_br49_trueOut_valid : std_logic;
  signal cond_br49_trueOut_ready : std_logic;
  signal cond_br49_falseOut : std_logic_vector(31 downto 0);
  signal cond_br49_falseOut_valid : std_logic;
  signal cond_br49_falseOut_ready : std_logic;
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal cond_br50_trueOut_valid : std_logic;
  signal cond_br50_trueOut_ready : std_logic;
  signal cond_br50_falseOut_valid : std_logic;
  signal cond_br50_falseOut_ready : std_logic;
  signal buffer40_outs : std_logic_vector(0 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal cond_br51_trueOut : std_logic_vector(31 downto 0);
  signal cond_br51_trueOut_valid : std_logic;
  signal cond_br51_trueOut_ready : std_logic;
  signal cond_br51_falseOut : std_logic_vector(31 downto 0);
  signal cond_br51_falseOut_valid : std_logic;
  signal cond_br51_falseOut_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(31 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(31 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal cond_br52_trueOut_valid : std_logic;
  signal cond_br52_trueOut_ready : std_logic;
  signal cond_br52_falseOut_valid : std_logic;
  signal cond_br52_falseOut_ready : std_logic;
  signal cond_br53_trueOut_valid : std_logic;
  signal cond_br53_trueOut_ready : std_logic;
  signal cond_br53_falseOut_valid : std_logic;
  signal cond_br53_falseOut_ready : std_logic;
  signal buffer43_outs : std_logic_vector(0 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cond_br54_trueOut_valid : std_logic;
  signal cond_br54_trueOut_ready : std_logic;
  signal cond_br54_falseOut_valid : std_logic;
  signal cond_br54_falseOut_ready : std_logic;
  signal buffer44_outs : std_logic_vector(0 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal cond_br55_trueOut_valid : std_logic;
  signal cond_br55_trueOut_ready : std_logic;
  signal cond_br55_falseOut_valid : std_logic;
  signal cond_br55_falseOut_ready : std_logic;
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal mux22_outs_valid : std_logic;
  signal mux22_outs_ready : std_logic;
  signal mux23_outs : std_logic_vector(31 downto 0);
  signal mux23_outs_valid : std_logic;
  signal mux23_outs_ready : std_logic;
  signal mux24_outs : std_logic_vector(31 downto 0);
  signal mux24_outs_valid : std_logic;
  signal mux24_outs_ready : std_logic;
  signal mux25_outs_valid : std_logic;
  signal mux25_outs_ready : std_logic;
  signal mux26_outs_valid : std_logic;
  signal mux26_outs_ready : std_logic;
  signal buffer52_outs : std_logic_vector(0 downto 0);
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal mux27_outs_valid : std_logic;
  signal mux27_outs_ready : std_logic;
  signal buffer53_outs : std_logic_vector(0 downto 0);
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal init12_outs : std_logic_vector(0 downto 0);
  signal init12_outs_valid : std_logic;
  signal init12_outs_ready : std_logic;
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
  signal fork17_outs_5 : std_logic_vector(0 downto 0);
  signal fork17_outs_5_valid : std_logic;
  signal fork17_outs_5_ready : std_logic;
  signal mux5_outs : std_logic_vector(5 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer38_outs : std_logic_vector(5 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal extsi24_outs : std_logic_vector(6 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer39_outs : std_logic_vector(31 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
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
  signal fork18_outs_3 : std_logic_vector(31 downto 0);
  signal fork18_outs_3_valid : std_logic;
  signal fork18_outs_3_ready : std_logic;
  signal fork18_outs_4 : std_logic_vector(31 downto 0);
  signal fork18_outs_4_valid : std_logic;
  signal fork18_outs_4_ready : std_logic;
  signal mux7_outs : std_logic_vector(31 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(5 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer45_outs : std_logic_vector(5 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal buffer46_outs : std_logic_vector(5 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(5 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(5 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(5 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal extsi25_outs : std_logic_vector(31 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal trunci1_outs : std_logic_vector(4 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(5 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal buffer47_outs : std_logic_vector(5 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal buffer48_outs : std_logic_vector(5 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(5 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(5 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal extsi26_outs : std_logic_vector(31 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(31 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(31 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal fork22_outs_2 : std_logic_vector(31 downto 0);
  signal fork22_outs_2_valid : std_logic;
  signal fork22_outs_2_ready : std_logic;
  signal fork22_outs_3 : std_logic_vector(31 downto 0);
  signal fork22_outs_3_valid : std_logic;
  signal fork22_outs_3_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(0 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(0 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal fork23_outs_2 : std_logic_vector(0 downto 0);
  signal fork23_outs_2_valid : std_logic;
  signal fork23_outs_2_ready : std_logic;
  signal fork23_outs_3 : std_logic_vector(0 downto 0);
  signal fork23_outs_3_valid : std_logic;
  signal fork23_outs_3_ready : std_logic;
  signal fork23_outs_4 : std_logic_vector(0 downto 0);
  signal fork23_outs_4_valid : std_logic;
  signal fork23_outs_4_ready : std_logic;
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal constant23_outs : std_logic_vector(1 downto 0);
  signal constant23_outs_valid : std_logic;
  signal constant23_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant24_outs : std_logic_vector(5 downto 0);
  signal constant24_outs_valid : std_logic;
  signal constant24_outs_ready : std_logic;
  signal extsi27_outs : std_logic_vector(6 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant25_outs : std_logic_vector(1 downto 0);
  signal constant25_outs_valid : std_logic;
  signal constant25_outs_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(1 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(1 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal extsi28_outs : std_logic_vector(6 downto 0);
  signal extsi28_outs_valid : std_logic;
  signal extsi28_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(31 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant26_outs : std_logic_vector(3 downto 0);
  signal constant26_outs_valid : std_logic;
  signal constant26_outs_ready : std_logic;
  signal extsi12_outs : std_logic_vector(31 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(31 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(31 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal fork26_outs_2 : std_logic_vector(31 downto 0);
  signal fork26_outs_2_valid : std_logic;
  signal fork26_outs_2_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant27_outs : std_logic_vector(2 downto 0);
  signal constant27_outs_valid : std_logic;
  signal constant27_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(31 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(31 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(31 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal fork27_outs_2 : std_logic_vector(31 downto 0);
  signal fork27_outs_2_valid : std_logic;
  signal fork27_outs_2_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer50_outs : std_logic_vector(31 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal buffer51_outs : std_logic_vector(31 downto 0);
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal addi9_result : std_logic_vector(31 downto 0);
  signal addi9_result_valid : std_logic;
  signal addi9_result_ready : std_logic;
  signal buffer54_outs : std_logic_vector(31 downto 0);
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(31 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(31 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(31 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal buffer33_outs : std_logic_vector(31 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal buffer55_outs : std_logic_vector(31 downto 0);
  signal buffer55_outs_valid : std_logic;
  signal buffer55_outs_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(0 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(0 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal cond_br39_trueOut_valid : std_logic;
  signal cond_br39_trueOut_ready : std_logic;
  signal cond_br39_falseOut_valid : std_logic;
  signal cond_br39_falseOut_ready : std_logic;
  signal buffer71_outs : std_logic_vector(0 downto 0);
  signal buffer71_outs_valid : std_logic;
  signal buffer71_outs_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal mux28_outs_valid : std_logic;
  signal mux28_outs_ready : std_logic;
  signal buffer56_outs_valid : std_logic;
  signal buffer56_outs_ready : std_logic;
  signal join0_outs_valid : std_logic;
  signal join0_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(8 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(8 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal load1_addrOut : std_logic_vector(4 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal shli2_result : std_logic_vector(31 downto 0);
  signal shli2_result_valid : std_logic;
  signal shli2_result_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer57_outs : std_logic_vector(31 downto 0);
  signal buffer57_outs_valid : std_logic;
  signal buffer57_outs_ready : std_logic;
  signal buffer58_outs : std_logic_vector(31 downto 0);
  signal buffer58_outs_valid : std_logic;
  signal buffer58_outs_ready : std_logic;
  signal addi10_result : std_logic_vector(31 downto 0);
  signal addi10_result_valid : std_logic;
  signal addi10_result_ready : std_logic;
  signal buffer59_outs : std_logic_vector(31 downto 0);
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal addi4_result : std_logic_vector(31 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal fork30_outs_0 : std_logic_vector(31 downto 0);
  signal fork30_outs_0_valid : std_logic;
  signal fork30_outs_0_ready : std_logic;
  signal fork30_outs_1 : std_logic_vector(31 downto 0);
  signal fork30_outs_1_valid : std_logic;
  signal fork30_outs_1_ready : std_logic;
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal gate2_outs : std_logic_vector(31 downto 0);
  signal gate2_outs_valid : std_logic;
  signal gate2_outs_ready : std_logic;
  signal buffer34_outs : std_logic_vector(31 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal buffer60_outs : std_logic_vector(31 downto 0);
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal cmpi4_result : std_logic_vector(0 downto 0);
  signal cmpi4_result_valid : std_logic;
  signal cmpi4_result_ready : std_logic;
  signal fork31_outs_0 : std_logic_vector(0 downto 0);
  signal fork31_outs_0_valid : std_logic;
  signal fork31_outs_0_ready : std_logic;
  signal fork31_outs_1 : std_logic_vector(0 downto 0);
  signal fork31_outs_1_valid : std_logic;
  signal fork31_outs_1_ready : std_logic;
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal cond_br40_trueOut_valid : std_logic;
  signal cond_br40_trueOut_ready : std_logic;
  signal cond_br40_falseOut_valid : std_logic;
  signal cond_br40_falseOut_ready : std_logic;
  signal buffer80_outs : std_logic_vector(0 downto 0);
  signal buffer80_outs_valid : std_logic;
  signal buffer80_outs_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal mux29_outs_valid : std_logic;
  signal mux29_outs_ready : std_logic;
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal join1_outs_valid : std_logic;
  signal join1_outs_ready : std_logic;
  signal gate3_outs : std_logic_vector(31 downto 0);
  signal gate3_outs_valid : std_logic;
  signal gate3_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(8 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(8 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal subi0_result : std_logic_vector(31 downto 0);
  signal subi0_result_valid : std_logic;
  signal subi0_result_ready : std_logic;
  signal shli4_result : std_logic_vector(31 downto 0);
  signal shli4_result_valid : std_logic;
  signal shli4_result_ready : std_logic;
  signal shli5_result : std_logic_vector(31 downto 0);
  signal shli5_result_valid : std_logic;
  signal shli5_result_ready : std_logic;
  signal buffer62_outs : std_logic_vector(31 downto 0);
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal buffer63_outs : std_logic_vector(31 downto 0);
  signal buffer63_outs_valid : std_logic;
  signal buffer63_outs_ready : std_logic;
  signal addi11_result : std_logic_vector(31 downto 0);
  signal addi11_result_valid : std_logic;
  signal addi11_result_ready : std_logic;
  signal buffer64_outs : std_logic_vector(31 downto 0);
  signal buffer64_outs_valid : std_logic;
  signal buffer64_outs_ready : std_logic;
  signal addi5_result : std_logic_vector(31 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal fork32_outs_0 : std_logic_vector(31 downto 0);
  signal fork32_outs_0_valid : std_logic;
  signal fork32_outs_0_ready : std_logic;
  signal fork32_outs_1 : std_logic_vector(31 downto 0);
  signal fork32_outs_1_valid : std_logic;
  signal fork32_outs_1_ready : std_logic;
  signal trunci4_outs : std_logic_vector(8 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal buffer88_outs : std_logic_vector(31 downto 0);
  signal buffer88_outs_valid : std_logic;
  signal buffer88_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer65_outs_valid : std_logic;
  signal buffer65_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork33_outs_0_valid : std_logic;
  signal fork33_outs_0_ready : std_logic;
  signal fork33_outs_1_valid : std_logic;
  signal fork33_outs_1_ready : std_logic;
  signal init18_outs_valid : std_logic;
  signal init18_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(8 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal buffer42_outs : std_logic_vector(31 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal addi6_result : std_logic_vector(6 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer67_outs : std_logic_vector(6 downto 0);
  signal buffer67_outs_valid : std_logic;
  signal buffer67_outs_ready : std_logic;
  signal fork34_outs_0 : std_logic_vector(6 downto 0);
  signal fork34_outs_0_valid : std_logic;
  signal fork34_outs_0_ready : std_logic;
  signal fork34_outs_1 : std_logic_vector(6 downto 0);
  signal fork34_outs_1_valid : std_logic;
  signal fork34_outs_1_ready : std_logic;
  signal trunci5_outs : std_logic_vector(5 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer68_outs : std_logic_vector(0 downto 0);
  signal buffer68_outs_valid : std_logic;
  signal buffer68_outs_ready : std_logic;
  signal fork35_outs_0 : std_logic_vector(0 downto 0);
  signal fork35_outs_0_valid : std_logic;
  signal fork35_outs_0_ready : std_logic;
  signal fork35_outs_1 : std_logic_vector(0 downto 0);
  signal fork35_outs_1_valid : std_logic;
  signal fork35_outs_1_ready : std_logic;
  signal fork35_outs_2 : std_logic_vector(0 downto 0);
  signal fork35_outs_2_valid : std_logic;
  signal fork35_outs_2_ready : std_logic;
  signal fork35_outs_3 : std_logic_vector(0 downto 0);
  signal fork35_outs_3_valid : std_logic;
  signal fork35_outs_3_ready : std_logic;
  signal fork35_outs_4 : std_logic_vector(0 downto 0);
  signal fork35_outs_4_valid : std_logic;
  signal fork35_outs_4_ready : std_logic;
  signal fork35_outs_5 : std_logic_vector(0 downto 0);
  signal fork35_outs_5_valid : std_logic;
  signal fork35_outs_5_ready : std_logic;
  signal fork35_outs_6 : std_logic_vector(0 downto 0);
  signal fork35_outs_6_valid : std_logic;
  signal fork35_outs_6_ready : std_logic;
  signal fork35_outs_7 : std_logic_vector(0 downto 0);
  signal fork35_outs_7_valid : std_logic;
  signal fork35_outs_7_ready : std_logic;
  signal fork35_outs_8 : std_logic_vector(0 downto 0);
  signal fork35_outs_8_valid : std_logic;
  signal fork35_outs_8_ready : std_logic;
  signal fork35_outs_9 : std_logic_vector(0 downto 0);
  signal fork35_outs_9_valid : std_logic;
  signal fork35_outs_9_ready : std_logic;
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
  signal buffer66_outs : std_logic_vector(31 downto 0);
  signal buffer66_outs_valid : std_logic;
  signal buffer66_outs_ready : std_logic;
  signal cond_br11_trueOut : std_logic_vector(31 downto 0);
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut : std_logic_vector(31 downto 0);
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(5 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(5 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal cond_br13_trueOut : std_logic_vector(5 downto 0);
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut : std_logic_vector(5 downto 0);
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal extsi29_outs : std_logic_vector(6 downto 0);
  signal extsi29_outs_valid : std_logic;
  signal extsi29_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant28_outs : std_logic_vector(1 downto 0);
  signal constant28_outs_valid : std_logic;
  signal constant28_outs_ready : std_logic;
  signal extsi30_outs : std_logic_vector(6 downto 0);
  signal extsi30_outs_valid : std_logic;
  signal extsi30_outs_ready : std_logic;
  signal addi8_result : std_logic_vector(6 downto 0);
  signal addi8_result_valid : std_logic;
  signal addi8_result_ready : std_logic;
  signal buffer69_outs : std_logic_vector(6 downto 0);
  signal buffer69_outs_valid : std_logic;
  signal buffer69_outs_ready : std_logic;
  signal cond_br56_trueOut : std_logic_vector(31 downto 0);
  signal cond_br56_trueOut_valid : std_logic;
  signal cond_br56_trueOut_ready : std_logic;
  signal cond_br56_falseOut : std_logic_vector(31 downto 0);
  signal cond_br56_falseOut_valid : std_logic;
  signal cond_br56_falseOut_ready : std_logic;
  signal cond_br57_trueOut : std_logic_vector(31 downto 0);
  signal cond_br57_trueOut_valid : std_logic;
  signal cond_br57_trueOut_ready : std_logic;
  signal cond_br57_falseOut : std_logic_vector(31 downto 0);
  signal cond_br57_falseOut_valid : std_logic;
  signal cond_br57_falseOut_ready : std_logic;
  signal cond_br58_trueOut_valid : std_logic;
  signal cond_br58_trueOut_ready : std_logic;
  signal cond_br58_falseOut_valid : std_logic;
  signal cond_br58_falseOut_ready : std_logic;
  signal cond_br59_trueOut_valid : std_logic;
  signal cond_br59_trueOut_ready : std_logic;
  signal cond_br59_falseOut_valid : std_logic;
  signal cond_br59_falseOut_ready : std_logic;
  signal cond_br60_trueOut_valid : std_logic;
  signal cond_br60_trueOut_ready : std_logic;
  signal cond_br60_falseOut_valid : std_logic;
  signal cond_br60_falseOut_ready : std_logic;
  signal cond_br61_trueOut_valid : std_logic;
  signal cond_br61_trueOut_ready : std_logic;
  signal cond_br61_falseOut_valid : std_logic;
  signal cond_br61_falseOut_ready : std_logic;
  signal extsi31_outs : std_logic_vector(6 downto 0);
  signal extsi31_outs_valid : std_logic;
  signal extsi31_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant29_outs : std_logic_vector(5 downto 0);
  signal constant29_outs_valid : std_logic;
  signal constant29_outs_ready : std_logic;
  signal extsi32_outs : std_logic_vector(6 downto 0);
  signal extsi32_outs_valid : std_logic;
  signal extsi32_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant30_outs : std_logic_vector(1 downto 0);
  signal constant30_outs_valid : std_logic;
  signal constant30_outs_ready : std_logic;
  signal extsi33_outs : std_logic_vector(6 downto 0);
  signal extsi33_outs_valid : std_logic;
  signal extsi33_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(6 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer70_outs : std_logic_vector(6 downto 0);
  signal buffer70_outs_valid : std_logic;
  signal buffer70_outs_ready : std_logic;
  signal fork36_outs_0 : std_logic_vector(6 downto 0);
  signal fork36_outs_0_valid : std_logic;
  signal fork36_outs_0_ready : std_logic;
  signal fork36_outs_1 : std_logic_vector(6 downto 0);
  signal fork36_outs_1_valid : std_logic;
  signal fork36_outs_1_ready : std_logic;
  signal trunci6_outs : std_logic_vector(5 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer72_outs : std_logic_vector(0 downto 0);
  signal buffer72_outs_valid : std_logic;
  signal buffer72_outs_ready : std_logic;
  signal fork37_outs_0 : std_logic_vector(0 downto 0);
  signal fork37_outs_0_valid : std_logic;
  signal fork37_outs_0_ready : std_logic;
  signal fork37_outs_1 : std_logic_vector(0 downto 0);
  signal fork37_outs_1_valid : std_logic;
  signal fork37_outs_1_ready : std_logic;
  signal fork37_outs_2 : std_logic_vector(0 downto 0);
  signal fork37_outs_2_valid : std_logic;
  signal fork37_outs_2_ready : std_logic;
  signal fork37_outs_3 : std_logic_vector(0 downto 0);
  signal fork37_outs_3_valid : std_logic;
  signal fork37_outs_3_ready : std_logic;
  signal fork37_outs_4 : std_logic_vector(0 downto 0);
  signal fork37_outs_4_valid : std_logic;
  signal fork37_outs_4_ready : std_logic;
  signal fork37_outs_5 : std_logic_vector(0 downto 0);
  signal fork37_outs_5_valid : std_logic;
  signal fork37_outs_5_ready : std_logic;
  signal fork37_outs_6 : std_logic_vector(0 downto 0);
  signal fork37_outs_6_valid : std_logic;
  signal fork37_outs_6_ready : std_logic;
  signal fork37_outs_7 : std_logic_vector(0 downto 0);
  signal fork37_outs_7_valid : std_logic;
  signal fork37_outs_7_ready : std_logic;
  signal fork37_outs_8 : std_logic_vector(0 downto 0);
  signal fork37_outs_8_valid : std_logic;
  signal fork37_outs_8_ready : std_logic;
  signal fork37_outs_9 : std_logic_vector(0 downto 0);
  signal fork37_outs_9_valid : std_logic;
  signal fork37_outs_9_ready : std_logic;
  signal cond_br15_trueOut : std_logic_vector(5 downto 0);
  signal cond_br15_trueOut_valid : std_logic;
  signal cond_br15_trueOut_ready : std_logic;
  signal cond_br15_falseOut : std_logic_vector(5 downto 0);
  signal cond_br15_falseOut_valid : std_logic;
  signal cond_br15_falseOut_ready : std_logic;
  signal cond_br16_trueOut : std_logic_vector(31 downto 0);
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut : std_logic_vector(31 downto 0);
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal cond_br17_trueOut_valid : std_logic;
  signal cond_br17_trueOut_ready : std_logic;
  signal cond_br17_falseOut_valid : std_logic;
  signal cond_br17_falseOut_ready : std_logic;
  signal fork38_outs_0_valid : std_logic;
  signal fork38_outs_0_ready : std_logic;
  signal fork38_outs_1_valid : std_logic;
  signal fork38_outs_1_ready : std_logic;

begin

  out0 <= cond_br16_falseOut;
  out0_valid <= cond_br16_falseOut_valid;
  cond_br16_falseOut_ready <= out0_ready;
  c_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= c_end_ready;
  a_end_valid <= mem_controller1_memEnd_valid;
  mem_controller1_memEnd_ready <= a_end_ready;
  end_valid <= fork0_outs_3_valid;
  fork0_outs_3_ready <= end_ready;
  c_loadEn <= mem_controller2_loadEn;
  c_loadAddr <= mem_controller2_loadAddr;
  c_storeEn <= mem_controller2_storeEn;
  c_storeAddr <= mem_controller2_storeAddr;
  c_storeData <= mem_controller2_storeData;
  a_loadEn <= mem_controller1_loadEn;
  a_loadAddr <= mem_controller1_loadAddr;
  a_storeEn <= mem_controller1_storeEn;
  a_storeAddr <= mem_controller1_storeAddr;
  a_storeData <= mem_controller1_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(9)
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
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready,
      outs_ready(3) => fork0_outs_3_ready,
      outs_ready(4) => fork0_outs_4_ready,
      outs_ready(5) => fork0_outs_5_ready,
      outs_ready(6) => fork0_outs_6_ready,
      outs_ready(7) => fork0_outs_7_ready,
      outs_ready(8) => fork0_outs_8_ready
    );

  mem_controller1 : entity work.mem_controller(arch) generic map(1, 2, 1, 32, 9)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ctrl(0) => extsi8_outs,
      ctrl_valid(0) => extsi8_outs_valid,
      ctrl_ready(0) => extsi8_outs_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr(1) => load2_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_valid(1) => load2_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ldAddr_ready(1) => load2_addrOut_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => fork38_outs_1_valid,
      ctrlEnd_ready => fork38_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller1_ldData_0,
      ldData(1) => mem_controller1_ldData_1,
      ldData_valid(0) => mem_controller1_ldData_0_valid,
      ldData_valid(1) => mem_controller1_ldData_1_valid,
      ldData_ready(0) => mem_controller1_ldData_0_ready,
      ldData_ready(1) => mem_controller1_ldData_1_ready,
      stDone_valid(0) => mem_controller1_stDone_0_valid,
      stDone_ready(0) => mem_controller1_stDone_0_ready,
      memEnd_valid => mem_controller1_memEnd_valid,
      memEnd_ready => mem_controller1_memEnd_ready,
      loadEn => mem_controller1_loadEn,
      loadAddr => mem_controller1_loadAddr,
      storeEn => mem_controller1_storeEn,
      storeAddr => mem_controller1_storeAddr,
      storeData => mem_controller1_storeData
    );

  mem_controller2 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => c_loadData,
      memStart_valid => c_start_valid,
      memStart_ready => c_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork38_outs_0_valid,
      ctrlEnd_ready => fork38_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller2_ldData_0,
      ldData_valid(0) => mem_controller2_ldData_0_valid,
      ldData_ready(0) => mem_controller2_ldData_0_ready,
      memEnd_valid => mem_controller2_memEnd_valid,
      memEnd_ready => mem_controller2_memEnd_ready,
      loadEn => mem_controller2_loadEn,
      loadAddr => mem_controller2_loadAddr,
      storeEn => mem_controller2_storeEn,
      storeAddr => mem_controller2_storeAddr,
      storeData => mem_controller2_storeData
    );

  constant5 : entity work.handshake_constant_0(arch) generic map(11)
    port map(
      ctrl_valid => fork0_outs_2_valid,
      ctrl_ready => fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 11)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
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

  constant6 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  constant19 : entity work.handshake_constant_2(arch) generic map(2)
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

  extsi19 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant6_outs,
      ins_valid => constant6_outs_valid,
      ins_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  mux10 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins_valid(0) => fork0_outs_7_valid,
      ins_valid(1) => cond_br58_trueOut_valid,
      ins_ready(0) => fork0_outs_7_ready,
      ins_ready(1) => cond_br58_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  mux11 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => extsi0_outs,
      ins(1) => cond_br56_trueOut,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => cond_br56_trueOut_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => cond_br56_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux11_outs,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  mux12 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_2,
      index_valid => fork2_outs_2_valid,
      index_ready => fork2_outs_2_ready,
      ins(0) => extsi1_outs,
      ins(1) => cond_br57_trueOut,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => cond_br57_trueOut_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => cond_br57_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux12_outs,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  mux13 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_3,
      index_valid => fork2_outs_3_valid,
      index_ready => fork2_outs_3_ready,
      ins_valid(0) => fork0_outs_6_valid,
      ins_valid(1) => cond_br61_trueOut_valid,
      ins_ready(0) => fork0_outs_6_ready,
      ins_ready(1) => cond_br61_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  mux14 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_4,
      index_valid => fork2_outs_4_valid,
      index_ready => fork2_outs_4_ready,
      ins_valid(0) => fork0_outs_5_valid,
      ins_valid(1) => cond_br60_trueOut_valid,
      ins_ready(0) => fork0_outs_5_ready,
      ins_ready(1) => cond_br60_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux14_outs_valid,
      outs_ready => mux14_outs_ready
    );

  mux15 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_5,
      index_valid => fork2_outs_5_valid,
      index_ready => fork2_outs_5_ready,
      ins_valid(0) => fork0_outs_4_valid,
      ins_valid(1) => cond_br59_trueOut_valid,
      ins_ready(0) => fork0_outs_4_ready,
      ins_ready(1) => cond_br59_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork37_outs_7,
      ins_valid => fork37_outs_7_valid,
      ins_ready => fork37_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(6, 1)
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
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_valid(5) => fork2_outs_5_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready,
      outs_ready(5) => fork2_outs_5_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => extsi18_outs,
      ins(1) => cond_br15_trueOut,
      ins_valid(0) => extsi18_outs_valid,
      ins_valid(1) => cond_br15_trueOut_valid,
      ins_ready(0) => extsi18_outs_ready,
      ins_ready(1) => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer8 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi20 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  buffer73 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br16_trueOut,
      ins_valid => cond_br16_trueOut_valid,
      ins_ready => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer73_outs,
      outs_valid => buffer73_outs_valid,
      outs_ready => buffer73_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => extsi19_outs,
      ins(1) => buffer73_outs,
      ins_valid(0) => extsi19_outs_valid,
      ins_valid(1) => buffer73_outs_valid,
      ins_ready(0) => extsi19_outs_ready,
      ins_ready(1) => buffer73_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_8_valid,
      ins_valid(1) => cond_br17_trueOut_valid,
      ins_ready(0) => fork0_outs_8_ready,
      ins_ready(1) => cond_br17_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant20 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  extsi21 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant20_outs,
      ins_valid => constant20_outs_valid,
      ins_ready => constant20_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi20_outs,
      lhs_valid => extsi20_outs_valid,
      lhs_ready => extsi20_outs_ready,
      rhs => extsi21_outs,
      rhs_valid => extsi21_outs_valid,
      rhs_ready => extsi21_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer10 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer9 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer2 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  mux16 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork5_outs_0,
      index_valid => fork5_outs_0_valid,
      index_ready => fork5_outs_0_ready,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => fork15_outs_0_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux16_outs_valid,
      outs_ready => mux16_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux11_outs,
      ins_valid => mux11_outs_valid,
      ins_ready => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  mux17 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork5_outs_1,
      index_valid => fork5_outs_1_valid,
      index_ready => fork5_outs_1_ready,
      ins(0) => buffer3_outs,
      ins(1) => fork11_outs_0,
      ins_valid(0) => buffer3_outs_valid,
      ins_valid(1) => fork11_outs_0_valid,
      ins_ready(0) => buffer3_outs_ready,
      ins_ready(1) => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux17_outs,
      outs_valid => mux17_outs_valid,
      outs_ready => mux17_outs_ready
    );

  buffer4 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux12_outs,
      ins_valid => mux12_outs_valid,
      ins_ready => mux12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  mux18 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork5_outs_2,
      index_valid => fork5_outs_2_valid,
      index_ready => fork5_outs_2_ready,
      ins(0) => buffer4_outs,
      ins(1) => fork11_outs_1,
      ins_valid(0) => buffer4_outs_valid,
      ins_valid(1) => fork11_outs_1_valid,
      ins_ready(0) => buffer4_outs_ready,
      ins_ready(1) => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux18_outs,
      outs_valid => mux18_outs_valid,
      outs_ready => mux18_outs_ready
    );

  buffer5 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux13_outs_valid,
      ins_ready => mux13_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  mux19 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork5_outs_3,
      index_valid => fork5_outs_3_valid,
      index_ready => fork5_outs_3_ready,
      ins_valid(0) => buffer5_outs_valid,
      ins_valid(1) => fork15_outs_1_valid,
      ins_ready(0) => buffer5_outs_ready,
      ins_ready(1) => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux19_outs_valid,
      outs_ready => mux19_outs_ready
    );

  buffer6 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux14_outs_valid,
      ins_ready => mux14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  mux20 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer21_outs,
      index_valid => buffer21_outs_valid,
      index_ready => buffer21_outs_ready,
      ins_valid(0) => buffer6_outs_valid,
      ins_valid(1) => fork13_outs_0_valid,
      ins_ready(0) => buffer6_outs_ready,
      ins_ready(1) => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux20_outs_valid,
      outs_ready => mux20_outs_ready
    );

  buffer21 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork5_outs_4,
      ins_valid => fork5_outs_4_valid,
      ins_ready => fork5_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  buffer7 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  mux21 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer22_outs,
      index_valid => buffer22_outs_valid,
      index_ready => buffer22_outs_ready,
      ins_valid(0) => buffer7_outs_valid,
      ins_valid(1) => fork13_outs_1_valid,
      ins_ready(0) => buffer7_outs_ready,
      ins_ready(1) => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux21_outs_valid,
      outs_ready => mux21_outs_ready
    );

  buffer22 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork5_outs_5,
      ins_valid => fork5_outs_5_valid,
      ins_ready => fork5_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  init6 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork10_outs_2,
      ins_valid => fork10_outs_2_valid,
      ins_ready => fork10_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => init6_outs,
      outs_valid => init6_outs_valid,
      outs_ready => init6_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => init6_outs,
      ins_valid => init6_outs_valid,
      ins_ready => init6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs(2) => fork5_outs_2,
      outs(3) => fork5_outs_3,
      outs(4) => fork5_outs_4,
      outs(5) => fork5_outs_5,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_valid(3) => fork5_outs_3_valid,
      outs_valid(4) => fork5_outs_4_valid,
      outs_valid(5) => fork5_outs_5_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready,
      outs_ready(3) => fork5_outs_3_ready,
      outs_ready(4) => fork5_outs_4_ready,
      outs_ready(5) => fork5_outs_5_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 7, 1)
    port map(
      index => fork7_outs_1,
      index_valid => fork7_outs_1_valid,
      index_ready => fork7_outs_1_ready,
      ins(0) => buffer10_outs,
      ins(1) => buffer69_outs,
      ins_valid(0) => buffer10_outs_valid,
      ins_valid(1) => buffer69_outs_valid,
      ins_ready(0) => buffer10_outs_ready,
      ins_ready(1) => buffer69_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer25 : entity work.tehb(arch) generic map(7)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  trunci0 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork7_outs_2,
      index_valid => fork7_outs_2_valid,
      index_ready => fork7_outs_2_ready,
      ins(0) => buffer9_outs,
      ins(1) => cond_br11_falseOut,
      ins_valid(0) => buffer9_outs_valid,
      ins_valid(1) => cond_br11_falseOut_valid,
      ins_ready(0) => buffer9_outs_ready,
      ins_ready(1) => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork7_outs_0,
      index_valid => fork7_outs_0_valid,
      index_ready => fork7_outs_0_ready,
      ins(0) => fork3_outs_0,
      ins(1) => cond_br12_falseOut,
      ins_valid(0) => fork3_outs_0_valid,
      ins_valid(1) => cond_br12_falseOut_valid,
      ins_ready(0) => fork3_outs_0_ready,
      ins_ready(1) => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => control_merge0_outs_valid,
      ins_valid(1) => cond_br14_falseOut_valid,
      ins_ready(0) => control_merge0_outs_ready,
      ins_ready(1) => cond_br14_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready
    );

  buffer30 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  fork8 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer30_outs_valid,
      ins_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant21 : entity work.handshake_constant_3(arch) generic map(6)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  extsi22 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant21_outs,
      ins_valid => constant21_outs_valid,
      ins_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  constant22 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork8_outs_0_valid,
      ctrl_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => extsi22_outs,
      rhs_valid => extsi22_outs_valid,
      rhs_ready => extsi22_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer31 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(13, 1)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs(2) => fork10_outs_2,
      outs(3) => fork10_outs_3,
      outs(4) => fork10_outs_4,
      outs(5) => fork10_outs_5,
      outs(6) => fork10_outs_6,
      outs(7) => fork10_outs_7,
      outs(8) => fork10_outs_8,
      outs(9) => fork10_outs_9,
      outs(10) => fork10_outs_10,
      outs(11) => fork10_outs_11,
      outs(12) => fork10_outs_12,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_valid(2) => fork10_outs_2_valid,
      outs_valid(3) => fork10_outs_3_valid,
      outs_valid(4) => fork10_outs_4_valid,
      outs_valid(5) => fork10_outs_5_valid,
      outs_valid(6) => fork10_outs_6_valid,
      outs_valid(7) => fork10_outs_7_valid,
      outs_valid(8) => fork10_outs_8_valid,
      outs_valid(9) => fork10_outs_9_valid,
      outs_valid(10) => fork10_outs_10_valid,
      outs_valid(11) => fork10_outs_11_valid,
      outs_valid(12) => fork10_outs_12_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready,
      outs_ready(2) => fork10_outs_2_ready,
      outs_ready(3) => fork10_outs_3_ready,
      outs_ready(4) => fork10_outs_4_ready,
      outs_ready(5) => fork10_outs_5_ready,
      outs_ready(6) => fork10_outs_6_ready,
      outs_ready(7) => fork10_outs_7_ready,
      outs_ready(8) => fork10_outs_8_ready,
      outs_ready(9) => fork10_outs_9_ready,
      outs_ready(10) => fork10_outs_10_ready,
      outs_ready(11) => fork10_outs_11_ready,
      outs_ready(12) => fork10_outs_12_ready
    );

  cond_br3 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork10_outs_12,
      condition_valid => fork10_outs_12_valid,
      condition_ready => fork10_outs_12_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(2)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst
    );

  extsi17 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  cond_br4 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork10_outs_11,
      condition_valid => fork10_outs_11_valid,
      condition_ready => fork10_outs_11_ready,
      data => fork9_outs_1,
      data_valid => fork9_outs_1_valid,
      data_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(2)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst
    );

  extsi23 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => cond_br4_trueOut,
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  buffer26 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  buffer27 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  cond_br5 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork10_outs_3,
      condition_valid => fork10_outs_3_valid,
      condition_ready => fork10_outs_3_ready,
      data => buffer27_outs,
      data_valid => buffer27_outs_valid,
      data_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer28 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  buffer29 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer28_outs,
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork10_outs_1,
      condition_valid => fork10_outs_1_valid,
      condition_ready => fork10_outs_1_ready,
      data => buffer29_outs,
      data_valid => buffer29_outs_valid,
      data_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork10_outs_0,
      condition_valid => fork10_outs_0_valid,
      condition_ready => fork10_outs_0_ready,
      data => trunci0_outs,
      data_valid => trunci0_outs_valid,
      data_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br8 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork10_outs_4,
      condition_valid => fork10_outs_4_valid,
      condition_ready => fork10_outs_4_ready,
      data_valid => fork8_outs_1_valid,
      data_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  buffer13 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux17_outs,
      ins_valid => mux17_outs_valid,
      ins_ready => mux17_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  buffer14 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  cond_br47 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork10_outs_5,
      condition_valid => fork10_outs_5_valid,
      condition_ready => fork10_outs_5_ready,
      data => buffer14_outs,
      data_valid => buffer14_outs_valid,
      data_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br47_trueOut,
      trueOut_valid => cond_br47_trueOut_valid,
      trueOut_ready => cond_br47_trueOut_ready,
      falseOut => cond_br47_falseOut,
      falseOut_valid => cond_br47_falseOut_valid,
      falseOut_ready => cond_br47_falseOut_ready
    );

  buffer11 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux16_outs_valid,
      ins_ready => mux16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  cond_br48 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork10_outs_6,
      condition_valid => fork10_outs_6_valid,
      condition_ready => fork10_outs_6_ready,
      data_valid => buffer12_outs_valid,
      data_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br48_trueOut_valid,
      trueOut_ready => cond_br48_trueOut_ready,
      falseOut_valid => cond_br48_falseOut_valid,
      falseOut_ready => cond_br48_falseOut_ready
    );

  buffer15 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux18_outs,
      ins_valid => mux18_outs_valid,
      ins_ready => mux18_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  buffer16 : entity work.tehb(arch) generic map(32)
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

  cond_br49 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork10_outs_7,
      condition_valid => fork10_outs_7_valid,
      condition_ready => fork10_outs_7_ready,
      data => buffer16_outs,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br49_trueOut,
      trueOut_valid => cond_br49_trueOut_valid,
      trueOut_ready => cond_br49_trueOut_ready,
      falseOut => cond_br49_falseOut,
      falseOut_valid => cond_br49_falseOut_valid,
      falseOut_ready => cond_br49_falseOut_ready
    );

  buffer23 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux21_outs_valid,
      ins_ready => mux21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  buffer24 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer23_outs_valid,
      ins_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  cond_br50 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer40_outs,
      condition_valid => buffer40_outs_valid,
      condition_ready => buffer40_outs_ready,
      data_valid => buffer24_outs_valid,
      data_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br50_trueOut_valid,
      trueOut_ready => cond_br50_trueOut_ready,
      falseOut_valid => cond_br50_falseOut_valid,
      falseOut_ready => cond_br50_falseOut_ready
    );

  buffer40 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork10_outs_8,
      ins_valid => fork10_outs_8_valid,
      ins_ready => fork10_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  cond_br51 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork35_outs_6,
      condition_valid => fork35_outs_6_valid,
      condition_ready => fork35_outs_6_ready,
      data => buffer0_outs,
      data_valid => buffer0_outs_valid,
      data_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br51_trueOut,
      trueOut_valid => cond_br51_trueOut_valid,
      trueOut_ready => cond_br51_trueOut_ready,
      falseOut => cond_br51_falseOut,
      falseOut_valid => cond_br51_falseOut_valid,
      falseOut_ready => cond_br51_falseOut_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br51_falseOut,
      ins_valid => cond_br51_falseOut_valid,
      ins_ready => cond_br51_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br51_trueOut,
      ins_valid => cond_br51_trueOut_valid,
      ins_ready => cond_br51_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  buffer17 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux19_outs_valid,
      ins_ready => mux19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer18 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  cond_br52 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork10_outs_9,
      condition_valid => fork10_outs_9_valid,
      condition_ready => fork10_outs_9_ready,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br52_trueOut_valid,
      trueOut_ready => cond_br52_trueOut_ready,
      falseOut_valid => cond_br52_falseOut_valid,
      falseOut_ready => cond_br52_falseOut_ready
    );

  cond_br53 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer43_outs,
      condition_valid => buffer43_outs_valid,
      condition_ready => buffer43_outs_ready,
      data_valid => fork33_outs_1_valid,
      data_ready => fork33_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br53_trueOut_valid,
      trueOut_ready => cond_br53_trueOut_ready,
      falseOut_valid => cond_br53_falseOut_valid,
      falseOut_ready => cond_br53_falseOut_ready
    );

  buffer43 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork35_outs_5,
      ins_valid => fork35_outs_5_valid,
      ins_ready => fork35_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  fork13 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br53_falseOut_valid,
      ins_ready => cond_br53_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  fork14 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br53_trueOut_valid,
      ins_ready => cond_br53_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  buffer19 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux20_outs_valid,
      ins_ready => mux20_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer20 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cond_br54 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer44_outs,
      condition_valid => buffer44_outs_valid,
      condition_ready => buffer44_outs_ready,
      data_valid => buffer20_outs_valid,
      data_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br54_trueOut_valid,
      trueOut_ready => cond_br54_trueOut_ready,
      falseOut_valid => cond_br54_falseOut_valid,
      falseOut_ready => cond_br54_falseOut_ready
    );

  buffer44 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork10_outs_10,
      ins_valid => fork10_outs_10_valid,
      ins_ready => fork10_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  cond_br55 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork35_outs_4,
      condition_valid => fork35_outs_4_valid,
      condition_ready => fork35_outs_4_ready,
      data_valid => init18_outs_valid,
      data_ready => init18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br55_trueOut_valid,
      trueOut_ready => cond_br55_trueOut_ready,
      falseOut_valid => cond_br55_falseOut_valid,
      falseOut_ready => cond_br55_falseOut_ready
    );

  fork15 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br55_falseOut_valid,
      ins_ready => cond_br55_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  fork16 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br55_trueOut_valid,
      ins_ready => cond_br55_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  mux22 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork17_outs_0,
      index_valid => fork17_outs_0_valid,
      index_ready => fork17_outs_0_ready,
      ins_valid(0) => cond_br48_trueOut_valid,
      ins_valid(1) => fork16_outs_1_valid,
      ins_ready(0) => cond_br48_trueOut_ready,
      ins_ready(1) => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux22_outs_valid,
      outs_ready => mux22_outs_ready
    );

  mux23 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork17_outs_1,
      index_valid => fork17_outs_1_valid,
      index_ready => fork17_outs_1_ready,
      ins(0) => cond_br47_trueOut,
      ins(1) => fork12_outs_1,
      ins_valid(0) => cond_br47_trueOut_valid,
      ins_valid(1) => fork12_outs_1_valid,
      ins_ready(0) => cond_br47_trueOut_ready,
      ins_ready(1) => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux23_outs,
      outs_valid => mux23_outs_valid,
      outs_ready => mux23_outs_ready
    );

  mux24 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork17_outs_2,
      index_valid => fork17_outs_2_valid,
      index_ready => fork17_outs_2_ready,
      ins(0) => cond_br49_trueOut,
      ins(1) => fork12_outs_0,
      ins_valid(0) => cond_br49_trueOut_valid,
      ins_valid(1) => fork12_outs_0_valid,
      ins_ready(0) => cond_br49_trueOut_ready,
      ins_ready(1) => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux24_outs,
      outs_valid => mux24_outs_valid,
      outs_ready => mux24_outs_ready
    );

  mux25 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork17_outs_3,
      index_valid => fork17_outs_3_valid,
      index_ready => fork17_outs_3_ready,
      ins_valid(0) => cond_br52_trueOut_valid,
      ins_valid(1) => fork16_outs_0_valid,
      ins_ready(0) => cond_br52_trueOut_ready,
      ins_ready(1) => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux25_outs_valid,
      outs_ready => mux25_outs_ready
    );

  mux26 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer52_outs,
      index_valid => buffer52_outs_valid,
      index_ready => buffer52_outs_ready,
      ins_valid(0) => cond_br54_trueOut_valid,
      ins_valid(1) => fork14_outs_1_valid,
      ins_ready(0) => cond_br54_trueOut_ready,
      ins_ready(1) => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux26_outs_valid,
      outs_ready => mux26_outs_ready
    );

  buffer52 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork17_outs_4,
      ins_valid => fork17_outs_4_valid,
      ins_ready => fork17_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer52_outs,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  mux27 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer53_outs,
      index_valid => buffer53_outs_valid,
      index_ready => buffer53_outs_ready,
      ins_valid(0) => cond_br50_trueOut_valid,
      ins_valid(1) => fork14_outs_0_valid,
      ins_ready(0) => cond_br50_trueOut_ready,
      ins_ready(1) => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux27_outs_valid,
      outs_ready => mux27_outs_ready
    );

  buffer53 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork17_outs_5,
      ins_valid => fork17_outs_5_valid,
      ins_ready => fork17_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer53_outs,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  init12 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork35_outs_3,
      ins_valid => fork35_outs_3_valid,
      ins_ready => fork35_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => init12_outs,
      outs_valid => init12_outs_valid,
      outs_ready => init12_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => init12_outs,
      ins_valid => init12_outs_valid,
      ins_ready => init12_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs(3) => fork17_outs_3,
      outs(4) => fork17_outs_4,
      outs(5) => fork17_outs_5,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_valid(3) => fork17_outs_3_valid,
      outs_valid(4) => fork17_outs_4_valid,
      outs_valid(5) => fork17_outs_5_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready,
      outs_ready(3) => fork17_outs_3_ready,
      outs_ready(4) => fork17_outs_4_ready,
      outs_ready(5) => fork17_outs_5_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork23_outs_2,
      index_valid => fork23_outs_2_valid,
      index_ready => fork23_outs_2_ready,
      ins(0) => extsi17_outs,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => extsi17_outs_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => extsi17_outs_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer38 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  extsi24 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => buffer38_outs,
      ins_valid => buffer38_outs_valid,
      ins_ready => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi24_outs,
      outs_valid => extsi24_outs_valid,
      outs_ready => extsi24_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork23_outs_3,
      index_valid => fork23_outs_3_valid,
      index_ready => fork23_outs_3_ready,
      ins(0) => extsi23_outs,
      ins(1) => cond_br10_trueOut,
      ins_valid(0) => extsi23_outs_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => extsi23_outs_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer39 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  buffer41 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer39_outs,
      ins_valid => buffer39_outs_valid,
      ins_ready => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => buffer41_outs,
      ins_valid => buffer41_outs_valid,
      ins_ready => buffer41_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs(2) => fork18_outs_2,
      outs(3) => fork18_outs_3,
      outs(4) => fork18_outs_4,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_valid(2) => fork18_outs_2_valid,
      outs_valid(3) => fork18_outs_3_valid,
      outs_valid(4) => fork18_outs_4_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready,
      outs_ready(2) => fork18_outs_2_ready,
      outs_ready(3) => fork18_outs_3_ready,
      outs_ready(4) => fork18_outs_4_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork23_outs_4,
      index_valid => fork23_outs_4_valid,
      index_ready => fork23_outs_4_ready,
      ins(0) => cond_br5_trueOut,
      ins(1) => cond_br11_trueOut,
      ins_valid(0) => cond_br5_trueOut_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => cond_br5_trueOut_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork23_outs_0,
      index_valid => fork23_outs_0_valid,
      index_ready => fork23_outs_0_ready,
      ins(0) => cond_br6_trueOut,
      ins(1) => cond_br12_trueOut,
      ins_valid(0) => cond_br6_trueOut_valid,
      ins_valid(1) => cond_br12_trueOut_valid,
      ins_ready(0) => cond_br6_trueOut_ready,
      ins_ready(1) => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  buffer45 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  buffer46 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer45_outs,
      ins_valid => buffer45_outs_valid,
      ins_ready => buffer45_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer46_outs,
      ins_valid => buffer46_outs_valid,
      ins_ready => buffer46_outs_ready,
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

  extsi25 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork19_outs_2,
      ins_valid => fork19_outs_2_valid,
      ins_ready => fork19_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi25_outs,
      ins_valid => extsi25_outs_valid,
      ins_ready => extsi25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  trunci1 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork23_outs_1,
      index_valid => fork23_outs_1_valid,
      index_ready => fork23_outs_1_ready,
      ins(0) => cond_br7_trueOut,
      ins(1) => cond_br13_trueOut,
      ins_valid(0) => cond_br7_trueOut_valid,
      ins_valid(1) => cond_br13_trueOut_valid,
      ins_ready(0) => cond_br7_trueOut_ready,
      ins_ready(1) => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  buffer47 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  buffer48 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer47_outs,
      ins_valid => buffer47_outs_valid,
      ins_ready => buffer47_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer48_outs,
      ins_valid => buffer48_outs_valid,
      ins_ready => buffer48_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  extsi26 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork21_outs_1,
      ins_valid => fork21_outs_1_valid,
      ins_ready => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  fork22 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi26_outs,
      ins_valid => extsi26_outs_valid,
      ins_ready => extsi26_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs(2) => fork22_outs_2,
      outs(3) => fork22_outs_3,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_valid(2) => fork22_outs_2_valid,
      outs_valid(3) => fork22_outs_3_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready,
      outs_ready(2) => fork22_outs_2_ready,
      outs_ready(3) => fork22_outs_3_ready
    );

  control_merge2 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => cond_br8_trueOut_valid,
      ins_valid(1) => cond_br14_trueOut_valid,
      ins_ready(0) => cond_br8_trueOut_ready,
      ins_ready(1) => cond_br14_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs(2) => fork23_outs_2,
      outs(3) => fork23_outs_3,
      outs(4) => fork23_outs_4,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_valid(2) => fork23_outs_2_valid,
      outs_valid(3) => fork23_outs_3_valid,
      outs_valid(4) => fork23_outs_4_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready,
      outs_ready(2) => fork23_outs_2_ready,
      outs_ready(3) => fork23_outs_3_ready,
      outs_ready(4) => fork23_outs_4_ready
    );

  buffer49 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  fork24 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer49_outs_valid,
      ins_ready => buffer49_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  constant23 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork24_outs_0_valid,
      ctrl_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant23_outs,
      outs_valid => constant23_outs_valid,
      outs_ready => constant23_outs_ready
    );

  extsi8 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant23_outs,
      ins_valid => constant23_outs_valid,
      ins_ready => constant23_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant24 : entity work.handshake_constant_4(arch) generic map(6)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant24_outs,
      outs_valid => constant24_outs_valid,
      outs_ready => constant24_outs_ready
    );

  extsi27 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant24_outs,
      ins_valid => constant24_outs_valid,
      ins_ready => constant24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi27_outs,
      outs_valid => extsi27_outs_valid,
      outs_ready => extsi27_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant25 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant25_outs,
      outs_valid => constant25_outs_valid,
      outs_ready => constant25_outs_ready
    );

  fork25 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant25_outs,
      ins_valid => constant25_outs_valid,
      ins_ready => constant25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

  extsi28 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => fork25_outs_0,
      ins_valid => fork25_outs_0_valid,
      ins_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi28_outs,
      outs_valid => extsi28_outs_valid,
      outs_ready => extsi28_outs_ready
    );

  extsi10 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => fork25_outs_1,
      ins_valid => fork25_outs_1_valid,
      ins_ready => fork25_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant26 : entity work.handshake_constant_5(arch) generic map(4)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant26_outs,
      outs_valid => constant26_outs_valid,
      outs_ready => constant26_outs_ready
    );

  extsi12 : entity work.extsi(arch) generic map(4, 32)
    port map(
      ins => constant26_outs,
      ins_valid => constant26_outs_valid,
      ins_ready => constant26_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  fork26 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi12_outs,
      ins_valid => extsi12_outs_valid,
      ins_ready => extsi12_outs_ready,
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

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant27 : entity work.handshake_constant_6(arch) generic map(3)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant27_outs,
      outs_valid => constant27_outs_valid,
      outs_ready => constant27_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant27_outs,
      ins_valid => constant27_outs_valid,
      ins_ready => constant27_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  fork27 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi13_outs,
      ins_valid => extsi13_outs_valid,
      ins_ready => extsi13_outs_ready,
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

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_0,
      lhs_valid => fork22_outs_0_valid,
      lhs_ready => fork22_outs_0_ready,
      rhs => fork27_outs_0,
      rhs_valid => fork27_outs_0_valid,
      rhs_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_1,
      lhs_valid => fork22_outs_1_valid,
      lhs_ready => fork22_outs_1_ready,
      rhs => fork26_outs_0,
      rhs_valid => fork26_outs_0_valid,
      rhs_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer50 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  buffer51 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer51_outs,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  addi9 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer50_outs,
      lhs_valid => buffer50_outs_valid,
      lhs_ready => buffer50_outs_ready,
      rhs => buffer51_outs,
      rhs_valid => buffer51_outs_valid,
      rhs_ready => buffer51_outs_ready,
      clk => clk,
      rst => rst,
      result => addi9_result,
      result_valid => addi9_result_valid,
      result_ready => addi9_result_ready
    );

  buffer54 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi9_result,
      ins_valid => addi9_result_valid,
      ins_ready => addi9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer54_outs,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork18_outs_4,
      lhs_valid => fork18_outs_4_valid,
      lhs_ready => fork18_outs_4_ready,
      rhs => buffer54_outs,
      rhs_valid => buffer54_outs_valid,
      rhs_ready => buffer54_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  fork28 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready
    );

  buffer32 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux22_outs_valid,
      ins_ready => mux22_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork28_outs_1,
      ins_valid(0) => fork28_outs_1_valid,
      ins_valid(1) => buffer32_outs_valid,
      ins_ready(0) => fork28_outs_1_ready,
      ins_ready(1) => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  buffer33 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux23_outs,
      ins_valid => mux23_outs_valid,
      ins_ready => mux23_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  buffer55 : entity work.oehb(arch) generic map(32)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer55_outs,
      outs_valid => buffer55_outs_valid,
      outs_ready => buffer55_outs_ready
    );

  cmpi3 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => buffer55_outs,
      lhs_valid => buffer55_outs_valid,
      lhs_ready => buffer55_outs_ready,
      rhs => buffer33_outs,
      rhs_valid => buffer33_outs_valid,
      rhs_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork29 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
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
      ins_valid => mux26_outs_valid,
      ins_ready => mux26_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  cond_br39 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer71_outs,
      condition_valid => buffer71_outs_valid,
      condition_ready => buffer71_outs_ready,
      data_valid => buffer36_outs_valid,
      data_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br39_trueOut_valid,
      trueOut_ready => cond_br39_trueOut_ready,
      falseOut_valid => cond_br39_falseOut_valid,
      falseOut_ready => cond_br39_falseOut_ready
    );

  buffer71 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork29_outs_1,
      ins_valid => fork29_outs_1_valid,
      ins_ready => fork29_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer71_outs,
      outs_valid => buffer71_outs_valid,
      outs_ready => buffer71_outs_ready
    );

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br39_trueOut_valid,
      ins_ready => cond_br39_trueOut_ready,
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
      index => fork29_outs_0,
      index_valid => fork29_outs_0_valid,
      index_ready => fork29_outs_0_ready,
      ins_valid(0) => cond_br39_falseOut_valid,
      ins_valid(1) => source10_outs_valid,
      ins_ready(0) => cond_br39_falseOut_ready,
      ins_ready(1) => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux28_outs_valid,
      outs_ready => mux28_outs_ready
    );

  buffer56 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux28_outs_valid,
      ins_ready => mux28_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer56_outs_valid,
      outs_ready => buffer56_outs_ready
    );

  join0 : entity work.join_handshake(arch) generic map(1)
    port map(
      ins_valid(0) => buffer56_outs_valid,
      ins_ready(0) => buffer56_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join0_outs_valid,
      outs_ready => join0_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork28_outs_0,
      ins_valid(0) => fork28_outs_0_valid,
      ins_valid(1) => join0_outs_valid,
      ins_ready(0) => fork28_outs_0_ready,
      ins_ready(1) => join0_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => gate1_outs,
      ins_valid => gate1_outs_valid,
      ins_ready => gate1_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataFromMem => mem_controller1_ldData_0,
      dataFromMem_valid => mem_controller1_ldData_0_valid,
      dataFromMem_ready => mem_controller1_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  load1 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller2_ldData_0,
      dataFromMem_valid => mem_controller2_ldData_0_valid,
      dataFromMem_ready => mem_controller2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  shli2 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork20_outs_0,
      lhs_valid => fork20_outs_0_valid,
      lhs_ready => fork20_outs_0_ready,
      rhs => fork27_outs_1,
      rhs_valid => fork27_outs_1_valid,
      rhs_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork20_outs_1,
      lhs_valid => fork20_outs_1_valid,
      lhs_ready => fork20_outs_1_ready,
      rhs => fork26_outs_1,
      rhs_valid => fork26_outs_1_valid,
      rhs_ready => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer57 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer57_outs,
      outs_valid => buffer57_outs_valid,
      outs_ready => buffer57_outs_ready
    );

  buffer58 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer58_outs,
      outs_valid => buffer58_outs_valid,
      outs_ready => buffer58_outs_ready
    );

  addi10 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer57_outs,
      lhs_valid => buffer57_outs_valid,
      lhs_ready => buffer57_outs_ready,
      rhs => buffer58_outs,
      rhs_valid => buffer58_outs_valid,
      rhs_ready => buffer58_outs_ready,
      clk => clk,
      rst => rst,
      result => addi10_result,
      result_valid => addi10_result_valid,
      result_ready => addi10_result_ready
    );

  buffer59 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi10_result,
      ins_valid => addi10_result_valid,
      ins_ready => addi10_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer59_outs,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  addi4 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork18_outs_3,
      lhs_valid => fork18_outs_3_valid,
      lhs_ready => fork18_outs_3_ready,
      rhs => buffer59_outs,
      rhs_valid => buffer59_outs_valid,
      rhs_ready => buffer59_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  fork30 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork30_outs_0,
      outs(1) => fork30_outs_1,
      outs_valid(0) => fork30_outs_0_valid,
      outs_valid(1) => fork30_outs_1_valid,
      outs_ready(0) => fork30_outs_0_ready,
      outs_ready(1) => fork30_outs_1_ready
    );

  buffer35 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux25_outs_valid,
      ins_ready => mux25_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  gate2 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork30_outs_1,
      ins_valid(0) => fork30_outs_1_valid,
      ins_valid(1) => buffer35_outs_valid,
      ins_ready(0) => fork30_outs_1_ready,
      ins_ready(1) => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate2_outs,
      outs_valid => gate2_outs_valid,
      outs_ready => gate2_outs_ready
    );

  buffer34 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux24_outs,
      ins_valid => mux24_outs_valid,
      ins_ready => mux24_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  buffer60 : entity work.oehb(arch) generic map(32)
    port map(
      ins => gate2_outs,
      ins_valid => gate2_outs_valid,
      ins_ready => gate2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer60_outs,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  cmpi4 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => buffer60_outs,
      lhs_valid => buffer60_outs_valid,
      lhs_ready => buffer60_outs_ready,
      rhs => buffer34_outs,
      rhs_valid => buffer34_outs_valid,
      rhs_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi4_result,
      result_valid => cmpi4_result_valid,
      result_ready => cmpi4_result_ready
    );

  fork31 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi4_result,
      ins_valid => cmpi4_result_valid,
      ins_ready => cmpi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork31_outs_0,
      outs(1) => fork31_outs_1,
      outs_valid(0) => fork31_outs_0_valid,
      outs_valid(1) => fork31_outs_1_valid,
      outs_ready(0) => fork31_outs_0_ready,
      outs_ready(1) => fork31_outs_1_ready
    );

  buffer37 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux27_outs_valid,
      ins_ready => mux27_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  cond_br40 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer80_outs,
      condition_valid => buffer80_outs_valid,
      condition_ready => buffer80_outs_ready,
      data_valid => buffer37_outs_valid,
      data_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br40_trueOut_valid,
      trueOut_ready => cond_br40_trueOut_ready,
      falseOut_valid => cond_br40_falseOut_valid,
      falseOut_ready => cond_br40_falseOut_ready
    );

  buffer80 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork31_outs_1,
      ins_valid => fork31_outs_1_valid,
      ins_ready => fork31_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer80_outs,
      outs_valid => buffer80_outs_valid,
      outs_ready => buffer80_outs_ready
    );

  sink4 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br40_trueOut_valid,
      ins_ready => cond_br40_trueOut_ready,
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

  mux29 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork31_outs_0,
      index_valid => fork31_outs_0_valid,
      index_ready => fork31_outs_0_ready,
      ins_valid(0) => cond_br40_falseOut_valid,
      ins_valid(1) => source11_outs_valid,
      ins_ready(0) => cond_br40_falseOut_ready,
      ins_ready(1) => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux29_outs_valid,
      outs_ready => mux29_outs_ready
    );

  buffer61 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux29_outs_valid,
      ins_ready => mux29_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  join1 : entity work.join_handshake(arch) generic map(1)
    port map(
      ins_valid(0) => buffer61_outs_valid,
      ins_ready(0) => buffer61_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join1_outs_valid,
      outs_ready => join1_outs_ready
    );

  gate3 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork30_outs_0,
      ins_valid(0) => fork30_outs_0_valid,
      ins_valid(1) => join1_outs_valid,
      ins_ready(0) => fork30_outs_0_ready,
      ins_ready(1) => join1_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate3_outs,
      outs_valid => gate3_outs_valid,
      outs_ready => gate3_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => gate3_outs,
      ins_valid => gate3_outs_valid,
      ins_ready => gate3_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => trunci3_outs,
      addrIn_valid => trunci3_outs_valid,
      addrIn_ready => trunci3_outs_ready,
      dataFromMem => mem_controller1_ldData_1,
      dataFromMem_valid => mem_controller1_ldData_1_valid,
      dataFromMem_ready => mem_controller1_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load1_dataOut,
      lhs_valid => load1_dataOut_valid,
      lhs_ready => load1_dataOut_ready,
      rhs => load2_dataOut,
      rhs_valid => load2_dataOut_valid,
      rhs_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  subi0 : entity work.subi(arch) generic map(32)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => muli0_result,
      rhs_valid => muli0_result_valid,
      rhs_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      result => subi0_result,
      result_valid => subi0_result_valid,
      result_ready => subi0_result_ready
    );

  shli4 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_2,
      lhs_valid => fork22_outs_2_valid,
      lhs_ready => fork22_outs_2_ready,
      rhs => fork27_outs_2,
      rhs_valid => fork27_outs_2_valid,
      rhs_ready => fork27_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli4_result,
      result_valid => shli4_result_valid,
      result_ready => shli4_result_ready
    );

  shli5 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_3,
      lhs_valid => fork22_outs_3_valid,
      lhs_ready => fork22_outs_3_ready,
      rhs => fork26_outs_2,
      rhs_valid => fork26_outs_2_valid,
      rhs_ready => fork26_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli5_result,
      result_valid => shli5_result_valid,
      result_ready => shli5_result_ready
    );

  buffer62 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli4_result,
      ins_valid => shli4_result_valid,
      ins_ready => shli4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer62_outs,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  buffer63 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli5_result,
      ins_valid => shli5_result_valid,
      ins_ready => shli5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer63_outs,
      outs_valid => buffer63_outs_valid,
      outs_ready => buffer63_outs_ready
    );

  addi11 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer62_outs,
      lhs_valid => buffer62_outs_valid,
      lhs_ready => buffer62_outs_ready,
      rhs => buffer63_outs,
      rhs_valid => buffer63_outs_valid,
      rhs_ready => buffer63_outs_ready,
      clk => clk,
      rst => rst,
      result => addi11_result,
      result_valid => addi11_result_valid,
      result_ready => addi11_result_ready
    );

  buffer64 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi11_result,
      ins_valid => addi11_result_valid,
      ins_ready => addi11_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer64_outs,
      outs_valid => buffer64_outs_valid,
      outs_ready => buffer64_outs_ready
    );

  addi5 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork18_outs_2,
      lhs_valid => fork18_outs_2_valid,
      lhs_ready => fork18_outs_2_ready,
      rhs => buffer64_outs,
      rhs_valid => buffer64_outs_valid,
      rhs_ready => buffer64_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  fork32 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork32_outs_0,
      outs(1) => fork32_outs_1,
      outs_valid(0) => fork32_outs_0_valid,
      outs_valid(1) => fork32_outs_1_valid,
      outs_ready(0) => fork32_outs_0_ready,
      outs_ready(1) => fork32_outs_1_ready
    );

  trunci4 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer88_outs,
      ins_valid => buffer88_outs_valid,
      ins_ready => buffer88_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  buffer88 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork32_outs_0,
      ins_valid => fork32_outs_0_valid,
      ins_ready => fork32_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer88_outs,
      outs_valid => buffer88_outs_valid,
      outs_ready => buffer88_outs_ready
    );

  buffer0 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork32_outs_1,
      ins_valid => fork32_outs_1_valid,
      ins_ready => fork32_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  buffer65 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => store0_doneOut_valid,
      ins_ready => store0_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer65_outs_valid,
      outs_ready => buffer65_outs_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer65_outs_valid,
      ins_ready => buffer65_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork33 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork33_outs_0_valid,
      outs_valid(1) => fork33_outs_1_valid,
      outs_ready(0) => fork33_outs_0_ready,
      outs_ready(1) => fork33_outs_1_ready
    );

  init18 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork33_outs_0_valid,
      ins_ready => fork33_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init18_outs_valid,
      outs_ready => init18_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 9)
    port map(
      addrIn => trunci4_outs,
      addrIn_valid => trunci4_outs_valid,
      addrIn_ready => trunci4_outs_ready,
      dataIn => subi0_result,
      dataIn_valid => subi0_result_valid,
      dataIn_ready => subi0_result_ready,
      doneFromMem_valid => mem_controller1_stDone_0_valid,
      doneFromMem_ready => mem_controller1_stDone_0_ready,
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

  buffer42 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer42_outs,
      lhs_valid => buffer42_outs_valid,
      lhs_ready => buffer42_outs_ready,
      rhs => fork18_outs_1,
      rhs_valid => fork18_outs_1_valid,
      rhs_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork18_outs_0,
      lhs_valid => fork18_outs_0_valid,
      lhs_ready => fork18_outs_0_ready,
      rhs => extsi10_outs,
      rhs_valid => extsi10_outs_valid,
      rhs_ready => extsi10_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  addi6 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi24_outs,
      lhs_valid => extsi24_outs_valid,
      lhs_ready => extsi24_outs_ready,
      rhs => extsi28_outs,
      rhs_valid => extsi28_outs_valid,
      rhs_ready => extsi28_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer67 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer67_outs,
      outs_valid => buffer67_outs_valid,
      outs_ready => buffer67_outs_ready
    );

  fork34 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer67_outs,
      ins_valid => buffer67_outs_valid,
      ins_ready => buffer67_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork34_outs_0,
      outs(1) => fork34_outs_1,
      outs_valid(0) => fork34_outs_0_valid,
      outs_valid(1) => fork34_outs_1_valid,
      outs_ready(0) => fork34_outs_0_ready,
      outs_ready(1) => fork34_outs_1_ready
    );

  trunci5 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork34_outs_0,
      ins_valid => fork34_outs_0_valid,
      ins_ready => fork34_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork34_outs_1,
      lhs_valid => fork34_outs_1_valid,
      lhs_ready => fork34_outs_1_ready,
      rhs => extsi27_outs,
      rhs_valid => extsi27_outs_valid,
      rhs_ready => extsi27_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer68 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer68_outs,
      outs_valid => buffer68_outs_valid,
      outs_ready => buffer68_outs_ready
    );

  fork35 : entity work.handshake_fork(arch) generic map(10, 1)
    port map(
      ins => buffer68_outs,
      ins_valid => buffer68_outs_valid,
      ins_ready => buffer68_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork35_outs_0,
      outs(1) => fork35_outs_1,
      outs(2) => fork35_outs_2,
      outs(3) => fork35_outs_3,
      outs(4) => fork35_outs_4,
      outs(5) => fork35_outs_5,
      outs(6) => fork35_outs_6,
      outs(7) => fork35_outs_7,
      outs(8) => fork35_outs_8,
      outs(9) => fork35_outs_9,
      outs_valid(0) => fork35_outs_0_valid,
      outs_valid(1) => fork35_outs_1_valid,
      outs_valid(2) => fork35_outs_2_valid,
      outs_valid(3) => fork35_outs_3_valid,
      outs_valid(4) => fork35_outs_4_valid,
      outs_valid(5) => fork35_outs_5_valid,
      outs_valid(6) => fork35_outs_6_valid,
      outs_valid(7) => fork35_outs_7_valid,
      outs_valid(8) => fork35_outs_8_valid,
      outs_valid(9) => fork35_outs_9_valid,
      outs_ready(0) => fork35_outs_0_ready,
      outs_ready(1) => fork35_outs_1_ready,
      outs_ready(2) => fork35_outs_2_ready,
      outs_ready(3) => fork35_outs_3_ready,
      outs_ready(4) => fork35_outs_4_ready,
      outs_ready(5) => fork35_outs_5_ready,
      outs_ready(6) => fork35_outs_6_ready,
      outs_ready(7) => fork35_outs_7_ready,
      outs_ready(8) => fork35_outs_8_ready,
      outs_ready(9) => fork35_outs_9_ready
    );

  cond_br9 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork35_outs_0,
      condition_valid => fork35_outs_0_valid,
      condition_ready => fork35_outs_0_ready,
      data => trunci5_outs,
      data_valid => trunci5_outs_valid,
      data_ready => trunci5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br10 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork35_outs_7,
      condition_valid => fork35_outs_7_valid,
      condition_ready => fork35_outs_7_ready,
      data => addi1_result,
      data_valid => addi1_result_valid,
      data_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink6 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer66 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer66_outs,
      outs_valid => buffer66_outs_valid,
      outs_ready => buffer66_outs_ready
    );

  cond_br11 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork35_outs_8,
      condition_valid => fork35_outs_8_valid,
      condition_ready => fork35_outs_8_ready,
      data => buffer66_outs,
      data_valid => buffer66_outs_valid,
      data_ready => buffer66_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br11_trueOut,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut => cond_br11_falseOut,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  cond_br12 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork35_outs_1,
      condition_valid => fork35_outs_1_valid,
      condition_ready => fork35_outs_1_ready,
      data => fork19_outs_1,
      data_valid => fork19_outs_1_valid,
      data_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br12_trueOut,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut => cond_br12_falseOut,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  cond_br13 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork35_outs_2,
      condition_valid => fork35_outs_2_valid,
      condition_ready => fork35_outs_2_ready,
      data => fork21_outs_0,
      data_valid => fork21_outs_0_valid,
      data_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  cond_br14 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork35_outs_9,
      condition_valid => fork35_outs_9_valid,
      condition_ready => fork35_outs_9_ready,
      data_valid => fork24_outs_1_valid,
      data_ready => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  extsi29 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => cond_br13_falseOut,
      ins_valid => cond_br13_falseOut_valid,
      ins_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi29_outs,
      outs_valid => extsi29_outs_valid,
      outs_ready => extsi29_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant28 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant28_outs,
      outs_valid => constant28_outs_valid,
      outs_ready => constant28_outs_ready
    );

  extsi30 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant28_outs,
      ins_valid => constant28_outs_valid,
      ins_ready => constant28_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi30_outs,
      outs_valid => extsi30_outs_valid,
      outs_ready => extsi30_outs_ready
    );

  addi8 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi29_outs,
      lhs_valid => extsi29_outs_valid,
      lhs_ready => extsi29_outs_ready,
      rhs => extsi30_outs,
      rhs_valid => extsi30_outs_valid,
      rhs_ready => extsi30_outs_ready,
      clk => clk,
      rst => rst,
      result => addi8_result,
      result_valid => addi8_result_valid,
      result_ready => addi8_result_ready
    );

  buffer69 : entity work.oehb(arch) generic map(7)
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

  cond_br56 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork37_outs_6,
      condition_valid => fork37_outs_6_valid,
      condition_ready => fork37_outs_6_ready,
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

  sink8 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br56_falseOut,
      ins_valid => cond_br56_falseOut_valid,
      ins_ready => cond_br56_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br57 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork37_outs_5,
      condition_valid => fork37_outs_5_valid,
      condition_ready => fork37_outs_5_ready,
      data => cond_br49_falseOut,
      data_valid => cond_br49_falseOut_valid,
      data_ready => cond_br49_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br57_trueOut,
      trueOut_valid => cond_br57_trueOut_valid,
      trueOut_ready => cond_br57_trueOut_ready,
      falseOut => cond_br57_falseOut,
      falseOut_valid => cond_br57_falseOut_valid,
      falseOut_ready => cond_br57_falseOut_ready
    );

  sink9 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br57_falseOut,
      ins_valid => cond_br57_falseOut_valid,
      ins_ready => cond_br57_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br58 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork37_outs_4,
      condition_valid => fork37_outs_4_valid,
      condition_ready => fork37_outs_4_ready,
      data_valid => cond_br48_falseOut_valid,
      data_ready => cond_br48_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br58_trueOut_valid,
      trueOut_ready => cond_br58_trueOut_ready,
      falseOut_valid => cond_br58_falseOut_valid,
      falseOut_ready => cond_br58_falseOut_ready
    );

  sink10 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br58_falseOut_valid,
      ins_ready => cond_br58_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br59 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork37_outs_3,
      condition_valid => fork37_outs_3_valid,
      condition_ready => fork37_outs_3_ready,
      data_valid => cond_br50_falseOut_valid,
      data_ready => cond_br50_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br59_trueOut_valid,
      trueOut_ready => cond_br59_trueOut_ready,
      falseOut_valid => cond_br59_falseOut_valid,
      falseOut_ready => cond_br59_falseOut_ready
    );

  sink11 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br59_falseOut_valid,
      ins_ready => cond_br59_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br60 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork37_outs_2,
      condition_valid => fork37_outs_2_valid,
      condition_ready => fork37_outs_2_ready,
      data_valid => cond_br54_falseOut_valid,
      data_ready => cond_br54_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br60_trueOut_valid,
      trueOut_ready => cond_br60_trueOut_ready,
      falseOut_valid => cond_br60_falseOut_valid,
      falseOut_ready => cond_br60_falseOut_ready
    );

  sink12 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br60_falseOut_valid,
      ins_ready => cond_br60_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br61 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork37_outs_1,
      condition_valid => fork37_outs_1_valid,
      condition_ready => fork37_outs_1_ready,
      data_valid => cond_br52_falseOut_valid,
      data_ready => cond_br52_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br61_trueOut_valid,
      trueOut_ready => cond_br61_trueOut_ready,
      falseOut_valid => cond_br61_falseOut_valid,
      falseOut_ready => cond_br61_falseOut_ready
    );

  sink13 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br61_falseOut_valid,
      ins_ready => cond_br61_falseOut_ready,
      clk => clk,
      rst => rst
    );

  extsi31 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi31_outs,
      outs_valid => extsi31_outs_valid,
      outs_ready => extsi31_outs_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant29 : entity work.handshake_constant_3(arch) generic map(6)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant29_outs,
      outs_valid => constant29_outs_valid,
      outs_ready => constant29_outs_ready
    );

  extsi32 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant29_outs,
      ins_valid => constant29_outs_valid,
      ins_ready => constant29_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi32_outs,
      outs_valid => extsi32_outs_valid,
      outs_ready => extsi32_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant30 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant30_outs,
      outs_valid => constant30_outs_valid,
      outs_ready => constant30_outs_ready
    );

  extsi33 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant30_outs,
      ins_valid => constant30_outs_valid,
      ins_ready => constant30_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi33_outs,
      outs_valid => extsi33_outs_valid,
      outs_ready => extsi33_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi31_outs,
      lhs_valid => extsi31_outs_valid,
      lhs_ready => extsi31_outs_ready,
      rhs => extsi33_outs,
      rhs_valid => extsi33_outs_valid,
      rhs_ready => extsi33_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer70 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer70_outs,
      outs_valid => buffer70_outs_valid,
      outs_ready => buffer70_outs_ready
    );

  fork36 : entity work.handshake_fork(arch) generic map(2, 7)
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

  trunci6 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork36_outs_0,
      ins_valid => fork36_outs_0_valid,
      ins_ready => fork36_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork36_outs_1,
      lhs_valid => fork36_outs_1_valid,
      lhs_ready => fork36_outs_1_ready,
      rhs => extsi32_outs,
      rhs_valid => extsi32_outs_valid,
      rhs_ready => extsi32_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer72 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer72_outs,
      outs_valid => buffer72_outs_valid,
      outs_ready => buffer72_outs_ready
    );

  fork37 : entity work.handshake_fork(arch) generic map(10, 1)
    port map(
      ins => buffer72_outs,
      ins_valid => buffer72_outs_valid,
      ins_ready => buffer72_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork37_outs_0,
      outs(1) => fork37_outs_1,
      outs(2) => fork37_outs_2,
      outs(3) => fork37_outs_3,
      outs(4) => fork37_outs_4,
      outs(5) => fork37_outs_5,
      outs(6) => fork37_outs_6,
      outs(7) => fork37_outs_7,
      outs(8) => fork37_outs_8,
      outs(9) => fork37_outs_9,
      outs_valid(0) => fork37_outs_0_valid,
      outs_valid(1) => fork37_outs_1_valid,
      outs_valid(2) => fork37_outs_2_valid,
      outs_valid(3) => fork37_outs_3_valid,
      outs_valid(4) => fork37_outs_4_valid,
      outs_valid(5) => fork37_outs_5_valid,
      outs_valid(6) => fork37_outs_6_valid,
      outs_valid(7) => fork37_outs_7_valid,
      outs_valid(8) => fork37_outs_8_valid,
      outs_valid(9) => fork37_outs_9_valid,
      outs_ready(0) => fork37_outs_0_ready,
      outs_ready(1) => fork37_outs_1_ready,
      outs_ready(2) => fork37_outs_2_ready,
      outs_ready(3) => fork37_outs_3_ready,
      outs_ready(4) => fork37_outs_4_ready,
      outs_ready(5) => fork37_outs_5_ready,
      outs_ready(6) => fork37_outs_6_ready,
      outs_ready(7) => fork37_outs_7_ready,
      outs_ready(8) => fork37_outs_8_ready,
      outs_ready(9) => fork37_outs_9_ready
    );

  cond_br15 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork37_outs_0,
      condition_valid => fork37_outs_0_valid,
      condition_ready => fork37_outs_0_ready,
      data => trunci6_outs,
      data_valid => trunci6_outs_valid,
      data_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  sink15 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br15_falseOut,
      ins_valid => cond_br15_falseOut_valid,
      ins_ready => cond_br15_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br16 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork37_outs_8,
      condition_valid => fork37_outs_8_valid,
      condition_ready => fork37_outs_8_ready,
      data => cond_br5_falseOut,
      data_valid => cond_br5_falseOut_valid,
      data_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br16_trueOut,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut => cond_br16_falseOut,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  cond_br17 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork37_outs_9,
      condition_valid => fork37_outs_9_valid,
      condition_ready => fork37_outs_9_ready,
      data_valid => cond_br8_falseOut_valid,
      data_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  fork38 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br17_falseOut_valid,
      ins_ready => cond_br17_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork38_outs_0_valid,
      outs_valid(1) => fork38_outs_1_valid,
      outs_ready(0) => fork38_outs_0_ready,
      outs_ready(1) => fork38_outs_1_ready
    );

end architecture;
