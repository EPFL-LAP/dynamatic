library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity histogram is
  port (
    feature_loadData : in std_logic_vector(31 downto 0);
    weight_loadData : in std_logic_vector(31 downto 0);
    hist_loadData : in std_logic_vector(31 downto 0);
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    feature_start_valid : in std_logic;
    weight_start_valid : in std_logic;
    hist_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    feature_end_ready : in std_logic;
    weight_end_ready : in std_logic;
    hist_end_ready : in std_logic;
    end_ready : in std_logic;
    n_ready : out std_logic;
    feature_start_ready : out std_logic;
    weight_start_ready : out std_logic;
    hist_start_ready : out std_logic;
    start_ready : out std_logic;
    feature_end_valid : out std_logic;
    weight_end_valid : out std_logic;
    hist_end_valid : out std_logic;
    end_valid : out std_logic;
    feature_loadEn : out std_logic;
    feature_loadAddr : out std_logic_vector(9 downto 0);
    feature_storeEn : out std_logic;
    feature_storeAddr : out std_logic_vector(9 downto 0);
    feature_storeData : out std_logic_vector(31 downto 0);
    weight_loadEn : out std_logic;
    weight_loadAddr : out std_logic_vector(9 downto 0);
    weight_storeEn : out std_logic;
    weight_storeAddr : out std_logic_vector(9 downto 0);
    weight_storeData : out std_logic_vector(31 downto 0);
    hist_loadEn : out std_logic;
    hist_loadAddr : out std_logic_vector(9 downto 0);
    hist_storeEn : out std_logic;
    hist_storeAddr : out std_logic_vector(9 downto 0);
    hist_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of histogram is

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
  signal mem_controller2_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0_valid : std_logic;
  signal mem_controller2_ldData_0_ready : std_logic;
  signal mem_controller2_stDone_0_valid : std_logic;
  signal mem_controller2_stDone_0_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller4_ldData_0_valid : std_logic;
  signal mem_controller4_ldData_0_ready : std_logic;
  signal mem_controller4_memEnd_valid : std_logic;
  signal mem_controller4_memEnd_ready : std_logic;
  signal mem_controller4_loadEn : std_logic;
  signal mem_controller4_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeEn : std_logic;
  signal mem_controller4_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller4_storeData : std_logic_vector(31 downto 0);
  signal constant9_outs : std_logic_vector(10 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
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
  signal constant10_outs : std_logic_vector(0 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(31 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(31 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(0 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal mux10_outs_valid : std_logic;
  signal mux10_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal mux11_outs_valid : std_logic;
  signal mux11_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(0 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal mux12_outs_valid : std_logic;
  signal mux12_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(0 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(0 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal mux14_outs_valid : std_logic;
  signal mux14_outs_ready : std_logic;
  signal buffer25_outs : std_logic_vector(0 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(0 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
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
  signal buffer50_outs : std_logic_vector(31 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(31 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(31 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(31 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(31 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer28_outs : std_logic_vector(0 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
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
  signal fork6_outs_4 : std_logic_vector(0 downto 0);
  signal fork6_outs_4_valid : std_logic;
  signal fork6_outs_4_ready : std_logic;
  signal fork6_outs_5 : std_logic_vector(0 downto 0);
  signal fork6_outs_5_valid : std_logic;
  signal fork6_outs_5_ready : std_logic;
  signal fork6_outs_6 : std_logic_vector(0 downto 0);
  signal fork6_outs_6_valid : std_logic;
  signal fork6_outs_6_ready : std_logic;
  signal fork6_outs_7 : std_logic_vector(0 downto 0);
  signal fork6_outs_7_valid : std_logic;
  signal fork6_outs_7_ready : std_logic;
  signal fork6_outs_8 : std_logic_vector(0 downto 0);
  signal fork6_outs_8_valid : std_logic;
  signal fork6_outs_8_ready : std_logic;
  signal fork6_outs_9 : std_logic_vector(0 downto 0);
  signal fork6_outs_9_valid : std_logic;
  signal fork6_outs_9_ready : std_logic;
  signal fork6_outs_10 : std_logic_vector(0 downto 0);
  signal fork6_outs_10_valid : std_logic;
  signal fork6_outs_10_ready : std_logic;
  signal fork6_outs_11 : std_logic_vector(0 downto 0);
  signal fork6_outs_11_valid : std_logic;
  signal fork6_outs_11_ready : std_logic;
  signal fork6_outs_12 : std_logic_vector(0 downto 0);
  signal fork6_outs_12_valid : std_logic;
  signal fork6_outs_12_ready : std_logic;
  signal fork6_outs_13 : std_logic_vector(0 downto 0);
  signal fork6_outs_13_valid : std_logic;
  signal fork6_outs_13_ready : std_logic;
  signal fork6_outs_14 : std_logic_vector(0 downto 0);
  signal fork6_outs_14_valid : std_logic;
  signal fork6_outs_14_ready : std_logic;
  signal fork6_outs_15 : std_logic_vector(0 downto 0);
  signal fork6_outs_15_valid : std_logic;
  signal fork6_outs_15_ready : std_logic;
  signal fork6_outs_16 : std_logic_vector(0 downto 0);
  signal fork6_outs_16_valid : std_logic;
  signal fork6_outs_16_ready : std_logic;
  signal cond_br1_trueOut : std_logic_vector(31 downto 0);
  signal cond_br1_trueOut_valid : std_logic;
  signal cond_br1_trueOut_ready : std_logic;
  signal cond_br1_falseOut : std_logic_vector(31 downto 0);
  signal cond_br1_falseOut_valid : std_logic;
  signal cond_br1_falseOut_ready : std_logic;
  signal cond_br2_trueOut : std_logic_vector(31 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(31 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal cond_br22_trueOut : std_logic_vector(31 downto 0);
  signal cond_br22_trueOut_valid : std_logic;
  signal cond_br22_trueOut_ready : std_logic;
  signal cond_br22_falseOut : std_logic_vector(31 downto 0);
  signal cond_br22_falseOut_valid : std_logic;
  signal cond_br22_falseOut_ready : std_logic;
  signal buffer36_outs : std_logic_vector(0 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal cond_br23_trueOut : std_logic_vector(31 downto 0);
  signal cond_br23_trueOut_valid : std_logic;
  signal cond_br23_trueOut_ready : std_logic;
  signal cond_br23_falseOut : std_logic_vector(31 downto 0);
  signal cond_br23_falseOut_valid : std_logic;
  signal cond_br23_falseOut_ready : std_logic;
  signal buffer37_outs : std_logic_vector(0 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal cond_br24_trueOut_valid : std_logic;
  signal cond_br24_trueOut_ready : std_logic;
  signal cond_br24_falseOut_valid : std_logic;
  signal cond_br24_falseOut_ready : std_logic;
  signal buffer38_outs : std_logic_vector(0 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal cond_br25_trueOut_valid : std_logic;
  signal cond_br25_trueOut_ready : std_logic;
  signal cond_br25_falseOut_valid : std_logic;
  signal cond_br25_falseOut_ready : std_logic;
  signal buffer39_outs : std_logic_vector(0 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal cond_br26_trueOut : std_logic_vector(31 downto 0);
  signal cond_br26_trueOut_valid : std_logic;
  signal cond_br26_trueOut_ready : std_logic;
  signal cond_br26_falseOut : std_logic_vector(31 downto 0);
  signal cond_br26_falseOut_valid : std_logic;
  signal cond_br26_falseOut_ready : std_logic;
  signal buffer40_outs : std_logic_vector(0 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal cond_br27_trueOut : std_logic_vector(31 downto 0);
  signal cond_br27_trueOut_valid : std_logic;
  signal cond_br27_trueOut_ready : std_logic;
  signal cond_br27_falseOut : std_logic_vector(31 downto 0);
  signal cond_br27_falseOut_valid : std_logic;
  signal cond_br27_falseOut_ready : std_logic;
  signal buffer41_outs : std_logic_vector(0 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal cond_br28_trueOut_valid : std_logic;
  signal cond_br28_trueOut_ready : std_logic;
  signal cond_br28_falseOut_valid : std_logic;
  signal cond_br28_falseOut_ready : std_logic;
  signal buffer42_outs : std_logic_vector(0 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br29_trueOut : std_logic_vector(31 downto 0);
  signal cond_br29_trueOut_valid : std_logic;
  signal cond_br29_trueOut_ready : std_logic;
  signal cond_br29_falseOut : std_logic_vector(31 downto 0);
  signal cond_br29_falseOut_valid : std_logic;
  signal cond_br29_falseOut_ready : std_logic;
  signal buffer43_outs : std_logic_vector(0 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal cond_br30_trueOut_valid : std_logic;
  signal cond_br30_trueOut_ready : std_logic;
  signal cond_br30_falseOut_valid : std_logic;
  signal cond_br30_falseOut_ready : std_logic;
  signal buffer44_outs : std_logic_vector(0 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal cond_br31_trueOut_valid : std_logic;
  signal cond_br31_trueOut_ready : std_logic;
  signal cond_br31_falseOut_valid : std_logic;
  signal cond_br31_falseOut_ready : std_logic;
  signal buffer45_outs : std_logic_vector(0 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal cond_br32_trueOut_valid : std_logic;
  signal cond_br32_trueOut_ready : std_logic;
  signal cond_br32_falseOut_valid : std_logic;
  signal cond_br32_falseOut_ready : std_logic;
  signal buffer46_outs : std_logic_vector(0 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal cond_br33_trueOut_valid : std_logic;
  signal cond_br33_trueOut_ready : std_logic;
  signal cond_br33_falseOut_valid : std_logic;
  signal cond_br33_falseOut_ready : std_logic;
  signal buffer47_outs : std_logic_vector(0 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal cond_br34_trueOut : std_logic_vector(31 downto 0);
  signal cond_br34_trueOut_valid : std_logic;
  signal cond_br34_trueOut_ready : std_logic;
  signal cond_br34_falseOut : std_logic_vector(31 downto 0);
  signal cond_br34_falseOut_valid : std_logic;
  signal cond_br34_falseOut_ready : std_logic;
  signal buffer48_outs : std_logic_vector(0 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(31 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal buffer49_outs : std_logic_vector(31 downto 0);
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal constant11_outs : std_logic_vector(1 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(1 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(9 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_2 : std_logic_vector(31 downto 0);
  signal fork9_outs_2_valid : std_logic;
  signal fork9_outs_2_ready : std_logic;
  signal fork9_outs_3 : std_logic_vector(31 downto 0);
  signal fork9_outs_3_valid : std_logic;
  signal fork9_outs_3_ready : std_logic;
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal buffer51_outs : std_logic_vector(31 downto 0);
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal fork10_outs_2 : std_logic_vector(31 downto 0);
  signal fork10_outs_2_valid : std_logic;
  signal fork10_outs_2_ready : std_logic;
  signal fork10_outs_3 : std_logic_vector(31 downto 0);
  signal fork10_outs_3_valid : std_logic;
  signal fork10_outs_3_ready : std_logic;
  signal fork10_outs_4 : std_logic_vector(31 downto 0);
  signal fork10_outs_4_valid : std_logic;
  signal fork10_outs_4_ready : std_logic;
  signal fork10_outs_5 : std_logic_vector(31 downto 0);
  signal fork10_outs_5_valid : std_logic;
  signal fork10_outs_5_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal cmpi4_result : std_logic_vector(0 downto 0);
  signal cmpi4_result_valid : std_logic;
  signal cmpi4_result_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(0 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(0 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal cmpi5_result : std_logic_vector(0 downto 0);
  signal cmpi5_result_valid : std_logic;
  signal cmpi5_result_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(0 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(0 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal cmpi6_result : std_logic_vector(0 downto 0);
  signal cmpi6_result_valid : std_logic;
  signal cmpi6_result_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(0 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(0 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal buffer59_outs : std_logic_vector(0 downto 0);
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal cond_br17_trueOut_valid : std_logic;
  signal cond_br17_trueOut_ready : std_logic;
  signal cond_br17_falseOut_valid : std_logic;
  signal cond_br17_falseOut_ready : std_logic;
  signal buffer60_outs : std_logic_vector(0 downto 0);
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal cond_br18_trueOut_valid : std_logic;
  signal cond_br18_trueOut_ready : std_logic;
  signal cond_br18_falseOut_valid : std_logic;
  signal cond_br18_falseOut_ready : std_logic;
  signal buffer61_outs : std_logic_vector(0 downto 0);
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal cond_br19_trueOut_valid : std_logic;
  signal cond_br19_trueOut_ready : std_logic;
  signal cond_br19_falseOut_valid : std_logic;
  signal cond_br19_falseOut_ready : std_logic;
  signal buffer62_outs : std_logic_vector(0 downto 0);
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal cond_br20_trueOut_valid : std_logic;
  signal cond_br20_trueOut_ready : std_logic;
  signal cond_br20_falseOut_valid : std_logic;
  signal cond_br20_falseOut_ready : std_logic;
  signal buffer63_outs : std_logic_vector(0 downto 0);
  signal buffer63_outs_valid : std_logic;
  signal buffer63_outs_ready : std_logic;
  signal cond_br21_trueOut_valid : std_logic;
  signal cond_br21_trueOut_ready : std_logic;
  signal cond_br21_falseOut_valid : std_logic;
  signal cond_br21_falseOut_ready : std_logic;
  signal buffer64_outs : std_logic_vector(0 downto 0);
  signal buffer64_outs_valid : std_logic;
  signal buffer64_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal mux15_outs_valid : std_logic;
  signal mux15_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal mux16_outs_valid : std_logic;
  signal mux16_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal mux17_outs_valid : std_logic;
  signal mux17_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal mux18_outs_valid : std_logic;
  signal mux18_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal mux19_outs_valid : std_logic;
  signal mux19_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal mux20_outs_valid : std_logic;
  signal mux20_outs_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal join0_outs_valid : std_logic;
  signal join0_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(9 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal load5_addrOut : std_logic_vector(9 downto 0);
  signal load5_addrOut_valid : std_logic;
  signal load5_addrOut_ready : std_logic;
  signal load5_dataOut : std_logic_vector(31 downto 0);
  signal load5_dataOut_valid : std_logic;
  signal load5_dataOut_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(31 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(31 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal init13_outs : std_logic_vector(31 downto 0);
  signal init13_outs_valid : std_logic;
  signal init13_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(31 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(31 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal init14_outs : std_logic_vector(31 downto 0);
  signal init14_outs_valid : std_logic;
  signal init14_outs_ready : std_logic;
  signal buffer74_outs : std_logic_vector(31 downto 0);
  signal buffer74_outs_valid : std_logic;
  signal buffer74_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(31 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(31 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal init15_outs : std_logic_vector(31 downto 0);
  signal init15_outs_valid : std_logic;
  signal init15_outs_ready : std_logic;
  signal buffer75_outs : std_logic_vector(31 downto 0);
  signal buffer75_outs_valid : std_logic;
  signal buffer75_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal init16_outs : std_logic_vector(31 downto 0);
  signal init16_outs_valid : std_logic;
  signal init16_outs_ready : std_logic;
  signal buffer76_outs : std_logic_vector(31 downto 0);
  signal buffer76_outs_valid : std_logic;
  signal buffer76_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(31 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(31 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal init17_outs : std_logic_vector(31 downto 0);
  signal init17_outs_valid : std_logic;
  signal init17_outs_ready : std_logic;
  signal buffer77_outs : std_logic_vector(31 downto 0);
  signal buffer77_outs_valid : std_logic;
  signal buffer77_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal init18_outs_valid : std_logic;
  signal init18_outs_ready : std_logic;
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal init19_outs_valid : std_logic;
  signal init19_outs_ready : std_logic;
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal init20_outs_valid : std_logic;
  signal init20_outs_ready : std_logic;
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal init21_outs_valid : std_logic;
  signal init21_outs_ready : std_logic;
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal init22_outs_valid : std_logic;
  signal init22_outs_ready : std_logic;
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal init23_outs_valid : std_logic;
  signal init23_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(9 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal store1_doneOut_valid : std_logic;
  signal store1_doneOut_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal fork28_outs_2_valid : std_logic;
  signal fork28_outs_2_ready : std_logic;

begin

  feature_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= feature_end_ready;
  weight_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= weight_end_ready;
  hist_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= hist_end_ready;
  end_valid <= fork0_outs_2_valid;
  fork0_outs_2_ready <= end_ready;
  feature_loadEn <= mem_controller4_loadEn;
  feature_loadAddr <= mem_controller4_loadAddr;
  feature_storeEn <= mem_controller4_storeEn;
  feature_storeAddr <= mem_controller4_storeAddr;
  feature_storeData <= mem_controller4_storeData;
  weight_loadEn <= mem_controller3_loadEn;
  weight_loadAddr <= mem_controller3_loadAddr;
  weight_storeEn <= mem_controller3_storeEn;
  weight_storeAddr <= mem_controller3_storeAddr;
  weight_storeData <= mem_controller3_storeData;
  hist_loadEn <= mem_controller2_loadEn;
  hist_loadAddr <= mem_controller2_loadAddr;
  hist_storeEn <= mem_controller2_storeEn;
  hist_storeAddr <= mem_controller2_storeAddr;
  hist_storeData <= mem_controller2_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(11)
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
      outs_ready(10) => fork0_outs_10_ready
    );

  mem_controller2 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 10)
    port map(
      loadData => hist_loadData,
      memStart_valid => hist_start_valid,
      memStart_ready => hist_start_ready,
      ctrl(0) => extsi7_outs,
      ctrl_valid(0) => extsi7_outs_valid,
      ctrl_ready(0) => extsi7_outs_ready,
      ldAddr(0) => load5_addrOut,
      ldAddr_valid(0) => load5_addrOut_valid,
      ldAddr_ready(0) => load5_addrOut_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork28_outs_2_valid,
      ctrlEnd_ready => fork28_outs_2_ready,
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

  mem_controller3 : entity work.mem_controller_storeless(arch) generic map(1, 32, 10)
    port map(
      loadData => weight_loadData,
      memStart_valid => weight_start_valid,
      memStart_ready => weight_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork28_outs_1_valid,
      ctrlEnd_ready => fork28_outs_1_ready,
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

  mem_controller4 : entity work.mem_controller_storeless(arch) generic map(1, 32, 10)
    port map(
      loadData => feature_loadData,
      memStart_valid => feature_start_valid,
      memStart_ready => feature_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrlEnd_valid => fork28_outs_0_valid,
      ctrlEnd_ready => fork28_outs_0_ready,
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

  constant9 : entity work.handshake_constant_0(arch) generic map(11)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(6, 11)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs(2) => fork1_outs_2,
      outs(3) => fork1_outs_3,
      outs(4) => fork1_outs_4,
      outs(5) => fork1_outs_5,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_valid(2) => fork1_outs_2_valid,
      outs_valid(3) => fork1_outs_3_valid,
      outs_valid(4) => fork1_outs_4_valid,
      outs_valid(5) => fork1_outs_5_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready,
      outs_ready(2) => fork1_outs_2_ready,
      outs_ready(3) => fork1_outs_3_ready,
      outs_ready(4) => fork1_outs_4_ready,
      outs_ready(5) => fork1_outs_5_ready
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

  constant10 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  extsi9 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant10_outs,
      ins_valid => constant10_outs_valid,
      ins_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => extsi0_outs,
      ins(1) => buffer8_outs,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => buffer8_outs_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer8 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork18_outs_1,
      ins_valid => fork18_outs_1_valid,
      ins_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => extsi1_outs,
      ins(1) => fork20_outs_1,
      ins_valid(0) => extsi1_outs_valid,
      ins_valid(1) => fork20_outs_1_valid,
      ins_ready(0) => extsi1_outs_ready,
      ins_ready(1) => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_2,
      index_valid => fork2_outs_2_valid,
      index_ready => fork2_outs_2_ready,
      ins(0) => extsi2_outs,
      ins(1) => fork19_outs_1,
      ins_valid(0) => extsi2_outs_valid,
      ins_valid(1) => fork19_outs_1_valid,
      ins_ready(0) => extsi2_outs_ready,
      ins_ready(1) => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_3,
      index_valid => fork2_outs_3_valid,
      index_ready => fork2_outs_3_ready,
      ins(0) => extsi3_outs,
      ins(1) => fork17_outs_1,
      ins_valid(0) => extsi3_outs_valid,
      ins_valid(1) => fork17_outs_1_valid,
      ins_ready(0) => extsi3_outs_ready,
      ins_ready(1) => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_4,
      index_valid => fork2_outs_4_valid,
      index_ready => fork2_outs_4_ready,
      ins(0) => extsi4_outs,
      ins(1) => fork21_outs_1,
      ins_valid(0) => extsi4_outs_valid,
      ins_valid(1) => fork21_outs_1_valid,
      ins_ready(0) => extsi4_outs_ready,
      ins_ready(1) => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  mux7 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork2_outs_5,
      index_valid => fork2_outs_5_valid,
      index_ready => fork2_outs_5_ready,
      ins_valid(0) => fork0_outs_9_valid,
      ins_valid(1) => init23_outs_valid,
      ins_ready(0) => fork0_outs_9_ready,
      ins_ready(1) => init23_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_6,
      index_valid => fork2_outs_6_valid,
      index_ready => fork2_outs_6_ready,
      ins(0) => extsi5_outs,
      ins(1) => init17_outs,
      ins_valid(0) => extsi5_outs_valid,
      ins_valid(1) => init17_outs_valid,
      ins_ready(0) => extsi5_outs_ready,
      ins_ready(1) => init17_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  mux9 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer20_outs,
      index_valid => buffer20_outs_valid,
      index_ready => buffer20_outs_ready,
      ins_valid(0) => fork0_outs_8_valid,
      ins_valid(1) => fork27_outs_1_valid,
      ins_ready(0) => fork0_outs_8_ready,
      ins_ready(1) => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  buffer20 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork2_outs_7,
      ins_valid => fork2_outs_7_valid,
      ins_ready => fork2_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  mux10 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer21_outs,
      index_valid => buffer21_outs_valid,
      index_ready => buffer21_outs_ready,
      ins_valid(0) => fork0_outs_7_valid,
      ins_valid(1) => fork23_outs_1_valid,
      ins_ready(0) => fork0_outs_7_ready,
      ins_ready(1) => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux10_outs_valid,
      outs_ready => mux10_outs_ready
    );

  buffer21 : entity work.tfifo(arch) generic map(5, 1)
    port map(
      ins => fork2_outs_8,
      ins_valid => fork2_outs_8_valid,
      ins_ready => fork2_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  mux11 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer22_outs,
      index_valid => buffer22_outs_valid,
      index_ready => buffer22_outs_ready,
      ins_valid(0) => fork0_outs_6_valid,
      ins_valid(1) => fork26_outs_1_valid,
      ins_ready(0) => fork0_outs_6_ready,
      ins_ready(1) => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux11_outs_valid,
      outs_ready => mux11_outs_ready
    );

  buffer22 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork2_outs_9,
      ins_valid => fork2_outs_9_valid,
      ins_ready => fork2_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  mux12 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer23_outs,
      index_valid => buffer23_outs_valid,
      index_ready => buffer23_outs_ready,
      ins_valid(0) => fork0_outs_5_valid,
      ins_valid(1) => fork24_outs_1_valid,
      ins_ready(0) => fork0_outs_5_ready,
      ins_ready(1) => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux12_outs_valid,
      outs_ready => mux12_outs_ready
    );

  buffer23 : entity work.tfifo(arch) generic map(4, 1)
    port map(
      ins => fork2_outs_10,
      ins_valid => fork2_outs_10_valid,
      ins_ready => fork2_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  mux13 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer24_outs,
      index_valid => buffer24_outs_valid,
      index_ready => buffer24_outs_ready,
      ins_valid(0) => fork0_outs_4_valid,
      ins_valid(1) => fork22_outs_1_valid,
      ins_ready(0) => fork0_outs_4_ready,
      ins_ready(1) => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  buffer24 : entity work.tfifo(arch) generic map(6, 1)
    port map(
      ins => fork2_outs_11,
      ins_valid => fork2_outs_11_valid,
      ins_ready => fork2_outs_11_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  mux14 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer25_outs,
      index_valid => buffer25_outs_valid,
      index_ready => buffer25_outs_ready,
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => fork25_outs_1_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => fork25_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux14_outs_valid,
      outs_ready => mux14_outs_ready
    );

  buffer25 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork2_outs_12,
      ins_valid => fork2_outs_12_valid,
      ins_ready => fork2_outs_12_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  buffer26 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork6_outs_16,
      ins_valid => fork6_outs_16_valid,
      ins_ready => fork6_outs_16_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(13, 1)
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
      outs_ready(12) => fork2_outs_12_ready
    );

  buffer50 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork5_outs_0,
      index_valid => fork5_outs_0_valid,
      index_ready => fork5_outs_0_ready,
      ins(0) => extsi9_outs,
      ins(1) => buffer50_outs,
      ins_valid(0) => extsi9_outs_valid,
      ins_valid(1) => buffer50_outs_valid,
      ins_ready(0) => extsi9_outs_ready,
      ins_ready(1) => buffer50_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer17 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer18 : entity work.tehb(arch) generic map(32)
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

  fork3 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork5_outs_1,
      index_valid => fork5_outs_1_valid,
      index_ready => fork5_outs_1_ready,
      ins(0) => n,
      ins(1) => cond_br1_trueOut,
      ins_valid(0) => n_valid,
      ins_valid(1) => cond_br1_trueOut_valid,
      ins_ready(0) => n_ready,
      ins_ready(1) => cond_br1_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer19 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer27 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_10_valid,
      ins_valid(1) => buffer29_outs_valid,
      ins_ready(0) => fork0_outs_10_ready,
      ins_ready(1) => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork3_outs_1,
      lhs_valid => fork3_outs_1_valid,
      lhs_ready => fork3_outs_1_ready,
      rhs => fork4_outs_1,
      rhs_valid => fork4_outs_1_valid,
      rhs_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer28 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(17, 1)
    port map(
      ins => buffer28_outs,
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs(3) => fork6_outs_3,
      outs(4) => fork6_outs_4,
      outs(5) => fork6_outs_5,
      outs(6) => fork6_outs_6,
      outs(7) => fork6_outs_7,
      outs(8) => fork6_outs_8,
      outs(9) => fork6_outs_9,
      outs(10) => fork6_outs_10,
      outs(11) => fork6_outs_11,
      outs(12) => fork6_outs_12,
      outs(13) => fork6_outs_13,
      outs(14) => fork6_outs_14,
      outs(15) => fork6_outs_15,
      outs(16) => fork6_outs_16,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_valid(3) => fork6_outs_3_valid,
      outs_valid(4) => fork6_outs_4_valid,
      outs_valid(5) => fork6_outs_5_valid,
      outs_valid(6) => fork6_outs_6_valid,
      outs_valid(7) => fork6_outs_7_valid,
      outs_valid(8) => fork6_outs_8_valid,
      outs_valid(9) => fork6_outs_9_valid,
      outs_valid(10) => fork6_outs_10_valid,
      outs_valid(11) => fork6_outs_11_valid,
      outs_valid(12) => fork6_outs_12_valid,
      outs_valid(13) => fork6_outs_13_valid,
      outs_valid(14) => fork6_outs_14_valid,
      outs_valid(15) => fork6_outs_15_valid,
      outs_valid(16) => fork6_outs_16_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready,
      outs_ready(3) => fork6_outs_3_ready,
      outs_ready(4) => fork6_outs_4_ready,
      outs_ready(5) => fork6_outs_5_ready,
      outs_ready(6) => fork6_outs_6_ready,
      outs_ready(7) => fork6_outs_7_ready,
      outs_ready(8) => fork6_outs_8_ready,
      outs_ready(9) => fork6_outs_9_ready,
      outs_ready(10) => fork6_outs_10_ready,
      outs_ready(11) => fork6_outs_11_ready,
      outs_ready(12) => fork6_outs_12_ready,
      outs_ready(13) => fork6_outs_13_ready,
      outs_ready(14) => fork6_outs_14_ready,
      outs_ready(15) => fork6_outs_15_ready,
      outs_ready(16) => fork6_outs_16_ready
    );

  cond_br1 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork6_outs_15,
      condition_valid => fork6_outs_15_valid,
      condition_ready => fork6_outs_15_ready,
      data => fork4_outs_0,
      data_valid => fork4_outs_0_valid,
      data_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br1_trueOut,
      trueOut_valid => cond_br1_trueOut_valid,
      trueOut_ready => cond_br1_trueOut_ready,
      falseOut => cond_br1_falseOut,
      falseOut_valid => cond_br1_falseOut_valid,
      falseOut_ready => cond_br1_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br1_falseOut,
      ins_valid => cond_br1_falseOut_valid,
      ins_ready => cond_br1_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br2 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork6_outs_14,
      condition_valid => fork6_outs_14_valid,
      condition_ready => fork6_outs_14_ready,
      data => fork3_outs_0,
      data_valid => fork3_outs_0_valid,
      data_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br3 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork6_outs_13,
      condition_valid => fork6_outs_13_valid,
      condition_ready => fork6_outs_13_ready,
      data_valid => control_merge0_outs_valid,
      data_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer2 : entity work.oehb(arch) generic map(32)
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

  cond_br22 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer36_outs,
      condition_valid => buffer36_outs_valid,
      condition_ready => buffer36_outs_ready,
      data => buffer2_outs,
      data_valid => buffer2_outs_valid,
      data_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br22_trueOut,
      trueOut_valid => cond_br22_trueOut_valid,
      trueOut_ready => cond_br22_trueOut_ready,
      falseOut => cond_br22_falseOut,
      falseOut_valid => cond_br22_falseOut_valid,
      falseOut_ready => cond_br22_falseOut_ready
    );

  buffer36 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_12,
      ins_valid => fork6_outs_12_valid,
      ins_ready => fork6_outs_12_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  sink2 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br22_falseOut,
      ins_valid => cond_br22_falseOut_valid,
      ins_ready => cond_br22_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer6 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  cond_br23 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer37_outs,
      condition_valid => buffer37_outs_valid,
      condition_ready => buffer37_outs_ready,
      data => buffer6_outs,
      data_valid => buffer6_outs_valid,
      data_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br23_trueOut,
      trueOut_valid => cond_br23_trueOut_valid,
      trueOut_ready => cond_br23_trueOut_ready,
      falseOut => cond_br23_falseOut,
      falseOut_valid => cond_br23_falseOut_valid,
      falseOut_ready => cond_br23_falseOut_ready
    );

  buffer37 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_11,
      ins_valid => fork6_outs_11_valid,
      ins_ready => fork6_outs_11_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  sink3 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br23_falseOut,
      ins_valid => cond_br23_falseOut_valid,
      ins_ready => cond_br23_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer16 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux14_outs_valid,
      ins_ready => mux14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  cond_br24 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer38_outs,
      condition_valid => buffer38_outs_valid,
      condition_ready => buffer38_outs_ready,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br24_trueOut_valid,
      trueOut_ready => cond_br24_trueOut_ready,
      falseOut_valid => cond_br24_falseOut_valid,
      falseOut_ready => cond_br24_falseOut_ready
    );

  buffer38 : entity work.tfifo(arch) generic map(4, 1)
    port map(
      ins => fork6_outs_10,
      ins_valid => fork6_outs_10_valid,
      ins_ready => fork6_outs_10_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  sink4 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br24_falseOut_valid,
      ins_ready => cond_br24_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer15 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux13_outs_valid,
      ins_ready => mux13_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  cond_br25 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer39_outs,
      condition_valid => buffer39_outs_valid,
      condition_ready => buffer39_outs_ready,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br25_trueOut_valid,
      trueOut_ready => cond_br25_trueOut_ready,
      falseOut_valid => cond_br25_falseOut_valid,
      falseOut_ready => cond_br25_falseOut_ready
    );

  buffer39 : entity work.tfifo(arch) generic map(7, 1)
    port map(
      ins => fork6_outs_9,
      ins_valid => fork6_outs_9_valid,
      ins_ready => fork6_outs_9_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  sink5 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br25_falseOut_valid,
      ins_ready => cond_br25_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer5 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  cond_br26 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer40_outs,
      condition_valid => buffer40_outs_valid,
      condition_ready => buffer40_outs_ready,
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br26_trueOut,
      trueOut_valid => cond_br26_trueOut_valid,
      trueOut_ready => cond_br26_trueOut_ready,
      falseOut => cond_br26_falseOut,
      falseOut_valid => cond_br26_falseOut_valid,
      falseOut_ready => cond_br26_falseOut_ready
    );

  buffer40 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_8,
      ins_valid => fork6_outs_8_valid,
      ins_ready => fork6_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  sink6 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br26_falseOut,
      ins_valid => cond_br26_falseOut_valid,
      ins_ready => cond_br26_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer10 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  cond_br27 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer41_outs,
      condition_valid => buffer41_outs_valid,
      condition_ready => buffer41_outs_ready,
      data => buffer10_outs,
      data_valid => buffer10_outs_valid,
      data_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br27_trueOut,
      trueOut_valid => cond_br27_trueOut_valid,
      trueOut_ready => cond_br27_trueOut_ready,
      falseOut => cond_br27_falseOut,
      falseOut_valid => cond_br27_falseOut_valid,
      falseOut_ready => cond_br27_falseOut_ready
    );

  buffer41 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_7,
      ins_valid => fork6_outs_7_valid,
      ins_ready => fork6_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  sink7 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br27_falseOut,
      ins_valid => cond_br27_falseOut_valid,
      ins_ready => cond_br27_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer14 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux12_outs_valid,
      ins_ready => mux12_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  cond_br28 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer42_outs,
      condition_valid => buffer42_outs_valid,
      condition_ready => buffer42_outs_ready,
      data_valid => buffer14_outs_valid,
      data_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br28_trueOut_valid,
      trueOut_ready => cond_br28_trueOut_ready,
      falseOut_valid => cond_br28_falseOut_valid,
      falseOut_ready => cond_br28_falseOut_ready
    );

  buffer42 : entity work.tfifo(arch) generic map(5, 1)
    port map(
      ins => fork6_outs_6,
      ins_valid => fork6_outs_6_valid,
      ins_ready => fork6_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  sink8 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br28_falseOut_valid,
      ins_ready => cond_br28_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer3 : entity work.oehb(arch) generic map(32)
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

  cond_br29 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer43_outs,
      condition_valid => buffer43_outs_valid,
      condition_ready => buffer43_outs_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br29_trueOut,
      trueOut_valid => cond_br29_trueOut_valid,
      trueOut_ready => cond_br29_trueOut_ready,
      falseOut => cond_br29_falseOut,
      falseOut_valid => cond_br29_falseOut_valid,
      falseOut_ready => cond_br29_falseOut_ready
    );

  buffer43 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_5,
      ins_valid => fork6_outs_5_valid,
      ins_ready => fork6_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  sink9 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br29_falseOut,
      ins_valid => cond_br29_falseOut_valid,
      ins_ready => cond_br29_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer13 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux11_outs_valid,
      ins_ready => mux11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  cond_br30 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer44_outs,
      condition_valid => buffer44_outs_valid,
      condition_ready => buffer44_outs_ready,
      data_valid => buffer13_outs_valid,
      data_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br30_trueOut_valid,
      trueOut_ready => cond_br30_trueOut_ready,
      falseOut_valid => cond_br30_falseOut_valid,
      falseOut_ready => cond_br30_falseOut_ready
    );

  buffer44 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork6_outs_4,
      ins_valid => fork6_outs_4_valid,
      ins_ready => fork6_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  sink10 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br30_falseOut_valid,
      ins_ready => cond_br30_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer12 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux10_outs_valid,
      ins_ready => mux10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  cond_br31 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer45_outs,
      condition_valid => buffer45_outs_valid,
      condition_ready => buffer45_outs_ready,
      data_valid => buffer12_outs_valid,
      data_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br31_trueOut_valid,
      trueOut_ready => cond_br31_trueOut_ready,
      falseOut_valid => cond_br31_falseOut_valid,
      falseOut_ready => cond_br31_falseOut_ready
    );

  buffer45 : entity work.tfifo(arch) generic map(6, 1)
    port map(
      ins => fork6_outs_3,
      ins_valid => fork6_outs_3_valid,
      ins_ready => fork6_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  sink11 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br31_falseOut_valid,
      ins_ready => cond_br31_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer7 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer9 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  cond_br32 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer46_outs,
      condition_valid => buffer46_outs_valid,
      condition_ready => buffer46_outs_ready,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br32_trueOut_valid,
      trueOut_ready => cond_br32_trueOut_ready,
      falseOut_valid => cond_br32_falseOut_valid,
      falseOut_ready => cond_br32_falseOut_ready
    );

  buffer46 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_2,
      ins_valid => fork6_outs_2_valid,
      ins_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  sink12 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br32_falseOut_valid,
      ins_ready => cond_br32_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer11 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  cond_br33 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer47_outs,
      condition_valid => buffer47_outs_valid,
      condition_ready => buffer47_outs_ready,
      data_valid => buffer11_outs_valid,
      data_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br33_trueOut_valid,
      trueOut_ready => cond_br33_trueOut_ready,
      falseOut_valid => cond_br33_falseOut_valid,
      falseOut_ready => cond_br33_falseOut_ready
    );

  buffer47 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork6_outs_1,
      ins_valid => fork6_outs_1_valid,
      ins_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  sink13 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br33_falseOut_valid,
      ins_ready => cond_br33_falseOut_ready,
      clk => clk,
      rst => rst
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

  cond_br34 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer48_outs,
      condition_valid => buffer48_outs_valid,
      condition_ready => buffer48_outs_ready,
      data => buffer4_outs,
      data_valid => buffer4_outs_valid,
      data_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br34_trueOut,
      trueOut_valid => cond_br34_trueOut_valid,
      trueOut_ready => cond_br34_trueOut_ready,
      falseOut => cond_br34_falseOut,
      falseOut_valid => cond_br34_falseOut_valid,
      falseOut_ready => cond_br34_falseOut_ready
    );

  buffer48 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  sink14 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br34_falseOut,
      ins_valid => cond_br34_falseOut_valid,
      ins_ready => cond_br34_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork7 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => cond_br2_trueOut,
      ins_valid => cond_br2_trueOut_valid,
      ins_ready => cond_br2_trueOut_ready,
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

  trunci0 : entity work.trunci(arch) generic map(32, 10)
    port map(
      ins => buffer49_outs,
      ins_valid => buffer49_outs_valid,
      ins_ready => buffer49_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  buffer49 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork7_outs_0,
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer49_outs,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 10)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  fork8 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  constant11 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork8_outs_0_valid,
      ctrl_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready
    );

  extsi7 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant11_outs,
      ins_valid => constant11_outs_valid,
      ins_ready => constant11_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant12 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  extsi8 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant12_outs,
      ins_valid => constant12_outs_valid,
      ins_ready => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs(2) => fork9_outs_2,
      outs(3) => fork9_outs_3,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_valid(2) => fork9_outs_2_valid,
      outs_valid(3) => fork9_outs_3_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready,
      outs_ready(2) => fork9_outs_2_ready,
      outs_ready(3) => fork9_outs_3_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 10)
    port map(
      ins => buffer51_outs,
      ins_valid => buffer51_outs_valid,
      ins_ready => buffer51_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  buffer51 : entity work.tfifo(arch) generic map(6, 32)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer51_outs,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  load1 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
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

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork9_outs_1,
      ins_valid(0) => fork9_outs_1_valid,
      ins_valid(1) => cond_br32_trueOut_valid,
      ins_ready(0) => fork9_outs_1_ready,
      ins_ready(1) => cond_br32_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(6, 32)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs(2) => fork10_outs_2,
      outs(3) => fork10_outs_3,
      outs(4) => fork10_outs_4,
      outs(5) => fork10_outs_5,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_valid(2) => fork10_outs_2_valid,
      outs_valid(3) => fork10_outs_3_valid,
      outs_valid(4) => fork10_outs_4_valid,
      outs_valid(5) => fork10_outs_5_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready,
      outs_ready(2) => fork10_outs_2_ready,
      outs_ready(3) => fork10_outs_3_ready,
      outs_ready(4) => fork10_outs_4_ready,
      outs_ready(5) => fork10_outs_5_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_5,
      lhs_valid => fork10_outs_5_valid,
      lhs_ready => fork10_outs_5_ready,
      rhs => cond_br26_trueOut,
      rhs_valid => cond_br26_trueOut_valid,
      rhs_ready => cond_br26_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_4,
      lhs_valid => fork10_outs_4_valid,
      lhs_ready => fork10_outs_4_ready,
      rhs => cond_br22_trueOut,
      rhs_valid => cond_br22_trueOut_valid,
      rhs_ready => cond_br22_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  cmpi3 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_3,
      lhs_valid => fork10_outs_3_valid,
      lhs_ready => fork10_outs_3_ready,
      rhs => cond_br34_trueOut,
      rhs_valid => cond_br34_trueOut_valid,
      rhs_ready => cond_br34_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  cmpi4 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_2,
      lhs_valid => fork10_outs_2_valid,
      lhs_ready => fork10_outs_2_ready,
      rhs => cond_br29_trueOut,
      rhs_valid => cond_br29_trueOut_valid,
      rhs_ready => cond_br29_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi4_result,
      result_valid => cmpi4_result_valid,
      result_ready => cmpi4_result_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi4_result,
      ins_valid => cmpi4_result_valid,
      ins_ready => cmpi4_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  cmpi5 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => cond_br23_trueOut,
      rhs_valid => cond_br23_trueOut_valid,
      rhs_ready => cond_br23_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi5_result,
      result_valid => cmpi5_result_valid,
      result_ready => cmpi5_result_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi5_result,
      ins_valid => cmpi5_result_valid,
      ins_ready => cmpi5_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  cmpi6 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_0,
      lhs_valid => fork10_outs_0_valid,
      lhs_ready => fork10_outs_0_ready,
      rhs => cond_br27_trueOut,
      rhs_valid => cond_br27_trueOut_valid,
      rhs_ready => cond_br27_trueOut_ready,
      clk => clk,
      rst => rst,
      result => cmpi6_result,
      result_valid => cmpi6_result_valid,
      result_ready => cmpi6_result_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi6_result,
      ins_valid => cmpi6_result_valid,
      ins_ready => cmpi6_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  cond_br16 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer59_outs,
      condition_valid => buffer59_outs_valid,
      condition_ready => buffer59_outs_ready,
      data_valid => cond_br25_trueOut_valid,
      data_ready => cond_br25_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  buffer59 : entity work.tfifo(arch) generic map(6, 1)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer59_outs,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  sink16 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br16_trueOut_valid,
      ins_ready => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br17 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer60_outs,
      condition_valid => buffer60_outs_valid,
      condition_ready => buffer60_outs_ready,
      data_valid => cond_br31_trueOut_valid,
      data_ready => cond_br31_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  buffer60 : entity work.tfifo(arch) generic map(5, 1)
    port map(
      ins => fork12_outs_1,
      ins_valid => fork12_outs_1_valid,
      ins_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer60_outs,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  sink17 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br17_trueOut_valid,
      ins_ready => cond_br17_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br18 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer61_outs,
      condition_valid => buffer61_outs_valid,
      condition_ready => buffer61_outs_ready,
      data_valid => cond_br28_trueOut_valid,
      data_ready => cond_br28_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br18_trueOut_valid,
      trueOut_ready => cond_br18_trueOut_ready,
      falseOut_valid => cond_br18_falseOut_valid,
      falseOut_ready => cond_br18_falseOut_ready
    );

  buffer61 : entity work.tfifo(arch) generic map(4, 1)
    port map(
      ins => fork13_outs_1,
      ins_valid => fork13_outs_1_valid,
      ins_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer61_outs,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  sink18 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br18_trueOut_valid,
      ins_ready => cond_br18_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br19 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer62_outs,
      condition_valid => buffer62_outs_valid,
      condition_ready => buffer62_outs_ready,
      data_valid => cond_br24_trueOut_valid,
      data_ready => cond_br24_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br19_trueOut_valid,
      trueOut_ready => cond_br19_trueOut_ready,
      falseOut_valid => cond_br19_falseOut_valid,
      falseOut_ready => cond_br19_falseOut_ready
    );

  buffer62 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork14_outs_1,
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer62_outs,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  sink19 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br19_trueOut_valid,
      ins_ready => cond_br19_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br20 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer63_outs,
      condition_valid => buffer63_outs_valid,
      condition_ready => buffer63_outs_ready,
      data_valid => cond_br30_trueOut_valid,
      data_ready => cond_br30_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br20_trueOut_valid,
      trueOut_ready => cond_br20_trueOut_ready,
      falseOut_valid => cond_br20_falseOut_valid,
      falseOut_ready => cond_br20_falseOut_ready
    );

  buffer63 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer63_outs,
      outs_valid => buffer63_outs_valid,
      outs_ready => buffer63_outs_ready
    );

  sink20 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br20_trueOut_valid,
      ins_ready => cond_br20_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br21 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer64_outs,
      condition_valid => buffer64_outs_valid,
      condition_ready => buffer64_outs_ready,
      data_valid => cond_br33_trueOut_valid,
      data_ready => cond_br33_trueOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br21_trueOut_valid,
      trueOut_ready => cond_br21_trueOut_ready,
      falseOut_valid => cond_br21_falseOut_valid,
      falseOut_ready => cond_br21_falseOut_ready
    );

  buffer64 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork16_outs_1,
      ins_valid => fork16_outs_1_valid,
      ins_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer64_outs,
      outs_valid => buffer64_outs_valid,
      outs_ready => buffer64_outs_ready
    );

  sink21 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br21_trueOut_valid,
      ins_ready => cond_br21_trueOut_ready,
      clk => clk,
      rst => rst
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  mux15 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork11_outs_0,
      index_valid => fork11_outs_0_valid,
      index_ready => fork11_outs_0_ready,
      ins_valid(0) => cond_br16_falseOut_valid,
      ins_valid(1) => source1_outs_valid,
      ins_ready(0) => cond_br16_falseOut_ready,
      ins_ready(1) => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux15_outs_valid,
      outs_ready => mux15_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  mux16 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork12_outs_0,
      index_valid => fork12_outs_0_valid,
      index_ready => fork12_outs_0_ready,
      ins_valid(0) => cond_br17_falseOut_valid,
      ins_valid(1) => source2_outs_valid,
      ins_ready(0) => cond_br17_falseOut_ready,
      ins_ready(1) => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux16_outs_valid,
      outs_ready => mux16_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  mux17 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork13_outs_0,
      index_valid => fork13_outs_0_valid,
      index_ready => fork13_outs_0_ready,
      ins_valid(0) => cond_br18_falseOut_valid,
      ins_valid(1) => source3_outs_valid,
      ins_ready(0) => cond_br18_falseOut_ready,
      ins_ready(1) => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux17_outs_valid,
      outs_ready => mux17_outs_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  mux18 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork14_outs_0,
      index_valid => fork14_outs_0_valid,
      index_ready => fork14_outs_0_ready,
      ins_valid(0) => cond_br19_falseOut_valid,
      ins_valid(1) => source4_outs_valid,
      ins_ready(0) => cond_br19_falseOut_ready,
      ins_ready(1) => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux18_outs_valid,
      outs_ready => mux18_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  mux19 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork15_outs_0,
      index_valid => fork15_outs_0_valid,
      index_ready => fork15_outs_0_ready,
      ins_valid(0) => cond_br20_falseOut_valid,
      ins_valid(1) => source5_outs_valid,
      ins_ready(0) => cond_br20_falseOut_ready,
      ins_ready(1) => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux19_outs_valid,
      outs_ready => mux19_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  mux20 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => fork16_outs_0,
      index_valid => fork16_outs_0_valid,
      index_ready => fork16_outs_0_ready,
      ins_valid(0) => cond_br21_falseOut_valid,
      ins_valid(1) => source6_outs_valid,
      ins_ready(0) => cond_br21_falseOut_ready,
      ins_ready(1) => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux20_outs_valid,
      outs_ready => mux20_outs_ready
    );

  buffer30 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux15_outs_valid,
      ins_ready => mux15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  buffer31 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux16_outs_valid,
      ins_ready => mux16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  buffer32 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux17_outs_valid,
      ins_ready => mux17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  buffer33 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux18_outs_valid,
      ins_ready => mux18_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  buffer34 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux19_outs_valid,
      ins_ready => mux19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  buffer35 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux20_outs_valid,
      ins_ready => mux20_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  join0 : entity work.join_handshake(arch) generic map(6)
    port map(
      ins_valid(0) => buffer30_outs_valid,
      ins_valid(1) => buffer31_outs_valid,
      ins_valid(2) => buffer32_outs_valid,
      ins_valid(3) => buffer33_outs_valid,
      ins_valid(4) => buffer34_outs_valid,
      ins_valid(5) => buffer35_outs_valid,
      ins_ready(0) => buffer30_outs_ready,
      ins_ready(1) => buffer31_outs_ready,
      ins_ready(2) => buffer32_outs_ready,
      ins_ready(3) => buffer33_outs_ready,
      ins_ready(4) => buffer34_outs_ready,
      ins_ready(5) => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => join0_outs_valid,
      outs_ready => join0_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork9_outs_2,
      ins_valid(0) => fork9_outs_2_valid,
      ins_valid(1) => join0_outs_valid,
      ins_ready(0) => fork9_outs_2_ready,
      ins_ready(1) => join0_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 10)
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

  load5 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => trunci3_outs,
      addrIn_valid => trunci3_outs_valid,
      addrIn_ready => trunci3_outs_ready,
      dataFromMem => mem_controller2_ldData_0,
      dataFromMem_valid => mem_controller2_ldData_0_valid,
      dataFromMem_ready => mem_controller2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load5_addrOut,
      addrOut_valid => load5_addrOut_valid,
      addrOut_ready => load5_addrOut_ready,
      dataOut => load5_dataOut,
      dataOut_valid => load5_dataOut_valid,
      dataOut_ready => load5_dataOut_ready
    );

  addf0 : entity work.addf(arch_32_2_922000) generic map(32)
    port map(
      lhs => load5_dataOut,
      lhs_valid => load5_dataOut_valid,
      lhs_ready => load5_dataOut_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  buffer0 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork9_outs_3,
      ins_valid => fork9_outs_3_valid,
      ins_ready => fork9_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  init13 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => fork17_outs_0,
      ins_valid => fork17_outs_0_valid,
      ins_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => init13_outs,
      outs_valid => init13_outs_valid,
      outs_ready => init13_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init13_outs,
      ins_valid => init13_outs_valid,
      ins_ready => init13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  init14 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer74_outs,
      ins_valid => buffer74_outs_valid,
      ins_ready => buffer74_outs_ready,
      clk => clk,
      rst => rst,
      outs => init14_outs,
      outs_valid => init14_outs_valid,
      outs_ready => init14_outs_ready
    );

  buffer74 : entity work.tfifo(arch) generic map(2, 32)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer74_outs,
      outs_valid => buffer74_outs_valid,
      outs_ready => buffer74_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init14_outs,
      ins_valid => init14_outs_valid,
      ins_ready => init14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready
    );

  init15 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer75_outs,
      ins_valid => buffer75_outs_valid,
      ins_ready => buffer75_outs_ready,
      clk => clk,
      rst => rst,
      outs => init15_outs,
      outs_valid => init15_outs_valid,
      outs_ready => init15_outs_ready
    );

  buffer75 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer75_outs,
      outs_valid => buffer75_outs_valid,
      outs_ready => buffer75_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init15_outs,
      ins_valid => init15_outs_valid,
      ins_ready => init15_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  init16 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer76_outs,
      ins_valid => buffer76_outs_valid,
      ins_ready => buffer76_outs_ready,
      clk => clk,
      rst => rst,
      outs => init16_outs,
      outs_valid => init16_outs_valid,
      outs_ready => init16_outs_ready
    );

  buffer76 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer76_outs,
      outs_valid => buffer76_outs_valid,
      outs_ready => buffer76_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => init16_outs,
      ins_valid => init16_outs_valid,
      ins_ready => init16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  init17 : entity work.tehb_init(arch) generic map(32, 0)
    port map(
      ins => buffer77_outs,
      ins_valid => buffer77_outs_valid,
      ins_ready => buffer77_outs_ready,
      clk => clk,
      rst => rst,
      outs => init17_outs,
      outs_valid => init17_outs_valid,
      outs_ready => init17_outs_ready
    );

  buffer77 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork21_outs_0,
      ins_valid => fork21_outs_0_valid,
      ins_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer77_outs,
      outs_valid => buffer77_outs_valid,
      outs_ready => buffer77_outs_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => store1_doneOut_valid,
      ins_ready => store1_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork22 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  init18 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork22_outs_0_valid,
      ins_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init18_outs_valid,
      outs_ready => init18_outs_ready
    );

  fork23 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init18_outs_valid,
      ins_ready => init18_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  init19 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init19_outs_valid,
      outs_ready => init19_outs_ready
    );

  fork24 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init19_outs_valid,
      ins_ready => init19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  init20 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork24_outs_0_valid,
      ins_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init20_outs_valid,
      outs_ready => init20_outs_ready
    );

  fork25 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init20_outs_valid,
      ins_ready => init20_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

  init21 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork25_outs_0_valid,
      ins_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init21_outs_valid,
      outs_ready => init21_outs_ready
    );

  fork26 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init21_outs_valid,
      ins_ready => init21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready
    );

  init22 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork26_outs_0_valid,
      ins_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init22_outs_valid,
      outs_ready => init22_outs_ready
    );

  fork27 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => init22_outs_valid,
      ins_ready => init22_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready
    );

  init23 : entity work.tehb_dataless_init(arch)
    port map(
      ins_valid => fork27_outs_0_valid,
      ins_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => init23_outs_valid,
      outs_ready => init23_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 10)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataIn => addf0_result,
      dataIn_valid => addf0_result_valid,
      dataIn_ready => addf0_result_ready,
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

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork7_outs_2,
      lhs_valid => fork7_outs_2_valid,
      lhs_ready => fork7_outs_2_ready,
      rhs => extsi8_outs,
      rhs_valid => extsi8_outs_valid,
      rhs_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  buffer29 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  fork28 : entity work.fork_dataless(arch) generic map(3)
    port map(
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_valid(2) => fork28_outs_2_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready,
      outs_ready(2) => fork28_outs_2_ready
    );

end architecture;
