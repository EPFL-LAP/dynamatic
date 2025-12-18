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
  signal lsq1_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_0_valid : std_logic;
  signal lsq1_ldData_0_ready : std_logic;
  signal lsq1_ldData_1 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_1_valid : std_logic;
  signal lsq1_ldData_1_ready : std_logic;
  signal lsq1_memEnd_valid : std_logic;
  signal lsq1_memEnd_ready : std_logic;
  signal lsq1_loadEn : std_logic;
  signal lsq1_loadAddr : std_logic_vector(8 downto 0);
  signal lsq1_storeEn : std_logic;
  signal lsq1_storeAddr : std_logic_vector(8 downto 0);
  signal lsq1_storeData : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0_valid : std_logic;
  signal mem_controller1_ldData_0_ready : std_logic;
  signal mem_controller1_memEnd_valid : std_logic;
  signal mem_controller1_memEnd_ready : std_logic;
  signal mem_controller1_loadEn : std_logic;
  signal mem_controller1_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller1_storeEn : std_logic;
  signal mem_controller1_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller1_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi15_outs : std_logic_vector(5 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal extsi16_outs : std_logic_vector(31 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(5 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(5 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(5 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(5 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal extsi17_outs : std_logic_vector(6 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(1 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal extsi18_outs : std_logic_vector(6 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(6 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(6 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(6 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(6 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(6 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
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
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(0 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(5 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal extsi19_outs : std_logic_vector(6 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(1 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(1 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(1 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(0 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(0 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(0 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal fork7_outs_3 : std_logic_vector(0 downto 0);
  signal fork7_outs_3_valid : std_logic;
  signal fork7_outs_3_ready : std_logic;
  signal fork7_outs_4 : std_logic_vector(0 downto 0);
  signal fork7_outs_4_valid : std_logic;
  signal fork7_outs_4_ready : std_logic;
  signal fork7_outs_5 : std_logic_vector(0 downto 0);
  signal fork7_outs_5_valid : std_logic;
  signal fork7_outs_5_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(1 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(1 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal extsi14_outs : std_logic_vector(5 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(1 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(1 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal extsi20_outs : std_logic_vector(31 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer8_outs : std_logic_vector(5 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(5 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
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
  signal mux5_outs : std_logic_vector(5 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(5 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal extsi21_outs : std_logic_vector(6 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(31 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(31 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal fork8_outs_4 : std_logic_vector(31 downto 0);
  signal fork8_outs_4_valid : std_logic;
  signal fork8_outs_4_ready : std_logic;
  signal trunci1_outs : std_logic_vector(8 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(8 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(8 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(31 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal mux8_outs : std_logic_vector(5 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(5 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(5 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(5 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(5 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_2 : std_logic_vector(5 downto 0);
  signal fork9_outs_2_valid : std_logic;
  signal fork9_outs_2_ready : std_logic;
  signal extsi22_outs : std_logic_vector(31 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal trunci4_outs : std_logic_vector(4 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal mux9_outs : std_logic_vector(5 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(5 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(5 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(5 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(5 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal extsi23_outs : std_logic_vector(31 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
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
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork13_outs_2 : std_logic_vector(0 downto 0);
  signal fork13_outs_2_valid : std_logic;
  signal fork13_outs_2_ready : std_logic;
  signal fork13_outs_3 : std_logic_vector(0 downto 0);
  signal fork13_outs_3_valid : std_logic;
  signal fork13_outs_3_ready : std_logic;
  signal fork13_outs_4 : std_logic_vector(0 downto 0);
  signal fork13_outs_4_valid : std_logic;
  signal fork13_outs_4_ready : std_logic;
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant19_outs : std_logic_vector(5 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal extsi24_outs : std_logic_vector(6 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant20_outs : std_logic_vector(1 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(1 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(1 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal extsi25_outs : std_logic_vector(6 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant21_outs : std_logic_vector(3 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(31 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(31 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(31 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal fork16_outs_2 : std_logic_vector(31 downto 0);
  signal fork16_outs_2_valid : std_logic;
  signal fork16_outs_2_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant22_outs : std_logic_vector(2 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(31 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(31 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(31 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_2 : std_logic_vector(31 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal trunci5_outs : std_logic_vector(8 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(8 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal addi9_result : std_logic_vector(8 downto 0);
  signal addi9_result_valid : std_logic;
  signal addi9_result_ready : std_logic;
  signal buffer23_outs : std_logic_vector(8 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(8 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer24_outs : std_logic_vector(8 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
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
  signal buffer25_outs : std_logic_vector(31 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(8 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer26_outs : std_logic_vector(31 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal trunci8_outs : std_logic_vector(8 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal addi10_result : std_logic_vector(8 downto 0);
  signal addi10_result_valid : std_logic;
  signal addi10_result_ready : std_logic;
  signal buffer27_outs : std_logic_vector(8 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal addi4_result : std_logic_vector(8 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer28_outs : std_logic_vector(8 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
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
  signal buffer30_outs : std_logic_vector(31 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal trunci9_outs : std_logic_vector(8 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal shli5_result : std_logic_vector(31 downto 0);
  signal shli5_result_valid : std_logic;
  signal shli5_result_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal trunci10_outs : std_logic_vector(8 downto 0);
  signal trunci10_outs_valid : std_logic;
  signal trunci10_outs_ready : std_logic;
  signal addi11_result : std_logic_vector(8 downto 0);
  signal addi11_result_valid : std_logic;
  signal addi11_result_ready : std_logic;
  signal buffer32_outs : std_logic_vector(8 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal addi5_result : std_logic_vector(8 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer29_outs : std_logic_vector(31 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal buffer33_outs : std_logic_vector(8 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(8 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal addi6_result : std_logic_vector(6 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer35_outs : std_logic_vector(6 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(6 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(6 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal trunci11_outs : std_logic_vector(5 downto 0);
  signal trunci11_outs_valid : std_logic;
  signal trunci11_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer36_outs : std_logic_vector(0 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
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
  signal buffer34_outs : std_logic_vector(31 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
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
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal buffer37_outs : std_logic_vector(5 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(6 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant23_outs : std_logic_vector(1 downto 0);
  signal constant23_outs_valid : std_logic;
  signal constant23_outs_ready : std_logic;
  signal extsi27_outs : std_logic_vector(6 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal addi8_result : std_logic_vector(6 downto 0);
  signal addi8_result_valid : std_logic;
  signal addi8_result_ready : std_logic;
  signal extsi28_outs : std_logic_vector(6 downto 0);
  signal extsi28_outs_valid : std_logic;
  signal extsi28_outs_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant24_outs : std_logic_vector(5 downto 0);
  signal constant24_outs_valid : std_logic;
  signal constant24_outs_ready : std_logic;
  signal extsi29_outs : std_logic_vector(6 downto 0);
  signal extsi29_outs_valid : std_logic;
  signal extsi29_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant25_outs : std_logic_vector(1 downto 0);
  signal constant25_outs_valid : std_logic;
  signal constant25_outs_ready : std_logic;
  signal extsi30_outs : std_logic_vector(6 downto 0);
  signal extsi30_outs_valid : std_logic;
  signal extsi30_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(6 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer38_outs : std_logic_vector(6 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(6 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(6 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal trunci12_outs : std_logic_vector(5 downto 0);
  signal trunci12_outs_valid : std_logic;
  signal trunci12_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(0 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(0 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork21_outs_2 : std_logic_vector(0 downto 0);
  signal fork21_outs_2_valid : std_logic;
  signal fork21_outs_2_ready : std_logic;
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
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;

begin

  out0 <= cond_br16_falseOut;
  out0_valid <= cond_br16_falseOut_valid;
  cond_br16_falseOut_ready <= out0_ready;
  c_end_valid <= mem_controller1_memEnd_valid;
  mem_controller1_memEnd_ready <= c_end_ready;
  a_end_valid <= lsq1_memEnd_valid;
  lsq1_memEnd_ready <= a_end_ready;
  end_valid <= fork0_outs_2_valid;
  fork0_outs_2_ready <= end_ready;
  c_loadEn <= mem_controller1_loadEn;
  c_loadAddr <= mem_controller1_loadAddr;
  c_storeEn <= mem_controller1_storeEn;
  c_storeAddr <= mem_controller1_storeAddr;
  c_storeData <= mem_controller1_storeData;
  a_loadEn <= lsq1_loadEn;
  a_loadAddr <= lsq1_loadAddr;
  a_storeEn <= lsq1_storeEn;
  a_storeAddr <= lsq1_storeAddr;
  a_storeData <= lsq1_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(4)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_valid(2) => fork0_outs_2_valid,
      outs_valid(3) => fork0_outs_3_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready,
      outs_ready(2) => fork0_outs_2_ready,
      outs_ready(3) => fork0_outs_3_ready
    );

  lsq1 : entity work.handshake_lsq_lsq1(arch)
    port map(
      io_loadData => a_loadData,
      io_memStart_valid => a_start_valid,
      io_memStart_ready => a_start_ready,
      io_ctrl_0_valid => fork14_outs_0_valid,
      io_ctrl_0_ready => fork14_outs_0_ready,
      io_ldAddr_0_bits => load0_addrOut,
      io_ldAddr_0_valid => load0_addrOut_valid,
      io_ldAddr_0_ready => load0_addrOut_ready,
      io_ldAddr_1_bits => load2_addrOut,
      io_ldAddr_1_valid => load2_addrOut_valid,
      io_ldAddr_1_ready => load2_addrOut_ready,
      io_stAddr_0_bits => store0_addrOut,
      io_stAddr_0_valid => store0_addrOut_valid,
      io_stAddr_0_ready => store0_addrOut_ready,
      io_stData_0_bits => store0_dataToMem,
      io_stData_0_valid => store0_dataToMem_valid,
      io_stData_0_ready => store0_dataToMem_ready,
      io_ctrlEnd_valid => fork22_outs_1_valid,
      io_ctrlEnd_ready => fork22_outs_1_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq1_ldData_0,
      io_ldData_0_valid => lsq1_ldData_0_valid,
      io_ldData_0_ready => lsq1_ldData_0_ready,
      io_ldData_1_bits => lsq1_ldData_1,
      io_ldData_1_valid => lsq1_ldData_1_valid,
      io_ldData_1_ready => lsq1_ldData_1_ready,
      io_memEnd_valid => lsq1_memEnd_valid,
      io_memEnd_ready => lsq1_memEnd_ready,
      io_loadEn => lsq1_loadEn,
      io_loadAddr => lsq1_loadAddr,
      io_storeEn => lsq1_storeEn,
      io_storeAddr => lsq1_storeAddr,
      io_storeData => lsq1_storeData
    );

  mem_controller1 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => c_loadData,
      memStart_valid => c_start_valid,
      memStart_ready => c_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork22_outs_0_valid,
      ctrlEnd_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller1_ldData_0,
      ldData_valid(0) => mem_controller1_ldData_0_valid,
      ldData_ready(0) => mem_controller1_ldData_0_ready,
      memEnd_valid => mem_controller1_memEnd_valid,
      memEnd_ready => mem_controller1_memEnd_ready,
      loadEn => mem_controller1_loadEn,
      loadAddr => mem_controller1_loadAddr,
      storeEn => mem_controller1_storeEn,
      storeAddr => mem_controller1_storeAddr,
      storeData => mem_controller1_storeData
    );

  constant0 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  constant1 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi15 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  extsi16 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => extsi15_outs,
      ins(1) => cond_br15_trueOut,
      ins_valid(0) => extsi15_outs_valid,
      ins_valid(1) => cond_br15_trueOut_valid,
      ins_ready(0) => extsi15_outs_ready,
      ins_ready(1) => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer0 : entity work.oehb(arch) generic map(6)
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

  buffer1 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  extsi17 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => extsi16_outs,
      ins(1) => cond_br16_trueOut,
      ins_valid(0) => extsi16_outs_valid,
      ins_valid(1) => cond_br16_trueOut_valid,
      ins_ready(0) => extsi16_outs_ready,
      ins_ready(1) => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer39 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br17_trueOut_valid,
      ins_ready => cond_br17_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => buffer39_outs_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant3 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  extsi18 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi17_outs,
      lhs_valid => extsi17_outs_valid,
      lhs_ready => extsi17_outs_ready,
      rhs => extsi18_outs,
      rhs_valid => extsi18_outs_valid,
      rhs_ready => extsi18_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer2 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 7, 1)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => addi2_result,
      ins(1) => addi8_result,
      ins_valid(0) => addi2_result_valid,
      ins_valid(1) => addi8_result_valid,
      ins_ready(0) => addi2_result_ready,
      ins_ready(1) => addi8_result_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer4 : entity work.oehb(arch) generic map(7)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.tehb(arch) generic map(7)
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

  fork3 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  trunci0 : entity work.trunci(arch) generic map(7, 6)
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

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork4_outs_2,
      index_valid => fork4_outs_2_valid,
      index_ready => fork4_outs_2_ready,
      ins(0) => buffer3_outs,
      ins(1) => cond_br11_falseOut,
      ins_valid(0) => buffer3_outs_valid,
      ins_valid(1) => cond_br11_falseOut_valid,
      ins_ready(0) => buffer3_outs_ready,
      ins_ready(1) => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br12_falseOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br12_falseOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
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

  fork4 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs(2) => fork4_outs_2,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready
    );

  fork5 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant5 : entity work.handshake_constant_2(arch) generic map(6)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  extsi19 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  constant6 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork5_outs_0_valid,
      ctrl_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant6_outs,
      ins_valid => constant6_outs_valid,
      ins_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork3_outs_1,
      lhs_valid => fork3_outs_1_valid,
      lhs_ready => fork3_outs_1_ready,
      rhs => extsi19_outs,
      rhs_valid => extsi19_outs_valid,
      rhs_ready => extsi19_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer10 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs(4) => fork7_outs_4,
      outs(5) => fork7_outs_5,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_valid(4) => fork7_outs_4_valid,
      outs_valid(5) => fork7_outs_5_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready,
      outs_ready(4) => fork7_outs_4_ready,
      outs_ready(5) => fork7_outs_5_ready
    );

  cond_br3 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork7_outs_5,
      condition_valid => fork7_outs_5_valid,
      condition_ready => fork7_outs_5_ready,
      data => fork6_outs_0,
      data_valid => fork6_outs_0_valid,
      data_ready => fork6_outs_0_ready,
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

  extsi14 : entity work.extsi(arch) generic map(2, 6)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  cond_br4 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork7_outs_4,
      condition_valid => fork7_outs_4_valid,
      condition_ready => fork7_outs_4_ready,
      data => fork6_outs_1,
      data_valid => fork6_outs_1_valid,
      data_ready => fork6_outs_1_ready,
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

  extsi20 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => cond_br4_trueOut,
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  buffer6 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer7 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  cond_br5 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_2,
      condition_valid => fork7_outs_2_valid,
      condition_ready => fork7_outs_2_ready,
      data => buffer7_outs,
      data_valid => buffer7_outs_valid,
      data_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer8 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork7_outs_1,
      condition_valid => fork7_outs_1_valid,
      condition_ready => fork7_outs_1_ready,
      data => buffer9_outs,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
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
      condition => fork7_outs_0,
      condition_valid => fork7_outs_0_valid,
      condition_ready => fork7_outs_0_ready,
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
      condition => fork7_outs_3,
      condition_valid => fork7_outs_3_valid,
      condition_ready => fork7_outs_3_ready,
      data_valid => fork5_outs_1_valid,
      data_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork13_outs_2,
      index_valid => fork13_outs_2_valid,
      index_ready => fork13_outs_2_ready,
      ins(0) => extsi14_outs,
      ins(1) => cond_br9_trueOut,
      ins_valid(0) => extsi14_outs_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => extsi14_outs_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  extsi21 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork13_outs_3,
      index_valid => fork13_outs_3_valid,
      index_ready => fork13_outs_3_ready,
      ins(0) => extsi20_outs,
      ins(1) => cond_br10_trueOut,
      ins_valid(0) => extsi20_outs_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => extsi20_outs_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer12 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs(4) => fork8_outs_4,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_valid(4) => fork8_outs_4_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready,
      outs_ready(4) => fork8_outs_4_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => fork8_outs_1,
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => fork8_outs_2,
      ins_valid => fork8_outs_2_valid,
      ins_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork13_outs_4,
      index_valid => fork13_outs_4_valid,
      index_ready => fork13_outs_4_ready,
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
      index => fork13_outs_0,
      index_valid => fork13_outs_0_valid,
      index_ready => fork13_outs_0_ready,
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

  buffer15 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
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

  fork9 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs(2) => fork9_outs_2,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_valid(2) => fork9_outs_2_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready,
      outs_ready(2) => fork9_outs_2_ready
    );

  extsi22 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork9_outs_2,
      ins_valid => fork9_outs_2_valid,
      ins_ready => fork9_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi22_outs,
      ins_valid => extsi22_outs_valid,
      ins_ready => extsi22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  trunci4 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork13_outs_1,
      index_valid => fork13_outs_1_valid,
      index_ready => fork13_outs_1_ready,
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

  buffer17 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer18 : entity work.tehb(arch) generic map(6)
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

  fork11 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  extsi23 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => extsi23_outs,
      ins_valid => extsi23_outs_valid,
      ins_ready => extsi23_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs(3) => fork12_outs_3,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_valid(3) => fork12_outs_3_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready,
      outs_ready(3) => fork12_outs_3_ready
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

  fork13 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs(3) => fork13_outs_3,
      outs(4) => fork13_outs_4,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_valid(3) => fork13_outs_3_valid,
      outs_valid(4) => fork13_outs_4_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready,
      outs_ready(3) => fork13_outs_3_ready,
      outs_ready(4) => fork13_outs_4_ready
    );

  fork14 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant19 : entity work.handshake_constant_3(arch) generic map(6)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant19_outs,
      outs_valid => constant19_outs_valid,
      outs_ready => constant19_outs_ready
    );

  extsi24 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant19_outs,
      ins_valid => constant19_outs_valid,
      ins_ready => constant19_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi24_outs,
      outs_valid => extsi24_outs_valid,
      outs_ready => extsi24_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant20 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => constant20_outs,
      ins_valid => constant20_outs_valid,
      ins_ready => constant20_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  extsi25 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
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

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant21 : entity work.handshake_constant_4(arch) generic map(4)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  extsi9 : entity work.extsi(arch) generic map(4, 32)
    port map(
      ins => constant21_outs,
      ins_valid => constant21_outs_valid,
      ins_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi9_outs,
      ins_valid => extsi9_outs_valid,
      ins_ready => extsi9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs(2) => fork16_outs_2,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_valid(2) => fork16_outs_2_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready,
      outs_ready(2) => fork16_outs_2_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant22 : entity work.handshake_constant_5(arch) generic map(3)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  extsi10 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi10_outs,
      ins_valid => extsi10_outs_valid,
      ins_ready => extsi10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork17_outs_0,
      outs(1) => fork17_outs_1,
      outs(2) => fork17_outs_2,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_valid(2) => fork17_outs_2_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready,
      outs_ready(2) => fork17_outs_2_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_0,
      lhs_valid => fork12_outs_0_valid,
      lhs_ready => fork12_outs_0_ready,
      rhs => fork17_outs_0,
      rhs_valid => fork17_outs_0_valid,
      rhs_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer21 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  trunci5 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => fork16_outs_0,
      rhs_valid => fork16_outs_0_valid,
      rhs_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer22 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer22_outs,
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  addi9 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci5_outs,
      lhs_valid => trunci5_outs_valid,
      lhs_ready => trunci5_outs_ready,
      rhs => trunci6_outs,
      rhs_valid => trunci6_outs_valid,
      rhs_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      result => addi9_result,
      result_valid => addi9_result_valid,
      result_ready => addi9_result_ready
    );

  buffer23 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi9_result,
      ins_valid => addi9_result_valid,
      ins_ready => addi9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci1_outs,
      lhs_valid => trunci1_outs_valid,
      lhs_ready => trunci1_outs_ready,
      rhs => buffer23_outs,
      rhs_valid => buffer23_outs_valid,
      rhs_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer24 : entity work.tfifo(arch) generic map(1, 9)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => buffer24_outs,
      addrIn_valid => buffer24_outs_valid,
      addrIn_ready => buffer24_outs_ready,
      dataFromMem => lsq1_ldData_0,
      dataFromMem_valid => lsq1_ldData_0_valid,
      dataFromMem_ready => lsq1_ldData_0_ready,
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
      addrIn => trunci4_outs,
      addrIn_valid => trunci4_outs_valid,
      addrIn_ready => trunci4_outs_ready,
      dataFromMem => mem_controller1_ldData_0,
      dataFromMem_valid => mem_controller1_ldData_0_valid,
      dataFromMem_ready => mem_controller1_ldData_0_ready,
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
      lhs => fork10_outs_0,
      lhs_valid => fork10_outs_0_valid,
      lhs_ready => fork10_outs_0_ready,
      rhs => fork17_outs_1,
      rhs_valid => fork17_outs_1_valid,
      rhs_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  buffer25 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => fork16_outs_1,
      rhs_valid => fork16_outs_1_valid,
      rhs_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer26 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  trunci8 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  addi10 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci7_outs,
      lhs_valid => trunci7_outs_valid,
      lhs_ready => trunci7_outs_ready,
      rhs => trunci8_outs,
      rhs_valid => trunci8_outs_valid,
      rhs_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      result => addi10_result,
      result_valid => addi10_result_valid,
      result_ready => addi10_result_ready
    );

  buffer27 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi10_result,
      ins_valid => addi10_result_valid,
      ins_ready => addi10_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  addi4 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci2_outs,
      lhs_valid => trunci2_outs_valid,
      lhs_ready => trunci2_outs_ready,
      rhs => buffer27_outs,
      rhs_valid => buffer27_outs_valid,
      rhs_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer28 : entity work.tfifo(arch) generic map(1, 9)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 9)
    port map(
      addrIn => buffer28_outs,
      addrIn_valid => buffer28_outs_valid,
      addrIn_ready => buffer28_outs_ready,
      dataFromMem => lsq1_ldData_1,
      dataFromMem_valid => lsq1_ldData_1_valid,
      dataFromMem_ready => lsq1_ldData_1_ready,
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
      lhs => fork12_outs_2,
      lhs_valid => fork12_outs_2_valid,
      lhs_ready => fork12_outs_2_ready,
      rhs => fork17_outs_2,
      rhs_valid => fork17_outs_2_valid,
      rhs_ready => fork17_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli4_result,
      result_valid => shli4_result_valid,
      result_ready => shli4_result_ready
    );

  buffer30 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli4_result,
      ins_valid => shli4_result_valid,
      ins_ready => shli4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  trunci9 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer30_outs,
      ins_valid => buffer30_outs_valid,
      ins_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  shli5 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork12_outs_3,
      lhs_valid => fork12_outs_3_valid,
      lhs_ready => fork12_outs_3_ready,
      rhs => fork16_outs_2,
      rhs_valid => fork16_outs_2_valid,
      rhs_ready => fork16_outs_2_ready,
      clk => clk,
      rst => rst,
      result => shli5_result,
      result_valid => shli5_result_valid,
      result_ready => shli5_result_ready
    );

  buffer31 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli5_result,
      ins_valid => shli5_result_valid,
      ins_ready => shli5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  trunci10 : entity work.trunci(arch) generic map(32, 9)
    port map(
      ins => buffer31_outs,
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci10_outs,
      outs_valid => trunci10_outs_valid,
      outs_ready => trunci10_outs_ready
    );

  addi11 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci9_outs,
      lhs_valid => trunci9_outs_valid,
      lhs_ready => trunci9_outs_ready,
      rhs => trunci10_outs,
      rhs_valid => trunci10_outs_valid,
      rhs_ready => trunci10_outs_ready,
      clk => clk,
      rst => rst,
      result => addi11_result,
      result_valid => addi11_result_valid,
      result_ready => addi11_result_ready
    );

  buffer32 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi11_result,
      ins_valid => addi11_result_valid,
      ins_ready => addi11_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  addi5 : entity work.addi(arch) generic map(9)
    port map(
      lhs => trunci3_outs,
      lhs_valid => trunci3_outs_valid,
      lhs_ready => trunci3_outs_ready,
      rhs => buffer32_outs,
      rhs_valid => buffer32_outs_valid,
      rhs_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer29 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => subi0_result,
      ins_valid => subi0_result_valid,
      ins_ready => subi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  buffer33 : entity work.tfifo(arch) generic map(1, 9)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 9)
    port map(
      addrIn => buffer33_outs,
      addrIn_valid => buffer33_outs_valid,
      addrIn_ready => buffer33_outs_ready,
      dataIn => buffer29_outs,
      dataIn_valid => buffer29_outs_valid,
      dataIn_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  buffer14 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer14_outs,
      lhs_valid => buffer14_outs_valid,
      lhs_ready => buffer14_outs_ready,
      rhs => fork8_outs_4,
      rhs_valid => fork8_outs_4_valid,
      rhs_ready => fork8_outs_4_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork8_outs_3,
      lhs_valid => fork8_outs_3_valid,
      lhs_ready => fork8_outs_3_ready,
      rhs => extsi7_outs,
      rhs_valid => extsi7_outs_valid,
      rhs_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  addi6 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi21_outs,
      lhs_valid => extsi21_outs_valid,
      lhs_ready => extsi21_outs_ready,
      rhs => extsi25_outs,
      rhs_valid => extsi25_outs_valid,
      rhs_ready => extsi25_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer35 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  trunci11 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci11_outs,
      outs_valid => trunci11_outs_valid,
      outs_ready => trunci11_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork18_outs_1,
      lhs_valid => fork18_outs_1_valid,
      lhs_ready => fork18_outs_1_ready,
      rhs => extsi24_outs,
      rhs_valid => extsi24_outs_valid,
      rhs_ready => extsi24_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer36 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => buffer36_outs,
      ins_valid => buffer36_outs_valid,
      ins_ready => buffer36_outs_ready,
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

  cond_br9 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork19_outs_0,
      condition_valid => fork19_outs_0_valid,
      condition_ready => fork19_outs_0_ready,
      data => trunci11_outs,
      data_valid => trunci11_outs_valid,
      data_ready => trunci11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  sink3 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br10 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork19_outs_3,
      condition_valid => fork19_outs_3_valid,
      condition_ready => fork19_outs_3_ready,
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

  sink4 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer34 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  cond_br11 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork19_outs_4,
      condition_valid => fork19_outs_4_valid,
      condition_ready => fork19_outs_4_ready,
      data => buffer34_outs,
      data_valid => buffer34_outs_valid,
      data_ready => buffer34_outs_ready,
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
      condition => fork19_outs_1,
      condition_valid => fork19_outs_1_valid,
      condition_ready => fork19_outs_1_ready,
      data => fork9_outs_1,
      data_valid => fork9_outs_1_valid,
      data_ready => fork9_outs_1_ready,
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
      condition => fork19_outs_2,
      condition_valid => fork19_outs_2_valid,
      condition_ready => fork19_outs_2_ready,
      data => fork11_outs_0,
      data_valid => fork11_outs_0_valid,
      data_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  buffer19 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer20 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cond_br14 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork19_outs_5,
      condition_valid => fork19_outs_5_valid,
      condition_ready => fork19_outs_5_ready,
      data_valid => buffer20_outs_valid,
      data_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  buffer37 : entity work.oehb(arch) generic map(6)
    port map(
      ins => cond_br13_falseOut,
      ins_valid => cond_br13_falseOut_valid,
      ins_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => buffer37_outs,
      ins_valid => buffer37_outs_valid,
      ins_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant23 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant23_outs,
      outs_valid => constant23_outs_valid,
      outs_ready => constant23_outs_ready
    );

  extsi27 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant23_outs,
      ins_valid => constant23_outs_valid,
      ins_ready => constant23_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi27_outs,
      outs_valid => extsi27_outs_valid,
      outs_ready => extsi27_outs_ready
    );

  addi8 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi26_outs,
      lhs_valid => extsi26_outs_valid,
      lhs_ready => extsi26_outs_ready,
      rhs => extsi27_outs,
      rhs_valid => extsi27_outs_valid,
      rhs_ready => extsi27_outs_ready,
      clk => clk,
      rst => rst,
      result => addi8_result,
      result_valid => addi8_result_valid,
      result_ready => addi8_result_ready
    );

  extsi28 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi28_outs,
      outs_valid => extsi28_outs_valid,
      outs_ready => extsi28_outs_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant24 : entity work.handshake_constant_2(arch) generic map(6)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant24_outs,
      outs_valid => constant24_outs_valid,
      outs_ready => constant24_outs_ready
    );

  extsi29 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant24_outs,
      ins_valid => constant24_outs_valid,
      ins_ready => constant24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi29_outs,
      outs_valid => extsi29_outs_valid,
      outs_ready => extsi29_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant25 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant25_outs,
      outs_valid => constant25_outs_valid,
      outs_ready => constant25_outs_ready
    );

  extsi30 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant25_outs,
      ins_valid => constant25_outs_valid,
      ins_ready => constant25_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi30_outs,
      outs_valid => extsi30_outs_valid,
      outs_ready => extsi30_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi28_outs,
      lhs_valid => extsi28_outs_valid,
      lhs_ready => extsi28_outs_ready,
      rhs => extsi30_outs,
      rhs_valid => extsi30_outs_valid,
      rhs_ready => extsi30_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer38 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer38_outs,
      ins_valid => buffer38_outs_valid,
      ins_ready => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  trunci12 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci12_outs,
      outs_valid => trunci12_outs_valid,
      outs_ready => trunci12_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork20_outs_1,
      lhs_valid => fork20_outs_1_valid,
      lhs_ready => fork20_outs_1_ready,
      rhs => extsi29_outs,
      rhs_valid => extsi29_outs_valid,
      rhs_ready => extsi29_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
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

  cond_br15 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork21_outs_0,
      condition_valid => fork21_outs_0_valid,
      condition_ready => fork21_outs_0_ready,
      data => trunci12_outs,
      data_valid => trunci12_outs_valid,
      data_ready => trunci12_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  sink7 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br15_falseOut,
      ins_valid => cond_br15_falseOut_valid,
      ins_ready => cond_br15_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br16 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork21_outs_1,
      condition_valid => fork21_outs_1_valid,
      condition_ready => fork21_outs_1_ready,
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
      condition => fork21_outs_2,
      condition_valid => fork21_outs_2_valid,
      condition_ready => fork21_outs_2_ready,
      data_valid => cond_br8_falseOut_valid,
      data_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  fork22 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br17_falseOut_valid,
      ins_ready => cond_br17_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

end architecture;
