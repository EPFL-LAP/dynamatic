library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity triangular is
  port (
    x_loadData : in std_logic_vector(31 downto 0);
    n : in std_logic_vector(31 downto 0);
    n_valid : in std_logic;
    a_loadData : in std_logic_vector(31 downto 0);
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
    x_loadEn : out std_logic;
    x_loadAddr : out std_logic_vector(3 downto 0);
    x_storeEn : out std_logic;
    x_storeAddr : out std_logic_vector(3 downto 0);
    x_storeData : out std_logic_vector(31 downto 0);
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(6 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(6 downto 0);
    a_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of triangular is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal fork0_outs_3_valid : std_logic;
  signal fork0_outs_3_ready : std_logic;
  signal mem_controller2_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_0_valid : std_logic;
  signal mem_controller2_ldData_0_ready : std_logic;
  signal mem_controller2_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller2_ldData_1_valid : std_logic;
  signal mem_controller2_ldData_1_ready : std_logic;
  signal mem_controller2_stDone_0_valid : std_logic;
  signal mem_controller2_stDone_0_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(3 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(3 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal buffer62_outs : std_logic_vector(31 downto 0);
  signal buffer62_outs_valid : std_logic;
  signal buffer62_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(0 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(31 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(31 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(0 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(0 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(0 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork4_outs_3 : std_logic_vector(0 downto 0);
  signal fork4_outs_3_valid : std_logic;
  signal fork4_outs_3_ready : std_logic;
  signal fork4_outs_4 : std_logic_vector(0 downto 0);
  signal fork4_outs_4_valid : std_logic;
  signal fork4_outs_4_ready : std_logic;
  signal cond_br2_trueOut : std_logic_vector(31 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(31 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(31 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(31 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(31 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal subi1_result : std_logic_vector(31 downto 0);
  signal subi1_result_valid : std_logic;
  signal subi1_result_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal init1_outs : std_logic_vector(0 downto 0);
  signal init1_outs_valid : std_logic;
  signal init1_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(0 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
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
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal buffer25_outs : std_logic_vector(31 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal fork11_outs_3 : std_logic_vector(0 downto 0);
  signal fork11_outs_3_valid : std_logic;
  signal fork11_outs_3_ready : std_logic;
  signal fork11_outs_4 : std_logic_vector(0 downto 0);
  signal fork11_outs_4_valid : std_logic;
  signal fork11_outs_4_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer24_outs : std_logic_vector(31 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(0 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(0 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(0 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal fork12_outs_2 : std_logic_vector(0 downto 0);
  signal fork12_outs_2_valid : std_logic;
  signal fork12_outs_2_ready : std_logic;
  signal fork12_outs_3 : std_logic_vector(0 downto 0);
  signal fork12_outs_3_valid : std_logic;
  signal fork12_outs_3_ready : std_logic;
  signal fork12_outs_4 : std_logic_vector(0 downto 0);
  signal fork12_outs_4_valid : std_logic;
  signal fork12_outs_4_ready : std_logic;
  signal fork12_outs_5 : std_logic_vector(0 downto 0);
  signal fork12_outs_5_valid : std_logic;
  signal fork12_outs_5_ready : std_logic;
  signal fork12_outs_6 : std_logic_vector(0 downto 0);
  signal fork12_outs_6_valid : std_logic;
  signal fork12_outs_6_ready : std_logic;
  signal fork12_outs_7 : std_logic_vector(0 downto 0);
  signal fork12_outs_7_valid : std_logic;
  signal fork12_outs_7_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(31 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(31 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(31 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(31 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(31 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(31 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(31 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(31 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(31 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal buffer34_outs : std_logic_vector(0 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal buffer29_outs : std_logic_vector(31 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork13_outs_2 : std_logic_vector(31 downto 0);
  signal fork13_outs_2_valid : std_logic;
  signal fork13_outs_2_ready : std_logic;
  signal fork13_outs_3 : std_logic_vector(31 downto 0);
  signal fork13_outs_3_valid : std_logic;
  signal fork13_outs_3_ready : std_logic;
  signal fork13_outs_4 : std_logic_vector(31 downto 0);
  signal fork13_outs_4_valid : std_logic;
  signal fork13_outs_4_ready : std_logic;
  signal fork13_outs_5 : std_logic_vector(31 downto 0);
  signal fork13_outs_5_valid : std_logic;
  signal fork13_outs_5_ready : std_logic;
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal buffer35_outs : std_logic_vector(31 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(31 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(31 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2 : std_logic_vector(31 downto 0);
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal fork14_outs_3 : std_logic_vector(31 downto 0);
  signal fork14_outs_3_valid : std_logic;
  signal fork14_outs_3_ready : std_logic;
  signal buffer28_outs : std_logic_vector(31 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal fork15_outs_2 : std_logic_vector(31 downto 0);
  signal fork15_outs_2_valid : std_logic;
  signal fork15_outs_2_ready : std_logic;
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(3 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
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
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(31 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(31 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal fork18_outs_2 : std_logic_vector(31 downto 0);
  signal fork18_outs_2_valid : std_logic;
  signal fork18_outs_2_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(31 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(31 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(31 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(31 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal fork19_outs_3 : std_logic_vector(31 downto 0);
  signal fork19_outs_3_valid : std_logic;
  signal fork19_outs_3_ready : std_logic;
  signal fork19_outs_4 : std_logic_vector(31 downto 0);
  signal fork19_outs_4_valid : std_logic;
  signal fork19_outs_4_ready : std_logic;
  signal trunci3_outs : std_logic_vector(6 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(3 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(1 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal fork20_outs_2 : std_logic_vector(31 downto 0);
  signal fork20_outs_2_valid : std_logic;
  signal fork20_outs_2_ready : std_logic;
  signal fork20_outs_3 : std_logic_vector(31 downto 0);
  signal fork20_outs_3_valid : std_logic;
  signal fork20_outs_3_ready : std_logic;
  signal fork20_outs_4 : std_logic_vector(31 downto 0);
  signal fork20_outs_4_valid : std_logic;
  signal fork20_outs_4_ready : std_logic;
  signal fork20_outs_5 : std_logic_vector(31 downto 0);
  signal fork20_outs_5_valid : std_logic;
  signal fork20_outs_5_ready : std_logic;
  signal fork20_outs_6 : std_logic_vector(31 downto 0);
  signal fork20_outs_6_valid : std_logic;
  signal fork20_outs_6_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant13_outs : std_logic_vector(2 downto 0);
  signal constant13_outs_valid : std_logic;
  signal constant13_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(31 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(31 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork21_outs_2 : std_logic_vector(31 downto 0);
  signal fork21_outs_2_valid : std_logic;
  signal fork21_outs_2_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal buffer30_outs : std_logic_vector(31 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal xori0_result : std_logic_vector(31 downto 0);
  signal xori0_result_valid : std_logic;
  signal xori0_result_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal addi3_result : std_logic_vector(31 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer32_outs : std_logic_vector(31 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal addi4_result : std_logic_vector(31 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer33_outs : std_logic_vector(31 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(31 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(31 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal addi6_result : std_logic_vector(6 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer37_outs : std_logic_vector(31 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal trunci5_outs : std_logic_vector(6 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal shli1_result : std_logic_vector(31 downto 0);
  signal shli1_result_valid : std_logic;
  signal shli1_result_ready : std_logic;
  signal buffer38_outs : std_logic_vector(31 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(6 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal addi5_result : std_logic_vector(6 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer36_outs : std_logic_vector(6 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal buffer39_outs : std_logic_vector(6 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal addi15_result : std_logic_vector(6 downto 0);
  signal addi15_result_valid : std_logic;
  signal addi15_result_ready : std_logic;
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal addi8_result : std_logic_vector(3 downto 0);
  signal addi8_result_valid : std_logic;
  signal addi8_result_ready : std_logic;
  signal load1_addrOut : std_logic_vector(3 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal addi7_result : std_logic_vector(31 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer40_outs : std_logic_vector(31 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal xori1_result : std_logic_vector(31 downto 0);
  signal xori1_result_valid : std_logic;
  signal xori1_result_ready : std_logic;
  signal addi9_result : std_logic_vector(31 downto 0);
  signal addi9_result_valid : std_logic;
  signal addi9_result_ready : std_logic;
  signal buffer41_outs : std_logic_vector(31 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal addi10_result : std_logic_vector(31 downto 0);
  signal addi10_result_valid : std_logic;
  signal addi10_result_ready : std_logic;
  signal buffer54_outs : std_logic_vector(31 downto 0);
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal buffer42_outs : std_logic_vector(31 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal addi11_result : std_logic_vector(31 downto 0);
  signal addi11_result_valid : std_logic;
  signal addi11_result_ready : std_logic;
  signal buffer43_outs : std_logic_vector(31 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(31 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(31 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal shli2_result : std_logic_vector(31 downto 0);
  signal shli2_result_valid : std_logic;
  signal shli2_result_ready : std_logic;
  signal buffer56_outs : std_logic_vector(31 downto 0);
  signal buffer56_outs_valid : std_logic;
  signal buffer56_outs_ready : std_logic;
  signal buffer57_outs : std_logic_vector(31 downto 0);
  signal buffer57_outs_valid : std_logic;
  signal buffer57_outs_ready : std_logic;
  signal shli3_result : std_logic_vector(31 downto 0);
  signal shli3_result_valid : std_logic;
  signal shli3_result_ready : std_logic;
  signal buffer58_outs : std_logic_vector(31 downto 0);
  signal buffer58_outs_valid : std_logic;
  signal buffer58_outs_ready : std_logic;
  signal buffer59_outs : std_logic_vector(31 downto 0);
  signal buffer59_outs_valid : std_logic;
  signal buffer59_outs_ready : std_logic;
  signal buffer44_outs : std_logic_vector(31 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal buffer45_outs : std_logic_vector(31 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal addi12_result : std_logic_vector(31 downto 0);
  signal addi12_result_valid : std_logic;
  signal addi12_result_ready : std_logic;
  signal buffer46_outs : std_logic_vector(31 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal addi16_result : std_logic_vector(31 downto 0);
  signal addi16_result_valid : std_logic;
  signal addi16_result_ready : std_logic;
  signal buffer60_outs : std_logic_vector(31 downto 0);
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(6 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(6 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal subi0_result : std_logic_vector(31 downto 0);
  signal subi0_result_valid : std_logic;
  signal subi0_result_ready : std_logic;
  signal addi20_result : std_logic_vector(31 downto 0);
  signal addi20_result_valid : std_logic;
  signal addi20_result_ready : std_logic;
  signal buffer47_outs : std_logic_vector(31 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal xori2_result : std_logic_vector(31 downto 0);
  signal xori2_result_valid : std_logic;
  signal xori2_result_ready : std_logic;
  signal addi21_result : std_logic_vector(31 downto 0);
  signal addi21_result_valid : std_logic;
  signal addi21_result_ready : std_logic;
  signal buffer48_outs : std_logic_vector(31 downto 0);
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal addi13_result : std_logic_vector(31 downto 0);
  signal addi13_result_valid : std_logic;
  signal addi13_result_ready : std_logic;
  signal buffer49_outs : std_logic_vector(31 downto 0);
  signal buffer49_outs_valid : std_logic;
  signal buffer49_outs_ready : std_logic;
  signal addi14_result : std_logic_vector(31 downto 0);
  signal addi14_result_valid : std_logic;
  signal addi14_result_ready : std_logic;
  signal buffer66_outs : std_logic_vector(31 downto 0);
  signal buffer66_outs_valid : std_logic;
  signal buffer66_outs_ready : std_logic;
  signal buffer50_outs : std_logic_vector(31 downto 0);
  signal buffer50_outs_valid : std_logic;
  signal buffer50_outs_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(31 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(31 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal shli4_result : std_logic_vector(31 downto 0);
  signal shli4_result_valid : std_logic;
  signal shli4_result_ready : std_logic;
  signal buffer67_outs : std_logic_vector(31 downto 0);
  signal buffer67_outs_valid : std_logic;
  signal buffer67_outs_ready : std_logic;
  signal buffer68_outs : std_logic_vector(31 downto 0);
  signal buffer68_outs_valid : std_logic;
  signal buffer68_outs_ready : std_logic;
  signal buffer51_outs : std_logic_vector(31 downto 0);
  signal buffer51_outs_valid : std_logic;
  signal buffer51_outs_ready : std_logic;
  signal trunci8_outs : std_logic_vector(6 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal shli5_result : std_logic_vector(31 downto 0);
  signal shli5_result_valid : std_logic;
  signal shli5_result_ready : std_logic;
  signal buffer69_outs : std_logic_vector(31 downto 0);
  signal buffer69_outs_valid : std_logic;
  signal buffer69_outs_ready : std_logic;
  signal buffer70_outs : std_logic_vector(31 downto 0);
  signal buffer70_outs_valid : std_logic;
  signal buffer70_outs_ready : std_logic;
  signal buffer52_outs : std_logic_vector(31 downto 0);
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal trunci9_outs : std_logic_vector(6 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal addi22_result : std_logic_vector(6 downto 0);
  signal addi22_result_valid : std_logic;
  signal addi22_result_ready : std_logic;
  signal buffer53_outs : std_logic_vector(6 downto 0);
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal addi17_result : std_logic_vector(6 downto 0);
  signal addi17_result_valid : std_logic;
  signal addi17_result_ready : std_logic;
  signal buffer55_outs_valid : std_logic;
  signal buffer55_outs_ready : std_logic;
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal addi18_result : std_logic_vector(31 downto 0);
  signal addi18_result_valid : std_logic;
  signal addi18_result_ready : std_logic;
  signal buffer71_outs : std_logic_vector(31 downto 0);
  signal buffer71_outs_valid : std_logic;
  signal buffer71_outs_ready : std_logic;
  signal buffer61_outs : std_logic_vector(31 downto 0);
  signal buffer61_outs_valid : std_logic;
  signal buffer61_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant14_outs : std_logic_vector(1 downto 0);
  signal constant14_outs_valid : std_logic;
  signal constant14_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal addi19_result : std_logic_vector(31 downto 0);
  signal addi19_result_valid : std_logic;
  signal addi19_result_ready : std_logic;
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;

begin

  x_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= x_end_ready;
  a_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= a_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  x_loadEn <= mem_controller3_loadEn;
  x_loadAddr <= mem_controller3_loadAddr;
  x_storeEn <= mem_controller3_storeEn;
  x_storeAddr <= mem_controller3_storeAddr;
  x_storeData <= mem_controller3_storeData;
  a_loadEn <= mem_controller2_loadEn;
  a_loadAddr <= mem_controller2_loadAddr;
  a_storeEn <= mem_controller2_storeEn;
  a_storeAddr <= mem_controller2_storeAddr;
  a_storeData <= mem_controller2_storeData;

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

  mem_controller2 : entity work.mem_controller(arch) generic map(1, 2, 1, 32, 7)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ctrl(0) => extsi2_outs,
      ctrl_valid(0) => extsi2_outs_valid,
      ctrl_ready(0) => extsi2_outs_ready,
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
      ctrlEnd_valid => fork25_outs_1_valid,
      ctrlEnd_ready => fork25_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller2_ldData_0,
      ldData(1) => mem_controller2_ldData_1,
      ldData_valid(0) => mem_controller2_ldData_0_valid,
      ldData_valid(1) => mem_controller2_ldData_1_valid,
      ldData_ready(0) => mem_controller2_ldData_0_ready,
      ldData_ready(1) => mem_controller2_ldData_1_ready,
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

  mem_controller3 : entity work.mem_controller_storeless(arch) generic map(1, 32, 4)
    port map(
      loadData => x_loadData,
      memStart_valid => x_start_valid,
      memStart_ready => x_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork25_outs_0_valid,
      ctrlEnd_ready => fork25_outs_0_ready,
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

  extsi6 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  mux7 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => init0_outs,
      index_valid => init0_outs_valid,
      index_ready => init0_outs_ready,
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br13_falseOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork4_outs_4,
      ins_valid => fork4_outs_4_valid,
      ins_ready => fork4_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  buffer62 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi19_result,
      ins_valid => addi19_result_valid,
      ins_ready => addi19_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer62_outs,
      outs_valid => buffer62_outs_valid,
      outs_ready => buffer62_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      ins(0) => extsi6_outs,
      ins(1) => buffer62_outs,
      ins_valid(0) => extsi6_outs_valid,
      ins_valid(1) => buffer62_outs_valid,
      ins_ready(0) => extsi6_outs_ready,
      ins_ready(1) => buffer62_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer2 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer2_outs,
      ins_valid => buffer2_outs_valid,
      ins_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer3_outs,
      index_valid => buffer3_outs_valid,
      index_ready => buffer3_outs_ready,
      ins(0) => n,
      ins(1) => cond_br5_falseOut,
      ins_valid(0) => n_valid,
      ins_valid(1) => cond_br5_falseOut_valid,
      ins_ready(0) => n_ready,
      ins_ready(1) => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer3 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  buffer4 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer6 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => cond_br10_falseOut_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => buffer5_outs,
      lhs_valid => buffer5_outs_valid,
      lhs_ready => buffer5_outs_ready,
      rhs => fork2_outs_1,
      rhs_valid => fork2_outs_1_valid,
      rhs_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer5 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer7 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => buffer7_outs,
      ins_valid => buffer7_outs_valid,
      ins_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs(2) => fork4_outs_2,
      outs(3) => fork4_outs_3,
      outs(4) => fork4_outs_4,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_valid(3) => fork4_outs_3_valid,
      outs_valid(4) => fork4_outs_4_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready,
      outs_ready(3) => fork4_outs_3_ready,
      outs_ready(4) => fork4_outs_4_ready
    );

  cond_br2 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork4_outs_3,
      condition_valid => fork4_outs_3_valid,
      condition_ready => fork4_outs_3_ready,
      data => fork2_outs_0,
      data_valid => fork2_outs_0_valid,
      data_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br3 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork4_outs_2,
      condition_valid => fork4_outs_2_valid,
      condition_ready => fork4_outs_2_ready,
      data => fork1_outs_0,
      data_valid => fork1_outs_0_valid,
      data_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br4 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer10_outs,
      condition_valid => buffer10_outs_valid,
      condition_ready => buffer10_outs_ready,
      data_valid => control_merge0_outs_valid,
      data_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  buffer10 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork4_outs_1,
      ins_valid => fork4_outs_1_valid,
      ins_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br2_trueOut,
      ins_valid => cond_br2_trueOut_valid,
      ins_ready => cond_br2_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  buffer8 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  fork7 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant5 : entity work.handshake_constant_1(arch) generic map(32)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  constant3 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork7_outs_0_valid,
      ctrl_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  subi1 : entity work.subi(arch) generic map(32)
    port map(
      lhs => fork5_outs_1,
      lhs_valid => fork5_outs_1_valid,
      lhs_ready => fork5_outs_1_ready,
      rhs => fork6_outs_1,
      rhs_valid => fork6_outs_1_valid,
      rhs_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      result => subi1_result,
      result_valid => subi1_result_valid,
      result_ready => subi1_result_ready
    );

  buffer9 : entity work.oehb(arch) generic map(32)
    port map(
      ins => subi1_result,
      ins_valid => subi1_result_valid,
      ins_ready => subi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork8_outs_1,
      lhs_valid => fork8_outs_1_valid,
      lhs_ready => fork8_outs_1_ready,
      rhs => constant5_outs,
      rhs_valid => constant5_outs_valid,
      rhs_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  extsi7 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  buffer16 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer1 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  cond_br12 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork4_outs_0,
      condition_valid => fork4_outs_0_valid,
      condition_ready => fork4_outs_0_ready,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br12_falseOut_valid,
      ins_ready => cond_br12_falseOut_ready,
      clk => clk,
      rst => rst
    );

  mux8 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => init1_outs,
      index_valid => init1_outs_valid,
      index_ready => init1_outs_ready,
      ins_valid(0) => cond_br12_trueOut_valid,
      ins_valid(1) => buffer0_outs_valid,
      ins_ready(0) => cond_br12_trueOut_ready,
      ins_ready(1) => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  init1 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => init1_outs,
      outs_valid => init1_outs_valid,
      outs_ready => init1_outs_ready
    );

  buffer18 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork12_outs_7,
      ins_valid => fork12_outs_7_valid,
      ins_ready => fork12_outs_7_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer19_outs,
      index_valid => buffer19_outs_valid,
      index_ready => buffer19_outs_ready,
      ins(0) => extsi7_outs,
      ins(1) => buffer61_outs,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => buffer61_outs_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => buffer61_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer19 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork11_outs_0,
      ins_valid => fork11_outs_0_valid,
      ins_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer13 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
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

  fork9 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_1,
      index_valid => fork11_outs_1_valid,
      index_ready => fork11_outs_1_ready,
      ins(0) => fork5_outs_0,
      ins(1) => fork13_outs_1,
      ins_valid(0) => fork5_outs_0_valid,
      ins_valid(1) => fork13_outs_1_valid,
      ins_ready(0) => fork5_outs_0_ready,
      ins_ready(1) => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_2,
      index_valid => fork11_outs_2_valid,
      index_ready => fork11_outs_2_ready,
      ins(0) => fork6_outs_0,
      ins(1) => fork14_outs_0,
      ins_valid(0) => fork6_outs_0_valid,
      ins_valid(1) => fork14_outs_0_valid,
      ins_ready(0) => fork6_outs_0_ready,
      ins_ready(1) => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_3,
      index_valid => fork11_outs_3_valid,
      index_ready => fork11_outs_3_ready,
      ins(0) => buffer16_outs,
      ins(1) => fork15_outs_2,
      ins_valid(0) => buffer16_outs_valid,
      ins_valid(1) => fork15_outs_2_valid,
      ins_ready(0) => buffer16_outs_ready,
      ins_ready(1) => fork15_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_4,
      index_valid => fork11_outs_4_valid,
      index_ready => fork11_outs_4_ready,
      ins(0) => addi1_result,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => addi1_result_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => addi1_result_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer23 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  buffer25 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer23_outs,
      ins_valid => buffer23_outs_valid,
      ins_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  control_merge2 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork7_outs_1_valid,
      ins_valid(1) => fork17_outs_1_valid,
      ins_ready(0) => fork7_outs_1_ready,
      ins_ready(1) => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs(2) => fork11_outs_2,
      outs(3) => fork11_outs_3,
      outs(4) => fork11_outs_4,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_valid(3) => fork11_outs_3_valid,
      outs_valid(4) => fork11_outs_4_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready,
      outs_ready(3) => fork11_outs_3_ready,
      outs_ready(4) => fork11_outs_4_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork9_outs_1,
      lhs_valid => fork9_outs_1_valid,
      lhs_ready => fork9_outs_1_ready,
      rhs => buffer24_outs,
      rhs_valid => buffer24_outs_valid,
      rhs_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer24 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork10_outs_1,
      ins_valid => fork10_outs_1_valid,
      ins_ready => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  buffer27 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(8, 1)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs(3) => fork12_outs_3,
      outs(4) => fork12_outs_4,
      outs(5) => fork12_outs_5,
      outs(6) => fork12_outs_6,
      outs(7) => fork12_outs_7,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_valid(3) => fork12_outs_3_valid,
      outs_valid(4) => fork12_outs_4_valid,
      outs_valid(5) => fork12_outs_5_valid,
      outs_valid(6) => fork12_outs_6_valid,
      outs_valid(7) => fork12_outs_7_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready,
      outs_ready(3) => fork12_outs_3_ready,
      outs_ready(4) => fork12_outs_4_ready,
      outs_ready(5) => fork12_outs_5_ready,
      outs_ready(6) => fork12_outs_6_ready,
      outs_ready(7) => fork12_outs_7_ready
    );

  buffer15 : entity work.oehb(arch) generic map(32)
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

  buffer17 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  cond_br5 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_6,
      condition_valid => fork12_outs_6_valid,
      condition_ready => fork12_outs_6_ready,
      data => buffer17_outs,
      data_valid => buffer17_outs_valid,
      data_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer20 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  buffer21 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_5,
      condition_valid => fork12_outs_5_valid,
      condition_ready => fork12_outs_5_ready,
      data => buffer21_outs,
      data_valid => buffer21_outs_valid,
      data_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  buffer22 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_4,
      condition_valid => fork12_outs_4_valid,
      condition_ready => fork12_outs_4_ready,
      data => buffer22_outs,
      data_valid => buffer22_outs_valid,
      data_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink4 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br8 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_3,
      condition_valid => fork12_outs_3_valid,
      condition_ready => fork12_outs_3_ready,
      data => fork10_outs_0,
      data_valid => fork10_outs_0_valid,
      data_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br9 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_2,
      condition_valid => fork12_outs_2_valid,
      condition_ready => fork12_outs_2_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  sink6 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br9_falseOut,
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer26 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  cond_br10 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork12_outs_1,
      condition_valid => fork12_outs_1_valid,
      condition_ready => fork12_outs_1_ready,
      data_valid => buffer26_outs_valid,
      data_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  buffer11 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
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

  cond_br13 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer34_outs,
      condition_valid => buffer34_outs_valid,
      condition_ready => buffer34_outs_ready,
      data_valid => buffer12_outs_valid,
      data_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  buffer34 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  buffer29 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br5_trueOut,
      ins_valid => cond_br5_trueOut_valid,
      ins_ready => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(6, 32)
    port map(
      ins => buffer29_outs,
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs(3) => fork13_outs_3,
      outs(4) => fork13_outs_4,
      outs(5) => fork13_outs_5,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_valid(3) => fork13_outs_3_valid,
      outs_valid(4) => fork13_outs_4_valid,
      outs_valid(5) => fork13_outs_5_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready,
      outs_ready(3) => fork13_outs_3_ready,
      outs_ready(4) => fork13_outs_4_ready,
      outs_ready(5) => fork13_outs_5_ready
    );

  trunci0 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  buffer35 : entity work.tfifo(arch) generic map(2, 32)
    port map(
      ins => fork13_outs_0,
      ins_valid => fork13_outs_0_valid,
      ins_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => cond_br6_trueOut,
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs(2) => fork14_outs_2,
      outs(3) => fork14_outs_3,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_valid(3) => fork14_outs_3_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready,
      outs_ready(3) => fork14_outs_3_ready
    );

  buffer28 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br7_trueOut,
      ins_valid => cond_br7_trueOut_valid,
      ins_ready => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => buffer28_outs,
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs(2) => fork15_outs_2,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_valid(2) => fork15_outs_2_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready,
      outs_ready(2) => fork15_outs_2_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => fork15_outs_0,
      ins_valid => fork15_outs_0_valid,
      ins_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork15_outs_1,
      ins_valid => fork15_outs_1_valid,
      ins_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => cond_br9_trueOut,
      ins_valid => cond_br9_trueOut_valid,
      ins_ready => cond_br9_trueOut_ready,
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

  fork17 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  constant4 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork17_outs_0_valid,
      ctrl_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant7 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => constant7_outs,
      ins_valid => constant7_outs_valid,
      ins_ready => constant7_outs_ready,
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

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant8 : entity work.handshake_constant_1(arch) generic map(32)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => constant8_outs,
      ins_valid => constant8_outs_valid,
      ins_ready => constant8_outs_ready,
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

  trunci3 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  trunci4 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork19_outs_1,
      ins_valid => fork19_outs_1_valid,
      ins_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant12 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant12_outs,
      ins_valid => constant12_outs_valid,
      ins_ready => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(7, 32)
    port map(
      ins => extsi3_outs,
      ins_valid => extsi3_outs_valid,
      ins_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs(2) => fork20_outs_2,
      outs(3) => fork20_outs_3,
      outs(4) => fork20_outs_4,
      outs(5) => fork20_outs_5,
      outs(6) => fork20_outs_6,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_valid(2) => fork20_outs_2_valid,
      outs_valid(3) => fork20_outs_3_valid,
      outs_valid(4) => fork20_outs_4_valid,
      outs_valid(5) => fork20_outs_5_valid,
      outs_valid(6) => fork20_outs_6_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready,
      outs_ready(2) => fork20_outs_2_ready,
      outs_ready(3) => fork20_outs_3_ready,
      outs_ready(4) => fork20_outs_4_ready,
      outs_ready(5) => fork20_outs_5_ready,
      outs_ready(6) => fork20_outs_6_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant13 : entity work.handshake_constant_4(arch) generic map(3)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant13_outs,
      outs_valid => constant13_outs_valid,
      outs_ready => constant13_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => constant13_outs,
      ins_valid => constant13_outs_valid,
      ins_ready => constant13_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi4_outs,
      ins_valid => extsi4_outs_valid,
      ins_ready => extsi4_outs_ready,
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

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork14_outs_3,
      lhs_valid => fork14_outs_3_valid,
      lhs_ready => fork14_outs_3_ready,
      rhs => fork16_outs_3,
      rhs_valid => fork16_outs_3_valid,
      rhs_ready => fork16_outs_3_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  buffer30 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  xori0 : entity work.xori(arch) generic map(32)
    port map(
      lhs => buffer30_outs,
      lhs_valid => buffer30_outs_valid,
      lhs_ready => buffer30_outs_ready,
      rhs => fork19_outs_4,
      rhs_valid => fork19_outs_4_valid,
      rhs_ready => fork19_outs_4_ready,
      clk => clk,
      rst => rst,
      result => xori0_result,
      result_valid => xori0_result_valid,
      result_ready => xori0_result_ready
    );

  addi2 : entity work.addi(arch) generic map(32)
    port map(
      lhs => xori0_result,
      lhs_valid => xori0_result_valid,
      lhs_ready => xori0_result_ready,
      rhs => fork20_outs_0,
      rhs_valid => fork20_outs_0_valid,
      rhs_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer31 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  addi3 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer31_outs,
      lhs_valid => buffer31_outs_valid,
      lhs_ready => buffer31_outs_ready,
      rhs => fork13_outs_5,
      rhs_valid => fork13_outs_5_valid,
      rhs_ready => fork13_outs_5_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer32 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  addi4 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer32_outs,
      lhs_valid => buffer32_outs_valid,
      lhs_ready => buffer32_outs_ready,
      rhs => fork18_outs_2,
      rhs_valid => fork18_outs_2_valid,
      rhs_ready => fork18_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer33 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  fork22 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer33_outs,
      ins_valid => buffer33_outs_valid,
      ins_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  addi6 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci1_outs,
      lhs_valid => trunci1_outs_valid,
      lhs_ready => trunci1_outs_ready,
      rhs => trunci3_outs,
      rhs_valid => trunci3_outs_valid,
      rhs_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_1,
      lhs_valid => fork22_outs_1_valid,
      lhs_ready => fork22_outs_1_ready,
      rhs => fork20_outs_1,
      rhs_valid => fork20_outs_1_valid,
      rhs_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer37 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  trunci5 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer37_outs,
      ins_valid => buffer37_outs_valid,
      ins_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  shli1 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork22_outs_0,
      lhs_valid => fork22_outs_0_valid,
      lhs_ready => fork22_outs_0_ready,
      rhs => fork21_outs_0,
      rhs_valid => fork21_outs_0_valid,
      rhs_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      result => shli1_result,
      result_valid => shli1_result_valid,
      result_ready => shli1_result_ready
    );

  buffer38 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli1_result,
      ins_valid => shli1_result_valid,
      ins_ready => shli1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer38_outs,
      ins_valid => buffer38_outs_valid,
      ins_ready => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  addi5 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci5_outs,
      lhs_valid => trunci5_outs_valid,
      lhs_ready => trunci5_outs_ready,
      rhs => trunci6_outs,
      rhs_valid => trunci6_outs_valid,
      rhs_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer36 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  buffer39 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  addi15 : entity work.addi(arch) generic map(7)
    port map(
      lhs => buffer36_outs,
      lhs_valid => buffer36_outs_valid,
      lhs_ready => buffer36_outs_ready,
      rhs => buffer39_outs,
      rhs_valid => buffer39_outs_valid,
      rhs_ready => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      result => addi15_result,
      result_valid => addi15_result_valid,
      result_ready => addi15_result_ready
    );

  load0 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => addi15_result,
      addrIn_valid => addi15_result_valid,
      addrIn_ready => addi15_result_ready,
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

  addi8 : entity work.addi(arch) generic map(4)
    port map(
      lhs => trunci2_outs,
      lhs_valid => trunci2_outs_valid,
      lhs_ready => trunci2_outs_ready,
      rhs => trunci4_outs,
      rhs_valid => trunci4_outs_valid,
      rhs_ready => trunci4_outs_ready,
      clk => clk,
      rst => rst,
      result => addi8_result,
      result_valid => addi8_result_valid,
      result_ready => addi8_result_ready
    );

  load1 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => addi8_result,
      addrIn_valid => addi8_result_valid,
      addrIn_ready => addi8_result_ready,
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

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  addi7 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork14_outs_2,
      lhs_valid => fork14_outs_2_valid,
      lhs_ready => fork14_outs_2_ready,
      rhs => fork16_outs_2,
      rhs_valid => fork16_outs_2_valid,
      rhs_ready => fork16_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer40 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  xori1 : entity work.xori(arch) generic map(32)
    port map(
      lhs => buffer40_outs,
      lhs_valid => buffer40_outs_valid,
      lhs_ready => buffer40_outs_ready,
      rhs => fork19_outs_3,
      rhs_valid => fork19_outs_3_valid,
      rhs_ready => fork19_outs_3_ready,
      clk => clk,
      rst => rst,
      result => xori1_result,
      result_valid => xori1_result_valid,
      result_ready => xori1_result_ready
    );

  addi9 : entity work.addi(arch) generic map(32)
    port map(
      lhs => xori1_result,
      lhs_valid => xori1_result_valid,
      lhs_ready => xori1_result_ready,
      rhs => fork20_outs_2,
      rhs_valid => fork20_outs_2_valid,
      rhs_ready => fork20_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi9_result,
      result_valid => addi9_result_valid,
      result_ready => addi9_result_ready
    );

  buffer41 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi9_result,
      ins_valid => addi9_result_valid,
      ins_ready => addi9_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  addi10 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer41_outs,
      lhs_valid => buffer41_outs_valid,
      lhs_ready => buffer41_outs_ready,
      rhs => buffer54_outs,
      rhs_valid => buffer54_outs_valid,
      rhs_ready => buffer54_outs_ready,
      clk => clk,
      rst => rst,
      result => addi10_result,
      result_valid => addi10_result_valid,
      result_ready => addi10_result_ready
    );

  buffer54 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork13_outs_4,
      ins_valid => fork13_outs_4_valid,
      ins_ready => fork13_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer54_outs,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  buffer42 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi10_result,
      ins_valid => addi10_result_valid,
      ins_ready => addi10_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  addi11 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer42_outs,
      lhs_valid => buffer42_outs_valid,
      lhs_ready => buffer42_outs_ready,
      rhs => fork18_outs_1,
      rhs_valid => fork18_outs_1_valid,
      rhs_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi11_result,
      result_valid => addi11_result_valid,
      result_ready => addi11_result_ready
    );

  buffer43 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi11_result,
      ins_valid => addi11_result_valid,
      ins_ready => addi11_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer43_outs,
      ins_valid => buffer43_outs_valid,
      ins_ready => buffer43_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  shli2 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer56_outs,
      lhs_valid => buffer56_outs_valid,
      lhs_ready => buffer56_outs_ready,
      rhs => buffer57_outs,
      rhs_valid => buffer57_outs_valid,
      rhs_ready => buffer57_outs_ready,
      clk => clk,
      rst => rst,
      result => shli2_result,
      result_valid => shli2_result_valid,
      result_ready => shli2_result_ready
    );

  buffer56 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer56_outs,
      outs_valid => buffer56_outs_valid,
      outs_ready => buffer56_outs_ready
    );

  buffer57 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork20_outs_3,
      ins_valid => fork20_outs_3_valid,
      ins_ready => fork20_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer57_outs,
      outs_valid => buffer57_outs_valid,
      outs_ready => buffer57_outs_ready
    );

  shli3 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer58_outs,
      lhs_valid => buffer58_outs_valid,
      lhs_ready => buffer58_outs_ready,
      rhs => buffer59_outs,
      rhs_valid => buffer59_outs_valid,
      rhs_ready => buffer59_outs_ready,
      clk => clk,
      rst => rst,
      result => shli3_result,
      result_valid => shli3_result_valid,
      result_ready => shli3_result_ready
    );

  buffer58 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork23_outs_0,
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer58_outs,
      outs_valid => buffer58_outs_valid,
      outs_ready => buffer58_outs_ready
    );

  buffer59 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork21_outs_1,
      ins_valid => fork21_outs_1_valid,
      ins_ready => fork21_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer59_outs,
      outs_valid => buffer59_outs_valid,
      outs_ready => buffer59_outs_ready
    );

  buffer44 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli2_result,
      ins_valid => shli2_result_valid,
      ins_ready => shli2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  buffer45 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli3_result,
      ins_valid => shli3_result_valid,
      ins_ready => shli3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  addi12 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer44_outs,
      lhs_valid => buffer44_outs_valid,
      lhs_ready => buffer44_outs_ready,
      rhs => buffer45_outs,
      rhs_valid => buffer45_outs_valid,
      rhs_ready => buffer45_outs_ready,
      clk => clk,
      rst => rst,
      result => addi12_result,
      result_valid => addi12_result_valid,
      result_ready => addi12_result_ready
    );

  buffer46 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi12_result,
      ins_valid => addi12_result_valid,
      ins_ready => addi12_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  addi16 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer60_outs,
      lhs_valid => buffer60_outs_valid,
      lhs_ready => buffer60_outs_ready,
      rhs => buffer46_outs,
      rhs_valid => buffer46_outs_valid,
      rhs_ready => buffer46_outs_ready,
      clk => clk,
      rst => rst,
      result => addi16_result,
      result_valid => addi16_result_valid,
      result_ready => addi16_result_ready
    );

  buffer60 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork13_outs_3,
      ins_valid => fork13_outs_3_valid,
      ins_ready => fork13_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer60_outs,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => addi16_result,
      ins_valid(0) => addi16_result_valid,
      ins_valid(1) => cond_br13_trueOut_valid,
      ins_ready(0) => addi16_result_ready,
      ins_ready(1) => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => trunci7_outs,
      addrIn_valid => trunci7_outs_valid,
      addrIn_ready => trunci7_outs_ready,
      dataFromMem => mem_controller2_ldData_1,
      dataFromMem_valid => mem_controller2_ldData_1_valid,
      dataFromMem_ready => mem_controller2_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  subi0 : entity work.subi(arch) generic map(32)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => muli0_result,
      rhs_valid => muli0_result_valid,
      rhs_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      result => subi0_result,
      result_valid => subi0_result_valid,
      result_ready => subi0_result_ready
    );

  addi20 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork14_outs_1,
      lhs_valid => fork14_outs_1_valid,
      lhs_ready => fork14_outs_1_ready,
      rhs => fork16_outs_1,
      rhs_valid => fork16_outs_1_valid,
      rhs_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi20_result,
      result_valid => addi20_result_valid,
      result_ready => addi20_result_ready
    );

  buffer47 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi20_result,
      ins_valid => addi20_result_valid,
      ins_ready => addi20_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  xori2 : entity work.xori(arch) generic map(32)
    port map(
      lhs => buffer47_outs,
      lhs_valid => buffer47_outs_valid,
      lhs_ready => buffer47_outs_ready,
      rhs => fork19_outs_2,
      rhs_valid => fork19_outs_2_valid,
      rhs_ready => fork19_outs_2_ready,
      clk => clk,
      rst => rst,
      result => xori2_result,
      result_valid => xori2_result_valid,
      result_ready => xori2_result_ready
    );

  addi21 : entity work.addi(arch) generic map(32)
    port map(
      lhs => xori2_result,
      lhs_valid => xori2_result_valid,
      lhs_ready => xori2_result_ready,
      rhs => fork20_outs_4,
      rhs_valid => fork20_outs_4_valid,
      rhs_ready => fork20_outs_4_ready,
      clk => clk,
      rst => rst,
      result => addi21_result,
      result_valid => addi21_result_valid,
      result_ready => addi21_result_ready
    );

  buffer48 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi21_result,
      ins_valid => addi21_result_valid,
      ins_ready => addi21_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer48_outs,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  addi13 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer48_outs,
      lhs_valid => buffer48_outs_valid,
      lhs_ready => buffer48_outs_ready,
      rhs => fork13_outs_2,
      rhs_valid => fork13_outs_2_valid,
      rhs_ready => fork13_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi13_result,
      result_valid => addi13_result_valid,
      result_ready => addi13_result_ready
    );

  buffer49 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi13_result,
      ins_valid => addi13_result_valid,
      ins_ready => addi13_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer49_outs,
      outs_valid => buffer49_outs_valid,
      outs_ready => buffer49_outs_ready
    );

  addi14 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer49_outs,
      lhs_valid => buffer49_outs_valid,
      lhs_ready => buffer49_outs_ready,
      rhs => buffer66_outs,
      rhs_valid => buffer66_outs_valid,
      rhs_ready => buffer66_outs_ready,
      clk => clk,
      rst => rst,
      result => addi14_result,
      result_valid => addi14_result_valid,
      result_ready => addi14_result_ready
    );

  buffer66 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer66_outs,
      outs_valid => buffer66_outs_valid,
      outs_ready => buffer66_outs_ready
    );

  buffer50 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi14_result,
      ins_valid => addi14_result_valid,
      ins_ready => addi14_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer50_outs,
      outs_valid => buffer50_outs_valid,
      outs_ready => buffer50_outs_ready
    );

  fork24 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer50_outs,
      ins_valid => buffer50_outs_valid,
      ins_ready => buffer50_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  shli4 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer67_outs,
      lhs_valid => buffer67_outs_valid,
      lhs_ready => buffer67_outs_ready,
      rhs => buffer68_outs,
      rhs_valid => buffer68_outs_valid,
      rhs_ready => buffer68_outs_ready,
      clk => clk,
      rst => rst,
      result => shli4_result,
      result_valid => shli4_result_valid,
      result_ready => shli4_result_ready
    );

  buffer67 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork24_outs_1,
      ins_valid => fork24_outs_1_valid,
      ins_ready => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer67_outs,
      outs_valid => buffer67_outs_valid,
      outs_ready => buffer67_outs_ready
    );

  buffer68 : entity work.tfifo(arch) generic map(2, 32)
    port map(
      ins => fork20_outs_5,
      ins_valid => fork20_outs_5_valid,
      ins_ready => fork20_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer68_outs,
      outs_valid => buffer68_outs_valid,
      outs_ready => buffer68_outs_ready
    );

  buffer51 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli4_result,
      ins_valid => shli4_result_valid,
      ins_ready => shli4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer51_outs,
      outs_valid => buffer51_outs_valid,
      outs_ready => buffer51_outs_ready
    );

  trunci8 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer51_outs,
      ins_valid => buffer51_outs_valid,
      ins_ready => buffer51_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  shli5 : entity work.shli(arch) generic map(32)
    port map(
      lhs => buffer69_outs,
      lhs_valid => buffer69_outs_valid,
      lhs_ready => buffer69_outs_ready,
      rhs => buffer70_outs,
      rhs_valid => buffer70_outs_valid,
      rhs_ready => buffer70_outs_ready,
      clk => clk,
      rst => rst,
      result => shli5_result,
      result_valid => shli5_result_valid,
      result_ready => shli5_result_ready
    );

  buffer69 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork24_outs_0,
      ins_valid => fork24_outs_0_valid,
      ins_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer69_outs,
      outs_valid => buffer69_outs_valid,
      outs_ready => buffer69_outs_ready
    );

  buffer70 : entity work.tfifo(arch) generic map(2, 32)
    port map(
      ins => fork21_outs_2,
      ins_valid => fork21_outs_2_valid,
      ins_ready => fork21_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer70_outs,
      outs_valid => buffer70_outs_valid,
      outs_ready => buffer70_outs_ready
    );

  buffer52 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli5_result,
      ins_valid => shli5_result_valid,
      ins_ready => shli5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer52_outs,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  trunci9 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => buffer52_outs,
      ins_valid => buffer52_outs_valid,
      ins_ready => buffer52_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  addi22 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci8_outs,
      lhs_valid => trunci8_outs_valid,
      lhs_ready => trunci8_outs_ready,
      rhs => trunci9_outs,
      rhs_valid => trunci9_outs_valid,
      rhs_ready => trunci9_outs_ready,
      clk => clk,
      rst => rst,
      result => addi22_result,
      result_valid => addi22_result_valid,
      result_ready => addi22_result_ready
    );

  buffer53 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi22_result,
      ins_valid => addi22_result_valid,
      ins_ready => addi22_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer53_outs,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  addi17 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci0_outs,
      lhs_valid => trunci0_outs_valid,
      lhs_ready => trunci0_outs_ready,
      rhs => buffer53_outs,
      rhs_valid => buffer53_outs_valid,
      rhs_ready => buffer53_outs_ready,
      clk => clk,
      rst => rst,
      result => addi17_result,
      result_valid => addi17_result_valid,
      result_ready => addi17_result_ready
    );

  buffer55 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => store0_doneOut_valid,
      ins_ready => store0_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer55_outs_valid,
      outs_ready => buffer55_outs_ready
    );

  buffer0 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer55_outs_valid,
      ins_ready => buffer55_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => addi17_result,
      addrIn_valid => addi17_result_valid,
      addrIn_ready => addi17_result_ready,
      dataIn => subi0_result,
      dataIn_valid => subi0_result_valid,
      dataIn_ready => subi0_result_ready,
      doneFromMem_valid => mem_controller2_stDone_0_valid,
      doneFromMem_ready => mem_controller2_stDone_0_ready,
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

  addi18 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork16_outs_0,
      lhs_valid => fork16_outs_0_valid,
      lhs_ready => fork16_outs_0_ready,
      rhs => buffer71_outs,
      rhs_valid => buffer71_outs_valid,
      rhs_ready => buffer71_outs_ready,
      clk => clk,
      rst => rst,
      result => addi18_result,
      result_valid => addi18_result_valid,
      result_ready => addi18_result_ready
    );

  buffer71 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork20_outs_6,
      ins_valid => fork20_outs_6_valid,
      ins_ready => fork20_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer71_outs,
      outs_valid => buffer71_outs_valid,
      outs_ready => buffer71_outs_ready
    );

  buffer61 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi18_result,
      ins_valid => addi18_result_valid,
      ins_ready => addi18_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer61_outs,
      outs_valid => buffer61_outs_valid,
      outs_ready => buffer61_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant14 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant14_outs,
      outs_valid => constant14_outs_valid,
      outs_ready => constant14_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant14_outs,
      ins_valid => constant14_outs_valid,
      ins_ready => constant14_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  addi19 : entity work.addi(arch) generic map(32)
    port map(
      lhs => cond_br6_falseOut,
      lhs_valid => cond_br6_falseOut_valid,
      lhs_ready => cond_br6_falseOut_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi19_result,
      result_valid => addi19_result_valid,
      result_ready => addi19_result_ready
    );

  fork25 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

end architecture;
