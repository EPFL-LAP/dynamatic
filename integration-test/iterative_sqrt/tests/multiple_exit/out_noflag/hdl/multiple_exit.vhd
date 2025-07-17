library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multiple_exit is
  port (
    arr_loadData : in std_logic_vector(31 downto 0);
    arr_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    arr_end_ready : in std_logic;
    end_ready : in std_logic;
    arr_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    arr_end_valid : out std_logic;
    end_valid : out std_logic;
    arr_loadEn : out std_logic;
    arr_loadAddr : out std_logic_vector(3 downto 0);
    arr_storeEn : out std_logic;
    arr_storeAddr : out std_logic_vector(3 downto 0);
    arr_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of multiple_exit is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal fork0_outs_3_valid : std_logic;
  signal fork0_outs_3_ready : std_logic;
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal buffer48_outs_valid : std_logic;
  signal buffer48_outs_ready : std_logic;
  signal lsq1_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_0_valid : std_logic;
  signal lsq1_ldData_0_ready : std_logic;
  signal lsq1_ldData_1 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_1_valid : std_logic;
  signal lsq1_ldData_1_ready : std_logic;
  signal lsq1_ldData_2 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_2_valid : std_logic;
  signal lsq1_ldData_2_ready : std_logic;
  signal lsq1_memEnd_valid : std_logic;
  signal lsq1_memEnd_ready : std_logic;
  signal lsq1_loadEn : std_logic;
  signal lsq1_loadAddr : std_logic_vector(3 downto 0);
  signal lsq1_storeEn : std_logic;
  signal lsq1_storeAddr : std_logic_vector(3 downto 0);
  signal lsq1_storeData : std_logic_vector(31 downto 0);
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(0 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(0 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal constant6_outs : std_logic_vector(0 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal extsi7_outs : std_logic_vector(1 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(31 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal mux1_outs : std_logic_vector(0 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(1 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(0 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(1 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(1 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(1 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(1 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal fork4_outs_3 : std_logic_vector(1 downto 0);
  signal fork4_outs_3_valid : std_logic;
  signal fork4_outs_3_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(4 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer6_outs : std_logic_vector(0 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(0 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
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
  signal buffer2_outs : std_logic_vector(0 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(0 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(0 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(0 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer4_outs : std_logic_vector(1 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(1 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(1 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(1 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(31 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal trunci0_outs : std_logic_vector(3 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(3 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal lazy_fork0_outs_2_valid : std_logic;
  signal lazy_fork0_outs_2_ready : std_logic;
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal constant8_outs : std_logic_vector(0 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(31 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(3 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(0 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(0 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(0 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal fork8_outs_3 : std_logic_vector(0 downto 0);
  signal fork8_outs_3_valid : std_logic;
  signal fork8_outs_3_ready : std_logic;
  signal fork8_outs_4 : std_logic_vector(0 downto 0);
  signal fork8_outs_4_valid : std_logic;
  signal fork8_outs_4_ready : std_logic;
  signal fork8_outs_5 : std_logic_vector(0 downto 0);
  signal fork8_outs_5_valid : std_logic;
  signal fork8_outs_5_ready : std_logic;
  signal buffer10_outs : std_logic_vector(0 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(0 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(0 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer12_outs : std_logic_vector(1 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(1 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(1 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(1 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut : std_logic_vector(31 downto 0);
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut : std_logic_vector(31 downto 0);
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(3 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(3 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(0 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(0 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal buffer26_outs : std_logic_vector(3 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal buffer27_outs : std_logic_vector(3 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(3 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(3 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(1 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(0 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal load4_addrOut : std_logic_vector(3 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
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
  signal buffer20_outs : std_logic_vector(0 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal buffer22_outs : std_logic_vector(1 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(1 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal select0_result : std_logic_vector(1 downto 0);
  signal select0_result_valid : std_logic;
  signal select0_result_ready : std_logic;
  signal buffer24_outs : std_logic_vector(31 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal buffer25_outs : std_logic_vector(31 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal cond_br13_trueOut : std_logic_vector(31 downto 0);
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut : std_logic_vector(31 downto 0);
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal cond_br14_trueOut : std_logic_vector(3 downto 0);
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut : std_logic_vector(3 downto 0);
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal cond_br15_trueOut : std_logic_vector(0 downto 0);
  signal cond_br15_trueOut_valid : std_logic;
  signal cond_br15_trueOut_ready : std_logic;
  signal cond_br15_falseOut : std_logic_vector(0 downto 0);
  signal cond_br15_falseOut_valid : std_logic;
  signal cond_br15_falseOut_ready : std_logic;
  signal cond_br16_trueOut : std_logic_vector(1 downto 0);
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut : std_logic_vector(1 downto 0);
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal cond_br17_trueOut_valid : std_logic;
  signal cond_br17_trueOut_ready : std_logic;
  signal cond_br17_falseOut_valid : std_logic;
  signal cond_br17_falseOut_ready : std_logic;
  signal cond_br18_trueOut : std_logic_vector(0 downto 0);
  signal cond_br18_trueOut_valid : std_logic;
  signal cond_br18_trueOut_ready : std_logic;
  signal cond_br18_falseOut : std_logic_vector(0 downto 0);
  signal cond_br18_falseOut_valid : std_logic;
  signal cond_br18_falseOut_ready : std_logic;
  signal buffer33_outs : std_logic_vector(3 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal buffer34_outs : std_logic_vector(3 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(3 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(3 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal lazy_fork2_outs_0_valid : std_logic;
  signal lazy_fork2_outs_0_ready : std_logic;
  signal lazy_fork2_outs_1_valid : std_logic;
  signal lazy_fork2_outs_1_ready : std_logic;
  signal lazy_fork2_outs_2_valid : std_logic;
  signal lazy_fork2_outs_2_ready : std_logic;
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal constant12_outs : std_logic_vector(0 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(1 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal load5_addrOut : std_logic_vector(3 downto 0);
  signal load5_addrOut_valid : std_logic;
  signal load5_addrOut_ready : std_logic;
  signal load5_dataOut : std_logic_vector(31 downto 0);
  signal load5_dataOut_valid : std_logic;
  signal load5_dataOut_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal store1_addrOut : std_logic_vector(3 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal buffer32_outs : std_logic_vector(31 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal buffer35_outs : std_logic_vector(0 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal buffer36_outs : std_logic_vector(0 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal buffer37_outs : std_logic_vector(1 downto 0);
  signal buffer37_outs_valid : std_logic;
  signal buffer37_outs_ready : std_logic;
  signal buffer38_outs : std_logic_vector(1 downto 0);
  signal buffer38_outs_valid : std_logic;
  signal buffer38_outs_ready : std_logic;
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal buffer45_outs : std_logic_vector(1 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal buffer46_outs : std_logic_vector(1 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(2 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(2 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal buffer43_outs : std_logic_vector(0 downto 0);
  signal buffer43_outs_valid : std_logic;
  signal buffer43_outs_ready : std_logic;
  signal buffer44_outs : std_logic_vector(0 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal select1_result : std_logic_vector(2 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;
  signal extsi10_outs : std_logic_vector(31 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;

begin

  out0 <= extsi10_outs;
  out0_valid <= extsi10_outs_valid;
  extsi10_outs_ready <= out0_ready;
  arr_end_valid <= lsq1_memEnd_valid;
  lsq1_memEnd_ready <= arr_end_ready;
  end_valid <= fork0_outs_0_valid;
  fork0_outs_0_ready <= end_ready;
  arr_loadEn <= lsq1_loadEn;
  arr_loadAddr <= lsq1_loadAddr;
  arr_storeEn <= lsq1_storeEn;
  arr_storeAddr <= lsq1_storeAddr;
  arr_storeData <= lsq1_storeData;

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

  buffer47 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  buffer48 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer47_outs_valid,
      ins_ready => buffer47_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer48_outs_valid,
      outs_ready => buffer48_outs_ready
    );

  lsq1 : entity work.handshake_lsq_lsq1(arch)
    port map(
      io_loadData => arr_loadData,
      io_memStart_valid => arr_start_valid,
      io_memStart_ready => arr_start_ready,
      io_ctrl_0_valid => lazy_fork0_outs_1_valid,
      io_ctrl_0_ready => lazy_fork0_outs_1_ready,
      io_ldAddr_0_bits => load3_addrOut,
      io_ldAddr_0_valid => load3_addrOut_valid,
      io_ldAddr_0_ready => load3_addrOut_ready,
      io_ctrl_1_valid => lazy_fork1_outs_1_valid,
      io_ctrl_1_ready => lazy_fork1_outs_1_ready,
      io_ldAddr_1_bits => load4_addrOut,
      io_ldAddr_1_valid => load4_addrOut_valid,
      io_ldAddr_1_ready => load4_addrOut_ready,
      io_ctrl_2_valid => lazy_fork2_outs_1_valid,
      io_ctrl_2_ready => lazy_fork2_outs_1_ready,
      io_ldAddr_2_bits => load5_addrOut,
      io_ldAddr_2_valid => load5_addrOut_valid,
      io_ldAddr_2_ready => load5_addrOut_ready,
      io_stAddr_0_bits => store1_addrOut,
      io_stAddr_0_valid => store1_addrOut_valid,
      io_stAddr_0_ready => store1_addrOut_ready,
      io_stData_0_bits => store1_dataToMem,
      io_stData_0_valid => store1_dataToMem_valid,
      io_stData_0_ready => store1_dataToMem_ready,
      io_ctrlEnd_valid => buffer48_outs_valid,
      io_ctrlEnd_ready => buffer48_outs_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq1_ldData_0,
      io_ldData_0_valid => lsq1_ldData_0_valid,
      io_ldData_0_ready => lsq1_ldData_0_ready,
      io_ldData_1_bits => lsq1_ldData_1,
      io_ldData_1_valid => lsq1_ldData_1_valid,
      io_ldData_1_ready => lsq1_ldData_1_ready,
      io_ldData_2_bits => lsq1_ldData_2,
      io_ldData_2_valid => lsq1_ldData_2_valid,
      io_ldData_2_ready => lsq1_ldData_2_ready,
      io_memEnd_valid => lsq1_memEnd_valid,
      io_memEnd_ready => lsq1_memEnd_ready,
      io_loadEn => lsq1_loadEn,
      io_loadAddr => lsq1_loadAddr,
      io_storeEn => lsq1_storeEn,
      io_storeAddr => lsq1_storeAddr,
      io_storeData => lsq1_storeData
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_3_valid,
      ctrl_ready => fork0_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  constant6 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_2_valid,
      ctrl_ready => fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  extsi8 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant6_outs,
      ins_valid => constant6_outs_valid,
      ins_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  extsi7 : entity work.extsi(arch) generic map(1, 2)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(4, 32, 2)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => extsi8_outs,
      ins(1) => cond_br9_falseOut,
      ins(2) => cond_br13_falseOut,
      ins(3) => addi1_result,
      ins_valid(0) => extsi8_outs_valid,
      ins_valid(1) => cond_br9_falseOut_valid,
      ins_valid(2) => cond_br13_falseOut_valid,
      ins_valid(3) => addi1_result_valid,
      ins_ready(0) => extsi8_outs_ready,
      ins_ready(1) => cond_br9_falseOut_ready,
      ins_ready(2) => cond_br13_falseOut_ready,
      ins_ready(3) => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer0 : entity work.oehb(arch) generic map(32)
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

  buffer1 : entity work.tehb(arch) generic map(32)
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

  fork3 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  mux1 : entity work.mux(arch) generic map(4, 1, 2)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => fork2_outs_1,
      ins(1) => cond_br7_falseOut,
      ins(2) => cond_br15_falseOut,
      ins(3) => buffer36_outs,
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => cond_br7_falseOut_valid,
      ins_valid(2) => cond_br15_falseOut_valid,
      ins_valid(3) => buffer36_outs_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => cond_br7_falseOut_ready,
      ins_ready(2) => cond_br15_falseOut_ready,
      ins_ready(3) => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(4, 2, 2)
    port map(
      index => fork4_outs_2,
      index_valid => fork4_outs_2_valid,
      index_ready => fork4_outs_2_ready,
      ins(0) => extsi7_outs,
      ins(1) => cond_br8_falseOut,
      ins(2) => cond_br16_falseOut,
      ins(3) => buffer38_outs,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => cond_br8_falseOut_valid,
      ins_valid(2) => cond_br16_falseOut_valid,
      ins_valid(3) => buffer38_outs_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => cond_br8_falseOut_ready,
      ins_ready(2) => cond_br16_falseOut_ready,
      ins_ready(3) => buffer38_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(4, 1, 2)
    port map(
      index => fork4_outs_3,
      index_valid => fork4_outs_3_valid,
      index_ready => fork4_outs_3_ready,
      ins(0) => fork2_outs_0,
      ins(1) => cond_br12_falseOut,
      ins(2) => cond_br18_falseOut,
      ins(3) => constant12_outs,
      ins_valid(0) => fork2_outs_0_valid,
      ins_valid(1) => cond_br12_falseOut_valid,
      ins_valid(2) => cond_br18_falseOut_valid,
      ins_valid(3) => constant12_outs_valid,
      ins_ready(0) => fork2_outs_0_ready,
      ins_ready(1) => cond_br12_falseOut_ready,
      ins_ready(2) => cond_br18_falseOut_ready,
      ins_ready(3) => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(4, 2)
    port map(
      ins_valid(0) => fork0_outs_1_valid,
      ins_valid(1) => cond_br11_falseOut_valid,
      ins_valid(2) => cond_br17_falseOut_valid,
      ins_valid(3) => buffer41_outs_valid,
      ins_ready(0) => fork0_outs_1_ready,
      ins_ready(1) => cond_br11_falseOut_ready,
      ins_ready(2) => cond_br17_falseOut_ready,
      ins_ready(3) => buffer41_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(4, 2)
    port map(
      ins => control_merge0_index,
      ins_valid => control_merge0_index_valid,
      ins_ready => control_merge0_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs(2) => fork4_outs_2,
      outs(3) => fork4_outs_3,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_valid(2) => fork4_outs_2_valid,
      outs_valid(3) => fork4_outs_3_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready,
      outs_ready(2) => fork4_outs_2_ready,
      outs_ready(3) => fork4_outs_3_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant2 : entity work.handshake_constant_2(arch) generic map(5)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork3_outs_1,
      lhs_valid => fork3_outs_1_valid,
      lhs_ready => fork3_outs_1_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer6 : entity work.oehb(arch) generic map(1)
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

  buffer7 : entity work.tehb(arch) generic map(1)
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

  andi0 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      rhs => buffer7_outs,
      rhs_valid => buffer7_outs_valid,
      rhs_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs(2) => fork5_outs_2,
      outs(3) => fork5_outs_3,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_valid(2) => fork5_outs_2_valid,
      outs_valid(3) => fork5_outs_3_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready,
      outs_ready(2) => fork5_outs_2_ready,
      outs_ready(3) => fork5_outs_3_ready
    );

  buffer2 : entity work.oehb(arch) generic map(1)
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

  buffer3 : entity work.tehb(arch) generic map(1)
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

  cond_br3 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork5_outs_3,
      condition_valid => fork5_outs_3_valid,
      condition_ready => fork5_outs_3_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer4 : entity work.oehb(arch) generic map(2)
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

  buffer5 : entity work.tehb(arch) generic map(2)
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

  cond_br4 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork5_outs_2,
      condition_valid => fork5_outs_2_valid,
      condition_ready => fork5_outs_2_ready,
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  cond_br5 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork5_outs_1,
      condition_valid => fork5_outs_1_valid,
      condition_ready => fork5_outs_1_ready,
      data => fork3_outs_0,
      data_valid => fork3_outs_0_valid,
      data_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer8 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  cond_br6 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork5_outs_0,
      condition_valid => fork5_outs_0_valid,
      condition_ready => fork5_outs_0_ready,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  buffer14 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br5_trueOut,
      ins_valid => cond_br5_trueOut_valid,
      ins_ready => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  buffer15 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs(2) => fork6_outs_2,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_valid(2) => fork6_outs_2_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready,
      outs_ready(2) => fork6_outs_2_ready
    );

  trunci0 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork6_outs_2,
      ins_valid => fork6_outs_2_valid,
      ins_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork6_outs_1,
      ins_valid => fork6_outs_1_valid,
      ins_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  buffer16 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer17 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_valid(2) => lazy_fork0_outs_2_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready,
      outs_ready(2) => lazy_fork0_outs_2_ready
    );

  buffer19 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_2_valid,
      ins_ready => lazy_fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork7 : entity work.fork_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_ready(0) => fork7_outs_0_ready
    );

  constant8 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork7_outs_0_valid,
      ctrl_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant9 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => lsq1_ldData_0,
      dataFromMem_valid => lsq1_ldData_0_valid,
      dataFromMem_ready => lsq1_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => load3_dataOut,
      lhs_valid => load3_dataOut_valid,
      lhs_ready => load3_dataOut_ready,
      rhs => constant9_outs,
      rhs_valid => constant9_outs_valid,
      rhs_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(6, 1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs(3) => fork8_outs_3,
      outs(4) => fork8_outs_4,
      outs(5) => fork8_outs_5,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_valid(3) => fork8_outs_3_valid,
      outs_valid(4) => fork8_outs_4_valid,
      outs_valid(5) => fork8_outs_5_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready,
      outs_ready(3) => fork8_outs_3_ready,
      outs_ready(4) => fork8_outs_4_ready,
      outs_ready(5) => fork8_outs_5_ready
    );

  buffer10 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cond_br3_trueOut,
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork8_outs_5,
      condition_valid => fork8_outs_5_valid,
      condition_ready => fork8_outs_5_ready,
      data => buffer11_outs,
      data_valid => buffer11_outs_valid,
      data_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  buffer12 : entity work.oehb(arch) generic map(2)
    port map(
      ins => cond_br4_trueOut,
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(2)
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

  cond_br8 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork8_outs_4,
      condition_valid => fork8_outs_4_valid,
      condition_ready => fork8_outs_4_ready,
      data => buffer13_outs,
      data_valid => buffer13_outs_valid,
      data_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  cond_br9 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork8_outs_3,
      condition_valid => fork8_outs_3_valid,
      condition_ready => fork8_outs_3_ready,
      data => fork6_outs_0,
      data_valid => fork6_outs_0_valid,
      data_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br9_trueOut,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut => cond_br9_falseOut,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(4)
    port map(
      condition => fork8_outs_2,
      condition_valid => fork8_outs_2_valid,
      condition_ready => fork8_outs_2_ready,
      data => trunci0_outs,
      data_valid => trunci0_outs_valid,
      data_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(4)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer18 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_0_valid,
      ins_ready => lazy_fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  cond_br11 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork8_outs_1,
      condition_valid => fork8_outs_1_valid,
      condition_ready => fork8_outs_1_ready,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  cond_br12 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork8_outs_0,
      condition_valid => fork8_outs_0_valid,
      condition_ready => fork8_outs_0_ready,
      data => constant8_outs,
      data_valid => constant8_outs_valid,
      data_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br12_trueOut,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut => cond_br12_falseOut,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  sink3 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br12_trueOut,
      ins_valid => cond_br12_trueOut_valid,
      ins_ready => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer26 : entity work.oehb(arch) generic map(4)
    port map(
      ins => cond_br10_trueOut,
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  buffer27 : entity work.tehb(arch) generic map(4)
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

  fork9 : entity work.handshake_fork(arch) generic map(2, 4)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  buffer28 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br11_trueOut_valid,
      ins_ready => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  buffer29 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
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

  constant3 : entity work.handshake_constant_4(arch) generic map(2)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant4 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(1, 32)
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

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi4_outs,
      ins_valid => extsi4_outs_valid,
      ins_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  load4 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => fork9_outs_1,
      addrIn_valid => fork9_outs_1_valid,
      addrIn_ready => fork9_outs_1_ready,
      dataFromMem => lsq1_ldData_1,
      dataFromMem_valid => lsq1_ldData_1_valid,
      dataFromMem_ready => lsq1_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load4_dataOut,
      ins_valid => load4_dataOut_valid,
      ins_ready => load4_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_2(arch) generic map(32)
    port map(
      lhs => fork11_outs_1,
      lhs_valid => fork11_outs_1_valid,
      lhs_ready => fork11_outs_1_ready,
      rhs => fork10_outs_1,
      rhs_valid => fork10_outs_1_valid,
      rhs_ready => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  cmpi3 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork11_outs_0,
      lhs_valid => fork11_outs_0_valid,
      lhs_ready => fork11_outs_0_ready,
      rhs => fork10_outs_0,
      rhs_valid => fork10_outs_0_valid,
      rhs_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(8, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
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

  buffer20 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cond_br7_trueOut,
      ins_valid => cond_br7_trueOut_valid,
      ins_ready => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  buffer21 : entity work.tehb(arch) generic map(1)
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

  andi1 : entity work.andi(arch) generic map(1)
    port map(
      lhs => fork12_outs_7,
      lhs_valid => fork12_outs_7_valid,
      lhs_ready => fork12_outs_7_ready,
      rhs => buffer21_outs,
      rhs_valid => buffer21_outs_valid,
      rhs_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  buffer22 : entity work.oehb(arch) generic map(2)
    port map(
      ins => cond_br8_trueOut,
      ins_valid => cond_br8_trueOut_valid,
      ins_ready => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  buffer23 : entity work.tehb(arch) generic map(2)
    port map(
      ins => buffer22_outs,
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  select0 : entity work.selector(arch) generic map(2)
    port map(
      condition => cmpi2_result,
      condition_valid => cmpi2_result_valid,
      condition_ready => cmpi2_result_ready,
      trueValue => constant3_outs,
      trueValue_valid => constant3_outs_valid,
      trueValue_ready => constant3_outs_ready,
      falseValue => buffer23_outs,
      falseValue_valid => buffer23_outs_valid,
      falseValue_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      result => select0_result,
      result_valid => select0_result_valid,
      result_ready => select0_result_ready
    );

  buffer24 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br9_trueOut,
      ins_valid => cond_br9_trueOut_valid,
      ins_ready => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  buffer25 : entity work.tehb(arch) generic map(32)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  cond_br13 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork12_outs_6,
      condition_valid => fork12_outs_6_valid,
      condition_ready => fork12_outs_6_ready,
      data => buffer25_outs,
      data_valid => buffer25_outs_valid,
      data_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  cond_br14 : entity work.cond_br(arch) generic map(4)
    port map(
      condition => fork12_outs_5,
      condition_valid => fork12_outs_5_valid,
      condition_ready => fork12_outs_5_ready,
      data => fork9_outs_0,
      data_valid => fork9_outs_0_valid,
      data_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br14_trueOut,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut => cond_br14_falseOut,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(4)
    port map(
      ins => cond_br14_falseOut,
      ins_valid => cond_br14_falseOut_valid,
      ins_ready => cond_br14_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br15 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork12_outs_4,
      condition_valid => fork12_outs_4_valid,
      condition_ready => fork12_outs_4_ready,
      data => andi1_result,
      data_valid => andi1_result_valid,
      data_ready => andi1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  cond_br16 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork12_outs_3,
      condition_valid => fork12_outs_3_valid,
      condition_ready => fork12_outs_3_ready,
      data => select0_result,
      data_valid => select0_result_valid,
      data_ready => select0_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br16_trueOut,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut => cond_br16_falseOut,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  buffer30 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_0_valid,
      ins_ready => lazy_fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  cond_br17 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork12_outs_2,
      condition_valid => fork12_outs_2_valid,
      condition_ready => fork12_outs_2_ready,
      data_valid => buffer30_outs_valid,
      data_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br17_trueOut_valid,
      trueOut_ready => cond_br17_trueOut_ready,
      falseOut_valid => cond_br17_falseOut_valid,
      falseOut_ready => cond_br17_falseOut_ready
    );

  cond_br18 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork12_outs_0,
      condition_valid => fork12_outs_0_valid,
      condition_ready => fork12_outs_0_ready,
      data => fork12_outs_1,
      data_valid => fork12_outs_1_valid,
      data_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br18_trueOut,
      trueOut_valid => cond_br18_trueOut_valid,
      trueOut_ready => cond_br18_trueOut_ready,
      falseOut => cond_br18_falseOut,
      falseOut_valid => cond_br18_falseOut_valid,
      falseOut_ready => cond_br18_falseOut_ready
    );

  sink6 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br18_trueOut,
      ins_valid => cond_br18_trueOut_valid,
      ins_ready => cond_br18_trueOut_ready,
      clk => clk,
      rst => rst
    );

  buffer33 : entity work.oehb(arch) generic map(4)
    port map(
      ins => cond_br14_trueOut,
      ins_valid => cond_br14_trueOut_valid,
      ins_ready => cond_br14_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer33_outs,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  buffer34 : entity work.tehb(arch) generic map(4)
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

  fork13 : entity work.handshake_fork(arch) generic map(2, 4)
    port map(
      ins => buffer34_outs,
      ins_valid => buffer34_outs_valid,
      ins_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
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

  buffer40 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer39_outs_valid,
      ins_ready => buffer39_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  lazy_fork2 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => buffer40_outs_valid,
      ins_ready => buffer40_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork2_outs_0_valid,
      outs_valid(1) => lazy_fork2_outs_1_valid,
      outs_valid(2) => lazy_fork2_outs_2_valid,
      outs_ready(0) => lazy_fork2_outs_0_ready,
      outs_ready(1) => lazy_fork2_outs_1_ready,
      outs_ready(2) => lazy_fork2_outs_2_ready
    );

  buffer42 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => lazy_fork2_outs_2_valid,
      ins_ready => lazy_fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  fork14 : entity work.fork_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer42_outs_valid,
      ins_ready => buffer42_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_ready(0) => fork14_outs_0_ready
    );

  constant12 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => fork14_outs_0_valid,
      ctrl_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant5 : entity work.handshake_constant_4(arch) generic map(2)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  fork15 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi5_outs,
      ins_valid => extsi5_outs_valid,
      ins_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  load5 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => fork13_outs_1,
      addrIn_valid => fork13_outs_1_valid,
      addrIn_ready => fork13_outs_1_ready,
      dataFromMem => lsq1_ldData_2,
      dataFromMem_valid => lsq1_ldData_2_valid,
      dataFromMem_ready => lsq1_ldData_2_ready,
      clk => clk,
      rst => rst,
      addrOut => load5_addrOut,
      addrOut_valid => load5_addrOut_valid,
      addrOut_ready => load5_addrOut_ready,
      dataOut => load5_dataOut,
      dataOut_valid => load5_dataOut_valid,
      dataOut_ready => load5_dataOut_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load5_dataOut,
      lhs_valid => load5_dataOut_valid,
      lhs_ready => load5_dataOut_ready,
      rhs => fork15_outs_1,
      rhs_valid => fork15_outs_1_valid,
      rhs_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  store1 : entity work.store(arch) generic map(32, 4)
    port map(
      addrIn => fork13_outs_0,
      addrIn_valid => fork13_outs_0_valid,
      addrIn_ready => fork13_outs_0_ready,
      dataIn => addi0_result,
      dataIn_valid => addi0_result_valid,
      dataIn_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  buffer31 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br13_trueOut,
      ins_valid => cond_br13_trueOut_valid,
      ins_ready => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  buffer32 : entity work.tehb(arch) generic map(32)
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

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer32_outs,
      lhs_valid => buffer32_outs_valid,
      lhs_ready => buffer32_outs_ready,
      rhs => fork15_outs_0,
      rhs_valid => fork15_outs_0_valid,
      rhs_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  buffer35 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cond_br15_trueOut,
      ins_valid => cond_br15_trueOut_valid,
      ins_ready => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer35_outs,
      outs_valid => buffer35_outs_valid,
      outs_ready => buffer35_outs_ready
    );

  buffer36 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  buffer37 : entity work.oehb(arch) generic map(2)
    port map(
      ins => cond_br16_trueOut,
      ins_valid => cond_br16_trueOut_valid,
      ins_ready => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer37_outs,
      outs_valid => buffer37_outs_valid,
      outs_ready => buffer37_outs_ready
    );

  buffer38 : entity work.tehb(arch) generic map(2)
    port map(
      ins => buffer37_outs,
      ins_valid => buffer37_outs_valid,
      ins_ready => buffer37_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer38_outs,
      outs_valid => buffer38_outs_valid,
      outs_ready => buffer38_outs_ready
    );

  buffer41 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork2_outs_0_valid,
      ins_ready => lazy_fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  buffer45 : entity work.oehb(arch) generic map(2)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  buffer46 : entity work.tehb(arch) generic map(2)
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

  extsi9 : entity work.extsi(arch) generic map(2, 3)
    port map(
      ins => buffer46_outs,
      ins_valid => buffer46_outs_valid,
      ins_ready => buffer46_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant15 : entity work.handshake_constant_5(arch) generic map(3)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  buffer43 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer43_outs,
      outs_valid => buffer43_outs_valid,
      outs_ready => buffer43_outs_ready
    );

  buffer44 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer43_outs,
      ins_valid => buffer43_outs_valid,
      ins_ready => buffer43_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  select1 : entity work.selector(arch) generic map(3)
    port map(
      condition => buffer44_outs,
      condition_valid => buffer44_outs_valid,
      condition_ready => buffer44_outs_ready,
      trueValue => constant15_outs,
      trueValue_valid => constant15_outs_valid,
      trueValue_ready => constant15_outs_ready,
      falseValue => extsi9_outs,
      falseValue_valid => extsi9_outs_valid,
      falseValue_ready => extsi9_outs_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

  extsi10 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => select1_result,
      ins_valid => select1_result_valid,
      ins_ready => select1_result_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

end architecture;
