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
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
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
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi0_outs : std_logic_vector(31 downto 0);
  signal extsi0_outs_valid : std_logic;
  signal extsi0_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant6_outs : std_logic_vector(0 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(0 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(0 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal cond_br25_trueOut : std_logic_vector(31 downto 0);
  signal cond_br25_trueOut_valid : std_logic;
  signal cond_br25_trueOut_ready : std_logic;
  signal cond_br25_falseOut : std_logic_vector(31 downto 0);
  signal cond_br25_falseOut_valid : std_logic;
  signal cond_br25_falseOut_ready : std_logic;
  signal buffer8_outs : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(0 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal cond_br26_trueOut : std_logic_vector(0 downto 0);
  signal cond_br26_trueOut_valid : std_logic;
  signal cond_br26_trueOut_ready : std_logic;
  signal cond_br26_falseOut : std_logic_vector(0 downto 0);
  signal cond_br26_falseOut_valid : std_logic;
  signal cond_br26_falseOut_ready : std_logic;
  signal buffer6_outs : std_logic_vector(1 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(1 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal cond_br27_trueOut : std_logic_vector(1 downto 0);
  signal cond_br27_trueOut_valid : std_logic;
  signal cond_br27_trueOut_ready : std_logic;
  signal cond_br27_falseOut : std_logic_vector(1 downto 0);
  signal cond_br27_falseOut_valid : std_logic;
  signal cond_br27_falseOut_ready : std_logic;
  signal buffer2_outs : std_logic_vector(0 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(0 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br28_trueOut : std_logic_vector(0 downto 0);
  signal cond_br28_trueOut_valid : std_logic;
  signal cond_br28_trueOut_ready : std_logic;
  signal cond_br28_falseOut : std_logic_vector(0 downto 0);
  signal cond_br28_falseOut_valid : std_logic;
  signal cond_br28_falseOut_ready : std_logic;
  signal merge1_outs_valid : std_logic;
  signal merge1_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(0 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(0 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(0 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(0 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(1 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(0 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(31 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(31 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(0 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(0 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal extsi7_outs : std_logic_vector(1 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(0 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal merge0_outs : std_logic_vector(0 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(0 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal fork3_outs_3 : std_logic_vector(0 downto 0);
  signal fork3_outs_3_valid : std_logic;
  signal fork3_outs_3_ready : std_logic;
  signal mux7_outs : std_logic_vector(1 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(1 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal buffer19_outs : std_logic_vector(1 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(1 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(1 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal mux8_outs : std_logic_vector(0 downto 0);
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(0 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal mux9_outs : std_logic_vector(31 downto 0);
  signal mux9_outs_valid : std_logic;
  signal mux9_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(31 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(31 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(31 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(31 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal fork6_outs_3 : std_logic_vector(31 downto 0);
  signal fork6_outs_3_valid : std_logic;
  signal fork6_outs_3_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant3_outs : std_logic_vector(4 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(0 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer15_outs : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
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
  signal fork7_outs_6 : std_logic_vector(0 downto 0);
  signal fork7_outs_6_valid : std_logic;
  signal fork7_outs_6_ready : std_logic;
  signal fork7_outs_7 : std_logic_vector(0 downto 0);
  signal fork7_outs_7_valid : std_logic;
  signal fork7_outs_7_ready : std_logic;
  signal fork7_outs_8 : std_logic_vector(0 downto 0);
  signal fork7_outs_8_valid : std_logic;
  signal fork7_outs_8_ready : std_logic;
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(0 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
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
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal cond_br20_trueOut_valid : std_logic;
  signal cond_br20_trueOut_ready : std_logic;
  signal cond_br20_falseOut_valid : std_logic;
  signal cond_br20_falseOut_ready : std_logic;
  signal cond_br29_trueOut : std_logic_vector(1 downto 0);
  signal cond_br29_trueOut_valid : std_logic;
  signal cond_br29_trueOut_ready : std_logic;
  signal cond_br29_falseOut : std_logic_vector(1 downto 0);
  signal cond_br29_falseOut_valid : std_logic;
  signal cond_br29_falseOut_ready : std_logic;
  signal extsi8_outs : std_logic_vector(2 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal cond_br30_trueOut : std_logic_vector(0 downto 0);
  signal cond_br30_trueOut_valid : std_logic;
  signal cond_br30_trueOut_ready : std_logic;
  signal cond_br30_falseOut : std_logic_vector(0 downto 0);
  signal cond_br30_falseOut_valid : std_logic;
  signal cond_br30_falseOut_ready : std_logic;
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal source8_outs_valid : std_logic;
  signal source8_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal source9_outs_valid : std_logic;
  signal source9_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(0 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal load4_addrOut : std_logic_vector(3 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
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
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal select0_result : std_logic_vector(1 downto 0);
  signal select0_result_valid : std_logic;
  signal select0_result_ready : std_logic;
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal cond_br23_trueOut_valid : std_logic;
  signal cond_br23_trueOut_ready : std_logic;
  signal cond_br23_falseOut_valid : std_logic;
  signal cond_br23_falseOut_ready : std_logic;
  signal cond_br31_trueOut : std_logic_vector(31 downto 0);
  signal cond_br31_trueOut_valid : std_logic;
  signal cond_br31_trueOut_ready : std_logic;
  signal cond_br31_falseOut : std_logic_vector(31 downto 0);
  signal cond_br31_falseOut_valid : std_logic;
  signal cond_br31_falseOut_ready : std_logic;
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
  signal trunci0_outs : std_logic_vector(3 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(3 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(3 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(3 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal lazy_fork2_outs_0_valid : std_logic;
  signal lazy_fork2_outs_0_ready : std_logic;
  signal lazy_fork2_outs_1_valid : std_logic;
  signal lazy_fork2_outs_1_ready : std_logic;
  signal source10_outs_valid : std_logic;
  signal source10_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(0 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal source11_outs_valid : std_logic;
  signal source11_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(1 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
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
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal source12_outs_valid : std_logic;
  signal source12_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(2 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal select1_result : std_logic_vector(2 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;
  signal extsi9_outs : std_logic_vector(31 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;

begin

  out0 <= extsi9_outs;
  out0_valid <= extsi9_outs_valid;
  extsi9_outs_ready <= out0_ready;
  arr_end_valid <= lsq1_memEnd_valid;
  lsq1_memEnd_ready <= arr_end_ready;
  end_valid <= fork0_outs_0_valid;
  fork0_outs_0_ready <= end_ready;
  arr_loadEn <= lsq1_loadEn;
  arr_loadAddr <= lsq1_loadAddr;
  arr_storeEn <= lsq1_storeEn;
  arr_storeAddr <= lsq1_storeAddr;
  arr_storeData <= lsq1_storeData;

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

  buffer33 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br9_falseOut_valid,
      ins_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer33_outs_valid,
      outs_ready => buffer33_outs_ready
    );

  buffer34 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer33_outs_valid,
      ins_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
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
      io_ctrlEnd_valid => buffer34_outs_valid,
      io_ctrlEnd_ready => buffer34_outs_ready,
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

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant2 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi0 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi0_outs,
      outs_valid => extsi0_outs_valid,
      outs_ready => extsi0_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant6 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  fork1 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant6_outs,
      ins_valid => constant6_outs_valid,
      ins_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  buffer10 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(32)
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

  cond_br25 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_4,
      condition_valid => fork7_outs_4_valid,
      condition_ready => fork7_outs_4_ready,
      data => buffer11_outs,
      data_valid => buffer11_outs_valid,
      data_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br25_trueOut,
      trueOut_valid => cond_br25_trueOut_valid,
      trueOut_ready => cond_br25_trueOut_ready,
      falseOut => cond_br25_falseOut,
      falseOut_valid => cond_br25_falseOut_valid,
      falseOut_ready => cond_br25_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br25_falseOut,
      ins_valid => cond_br25_falseOut_valid,
      ins_ready => cond_br25_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer8 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  buffer9 : entity work.tehb(arch) generic map(1)
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

  cond_br26 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork7_outs_5,
      condition_valid => fork7_outs_5_valid,
      condition_ready => fork7_outs_5_ready,
      data => buffer9_outs,
      data_valid => buffer9_outs_valid,
      data_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br26_trueOut,
      trueOut_valid => cond_br26_trueOut_valid,
      trueOut_ready => cond_br26_trueOut_ready,
      falseOut => cond_br26_falseOut,
      falseOut_valid => cond_br26_falseOut_valid,
      falseOut_ready => cond_br26_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br26_falseOut,
      ins_valid => cond_br26_falseOut_valid,
      ins_ready => cond_br26_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer6 : entity work.oehb(arch) generic map(2)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  buffer7 : entity work.tehb(arch) generic map(2)
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

  cond_br27 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork7_outs_6,
      condition_valid => fork7_outs_6_valid,
      condition_ready => fork7_outs_6_ready,
      data => buffer7_outs,
      data_valid => buffer7_outs_valid,
      data_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br27_trueOut,
      trueOut_valid => cond_br27_trueOut_valid,
      trueOut_ready => cond_br27_trueOut_ready,
      falseOut => cond_br27_falseOut,
      falseOut_valid => cond_br27_falseOut_valid,
      falseOut_ready => cond_br27_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(2)
    port map(
      ins => cond_br27_falseOut,
      ins_valid => cond_br27_falseOut_valid,
      ins_ready => cond_br27_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer2 : entity work.oehb(arch) generic map(1)
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

  cond_br28 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork7_outs_7,
      condition_valid => fork7_outs_7_valid,
      condition_ready => fork7_outs_7_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br28_trueOut,
      trueOut_valid => cond_br28_trueOut_valid,
      trueOut_ready => cond_br28_trueOut_ready,
      falseOut => cond_br28_falseOut,
      falseOut_valid => cond_br28_falseOut_valid,
      falseOut_ready => cond_br28_falseOut_ready
    );

  sink3 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br28_falseOut,
      ins_valid => cond_br28_falseOut_valid,
      ins_ready => cond_br28_falseOut_ready,
      clk => clk,
      rst => rst
    );

  merge1 : entity work.merge_dataless(arch) generic map(4)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br20_falseOut_valid,
      ins_valid(2) => cond_br23_falseOut_valid,
      ins_valid(3) => buffer32_outs_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br20_falseOut_ready,
      ins_ready(2) => cond_br23_falseOut_ready,
      ins_ready(3) => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => merge1_outs_valid,
      outs_ready => merge1_outs_ready
    );

  buffer4 : entity work.oehb(arch) generic map(1)
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

  buffer5 : entity work.tehb(arch) generic map(1)
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

  mux0 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork8_outs_1,
      index_valid => fork8_outs_1_valid,
      index_ready => fork8_outs_1_ready,
      ins(0) => constant8_outs,
      ins(1) => buffer5_outs,
      ins_valid(0) => constant8_outs_valid,
      ins_valid(1) => buffer5_outs_valid,
      ins_ready(0) => constant8_outs_ready,
      ins_ready(1) => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork11_outs_2,
      index_valid => fork11_outs_2_valid,
      index_ready => fork11_outs_2_ready,
      ins(0) => fork11_outs_3,
      ins(1) => constant12_outs,
      ins_valid(0) => fork11_outs_3_valid,
      ins_valid(1) => constant12_outs_valid,
      ins_ready(0) => fork11_outs_3_ready,
      ins_ready(1) => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 2, 1)
    port map(
      index => fork8_outs_2,
      index_valid => fork8_outs_2_valid,
      index_ready => fork8_outs_2_ready,
      ins(0) => fork4_outs_1,
      ins(1) => select0_result,
      ins_valid(0) => fork4_outs_1_valid,
      ins_valid(1) => select0_result_valid,
      ins_ready(0) => fork4_outs_1_ready,
      ins_ready(1) => select0_result_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork8_outs_3,
      index_valid => fork8_outs_3_valid,
      index_ready => fork8_outs_3_ready,
      ins(0) => fork5_outs_1,
      ins(1) => andi1_result,
      ins_valid(0) => fork5_outs_1_valid,
      ins_valid(1) => andi1_result_valid,
      ins_ready(0) => fork5_outs_1_ready,
      ins_ready(1) => andi1_result_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer12 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
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

  mux4 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork8_outs_4,
      index_valid => fork8_outs_4_valid,
      index_ready => fork8_outs_4_ready,
      ins(0) => fork6_outs_2,
      ins(1) => buffer13_outs,
      ins_valid(0) => fork6_outs_2_valid,
      ins_valid(1) => buffer13_outs_valid,
      ins_ready(0) => fork6_outs_2_ready,
      ins_ready(1) => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork11_outs_4,
      index_valid => fork11_outs_4_valid,
      index_ready => fork11_outs_4_ready,
      ins(0) => fork6_outs_3,
      ins(1) => addi1_result,
      ins_valid(0) => fork6_outs_3_valid,
      ins_valid(1) => addi1_result_valid,
      ins_ready(0) => fork6_outs_3_ready,
      ins_ready(1) => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_1_valid,
      ctrl_ready => fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
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
      ins => fork2_outs_1,
      ins_valid => fork2_outs_1_valid,
      ins_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork3_outs_3,
      index_valid => fork3_outs_3_valid,
      index_ready => fork3_outs_3_ready,
      ins(0) => fork1_outs_1,
      ins(1) => cond_br28_trueOut,
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => cond_br28_trueOut_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => cond_br28_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  merge0 : entity work.merge(arch) generic map(2, 1)
    port map(
      ins(0) => fork2_outs_0,
      ins(1) => fork7_outs_8,
      ins_valid(0) => fork2_outs_0_valid,
      ins_valid(1) => fork7_outs_8_valid,
      ins_ready(0) => fork2_outs_0_ready,
      ins_ready(1) => fork7_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  buffer16 : entity work.oehb(arch) generic map(1)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer17 : entity work.tehb(arch) generic map(1)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs(2) => fork3_outs_2,
      outs(3) => fork3_outs_3,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_valid(2) => fork3_outs_2_valid,
      outs_valid(3) => fork3_outs_3_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready,
      outs_ready(2) => fork3_outs_2_ready,
      outs_ready(3) => fork3_outs_3_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 2, 1)
    port map(
      index => fork3_outs_2,
      index_valid => fork3_outs_2_valid,
      index_ready => fork3_outs_2_ready,
      ins(0) => extsi7_outs,
      ins(1) => cond_br27_trueOut,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => cond_br27_trueOut_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => cond_br27_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  buffer18 : entity work.oehb(arch) generic map(2)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  buffer19 : entity work.tehb(arch) generic map(2)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(2, 2)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork4_outs_0,
      outs(1) => fork4_outs_1,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  mux8 : entity work.mux(arch) generic map(2, 1, 1)
    port map(
      index => fork3_outs_1,
      index_valid => fork3_outs_1_valid,
      index_ready => fork3_outs_1_ready,
      ins(0) => fork1_outs_0,
      ins(1) => cond_br26_trueOut,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => cond_br26_trueOut_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => cond_br26_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux8_outs,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  buffer20 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux8_outs,
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
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

  fork5 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  mux9 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      ins(0) => extsi0_outs,
      ins(1) => cond_br25_trueOut,
      ins_valid(0) => extsi0_outs_valid,
      ins_valid(1) => cond_br25_trueOut_valid,
      ins_ready(0) => extsi0_outs_ready,
      ins_ready(1) => cond_br25_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux9_outs,
      outs_valid => mux9_outs_valid,
      outs_ready => mux9_outs_ready
    );

  buffer22 : entity work.oehb(arch) generic map(32)
    port map(
      ins => mux9_outs,
      ins_valid => mux9_outs_valid,
      ins_ready => mux9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  buffer23 : entity work.tehb(arch) generic map(32)
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

  fork6 : entity work.handshake_fork(arch) generic map(4, 32)
    port map(
      ins => buffer23_outs,
      ins_valid => buffer23_outs_valid,
      ins_ready => buffer23_outs_ready,
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

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant3 : entity work.handshake_constant_2(arch) generic map(5)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  extsi2 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => extsi2_outs,
      rhs_valid => extsi2_outs_valid,
      rhs_ready => extsi2_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer14 : entity work.oehb(arch) generic map(1)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  buffer15 : entity work.tehb(arch) generic map(1)
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

  andi0 : entity work.andi(arch) generic map(1)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      rhs => buffer15_outs,
      rhs_valid => buffer15_outs_valid,
      rhs_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(9, 1)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs(4) => fork7_outs_4,
      outs(5) => fork7_outs_5,
      outs(6) => fork7_outs_6,
      outs(7) => fork7_outs_7,
      outs(8) => fork7_outs_8,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_valid(4) => fork7_outs_4_valid,
      outs_valid(5) => fork7_outs_5_valid,
      outs_valid(6) => fork7_outs_6_valid,
      outs_valid(7) => fork7_outs_7_valid,
      outs_valid(8) => fork7_outs_8_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready,
      outs_ready(4) => fork7_outs_4_ready,
      outs_ready(5) => fork7_outs_5_ready,
      outs_ready(6) => fork7_outs_6_ready,
      outs_ready(7) => fork7_outs_7_ready,
      outs_ready(8) => fork7_outs_8_ready
    );

  buffer0 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => merge1_outs_valid,
      ins_ready => merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  buffer1 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  cond_br9 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork7_outs_3,
      condition_valid => fork7_outs_3_valid,
      condition_ready => fork7_outs_3_ready,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  buffer24 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br9_trueOut_valid,
      ins_ready => cond_br9_trueOut_ready,
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

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant8 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant9 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => trunci3_outs,
      addrIn_valid => trunci3_outs_valid,
      addrIn_ready => trunci3_outs_ready,
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

  fork8 : entity work.handshake_fork(arch) generic map(5, 1)
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

  buffer26 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_0_valid,
      ins_ready => lazy_fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  cond_br20 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork8_outs_0,
      condition_valid => fork8_outs_0_valid,
      condition_ready => fork8_outs_0_ready,
      data_valid => buffer26_outs_valid,
      data_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br20_trueOut_valid,
      trueOut_ready => cond_br20_trueOut_ready,
      falseOut_valid => cond_br20_falseOut_valid,
      falseOut_ready => cond_br20_falseOut_ready
    );

  cond_br29 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork7_outs_2,
      condition_valid => fork7_outs_2_valid,
      condition_ready => fork7_outs_2_ready,
      data => fork4_outs_0,
      data_valid => fork4_outs_0_valid,
      data_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br29_trueOut,
      trueOut_valid => cond_br29_trueOut_valid,
      trueOut_ready => cond_br29_trueOut_ready,
      falseOut => cond_br29_falseOut,
      falseOut_valid => cond_br29_falseOut_valid,
      falseOut_ready => cond_br29_falseOut_ready
    );

  extsi8 : entity work.extsi(arch) generic map(2, 3)
    port map(
      ins => cond_br29_falseOut,
      ins_valid => cond_br29_falseOut_valid,
      ins_ready => cond_br29_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  cond_br30 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork7_outs_1,
      condition_valid => fork7_outs_1_valid,
      condition_ready => fork7_outs_1_ready,
      data => fork5_outs_0,
      data_valid => fork5_outs_0_valid,
      data_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br30_trueOut,
      trueOut_valid => cond_br30_trueOut_valid,
      trueOut_ready => cond_br30_trueOut_ready,
      falseOut => cond_br30_falseOut,
      falseOut_valid => cond_br30_falseOut_valid,
      falseOut_ready => cond_br30_falseOut_ready
    );

  buffer27 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br20_trueOut_valid,
      ins_ready => cond_br20_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  buffer28 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork1_outs_0_valid,
      outs_valid(1) => lazy_fork1_outs_1_valid,
      outs_ready(0) => lazy_fork1_outs_0_ready,
      outs_ready(1) => lazy_fork1_outs_1_ready
    );

  source8 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source8_outs_valid,
      outs_ready => source8_outs_ready
    );

  constant4 : entity work.handshake_constant_4(arch) generic map(2)
    port map(
      ctrl_valid => source8_outs_valid,
      ctrl_ready => source8_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  source9 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source9_outs_valid,
      outs_ready => source9_outs_ready
    );

  constant5 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => source9_outs_valid,
      ctrl_ready => source9_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(1, 32)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi4_outs,
      ins_valid => extsi4_outs_valid,
      ins_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  load4 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
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

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load4_dataOut,
      ins_valid => load4_dataOut_valid,
      ins_ready => load4_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  cmpi2 : entity work.handshake_cmpi_2(arch) generic map(32)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => fork9_outs_1,
      rhs_valid => fork9_outs_1_valid,
      rhs_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  cmpi3 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => fork10_outs_0,
      lhs_valid => fork10_outs_0_valid,
      lhs_ready => fork10_outs_0_ready,
      rhs => fork9_outs_0,
      rhs_valid => fork9_outs_0_valid,
      rhs_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(5, 1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
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

  andi1 : entity work.andi(arch) generic map(1)
    port map(
      lhs => fork11_outs_1,
      lhs_valid => fork11_outs_1_valid,
      lhs_ready => fork11_outs_1_ready,
      rhs => cond_br30_trueOut,
      rhs_valid => cond_br30_trueOut_valid,
      rhs_ready => cond_br30_trueOut_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  select0 : entity work.selector(arch) generic map(2)
    port map(
      condition => cmpi2_result,
      condition_valid => cmpi2_result_valid,
      condition_ready => cmpi2_result_ready,
      trueValue => constant4_outs,
      trueValue_valid => constant4_outs_valid,
      trueValue_ready => constant4_outs_ready,
      falseValue => cond_br29_trueOut,
      falseValue_valid => cond_br29_trueOut_valid,
      falseValue_ready => cond_br29_trueOut_ready,
      clk => clk,
      rst => rst,
      result => select0_result,
      result_valid => select0_result_valid,
      result_ready => select0_result_ready
    );

  buffer29 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_0_valid,
      ins_ready => lazy_fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  cond_br23 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork11_outs_0,
      condition_valid => fork11_outs_0_valid,
      condition_ready => fork11_outs_0_ready,
      data_valid => buffer29_outs_valid,
      data_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br23_trueOut_valid,
      trueOut_ready => cond_br23_trueOut_ready,
      falseOut_valid => cond_br23_falseOut_valid,
      falseOut_ready => cond_br23_falseOut_ready
    );

  cond_br31 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork7_outs_0,
      condition_valid => fork7_outs_0_valid,
      condition_ready => fork7_outs_0_ready,
      data => fork6_outs_0,
      data_valid => fork6_outs_0_valid,
      data_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br31_trueOut,
      trueOut_valid => cond_br31_trueOut_valid,
      trueOut_ready => cond_br31_trueOut_ready,
      falseOut => cond_br31_falseOut,
      falseOut_valid => cond_br31_falseOut_valid,
      falseOut_ready => cond_br31_falseOut_ready
    );

  sink7 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br31_falseOut,
      ins_valid => cond_br31_falseOut_valid,
      ins_ready => cond_br31_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork12 : entity work.handshake_fork(arch) generic map(5, 32)
    port map(
      ins => cond_br31_trueOut,
      ins_valid => cond_br31_trueOut_valid,
      ins_ready => cond_br31_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs(2) => fork12_outs_2,
      outs(3) => fork12_outs_3,
      outs(4) => fork12_outs_4,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_valid(2) => fork12_outs_2_valid,
      outs_valid(3) => fork12_outs_3_valid,
      outs_valid(4) => fork12_outs_4_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready,
      outs_ready(2) => fork12_outs_2_ready,
      outs_ready(3) => fork12_outs_3_ready,
      outs_ready(4) => fork12_outs_4_ready
    );

  trunci0 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork12_outs_4,
      ins_valid => fork12_outs_4_valid,
      ins_ready => fork12_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork12_outs_3,
      ins_valid => fork12_outs_3_valid,
      ins_ready => fork12_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork12_outs_2,
      ins_valid => fork12_outs_2_valid,
      ins_ready => fork12_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 4)
    port map(
      ins => fork12_outs_1,
      ins_valid => fork12_outs_1_valid,
      ins_ready => fork12_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  buffer30 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br23_trueOut_valid,
      ins_ready => cond_br23_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  buffer31 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer30_outs_valid,
      ins_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  lazy_fork2 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer31_outs_valid,
      ins_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork2_outs_0_valid,
      outs_valid(1) => lazy_fork2_outs_1_valid,
      outs_ready(0) => lazy_fork2_outs_0_ready,
      outs_ready(1) => lazy_fork2_outs_1_ready
    );

  source10 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source10_outs_valid,
      outs_ready => source10_outs_ready
    );

  constant12 : entity work.handshake_constant_1(arch) generic map(1)
    port map(
      ctrl_valid => source10_outs_valid,
      ctrl_ready => source10_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  source11 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source11_outs_valid,
      outs_ready => source11_outs_ready
    );

  constant15 : entity work.handshake_constant_4(arch) generic map(2)
    port map(
      ctrl_valid => source11_outs_valid,
      ctrl_ready => source11_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi5_outs,
      ins_valid => extsi5_outs_valid,
      ins_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  load5 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
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
      rhs => fork13_outs_1,
      rhs_valid => fork13_outs_1_valid,
      rhs_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  store1 : entity work.store(arch) generic map(32, 4)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
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

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork12_outs_0,
      lhs_valid => fork12_outs_0_valid,
      lhs_ready => fork12_outs_0_ready,
      rhs => fork13_outs_0,
      rhs_valid => fork13_outs_0_valid,
      rhs_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  buffer32 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork2_outs_0_valid,
      ins_ready => lazy_fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  source12 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source12_outs_valid,
      outs_ready => source12_outs_ready
    );

  constant16 : entity work.handshake_constant_5(arch) generic map(3)
    port map(
      ctrl_valid => source12_outs_valid,
      ctrl_ready => source12_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  select1 : entity work.selector(arch) generic map(3)
    port map(
      condition => cond_br30_falseOut,
      condition_valid => cond_br30_falseOut_valid,
      condition_ready => cond_br30_falseOut_ready,
      trueValue => constant16_outs,
      trueValue_valid => constant16_outs_valid,
      trueValue_ready => constant16_outs_ready,
      falseValue => extsi8_outs,
      falseValue_valid => extsi8_outs_valid,
      falseValue_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

  extsi9 : entity work.extsi(arch) generic map(3, 32)
    port map(
      ins => select1_result,
      ins_valid => select1_result_valid,
      ins_ready => select1_result_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

end architecture;
