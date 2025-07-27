library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity nested_loop is
  port (
    a_loadData : in std_logic_vector(31 downto 0);
    b_loadData : in std_logic_vector(31 downto 0);
    c_loadData : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    b_start_valid : in std_logic;
    c_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    a_end_ready : in std_logic;
    b_end_ready : in std_logic;
    c_end_ready : in std_logic;
    end_ready : in std_logic;
    a_start_ready : out std_logic;
    b_start_ready : out std_logic;
    c_start_ready : out std_logic;
    start_ready : out std_logic;
    a_end_valid : out std_logic;
    b_end_valid : out std_logic;
    c_end_valid : out std_logic;
    end_valid : out std_logic;
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(9 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(9 downto 0);
    a_storeData : out std_logic_vector(31 downto 0);
    b_loadEn : out std_logic;
    b_loadAddr : out std_logic_vector(9 downto 0);
    b_storeEn : out std_logic;
    b_storeAddr : out std_logic_vector(9 downto 0);
    b_storeData : out std_logic_vector(31 downto 0);
    c_loadEn : out std_logic;
    c_loadAddr : out std_logic_vector(9 downto 0);
    c_storeEn : out std_logic;
    c_storeAddr : out std_logic_vector(9 downto 0);
    c_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of nested_loop is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
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
  signal mem_controller5_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller5_ldData_0_valid : std_logic;
  signal mem_controller5_ldData_0_ready : std_logic;
  signal mem_controller5_memEnd_valid : std_logic;
  signal mem_controller5_memEnd_ready : std_logic;
  signal mem_controller5_loadEn : std_logic;
  signal mem_controller5_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeEn : std_logic;
  signal mem_controller5_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller5_storeData : std_logic_vector(31 downto 0);
  signal constant3_outs : std_logic_vector(0 downto 0);
  signal constant3_outs_valid : std_logic;
  signal constant3_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(1 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(1 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(1 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(1 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(1 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal extsi9_outs : std_logic_vector(11 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(9 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(11 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(0 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal muli0_result : std_logic_vector(11 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal extsi11_outs : std_logic_vector(31 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(31 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal fork3_outs_3 : std_logic_vector(31 downto 0);
  signal fork3_outs_3_valid : std_logic;
  signal fork3_outs_3_ready : std_logic;
  signal trunci0_outs : std_logic_vector(9 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(9 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(1 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(1 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(11 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer9_outs : std_logic_vector(11 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(11 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(11 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(11 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal trunci3_outs : std_logic_vector(9 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal fork20_outs_2_valid : std_logic;
  signal fork20_outs_2_ready : std_logic;
  signal constant6_outs : std_logic_vector(1 downto 0);
  signal constant6_outs_valid : std_logic;
  signal constant6_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant13_outs : std_logic_vector(1 downto 0);
  signal constant13_outs_valid : std_logic;
  signal constant13_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant14_outs : std_logic_vector(10 downto 0);
  signal constant14_outs_valid : std_logic;
  signal constant14_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(9 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal load3_addrOut : std_logic_vector(9 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal muli1_result : std_logic_vector(31 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal buffer6_outs : std_logic_vector(9 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(9 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal store1_addrOut : std_logic_vector(9 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer18_outs : std_logic_vector(0 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
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
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(0 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal merge1_outs : std_logic_vector(0 downto 0);
  signal merge1_outs_valid : std_logic;
  signal merge1_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(0 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork22_outs_0 : std_logic_vector(0 downto 0);
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1 : std_logic_vector(0 downto 0);
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal fork22_outs_2 : std_logic_vector(0 downto 0);
  signal fork22_outs_2_valid : std_logic;
  signal fork22_outs_2_ready : std_logic;
  signal fork22_outs_3 : std_logic_vector(0 downto 0);
  signal fork22_outs_3_valid : std_logic;
  signal fork22_outs_3_ready : std_logic;
  signal fork22_outs_4 : std_logic_vector(0 downto 0);
  signal fork22_outs_4_valid : std_logic;
  signal fork22_outs_4_ready : std_logic;
  signal fork22_outs_5 : std_logic_vector(0 downto 0);
  signal fork22_outs_5_valid : std_logic;
  signal fork22_outs_5_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(0 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(0 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal fork23_outs_2 : std_logic_vector(0 downto 0);
  signal fork23_outs_2_valid : std_logic;
  signal fork23_outs_2_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal spec_v2_resolver0_confirmSpec : std_logic_vector(0 downto 0);
  signal spec_v2_resolver0_confirmSpec_valid : std_logic;
  signal spec_v2_resolver0_confirmSpec_ready : std_logic;
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
  signal passer0_result : std_logic_vector(0 downto 0);
  signal passer0_result_valid : std_logic;
  signal passer0_result_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(0 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(0 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork25_outs_2 : std_logic_vector(0 downto 0);
  signal fork25_outs_2_valid : std_logic;
  signal fork25_outs_2_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal passer1_result : std_logic_vector(31 downto 0);
  signal passer1_result_valid : std_logic;
  signal passer1_result_ready : std_logic;
  signal passer2_result : std_logic_vector(31 downto 0);
  signal passer2_result_valid : std_logic;
  signal passer2_result_ready : std_logic;
  signal buffer7_outs : std_logic_vector(1 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(1 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(1 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(1 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal passer3_result : std_logic_vector(1 downto 0);
  signal passer3_result_valid : std_logic;
  signal passer3_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(1 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(1 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal passer4_result : std_logic_vector(1 downto 0);
  signal passer4_result_valid : std_logic;
  signal passer4_result_ready : std_logic;
  signal buffer15_outs : std_logic_vector(9 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal passer5_result : std_logic_vector(9 downto 0);
  signal passer5_result_valid : std_logic;
  signal passer5_result_ready : std_logic;
  signal passer6_result : std_logic_vector(11 downto 0);
  signal passer6_result_valid : std_logic;
  signal passer6_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(31 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal passer7_result : std_logic_vector(31 downto 0);
  signal passer7_result_valid : std_logic;
  signal passer7_result_ready : std_logic;
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal extsi12_outs : std_logic_vector(2 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(2 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(1 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(2 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(2 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(2 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(2 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal trunci4_outs : std_logic_vector(1 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(0 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(0 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(1 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(1 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;

begin

  a_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= a_end_ready;
  b_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= b_end_ready;
  c_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= c_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  a_loadEn <= mem_controller5_loadEn;
  a_loadAddr <= mem_controller5_loadAddr;
  a_storeEn <= mem_controller5_storeEn;
  a_storeAddr <= mem_controller5_storeAddr;
  a_storeData <= mem_controller5_storeData;
  b_loadEn <= mem_controller4_loadEn;
  b_loadAddr <= mem_controller4_loadAddr;
  b_storeEn <= mem_controller4_storeEn;
  b_storeAddr <= mem_controller4_storeAddr;
  b_storeData <= mem_controller4_storeData;
  c_loadEn <= mem_controller3_loadEn;
  c_loadAddr <= mem_controller3_loadAddr;
  c_storeEn <= mem_controller3_storeEn;
  c_storeAddr <= mem_controller3_storeAddr;
  c_storeData <= mem_controller3_storeData;

  fork0 : entity work.handshake_fork_0(arch)
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

  mem_controller3 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => c_loadData,
      memStart_valid => c_start_valid,
      memStart_ready => c_start_ready,
      ctrl(0) => passer7_result,
      ctrl_valid(0) => passer7_result_valid,
      ctrl_ready(0) => passer7_result_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork11_outs_2_valid,
      ctrlEnd_ready => fork11_outs_2_ready,
      clk => clk,
      rst => rst,
      memEnd_valid => mem_controller3_memEnd_valid,
      memEnd_ready => mem_controller3_memEnd_ready,
      loadEn => mem_controller3_loadEn,
      loadAddr => mem_controller3_loadAddr,
      storeEn => mem_controller3_storeEn,
      storeAddr => mem_controller3_storeAddr,
      storeData => mem_controller3_storeData
    );

  mem_controller4 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => b_loadData,
      memStart_valid => b_start_valid,
      memStart_ready => b_start_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      ctrlEnd_valid => fork11_outs_1_valid,
      ctrlEnd_ready => fork11_outs_1_ready,
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

  mem_controller5 : entity work.handshake_mem_controller_2(arch)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork11_outs_0_valid,
      ctrlEnd_ready => fork11_outs_0_ready,
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

  constant3 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant3_outs,
      outs_valid => constant3_outs_valid,
      outs_ready => constant3_outs_ready
    );

  extsi8 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant3_outs,
      ins_valid => constant3_outs_valid,
      ins_ready => constant3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi8_outs,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => extsi8_outs_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => extsi8_outs_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer1 : entity work.handshake_buffer_0(arch)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork1 : entity work.handshake_fork_1(arch)
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

  extsi9 : entity work.handshake_extsi_1(arch)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  buffer3 : entity work.handshake_buffer_1(arch)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
    port map(
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi10 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  constant5 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork2_outs_0_valid,
      ctrl_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  muli0 : entity work.handshake_muli_0(arch)
    port map(
      lhs => extsi9_outs,
      lhs_valid => extsi9_outs_valid,
      lhs_ready => extsi9_outs_ready,
      rhs => extsi10_outs,
      rhs_valid => extsi10_outs_valid,
      rhs_ready => extsi10_outs_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  extsi11 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  mux1 : entity work.handshake_mux_1(arch)
    port map(
      index => fork21_outs_3,
      index_valid => fork21_outs_3_valid,
      index_ready => fork21_outs_3_ready,
      ins(0) => extsi11_outs,
      ins(1) => passer2_result,
      ins_valid(0) => extsi11_outs_valid,
      ins_valid(1) => passer2_result_valid,
      ins_ready(0) => extsi11_outs_ready,
      ins_ready(1) => passer2_result_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer4 : entity work.handshake_buffer_2(arch)
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

  fork3 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer4_outs,
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
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

  trunci0 : entity work.handshake_trunci_0(arch)
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

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  trunci2 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork3_outs_2,
      ins_valid => fork3_outs_2_valid,
      ins_ready => fork3_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  buffer2 : entity work.handshake_buffer_3(arch)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  mux2 : entity work.handshake_mux_0(arch)
    port map(
      index => fork21_outs_2,
      index_valid => fork21_outs_2_valid,
      index_ready => fork21_outs_2_ready,
      ins(0) => buffer2_outs,
      ins(1) => passer3_result,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => passer3_result_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => passer3_result_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  mux3 : entity work.handshake_mux_2(arch)
    port map(
      index => fork21_outs_1,
      index_valid => fork21_outs_1_valid,
      index_ready => fork21_outs_1_ready,
      ins(0) => muli0_result,
      ins(1) => passer6_result,
      ins_valid(0) => muli0_result_valid,
      ins_valid(1) => passer6_result_valid,
      ins_ready(0) => muli0_result_ready,
      ins_ready(1) => passer6_result_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer9 : entity work.handshake_buffer_4(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer10 : entity work.handshake_buffer_5(arch)
    port map(
      ins => buffer9_outs,
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork19 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready
    );

  trunci3 : entity work.handshake_trunci_1(arch)
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

  mux4 : entity work.handshake_mux_3(arch)
    port map(
      index => fork21_outs_0,
      index_valid => fork21_outs_0_valid,
      index_ready => fork21_outs_0_ready,
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => passer8_result_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => passer8_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer11 : entity work.handshake_buffer_6(arch)
    port map(
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.handshake_buffer_7(arch)
    port map(
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  fork20 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_valid(2) => fork20_outs_2_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready,
      outs_ready(2) => fork20_outs_2_ready
    );

  constant6 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => fork20_outs_0_valid,
      ctrl_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant6_outs,
      outs_valid => constant6_outs_valid,
      outs_ready => constant6_outs_ready
    );

  extsi3 : entity work.handshake_extsi_4(arch)
    port map(
      ins => constant6_outs,
      ins_valid => constant6_outs_valid,
      ins_ready => constant6_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  source1 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant13 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant13_outs,
      outs_valid => constant13_outs_valid,
      outs_ready => constant13_outs_ready
    );

  extsi4 : entity work.handshake_extsi_4(arch)
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

  source2 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant14 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant14_outs,
      outs_valid => constant14_outs_valid,
      outs_ready => constant14_outs_ready
    );

  extsi5 : entity work.handshake_extsi_5(arch)
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

  load2 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  load3 : entity work.handshake_load_0(arch)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  muli1 : entity work.handshake_muli_1(arch)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => load3_dataOut,
      rhs_valid => load3_dataOut_valid,
      rhs_ready => load3_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  fork7 : entity work.handshake_fork_5(arch)
    port map(
      ins => muli1_result,
      ins_valid => muli1_result_valid,
      ins_ready => muli1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  buffer6 : entity work.handshake_buffer_8(arch)
    port map(
      ins => trunci0_outs,
      ins_valid => trunci0_outs_valid,
      ins_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => buffer6_outs,
      lhs_valid => buffer6_outs_valid,
      lhs_ready => buffer6_outs_ready,
      rhs => trunci3_outs,
      rhs_valid => trunci3_outs_valid,
      rhs_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  store1 : entity work.handshake_store_0(arch)
    port map(
      addrIn => passer5_result,
      addrIn_valid => passer5_result_valid,
      addrIn_ready => passer5_result_ready,
      dataIn => passer1_result,
      dataIn_valid => passer1_result_valid,
      dataIn_ready => passer1_result_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork7_outs_1,
      lhs_valid => fork7_outs_1_valid,
      lhs_ready => fork7_outs_1_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer18 : entity work.handshake_buffer_9(arch)
    port map(
      ins => fork22_outs_0,
      ins_valid => fork22_outs_0_valid,
      ins_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  init0 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  fork21 : entity work.handshake_fork_6(arch)
    port map(
      ins => init0_outs,
      ins_valid => init0_outs_valid,
      ins_ready => init0_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs(2) => fork21_outs_2,
      outs(3) => fork21_outs_3,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_valid(2) => fork21_outs_2_valid,
      outs_valid(3) => fork21_outs_3_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready,
      outs_ready(2) => fork21_outs_2_ready,
      outs_ready(3) => fork21_outs_3_ready
    );

  source5 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant0 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  buffer0 : entity work.handshake_buffer_10(arch)
    port map(
      ins => passer0_result,
      ins_valid => passer0_result_valid,
      ins_ready => passer0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer0_outs,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  merge1 : entity work.handshake_merge_0(arch)
    port map(
      ins(0) => buffer0_outs,
      ins(1) => constant0_outs,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => constant0_outs_valid,
      ins_ready(0) => buffer0_outs_ready,
      ins_ready(1) => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => merge1_outs,
      outs_valid => merge1_outs_valid,
      outs_ready => merge1_outs_ready
    );

  buffer16 : entity work.handshake_buffer_11(arch)
    port map(
      ins => merge1_outs,
      ins_valid => merge1_outs_valid,
      ins_ready => merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer17 : entity work.handshake_buffer_10(arch)
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

  fork22 : entity work.handshake_fork_7(arch)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork22_outs_0,
      outs(1) => fork22_outs_1,
      outs(2) => fork22_outs_2,
      outs(3) => fork22_outs_3,
      outs(4) => fork22_outs_4,
      outs(5) => fork22_outs_5,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_valid(2) => fork22_outs_2_valid,
      outs_valid(3) => fork22_outs_3_valid,
      outs_valid(4) => fork22_outs_4_valid,
      outs_valid(5) => fork22_outs_5_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready,
      outs_ready(2) => fork22_outs_2_ready,
      outs_ready(3) => fork22_outs_3_ready,
      outs_ready(4) => fork22_outs_4_ready,
      outs_ready(5) => fork22_outs_5_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not0_outs,
      lhs_valid => not0_outs_valid,
      lhs_ready => not0_outs_ready,
      rhs => fork24_outs_0,
      rhs_valid => fork24_outs_0_valid,
      rhs_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  fork23 : entity work.handshake_fork_8(arch)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs(2) => fork23_outs_2,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_valid(2) => fork23_outs_2_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready,
      outs_ready(2) => fork23_outs_2_ready
    );

  buffer19 : entity work.handshake_buffer_12(arch)
    port map(
      ins => fork22_outs_5,
      ins_valid => fork22_outs_5_valid,
      ins_ready => fork22_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  spec_v2_resolver0 : entity work.handshake_spec_v2_resolver_0(arch)
    port map(
      actualCondition => fork25_outs_1,
      actualCondition_valid => fork25_outs_1_valid,
      actualCondition_ready => fork25_outs_1_ready,
      generatedCondition => buffer19_outs,
      generatedCondition_valid => buffer19_outs_valid,
      generatedCondition_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      confirmSpec => spec_v2_resolver0_confirmSpec,
      confirmSpec_valid => spec_v2_resolver0_confirmSpec_valid,
      confirmSpec_ready => spec_v2_resolver0_confirmSpec_ready
    );

  fork24 : entity work.handshake_fork_6(arch)
    port map(
      ins => spec_v2_resolver0_confirmSpec,
      ins_valid => spec_v2_resolver0_confirmSpec_valid,
      ins_ready => spec_v2_resolver0_confirmSpec_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs(2) => fork24_outs_2,
      outs(3) => fork24_outs_3,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_valid(2) => fork24_outs_2_valid,
      outs_valid(3) => fork24_outs_3_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready,
      outs_ready(2) => fork24_outs_2_ready,
      outs_ready(3) => fork24_outs_3_ready
    );

  passer0 : entity work.handshake_passer_0(arch)
    port map(
      data => fork25_outs_2,
      data_valid => fork25_outs_2_valid,
      data_ready => fork25_outs_2_ready,
      ctrl => fork23_outs_0,
      ctrl_valid => fork23_outs_0_valid,
      ctrl_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer0_result,
      result_valid => passer0_result_valid,
      result_ready => passer0_result_ready
    );

  fork25 : entity work.handshake_fork_8(arch)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs(2) => fork25_outs_2,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_valid(2) => fork25_outs_2_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready,
      outs_ready(2) => fork25_outs_2_ready
    );

  buffer5 : entity work.handshake_buffer_13(arch)
    port map(
      ins => fork3_outs_3,
      ins_valid => fork3_outs_3_valid,
      ins_ready => fork3_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  addi1 : entity work.handshake_addi_1(arch)
    port map(
      lhs => buffer5_outs,
      lhs_valid => buffer5_outs_valid,
      lhs_ready => buffer5_outs_ready,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork25_outs_0,
      ins_valid => fork25_outs_0_valid,
      ins_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  passer1 : entity work.handshake_passer_1(arch)
    port map(
      data => fork7_outs_0,
      data_valid => fork7_outs_0_valid,
      data_ready => fork7_outs_0_ready,
      ctrl => fork24_outs_2,
      ctrl_valid => fork24_outs_2_valid,
      ctrl_ready => fork24_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer1_result,
      result_valid => passer1_result_valid,
      result_ready => passer1_result_ready
    );

  passer2 : entity work.handshake_passer_1(arch)
    port map(
      data => addi1_result,
      data_valid => addi1_result_valid,
      data_ready => addi1_result_ready,
      ctrl => fork22_outs_3,
      ctrl_valid => fork22_outs_3_valid,
      ctrl_ready => fork22_outs_3_ready,
      clk => clk,
      rst => rst,
      result => passer2_result,
      result_valid => passer2_result_valid,
      result_ready => passer2_result_ready
    );

  buffer7 : entity work.handshake_buffer_14(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer8 : entity work.handshake_buffer_0(arch)
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

  fork16 : entity work.handshake_fork_1(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  passer3 : entity work.handshake_passer_2(arch)
    port map(
      data => fork16_outs_1,
      data_valid => fork16_outs_1_valid,
      data_ready => fork16_outs_1_ready,
      ctrl => fork22_outs_2,
      ctrl_valid => fork22_outs_2_valid,
      ctrl_ready => fork22_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer3_result,
      result_valid => passer3_result_valid,
      result_ready => passer3_result_ready
    );

  buffer20 : entity work.handshake_buffer_14(arch)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  buffer21 : entity work.handshake_buffer_15(arch)
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

  passer4 : entity work.handshake_passer_2(arch)
    port map(
      data => buffer21_outs,
      data_valid => buffer21_outs_valid,
      data_ready => buffer21_outs_ready,
      ctrl => fork23_outs_2,
      ctrl_valid => fork23_outs_2_valid,
      ctrl_ready => fork23_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer4_result,
      result_valid => passer4_result_valid,
      result_ready => passer4_result_ready
    );

  buffer15 : entity work.handshake_buffer_16(arch)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  passer5 : entity work.handshake_passer_3(arch)
    port map(
      data => buffer15_outs,
      data_valid => buffer15_outs_valid,
      data_ready => buffer15_outs_ready,
      ctrl => fork24_outs_3,
      ctrl_valid => fork24_outs_3_valid,
      ctrl_ready => fork24_outs_3_ready,
      clk => clk,
      rst => rst,
      result => passer5_result,
      result_valid => passer5_result_valid,
      result_ready => passer5_result_ready
    );

  passer6 : entity work.handshake_passer_4(arch)
    port map(
      data => fork19_outs_1,
      data_valid => fork19_outs_1_valid,
      data_ready => fork19_outs_1_ready,
      ctrl => fork22_outs_4,
      ctrl_valid => fork22_outs_4_valid,
      ctrl_ready => fork22_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer6_result,
      result_valid => passer6_result_valid,
      result_ready => passer6_result_ready
    );

  buffer14 : entity work.handshake_buffer_17(arch)
    port map(
      ins => extsi3_outs,
      ins_valid => extsi3_outs_valid,
      ins_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  passer7 : entity work.handshake_passer_1(arch)
    port map(
      data => buffer14_outs,
      data_valid => buffer14_outs_valid,
      data_ready => buffer14_outs_ready,
      ctrl => fork24_outs_1,
      ctrl_valid => fork24_outs_1_valid,
      ctrl_ready => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      result => passer7_result,
      result_valid => passer7_result_valid,
      result_ready => passer7_result_ready
    );

  passer8 : entity work.handshake_passer_5(arch)
    port map(
      data_valid => fork20_outs_2_valid,
      data_ready => fork20_outs_2_ready,
      ctrl => fork22_outs_1,
      ctrl_valid => fork22_outs_1_valid,
      ctrl_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  buffer13 : entity work.handshake_buffer_18(arch)
    port map(
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  passer9 : entity work.handshake_passer_5(arch)
    port map(
      data_valid => buffer13_outs_valid,
      data_ready => buffer13_outs_ready,
      ctrl => fork23_outs_1,
      ctrl_valid => fork23_outs_1_valid,
      ctrl_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  extsi12 : entity work.handshake_extsi_6(arch)
    port map(
      ins => passer4_result,
      ins_valid => passer4_result_valid,
      ins_ready => passer4_result_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant15 : entity work.handshake_constant_5(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant16 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  extsi13 : entity work.handshake_extsi_6(arch)
    port map(
      ins => constant16_outs,
      ins_valid => constant16_outs_valid,
      ins_ready => constant16_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  addi2 : entity work.handshake_addi_2(arch)
    port map(
      lhs => extsi12_outs,
      lhs_valid => extsi12_outs_valid,
      lhs_ready => extsi12_outs_ready,
      rhs => extsi13_outs,
      rhs_valid => extsi13_outs_valid,
      rhs_ready => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  fork9 : entity work.handshake_fork_9(arch)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  trunci4 : entity work.handshake_trunci_2(arch)
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

  cmpi1 : entity work.handshake_cmpi_1(arch)
    port map(
      lhs => fork9_outs_1,
      lhs_valid => fork9_outs_1_valid,
      lhs_ready => fork9_outs_1_ready,
      rhs => constant15_outs,
      rhs_valid => constant15_outs_valid,
      rhs_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  fork10 : entity work.handshake_fork_10(arch)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  cond_br6 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork10_outs_0,
      condition_valid => fork10_outs_0_valid,
      condition_ready => fork10_outs_0_ready,
      data => trunci4_outs,
      data_valid => trunci4_outs_valid,
      data_ready => trunci4_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink3 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br7 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork10_outs_1,
      condition_valid => fork10_outs_1_valid,
      condition_ready => fork10_outs_1_ready,
      data_valid => passer9_result_valid,
      data_ready => passer9_result_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  fork11 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_valid(2) => fork11_outs_2_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready,
      outs_ready(2) => fork11_outs_2_ready
    );

end architecture;
