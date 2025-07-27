library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity if_convert is
  port (
    a_loadData : in std_logic_vector(31 downto 0);
    b_loadData : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    b_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    a_end_ready : in std_logic;
    b_end_ready : in std_logic;
    end_ready : in std_logic;
    a_start_ready : out std_logic;
    b_start_ready : out std_logic;
    start_ready : out std_logic;
    a_end_valid : out std_logic;
    b_end_valid : out std_logic;
    end_valid : out std_logic;
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(7 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(7 downto 0);
    a_storeData : out std_logic_vector(31 downto 0);
    b_loadEn : out std_logic;
    b_loadAddr : out std_logic_vector(7 downto 0);
    b_storeEn : out std_logic;
    b_storeAddr : out std_logic_vector(7 downto 0);
    b_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of if_convert is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal mem_controller2_memEnd_valid : std_logic;
  signal mem_controller2_memEnd_ready : std_logic;
  signal mem_controller2_loadEn : std_logic;
  signal mem_controller2_loadAddr : std_logic_vector(7 downto 0);
  signal mem_controller2_storeEn : std_logic;
  signal mem_controller2_storeAddr : std_logic_vector(7 downto 0);
  signal mem_controller2_storeData : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller3_ldData_0_valid : std_logic;
  signal mem_controller3_ldData_0_ready : std_logic;
  signal mem_controller3_memEnd_valid : std_logic;
  signal mem_controller3_memEnd_ready : std_logic;
  signal mem_controller3_loadEn : std_logic;
  signal mem_controller3_loadAddr : std_logic_vector(7 downto 0);
  signal mem_controller3_storeEn : std_logic;
  signal mem_controller3_storeAddr : std_logic_vector(7 downto 0);
  signal mem_controller3_storeData : std_logic_vector(31 downto 0);
  signal constant2_outs : std_logic_vector(1 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(31 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(31 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(31 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal mux0_outs : std_logic_vector(31 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(31 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
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
  signal buffer1_outs : std_logic_vector(31 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(31 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(31 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal fork23_outs_2 : std_logic_vector(31 downto 0);
  signal fork23_outs_2_valid : std_logic;
  signal fork23_outs_2_ready : std_logic;
  signal fork23_outs_3 : std_logic_vector(31 downto 0);
  signal fork23_outs_3_valid : std_logic;
  signal fork23_outs_3_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(31 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(31 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(31 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(31 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal fork24_outs_2 : std_logic_vector(31 downto 0);
  signal fork24_outs_2_valid : std_logic;
  signal fork24_outs_2_ready : std_logic;
  signal fork24_outs_3 : std_logic_vector(31 downto 0);
  signal fork24_outs_3_valid : std_logic;
  signal fork24_outs_3_ready : std_logic;
  signal fork24_outs_4 : std_logic_vector(31 downto 0);
  signal fork24_outs_4_valid : std_logic;
  signal fork24_outs_4_ready : std_logic;
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(8 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal andi0_result : std_logic_vector(0 downto 0);
  signal andi0_result_valid : std_logic;
  signal andi0_result_ready : std_logic;
  signal buffer13_outs : std_logic_vector(0 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(0 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(0 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork25_outs_2 : std_logic_vector(0 downto 0);
  signal fork25_outs_2_valid : std_logic;
  signal fork25_outs_2_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(0 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal merge0_outs : std_logic_vector(0 downto 0);
  signal merge0_outs_valid : std_logic;
  signal merge0_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(0 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(0 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal fork26_outs_2 : std_logic_vector(0 downto 0);
  signal fork26_outs_2_valid : std_logic;
  signal fork26_outs_2_ready : std_logic;
  signal fork26_outs_3 : std_logic_vector(0 downto 0);
  signal fork26_outs_3_valid : std_logic;
  signal fork26_outs_3_ready : std_logic;
  signal fork26_outs_4 : std_logic_vector(0 downto 0);
  signal fork26_outs_4_valid : std_logic;
  signal fork26_outs_4_ready : std_logic;
  signal andi1_result : std_logic_vector(0 downto 0);
  signal andi1_result_valid : std_logic;
  signal andi1_result_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(0 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(0 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal fork27_outs_2 : std_logic_vector(0 downto 0);
  signal fork27_outs_2_valid : std_logic;
  signal fork27_outs_2_ready : std_logic;
  signal spec_v2_resolver0_confirmSpec : std_logic_vector(0 downto 0);
  signal spec_v2_resolver0_confirmSpec_valid : std_logic;
  signal spec_v2_resolver0_confirmSpec_ready : std_logic;
  signal fork28_outs_0 : std_logic_vector(0 downto 0);
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1 : std_logic_vector(0 downto 0);
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;
  signal fork28_outs_2 : std_logic_vector(0 downto 0);
  signal fork28_outs_2_valid : std_logic;
  signal fork28_outs_2_ready : std_logic;
  signal fork28_outs_3 : std_logic_vector(0 downto 0);
  signal fork28_outs_3_valid : std_logic;
  signal fork28_outs_3_ready : std_logic;
  signal fork28_outs_4 : std_logic_vector(0 downto 0);
  signal fork28_outs_4_valid : std_logic;
  signal fork28_outs_4_ready : std_logic;
  signal fork28_outs_5 : std_logic_vector(0 downto 0);
  signal fork28_outs_5_valid : std_logic;
  signal fork28_outs_5_ready : std_logic;
  signal fork28_outs_6 : std_logic_vector(0 downto 0);
  signal fork28_outs_6_valid : std_logic;
  signal fork28_outs_6_ready : std_logic;
  signal passer0_result : std_logic_vector(0 downto 0);
  signal passer0_result_valid : std_logic;
  signal passer0_result_ready : std_logic;
  signal fork29_outs_0 : std_logic_vector(0 downto 0);
  signal fork29_outs_0_valid : std_logic;
  signal fork29_outs_0_ready : std_logic;
  signal fork29_outs_1 : std_logic_vector(0 downto 0);
  signal fork29_outs_1_valid : std_logic;
  signal fork29_outs_1_ready : std_logic;
  signal fork29_outs_2 : std_logic_vector(0 downto 0);
  signal fork29_outs_2_valid : std_logic;
  signal fork29_outs_2_ready : std_logic;
  signal fork29_outs_3 : std_logic_vector(0 downto 0);
  signal fork29_outs_3_valid : std_logic;
  signal fork29_outs_3_ready : std_logic;
  signal fork29_outs_4 : std_logic_vector(0 downto 0);
  signal fork29_outs_4_valid : std_logic;
  signal fork29_outs_4_ready : std_logic;
  signal fork29_outs_5 : std_logic_vector(0 downto 0);
  signal fork29_outs_5_valid : std_logic;
  signal fork29_outs_5_ready : std_logic;
  signal not0_outs : std_logic_vector(0 downto 0);
  signal not0_outs_valid : std_logic;
  signal not0_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(31 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal passer1_result : std_logic_vector(31 downto 0);
  signal passer1_result_valid : std_logic;
  signal passer1_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(0 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal passer2_result : std_logic_vector(31 downto 0);
  signal passer2_result_valid : std_logic;
  signal passer2_result_ready : std_logic;
  signal buffer15_outs : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal passer3_result : std_logic_vector(7 downto 0);
  signal passer3_result_valid : std_logic;
  signal passer3_result_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal passer4_result : std_logic_vector(7 downto 0);
  signal passer4_result_valid : std_logic;
  signal passer4_result_ready : std_logic;
  signal passer5_result : std_logic_vector(0 downto 0);
  signal passer5_result_valid : std_logic;
  signal passer5_result_ready : std_logic;
  signal passer6_result : std_logic_vector(0 downto 0);
  signal passer6_result_valid : std_logic;
  signal passer6_result_ready : std_logic;
  signal passer7_result : std_logic_vector(0 downto 0);
  signal passer7_result_valid : std_logic;
  signal passer7_result_ready : std_logic;
  signal passer8_result : std_logic_vector(31 downto 0);
  signal passer8_result_valid : std_logic;
  signal passer8_result_ready : std_logic;
  signal passer9_result : std_logic_vector(31 downto 0);
  signal passer9_result_valid : std_logic;
  signal passer9_result_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal fork22_outs_2_valid : std_logic;
  signal fork22_outs_2_ready : std_logic;
  signal passer10_result : std_logic_vector(31 downto 0);
  signal passer10_result_valid : std_logic;
  signal passer10_result_ready : std_logic;
  signal passer11_result : std_logic_vector(31 downto 0);
  signal passer11_result_valid : std_logic;
  signal passer11_result_ready : std_logic;
  signal passer12_result : std_logic_vector(31 downto 0);
  signal passer12_result_valid : std_logic;
  signal passer12_result_ready : std_logic;
  signal passer13_result_valid : std_logic;
  signal passer13_result_ready : std_logic;
  signal passer14_result_valid : std_logic;
  signal passer14_result_ready : std_logic;
  signal trunci0_outs : std_logic_vector(7 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(1 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(1 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(1 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal extsi2_outs : std_logic_vector(31 downto 0);
  signal extsi2_outs_valid : std_logic;
  signal extsi2_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(31 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant11_outs : std_logic_vector(14 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant12_outs : std_logic_vector(2 downto 0);
  signal constant12_outs_valid : std_logic;
  signal constant12_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal load1_addrOut : std_logic_vector(7 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal select1_result : std_logic_vector(31 downto 0);
  signal select1_result_valid : std_logic;
  signal select1_result_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(31 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(31 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal trunci1_outs : std_logic_vector(7 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(7 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(31 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(31 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant13_outs : std_logic_vector(8 downto 0);
  signal constant13_outs_valid : std_logic;
  signal constant13_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(0 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(0 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal fork15_outs_0 : std_logic_vector(31 downto 0);
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1 : std_logic_vector(31 downto 0);
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;

begin

  a_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= a_end_ready;
  b_end_valid <= mem_controller2_memEnd_valid;
  mem_controller2_memEnd_ready <= b_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  a_loadEn <= mem_controller3_loadEn;
  a_loadAddr <= mem_controller3_loadAddr;
  a_storeEn <= mem_controller3_storeEn;
  a_storeAddr <= mem_controller3_storeAddr;
  a_storeData <= mem_controller3_storeData;
  b_loadEn <= mem_controller2_loadEn;
  b_loadAddr <= mem_controller2_loadAddr;
  b_storeEn <= mem_controller2_storeEn;
  b_storeAddr <= mem_controller2_storeAddr;
  b_storeData <= mem_controller2_storeData;

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

  mem_controller2 : entity work.handshake_mem_controller_0(arch)
    port map(
      loadData => b_loadData,
      memStart_valid => b_start_valid,
      memStart_ready => b_start_ready,
      ctrl(0) => passer9_result,
      ctrl_valid(0) => passer9_result_valid,
      ctrl_ready(0) => passer9_result_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork16_outs_1_valid,
      ctrlEnd_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      memEnd_valid => mem_controller2_memEnd_valid,
      memEnd_ready => mem_controller2_memEnd_ready,
      loadEn => mem_controller2_loadEn,
      loadAddr => mem_controller2_loadAddr,
      storeEn => mem_controller2_storeEn,
      storeAddr => mem_controller2_storeAddr,
      storeData => mem_controller2_storeData
    );

  mem_controller3 : entity work.handshake_mem_controller_1(arch)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork16_outs_0_valid,
      ctrlEnd_ready => fork16_outs_0_ready,
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

  constant2 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi7 : entity work.handshake_extsi_0(arch)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  fork1 : entity work.handshake_fork_1(arch)
    port map(
      ins => extsi7_outs,
      ins_valid => extsi7_outs_valid,
      ins_ready => extsi7_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork1_outs_0,
      outs(1) => fork1_outs_1,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  mux0 : entity work.handshake_mux_0(arch)
    port map(
      index => fork2_outs_0,
      index_valid => fork2_outs_0_valid,
      index_ready => fork2_outs_0_ready,
      ins(0) => fork1_outs_0,
      ins(1) => fork15_outs_0,
      ins_valid(0) => fork1_outs_0_valid,
      ins_valid(1) => fork15_outs_0_valid,
      ins_ready(0) => fork1_outs_0_ready,
      ins_ready(1) => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  mux1 : entity work.handshake_mux_0(arch)
    port map(
      index => fork2_outs_1,
      index_valid => fork2_outs_1_valid,
      index_ready => fork2_outs_1_ready,
      ins(0) => fork1_outs_1,
      ins(1) => fork15_outs_1,
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => fork15_outs_1_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  control_merge0 : entity work.handshake_control_merge_0(arch)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork2 : entity work.handshake_fork_2(arch)
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

  mux2 : entity work.handshake_mux_0(arch)
    port map(
      index => fork25_outs_2,
      index_valid => fork25_outs_2_valid,
      index_ready => fork25_outs_2_ready,
      ins(0) => buffer1_outs,
      ins(1) => passer2_result,
      ins_valid(0) => buffer1_outs_valid,
      ins_valid(1) => passer2_result_valid,
      ins_ready(0) => buffer1_outs_ready,
      ins_ready(1) => passer2_result_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer3 : entity work.handshake_buffer_0(arch)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork23 : entity work.handshake_fork_3(arch)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs(2) => fork23_outs_2,
      outs(3) => fork23_outs_3,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_valid(2) => fork23_outs_2_valid,
      outs_valid(3) => fork23_outs_3_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready,
      outs_ready(2) => fork23_outs_2_ready,
      outs_ready(3) => fork23_outs_3_ready
    );

  buffer2 : entity work.handshake_buffer_0(arch)
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

  mux3 : entity work.handshake_mux_0(arch)
    port map(
      index => fork25_outs_1,
      index_valid => fork25_outs_1_valid,
      index_ready => fork25_outs_1_ready,
      ins(0) => buffer2_outs,
      ins(1) => passer12_result,
      ins_valid(0) => buffer2_outs_valid,
      ins_valid(1) => passer12_result_valid,
      ins_ready(0) => buffer2_outs_ready,
      ins_ready(1) => passer12_result_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer7 : entity work.handshake_buffer_1(arch)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
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

  fork24 : entity work.handshake_fork_4(arch)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
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

  mux4 : entity work.handshake_mux_1(arch)
    port map(
      index => fork25_outs_0,
      index_valid => fork25_outs_0_valid,
      index_ready => fork25_outs_0_ready,
      ins_valid(0) => control_merge0_outs_valid,
      ins_valid(1) => passer13_result_valid,
      ins_ready(0) => control_merge0_outs_ready,
      ins_ready(1) => passer13_result_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  source0 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant9 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  extsi1 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork24_outs_1,
      lhs_valid => fork24_outs_1_valid,
      lhs_ready => fork24_outs_1_ready,
      rhs => extsi1_outs,
      rhs_valid => extsi1_outs_valid,
      rhs_ready => extsi1_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer4 : entity work.handshake_buffer_1(arch)
    port map(
      ins => fork23_outs_0,
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer4_outs,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_1(arch)
    port map(
      lhs => fork24_outs_0,
      lhs_valid => fork24_outs_0_valid,
      lhs_ready => fork24_outs_0_ready,
      rhs => buffer4_outs,
      rhs_valid => buffer4_outs_valid,
      rhs_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  andi0 : entity work.handshake_andi_0(arch)
    port map(
      lhs => cmpi0_result,
      lhs_valid => cmpi0_result_valid,
      lhs_ready => cmpi0_result_ready,
      rhs => cmpi1_result,
      rhs_valid => cmpi1_result_valid,
      rhs_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      result => andi0_result,
      result_valid => andi0_result_valid,
      result_ready => andi0_result_ready
    );

  buffer13 : entity work.handshake_buffer_2(arch)
    port map(
      ins => fork26_outs_0,
      ins_valid => fork26_outs_0_valid,
      ins_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  init0 : entity work.handshake_init_0(arch)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  fork25 : entity work.handshake_fork_5(arch)
    port map(
      ins => init0_outs,
      ins_valid => init0_outs_valid,
      ins_ready => init0_outs_ready,
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

  source4 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant0 : entity work.handshake_constant_2(arch)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant0_outs,
      outs_valid => constant0_outs_valid,
      outs_ready => constant0_outs_ready
    );

  buffer0 : entity work.handshake_buffer_3(arch)
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

  merge0 : entity work.handshake_merge_0(arch)
    port map(
      ins(0) => buffer0_outs,
      ins(1) => constant0_outs,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => constant0_outs_valid,
      ins_ready(0) => buffer0_outs_ready,
      ins_ready(1) => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => merge0_outs,
      outs_valid => merge0_outs_valid,
      outs_ready => merge0_outs_ready
    );

  buffer11 : entity work.handshake_buffer_4(arch)
    port map(
      ins => merge0_outs,
      ins_valid => merge0_outs_valid,
      ins_ready => merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  buffer12 : entity work.handshake_buffer_3(arch)
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

  fork26 : entity work.handshake_fork_6(arch)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork26_outs_0,
      outs(1) => fork26_outs_1,
      outs(2) => fork26_outs_2,
      outs(3) => fork26_outs_3,
      outs(4) => fork26_outs_4,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_valid(2) => fork26_outs_2_valid,
      outs_valid(3) => fork26_outs_3_valid,
      outs_valid(4) => fork26_outs_4_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready,
      outs_ready(2) => fork26_outs_2_ready,
      outs_ready(3) => fork26_outs_3_ready,
      outs_ready(4) => fork26_outs_4_ready
    );

  andi1 : entity work.handshake_andi_0(arch)
    port map(
      lhs => not0_outs,
      lhs_valid => not0_outs_valid,
      lhs_ready => not0_outs_ready,
      rhs => fork28_outs_0,
      rhs_valid => fork28_outs_0_valid,
      rhs_ready => fork28_outs_0_ready,
      clk => clk,
      rst => rst,
      result => andi1_result,
      result_valid => andi1_result_valid,
      result_ready => andi1_result_ready
    );

  fork27 : entity work.handshake_fork_5(arch)
    port map(
      ins => andi1_result,
      ins_valid => andi1_result_valid,
      ins_ready => andi1_result_ready,
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

  spec_v2_resolver0 : entity work.handshake_spec_v2_resolver_0(arch)
    port map(
      actualCondition => fork29_outs_4,
      actualCondition_valid => fork29_outs_4_valid,
      actualCondition_ready => fork29_outs_4_ready,
      generatedCondition => fork26_outs_4,
      generatedCondition_valid => fork26_outs_4_valid,
      generatedCondition_ready => fork26_outs_4_ready,
      clk => clk,
      rst => rst,
      confirmSpec => spec_v2_resolver0_confirmSpec,
      confirmSpec_valid => spec_v2_resolver0_confirmSpec_valid,
      confirmSpec_ready => spec_v2_resolver0_confirmSpec_ready
    );

  fork28 : entity work.handshake_fork_7(arch)
    port map(
      ins => spec_v2_resolver0_confirmSpec,
      ins_valid => spec_v2_resolver0_confirmSpec_valid,
      ins_ready => spec_v2_resolver0_confirmSpec_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork28_outs_0,
      outs(1) => fork28_outs_1,
      outs(2) => fork28_outs_2,
      outs(3) => fork28_outs_3,
      outs(4) => fork28_outs_4,
      outs(5) => fork28_outs_5,
      outs(6) => fork28_outs_6,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_valid(2) => fork28_outs_2_valid,
      outs_valid(3) => fork28_outs_3_valid,
      outs_valid(4) => fork28_outs_4_valid,
      outs_valid(5) => fork28_outs_5_valid,
      outs_valid(6) => fork28_outs_6_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready,
      outs_ready(2) => fork28_outs_2_ready,
      outs_ready(3) => fork28_outs_3_ready,
      outs_ready(4) => fork28_outs_4_ready,
      outs_ready(5) => fork28_outs_5_ready,
      outs_ready(6) => fork28_outs_6_ready
    );

  passer0 : entity work.handshake_passer_0(arch)
    port map(
      data => fork29_outs_5,
      data_valid => fork29_outs_5_valid,
      data_ready => fork29_outs_5_ready,
      ctrl => fork27_outs_0,
      ctrl_valid => fork27_outs_0_valid,
      ctrl_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      result => passer0_result,
      result_valid => passer0_result_valid,
      result_ready => passer0_result_ready
    );

  fork29 : entity work.handshake_fork_8(arch)
    port map(
      ins => andi0_result,
      ins_valid => andi0_result_valid,
      ins_ready => andi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork29_outs_0,
      outs(1) => fork29_outs_1,
      outs(2) => fork29_outs_2,
      outs(3) => fork29_outs_3,
      outs(4) => fork29_outs_4,
      outs(5) => fork29_outs_5,
      outs_valid(0) => fork29_outs_0_valid,
      outs_valid(1) => fork29_outs_1_valid,
      outs_valid(2) => fork29_outs_2_valid,
      outs_valid(3) => fork29_outs_3_valid,
      outs_valid(4) => fork29_outs_4_valid,
      outs_valid(5) => fork29_outs_5_valid,
      outs_ready(0) => fork29_outs_0_ready,
      outs_ready(1) => fork29_outs_1_ready,
      outs_ready(2) => fork29_outs_2_ready,
      outs_ready(3) => fork29_outs_3_ready,
      outs_ready(4) => fork29_outs_4_ready,
      outs_ready(5) => fork29_outs_5_ready
    );

  not0 : entity work.handshake_not_0(arch)
    port map(
      ins => fork29_outs_0,
      ins_valid => fork29_outs_0_valid,
      ins_ready => fork29_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => not0_outs,
      outs_valid => not0_outs_valid,
      outs_ready => not0_outs_ready
    );

  buffer5 : entity work.handshake_buffer_1(arch)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  passer1 : entity work.handshake_passer_1(arch)
    port map(
      data => buffer5_outs,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      ctrl => fork27_outs_2,
      ctrl_valid => fork27_outs_2_valid,
      ctrl_ready => fork27_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer1_result,
      result_valid => passer1_result_valid,
      result_ready => passer1_result_ready
    );

  buffer14 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork26_outs_3,
      ins_valid => fork26_outs_3_valid,
      ins_ready => fork26_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  passer2 : entity work.handshake_passer_1(arch)
    port map(
      data => fork12_outs_1,
      data_valid => fork12_outs_1_valid,
      data_ready => fork12_outs_1_ready,
      ctrl => buffer14_outs,
      ctrl_valid => buffer14_outs_valid,
      ctrl_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      result => passer2_result,
      result_valid => passer2_result_valid,
      result_ready => passer2_result_ready
    );

  buffer15 : entity work.handshake_buffer_5(arch)
    port map(
      ins => fork28_outs_1,
      ins_valid => fork28_outs_1_valid,
      ins_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  passer3 : entity work.handshake_passer_2(arch)
    port map(
      data => trunci1_outs,
      data_valid => trunci1_outs_valid,
      data_ready => trunci1_outs_ready,
      ctrl => buffer15_outs,
      ctrl_valid => buffer15_outs_valid,
      ctrl_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      result => passer3_result,
      result_valid => passer3_result_valid,
      result_ready => passer3_result_ready
    );

  buffer16 : entity work.handshake_buffer_5(arch)
    port map(
      ins => passer7_result,
      ins_valid => passer7_result_valid,
      ins_ready => passer7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  passer4 : entity work.handshake_passer_2(arch)
    port map(
      data => passer3_result,
      data_valid => passer3_result_valid,
      data_ready => passer3_result_ready,
      ctrl => buffer16_outs,
      ctrl_valid => buffer16_outs_valid,
      ctrl_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      result => passer4_result,
      result_valid => passer4_result_valid,
      result_ready => passer4_result_ready
    );

  passer5 : entity work.handshake_passer_0(arch)
    port map(
      data => fork29_outs_1,
      data_valid => fork29_outs_1_valid,
      data_ready => fork29_outs_1_ready,
      ctrl => fork28_outs_4,
      ctrl_valid => fork28_outs_4_valid,
      ctrl_ready => fork28_outs_4_ready,
      clk => clk,
      rst => rst,
      result => passer5_result,
      result_valid => passer5_result_valid,
      result_ready => passer5_result_ready
    );

  passer6 : entity work.handshake_passer_0(arch)
    port map(
      data => fork29_outs_2,
      data_valid => fork29_outs_2_valid,
      data_ready => fork29_outs_2_ready,
      ctrl => fork28_outs_3,
      ctrl_valid => fork28_outs_3_valid,
      ctrl_ready => fork28_outs_3_ready,
      clk => clk,
      rst => rst,
      result => passer6_result,
      result_valid => passer6_result_valid,
      result_ready => passer6_result_ready
    );

  passer7 : entity work.handshake_passer_0(arch)
    port map(
      data => fork29_outs_3,
      data_valid => fork29_outs_3_valid,
      data_ready => fork29_outs_3_ready,
      ctrl => fork28_outs_2,
      ctrl_valid => fork28_outs_2_valid,
      ctrl_ready => fork28_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer7_result,
      result_valid => passer7_result_valid,
      result_ready => passer7_result_ready
    );

  passer8 : entity work.handshake_passer_1(arch)
    port map(
      data => passer10_result,
      data_valid => passer10_result_valid,
      data_ready => passer10_result_ready,
      ctrl => passer6_result,
      ctrl_valid => passer6_result_valid,
      ctrl_ready => passer6_result_ready,
      clk => clk,
      rst => rst,
      result => passer8_result,
      result_valid => passer8_result_valid,
      result_ready => passer8_result_ready
    );

  passer9 : entity work.handshake_passer_1(arch)
    port map(
      data => passer11_result,
      data_valid => passer11_result_valid,
      data_ready => passer11_result_ready,
      ctrl => passer5_result,
      ctrl_valid => passer5_result_valid,
      ctrl_ready => passer5_result_ready,
      clk => clk,
      rst => rst,
      result => passer9_result,
      result_valid => passer9_result_valid,
      result_ready => passer9_result_ready
    );

  buffer9 : entity work.handshake_buffer_6(arch)
    port map(
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  buffer10 : entity work.handshake_buffer_7(arch)
    port map(
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  fork22 : entity work.handshake_fork_0(arch)
    port map(
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_valid(2) => fork22_outs_2_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready,
      outs_ready(2) => fork22_outs_2_ready
    );

  passer10 : entity work.handshake_passer_1(arch)
    port map(
      data => fork11_outs_0,
      data_valid => fork11_outs_0_valid,
      data_ready => fork11_outs_0_ready,
      ctrl => fork28_outs_5,
      ctrl_valid => fork28_outs_5_valid,
      ctrl_ready => fork28_outs_5_ready,
      clk => clk,
      rst => rst,
      result => passer10_result,
      result_valid => passer10_result_valid,
      result_ready => passer10_result_ready
    );

  passer11 : entity work.handshake_passer_1(arch)
    port map(
      data => extsi3_outs,
      data_valid => extsi3_outs_valid,
      data_ready => extsi3_outs_ready,
      ctrl => fork28_outs_6,
      ctrl_valid => fork28_outs_6_valid,
      ctrl_ready => fork28_outs_6_ready,
      clk => clk,
      rst => rst,
      result => passer11_result,
      result_valid => passer11_result_valid,
      result_ready => passer11_result_ready
    );

  passer12 : entity work.handshake_passer_1(arch)
    port map(
      data => addi2_result,
      data_valid => addi2_result_valid,
      data_ready => addi2_result_ready,
      ctrl => fork26_outs_2,
      ctrl_valid => fork26_outs_2_valid,
      ctrl_ready => fork26_outs_2_ready,
      clk => clk,
      rst => rst,
      result => passer12_result,
      result_valid => passer12_result_valid,
      result_ready => passer12_result_ready
    );

  passer13 : entity work.handshake_passer_3(arch)
    port map(
      data_valid => fork22_outs_2_valid,
      data_ready => fork22_outs_2_ready,
      ctrl => fork26_outs_1,
      ctrl_valid => fork26_outs_1_valid,
      ctrl_ready => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      result_valid => passer13_result_valid,
      result_ready => passer13_result_ready
    );

  passer14 : entity work.handshake_passer_3(arch)
    port map(
      data_valid => fork22_outs_0_valid,
      data_ready => fork22_outs_0_ready,
      ctrl => fork27_outs_1,
      ctrl_valid => fork27_outs_1_valid,
      ctrl_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      result_valid => passer14_result_valid,
      result_ready => passer14_result_ready
    );

  trunci0 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork23_outs_2,
      ins_valid => fork23_outs_2_valid,
      ins_ready => fork23_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  constant10 : entity work.handshake_constant_0(arch)
    port map(
      ctrl_valid => fork22_outs_1_valid,
      ctrl_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  fork10 : entity work.handshake_fork_9(arch)
    port map(
      ins => constant10_outs,
      ins_valid => constant10_outs_valid,
      ins_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  extsi2 : entity work.handshake_extsi_0(arch)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi2_outs,
      outs_valid => extsi2_outs_valid,
      outs_ready => extsi2_outs_ready
    );

  fork11 : entity work.handshake_fork_10(arch)
    port map(
      ins => extsi2_outs,
      ins_valid => extsi2_outs_valid,
      ins_ready => extsi2_outs_ready,
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

  extsi3 : entity work.handshake_extsi_0(arch)
    port map(
      ins => fork10_outs_1,
      ins_valid => fork10_outs_1_valid,
      ins_ready => fork10_outs_1_ready,
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

  constant11 : entity work.handshake_constant_3(arch)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready
    );

  extsi4 : entity work.handshake_extsi_2(arch)
    port map(
      ins => constant11_outs,
      ins_valid => constant11_outs_valid,
      ins_ready => constant11_outs_ready,
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

  constant12 : entity work.handshake_constant_4(arch)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant12_outs,
      outs_valid => constant12_outs_valid,
      outs_ready => constant12_outs_ready
    );

  extsi5 : entity work.handshake_extsi_3(arch)
    port map(
      ins => constant12_outs,
      ins_valid => constant12_outs_valid,
      ins_ready => constant12_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  load1 : entity work.handshake_load_0(arch)
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

  buffer6 : entity work.handshake_buffer_8(arch)
    port map(
      ins => fork23_outs_3,
      ins_valid => fork23_outs_3_valid,
      ins_ready => fork23_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  muli0 : entity work.handshake_muli_0(arch)
    port map(
      lhs => buffer6_outs,
      lhs_valid => buffer6_outs_valid,
      lhs_ready => buffer6_outs_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => muli0_result,
      lhs_valid => muli0_result_valid,
      lhs_ready => muli0_result_ready,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  addi0 : entity work.handshake_addi_0(arch)
    port map(
      lhs => fork24_outs_4,
      lhs_valid => fork24_outs_4_valid,
      lhs_ready => fork24_outs_4_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  addi1 : entity work.handshake_addi_0(arch)
    port map(
      lhs => fork24_outs_3,
      lhs_valid => fork24_outs_3_valid,
      lhs_ready => fork24_outs_3_ready,
      rhs => fork11_outs_1,
      rhs_valid => fork11_outs_1_valid,
      rhs_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  buffer18 : entity work.handshake_buffer_9(arch)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  select1 : entity work.handshake_select_0(arch)
    port map(
      condition => cmpi2_result,
      condition_valid => cmpi2_result_valid,
      condition_ready => cmpi2_result_ready,
      trueValue => addi0_result,
      trueValue_valid => addi0_result_valid,
      trueValue_ready => addi0_result_ready,
      falseValue => buffer18_outs,
      falseValue_valid => buffer18_outs_valid,
      falseValue_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      result => select1_result,
      result_valid => select1_result_valid,
      result_ready => select1_result_ready
    );

  fork12 : entity work.handshake_fork_1(arch)
    port map(
      ins => select1_result,
      ins_valid => select1_result_valid,
      ins_ready => select1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  trunci1 : entity work.handshake_trunci_0(arch)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  buffer17 : entity work.handshake_buffer_9(arch)
    port map(
      ins => passer8_result,
      ins_valid => passer8_result_valid,
      ins_ready => passer8_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  store1 : entity work.handshake_store_0(arch)
    port map(
      addrIn => passer4_result,
      addrIn_valid => passer4_result_valid,
      addrIn_ready => passer4_result_ready,
      dataIn => buffer17_outs,
      dataIn_valid => buffer17_outs_valid,
      dataIn_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  addi2 : entity work.handshake_addi_0(arch)
    port map(
      lhs => fork24_outs_2,
      lhs_valid => fork24_outs_2_valid,
      lhs_ready => fork24_outs_2_ready,
      rhs => fork11_outs_2,
      rhs_valid => fork11_outs_2_valid,
      rhs_ready => fork11_outs_2_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  fork13 : entity work.handshake_fork_1(arch)
    port map(
      ins => passer1_result,
      ins_valid => passer1_result_valid,
      ins_ready => passer1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  source3 : entity work.handshake_source_0(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant13 : entity work.handshake_constant_1(arch)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant13_outs,
      outs_valid => constant13_outs_valid,
      outs_ready => constant13_outs_ready
    );

  extsi6 : entity work.handshake_extsi_1(arch)
    port map(
      ins => constant13_outs,
      ins_valid => constant13_outs_valid,
      ins_ready => constant13_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  cmpi3 : entity work.handshake_cmpi_0(arch)
    port map(
      lhs => fork13_outs_1,
      lhs_valid => fork13_outs_1_valid,
      lhs_ready => fork13_outs_1_ready,
      rhs => extsi6_outs,
      rhs_valid => extsi6_outs_valid,
      rhs_ready => extsi6_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  fork14 : entity work.handshake_fork_2(arch)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  cond_br5 : entity work.handshake_cond_br_0(arch)
    port map(
      condition => fork14_outs_1,
      condition_valid => fork14_outs_1_valid,
      condition_ready => fork14_outs_1_ready,
      data => fork13_outs_0,
      data_valid => fork13_outs_0_valid,
      data_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  sink3 : entity work.handshake_sink_0(arch)
    port map(
      ins => cond_br5_falseOut,
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst
    );

  fork15 : entity work.handshake_fork_1(arch)
    port map(
      ins => cond_br5_trueOut,
      ins_valid => cond_br5_trueOut_valid,
      ins_ready => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork15_outs_0,
      outs(1) => fork15_outs_1,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  cond_br6 : entity work.handshake_cond_br_1(arch)
    port map(
      condition => fork14_outs_0,
      condition_valid => fork14_outs_0_valid,
      condition_ready => fork14_outs_0_ready,
      data_valid => passer14_result_valid,
      data_ready => passer14_result_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  fork16 : entity work.handshake_fork_11(arch)
    port map(
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

end architecture;
