library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity jacobi_1d_imper is
  port (
    A_loadData : in std_logic_vector(31 downto 0);
    B_loadData : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    B_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    B_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    B_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    B_end_valid : out std_logic;
    end_valid : out std_logic;
    A_loadEn : out std_logic;
    A_loadAddr : out std_logic_vector(6 downto 0);
    A_storeEn : out std_logic;
    A_storeAddr : out std_logic_vector(6 downto 0);
    A_storeData : out std_logic_vector(31 downto 0);
    B_loadEn : out std_logic;
    B_loadAddr : out std_logic_vector(6 downto 0);
    B_storeEn : out std_logic;
    B_storeAddr : out std_logic_vector(6 downto 0);
    B_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of jacobi_1d_imper is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
  signal fork0_outs_3_valid : std_logic;
  signal fork0_outs_3_ready : std_logic;
  signal mem_controller0_stDone_0_valid : std_logic;
  signal mem_controller0_stDone_0_ready : std_logic;
  signal mem_controller0_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller0_ldData_0_valid : std_logic;
  signal mem_controller0_ldData_0_ready : std_logic;
  signal mem_controller0_memEnd_valid : std_logic;
  signal mem_controller0_memEnd_ready : std_logic;
  signal mem_controller0_loadEn : std_logic;
  signal mem_controller0_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller0_storeEn : std_logic;
  signal mem_controller0_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller0_storeData : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0_valid : std_logic;
  signal mem_controller1_ldData_0_ready : std_logic;
  signal mem_controller1_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_1_valid : std_logic;
  signal mem_controller1_ldData_1_ready : std_logic;
  signal mem_controller1_ldData_2 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_2_valid : std_logic;
  signal mem_controller1_ldData_2_ready : std_logic;
  signal mem_controller1_stDone_0_valid : std_logic;
  signal mem_controller1_stDone_0_ready : std_logic;
  signal mem_controller1_memEnd_valid : std_logic;
  signal mem_controller1_memEnd_ready : std_logic;
  signal mem_controller1_loadEn : std_logic;
  signal mem_controller1_loadAddr : std_logic_vector(6 downto 0);
  signal mem_controller1_storeEn : std_logic;
  signal mem_controller1_storeAddr : std_logic_vector(6 downto 0);
  signal mem_controller1_storeData : std_logic_vector(31 downto 0);
  signal constant2_outs : std_logic_vector(0 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(2 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal mux8_outs_valid : std_logic;
  signal mux8_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(2 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extsi12_outs : std_logic_vector(7 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(2 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal cond_br45_trueOut_valid : std_logic;
  signal cond_br45_trueOut_ready : std_logic;
  signal cond_br45_falseOut_valid : std_logic;
  signal cond_br45_falseOut_ready : std_logic;
  signal cond_br46_trueOut_valid : std_logic;
  signal cond_br46_trueOut_ready : std_logic;
  signal cond_br46_falseOut_valid : std_logic;
  signal cond_br46_falseOut_ready : std_logic;
  signal cond_br47_trueOut_valid : std_logic;
  signal cond_br47_trueOut_ready : std_logic;
  signal cond_br47_falseOut_valid : std_logic;
  signal cond_br47_falseOut_ready : std_logic;
  signal cond_br48_trueOut_valid : std_logic;
  signal cond_br48_trueOut_ready : std_logic;
  signal cond_br48_falseOut_valid : std_logic;
  signal cond_br48_falseOut_ready : std_logic;
  signal cond_br49_trueOut_valid : std_logic;
  signal cond_br49_trueOut_ready : std_logic;
  signal cond_br49_falseOut_valid : std_logic;
  signal cond_br49_falseOut_ready : std_logic;
  signal init8_outs : std_logic_vector(0 downto 0);
  signal init8_outs_valid : std_logic;
  signal init8_outs_ready : std_logic;
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal mux13_outs_valid : std_logic;
  signal mux13_outs_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_3_valid : std_logic;
  signal fork2_outs_3_ready : std_logic;
  signal fork2_outs_4_valid : std_logic;
  signal fork2_outs_4_ready : std_logic;
  signal unbundle0_outs_0_valid : std_logic;
  signal unbundle0_outs_0_ready : std_logic;
  signal unbundle0_outs_1 : std_logic_vector(31 downto 0);
  signal unbundle1_outs_0_valid : std_logic;
  signal unbundle1_outs_0_ready : std_logic;
  signal unbundle1_outs_1 : std_logic_vector(31 downto 0);
  signal unbundle2_outs_0_valid : std_logic;
  signal unbundle2_outs_0_ready : std_logic;
  signal unbundle2_outs_1 : std_logic_vector(31 downto 0);
  signal mux1_outs : std_logic_vector(7 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(7 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(7 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(7 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(7 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(7 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal extsi14_outs : std_logic_vector(8 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal extsi15_outs : std_logic_vector(8 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal extsi16_outs : std_logic_vector(31 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(31 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(31 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(31 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal mux2_outs : std_logic_vector(2 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal constant5_outs : std_logic_vector(1 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(1 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(1 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal fork7_outs_2 : std_logic_vector(1 downto 0);
  signal fork7_outs_2_valid : std_logic;
  signal fork7_outs_2_ready : std_logic;
  signal fork7_outs_3 : std_logic_vector(1 downto 0);
  signal fork7_outs_3_valid : std_logic;
  signal fork7_outs_3_ready : std_logic;
  signal extsi17_outs : std_logic_vector(8 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal extsi18_outs : std_logic_vector(8 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(7 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal extsi19_outs : std_logic_vector(8 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(1 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal extsi5_outs : std_logic_vector(31 downto 0);
  signal extsi5_outs_valid : std_logic;
  signal extsi5_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(31 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(31 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(31 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal load1_addrOut : std_logic_vector(6 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(31 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(31 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi3_result : std_logic_vector(8 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal extsi20_outs : std_logic_vector(31 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal gate2_outs : std_logic_vector(31 downto 0);
  signal gate2_outs_valid : std_logic;
  signal gate2_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(6 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(6 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(31 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(31 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal buffer15_outs : std_logic_vector(31 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal buffer16_outs : std_logic_vector(31 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(31 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal gate3_outs : std_logic_vector(31 downto 0);
  signal gate3_outs_valid : std_logic;
  signal gate3_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(6 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal addi4_result : std_logic_vector(8 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer19_outs : std_logic_vector(8 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(8 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(8 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal trunci4_outs : std_logic_vector(7 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
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
  signal fork13_outs_5 : std_logic_vector(0 downto 0);
  signal fork13_outs_5_valid : std_logic;
  signal fork13_outs_5_ready : std_logic;
  signal fork13_outs_6 : std_logic_vector(0 downto 0);
  signal fork13_outs_6_valid : std_logic;
  signal fork13_outs_6_ready : std_logic;
  signal fork13_outs_7 : std_logic_vector(0 downto 0);
  signal fork13_outs_7_valid : std_logic;
  signal fork13_outs_7_ready : std_logic;
  signal fork13_outs_8 : std_logic_vector(0 downto 0);
  signal fork13_outs_8_valid : std_logic;
  signal fork13_outs_8_ready : std_logic;
  signal fork13_outs_9 : std_logic_vector(0 downto 0);
  signal fork13_outs_9_valid : std_logic;
  signal fork13_outs_9_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(7 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(7 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer12_outs : std_logic_vector(2 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(2 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(2 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(2 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(1 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(1 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal extsi11_outs : std_logic_vector(7 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
  signal cond_br50_trueOut_valid : std_logic;
  signal cond_br50_trueOut_ready : std_logic;
  signal cond_br50_falseOut_valid : std_logic;
  signal cond_br50_falseOut_ready : std_logic;
  signal cond_br51_trueOut_valid : std_logic;
  signal cond_br51_trueOut_ready : std_logic;
  signal cond_br51_falseOut_valid : std_logic;
  signal cond_br51_falseOut_ready : std_logic;
  signal buffer39_outs : std_logic_vector(0 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal cond_br52_trueOut_valid : std_logic;
  signal cond_br52_trueOut_ready : std_logic;
  signal cond_br52_falseOut_valid : std_logic;
  signal cond_br52_falseOut_ready : std_logic;
  signal buffer40_outs : std_logic_vector(0 downto 0);
  signal buffer40_outs_valid : std_logic;
  signal buffer40_outs_ready : std_logic;
  signal cond_br53_trueOut_valid : std_logic;
  signal cond_br53_trueOut_ready : std_logic;
  signal cond_br53_falseOut_valid : std_logic;
  signal cond_br53_falseOut_ready : std_logic;
  signal buffer41_outs : std_logic_vector(0 downto 0);
  signal buffer41_outs_valid : std_logic;
  signal buffer41_outs_ready : std_logic;
  signal cond_br54_trueOut_valid : std_logic;
  signal cond_br54_trueOut_ready : std_logic;
  signal cond_br54_falseOut_valid : std_logic;
  signal cond_br54_falseOut_ready : std_logic;
  signal buffer42_outs : std_logic_vector(0 downto 0);
  signal buffer42_outs_valid : std_logic;
  signal buffer42_outs_ready : std_logic;
  signal init16_outs : std_logic_vector(0 downto 0);
  signal init16_outs_valid : std_logic;
  signal init16_outs_ready : std_logic;
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
  signal mux21_outs_valid : std_logic;
  signal mux21_outs_ready : std_logic;
  signal buffer44_outs : std_logic_vector(0 downto 0);
  signal buffer44_outs_valid : std_logic;
  signal buffer44_outs_ready : std_logic;
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal mux22_outs_valid : std_logic;
  signal mux22_outs_ready : std_logic;
  signal buffer45_outs : std_logic_vector(0 downto 0);
  signal buffer45_outs_valid : std_logic;
  signal buffer45_outs_ready : std_logic;
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal mux23_outs_valid : std_logic;
  signal mux23_outs_ready : std_logic;
  signal buffer46_outs : std_logic_vector(0 downto 0);
  signal buffer46_outs_valid : std_logic;
  signal buffer46_outs_ready : std_logic;
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal mux24_outs_valid : std_logic;
  signal mux24_outs_ready : std_logic;
  signal buffer47_outs : std_logic_vector(0 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal unbundle6_outs_0_valid : std_logic;
  signal unbundle6_outs_0_ready : std_logic;
  signal unbundle6_outs_1 : std_logic_vector(31 downto 0);
  signal mux3_outs : std_logic_vector(7 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer28_outs : std_logic_vector(7 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer29_outs : std_logic_vector(7 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(7 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(7 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal extsi21_outs : std_logic_vector(8 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal extsi22_outs : std_logic_vector(31 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(31 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(31 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal mux4_outs : std_logic_vector(2 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(0 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(0 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal fork22_outs_0_valid : std_logic;
  signal fork22_outs_0_ready : std_logic;
  signal fork22_outs_1_valid : std_logic;
  signal fork22_outs_1_ready : std_logic;
  signal constant17_outs : std_logic_vector(1 downto 0);
  signal constant17_outs_valid : std_logic;
  signal constant17_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant18_outs : std_logic_vector(7 downto 0);
  signal constant18_outs_valid : std_logic;
  signal constant18_outs_ready : std_logic;
  signal extsi23_outs : std_logic_vector(8 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant19_outs : std_logic_vector(1 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal extsi24_outs : std_logic_vector(8 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal gate4_outs : std_logic_vector(31 downto 0);
  signal gate4_outs_valid : std_logic;
  signal gate4_outs_ready : std_logic;
  signal buffer53_outs : std_logic_vector(31 downto 0);
  signal buffer53_outs_valid : std_logic;
  signal buffer53_outs_ready : std_logic;
  signal trunci5_outs : std_logic_vector(6 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(6 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(31 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(31 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal gate5_outs : std_logic_vector(31 downto 0);
  signal gate5_outs_valid : std_logic;
  signal gate5_outs_ready : std_logic;
  signal buffer54_outs : std_logic_vector(31 downto 0);
  signal buffer54_outs_valid : std_logic;
  signal buffer54_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(6 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(6 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal store1_doneOut_valid : std_logic;
  signal store1_doneOut_ready : std_logic;
  signal addi5_result : std_logic_vector(8 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer33_outs : std_logic_vector(8 downto 0);
  signal buffer33_outs_valid : std_logic;
  signal buffer33_outs_ready : std_logic;
  signal fork24_outs_0 : std_logic_vector(8 downto 0);
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1 : std_logic_vector(8 downto 0);
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal trunci7_outs : std_logic_vector(7 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer34_outs : std_logic_vector(0 downto 0);
  signal buffer34_outs_valid : std_logic;
  signal buffer34_outs_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(0 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(0 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal fork25_outs_2 : std_logic_vector(0 downto 0);
  signal fork25_outs_2_valid : std_logic;
  signal fork25_outs_2_ready : std_logic;
  signal fork25_outs_3 : std_logic_vector(0 downto 0);
  signal fork25_outs_3_valid : std_logic;
  signal fork25_outs_3_ready : std_logic;
  signal fork25_outs_4 : std_logic_vector(0 downto 0);
  signal fork25_outs_4_valid : std_logic;
  signal fork25_outs_4_ready : std_logic;
  signal fork25_outs_5 : std_logic_vector(0 downto 0);
  signal fork25_outs_5_valid : std_logic;
  signal fork25_outs_5_ready : std_logic;
  signal fork25_outs_6 : std_logic_vector(0 downto 0);
  signal fork25_outs_6_valid : std_logic;
  signal fork25_outs_6_ready : std_logic;
  signal fork25_outs_7 : std_logic_vector(0 downto 0);
  signal fork25_outs_7_valid : std_logic;
  signal fork25_outs_7_ready : std_logic;
  signal fork25_outs_8 : std_logic_vector(0 downto 0);
  signal fork25_outs_8_valid : std_logic;
  signal fork25_outs_8_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(7 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(7 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer30_outs : std_logic_vector(2 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal buffer31_outs : std_logic_vector(2 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(2 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(2 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal buffer60_outs : std_logic_vector(0 downto 0);
  signal buffer60_outs_valid : std_logic;
  signal buffer60_outs_ready : std_logic;
  signal cond_br55_trueOut_valid : std_logic;
  signal cond_br55_trueOut_ready : std_logic;
  signal cond_br55_falseOut_valid : std_logic;
  signal cond_br55_falseOut_ready : std_logic;
  signal extsi25_outs : std_logic_vector(3 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant20_outs : std_logic_vector(2 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(3 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant21_outs : std_logic_vector(1 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi27_outs : std_logic_vector(3 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal addi6_result : std_logic_vector(3 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer35_outs : std_logic_vector(3 downto 0);
  signal buffer35_outs_valid : std_logic;
  signal buffer35_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(3 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(3 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal trunci8_outs : std_logic_vector(2 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer36_outs : std_logic_vector(0 downto 0);
  signal buffer36_outs_valid : std_logic;
  signal buffer36_outs_ready : std_logic;
  signal fork27_outs_0 : std_logic_vector(0 downto 0);
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1 : std_logic_vector(0 downto 0);
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal fork27_outs_2 : std_logic_vector(0 downto 0);
  signal fork27_outs_2_valid : std_logic;
  signal fork27_outs_2_ready : std_logic;
  signal fork27_outs_3 : std_logic_vector(0 downto 0);
  signal fork27_outs_3_valid : std_logic;
  signal fork27_outs_3_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(2 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(2 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal fork28_outs_0_valid : std_logic;
  signal fork28_outs_0_ready : std_logic;
  signal fork28_outs_1_valid : std_logic;
  signal fork28_outs_1_ready : std_logic;

begin

  A_end_valid <= mem_controller1_memEnd_valid;
  mem_controller1_memEnd_ready <= A_end_ready;
  B_end_valid <= mem_controller0_memEnd_valid;
  mem_controller0_memEnd_ready <= B_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  A_loadEn <= mem_controller1_loadEn;
  A_loadAddr <= mem_controller1_loadAddr;
  A_storeEn <= mem_controller1_storeEn;
  A_storeAddr <= mem_controller1_storeAddr;
  A_storeData <= mem_controller1_storeData;
  B_loadEn <= mem_controller0_loadEn;
  B_loadAddr <= mem_controller0_loadAddr;
  B_storeEn <= mem_controller0_storeEn;
  B_storeAddr <= mem_controller0_storeAddr;
  B_storeData <= mem_controller0_storeData;

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

  mem_controller0 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 7)
    port map(
      loadData => B_loadData,
      memStart_valid => B_start_valid,
      memStart_ready => B_start_ready,
      ctrl(0) => extsi3_outs,
      ctrl_valid(0) => extsi3_outs_valid,
      ctrl_ready(0) => extsi3_outs_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      ctrlEnd_valid => fork28_outs_1_valid,
      ctrlEnd_ready => fork28_outs_1_ready,
      clk => clk,
      rst => rst,
      stDone_valid(0) => mem_controller0_stDone_0_valid,
      stDone_ready(0) => mem_controller0_stDone_0_ready,
      ldData(0) => mem_controller0_ldData_0,
      ldData_valid(0) => mem_controller0_ldData_0_valid,
      ldData_ready(0) => mem_controller0_ldData_0_ready,
      memEnd_valid => mem_controller0_memEnd_valid,
      memEnd_ready => mem_controller0_memEnd_ready,
      loadEn => mem_controller0_loadEn,
      loadAddr => mem_controller0_loadAddr,
      storeEn => mem_controller0_storeEn,
      storeAddr => mem_controller0_storeAddr,
      storeData => mem_controller0_storeData
    );

  mem_controller1 : entity work.mem_controller(arch) generic map(1, 3, 1, 32, 7)
    port map(
      loadData => A_loadData,
      memStart_valid => A_start_valid,
      memStart_ready => A_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr(1) => load1_addrOut,
      ldAddr(2) => load2_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_valid(1) => load1_addrOut_valid,
      ldAddr_valid(2) => load2_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ldAddr_ready(1) => load1_addrOut_ready,
      ldAddr_ready(2) => load2_addrOut_ready,
      ctrl(0) => extsi6_outs,
      ctrl_valid(0) => extsi6_outs_valid,
      ctrl_ready(0) => extsi6_outs_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork28_outs_0_valid,
      ctrlEnd_ready => fork28_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller1_ldData_0,
      ldData(1) => mem_controller1_ldData_1,
      ldData(2) => mem_controller1_ldData_2,
      ldData_valid(0) => mem_controller1_ldData_0_valid,
      ldData_valid(1) => mem_controller1_ldData_1_valid,
      ldData_valid(2) => mem_controller1_ldData_2_valid,
      ldData_ready(0) => mem_controller1_ldData_0_ready,
      ldData_ready(1) => mem_controller1_ldData_1_ready,
      ldData_ready(2) => mem_controller1_ldData_2_ready,
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

  constant2 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(1, 3)
    port map(
      ins => constant2_outs,
      ins_valid => constant2_outs_valid,
      ins_ready => constant2_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  mux8 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => init0_outs,
      index_valid => init0_outs_valid,
      index_ready => init0_outs_ready,
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br55_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br55_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux8_outs_valid,
      outs_ready => mux8_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork27_outs_2,
      ins_valid => fork27_outs_2_valid,
      ins_ready => fork27_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi13_outs,
      ins(1) => cond_br10_trueOut,
      ins_valid(0) => extsi13_outs_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => extsi13_outs_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_3_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => fork0_outs_3_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork1 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  constant4 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork1_outs_0_valid,
      ctrl_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extsi12 : entity work.extsi(arch) generic map(2, 8)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  buffer7 : entity work.tehb(arch) generic map(3)
    port map(
      ins => mux0_outs,
      ins_valid => mux0_outs_valid,
      ins_ready => mux0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  cond_br45 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_7,
      condition_valid => fork13_outs_7_valid,
      condition_ready => fork13_outs_7_ready,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br45_trueOut_valid,
      trueOut_ready => cond_br45_trueOut_ready,
      falseOut_valid => cond_br45_falseOut_valid,
      falseOut_ready => cond_br45_falseOut_ready
    );

  sink0 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br45_trueOut_valid,
      ins_ready => cond_br45_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br46 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_6,
      condition_valid => fork13_outs_6_valid,
      condition_ready => fork13_outs_6_ready,
      data_valid => buffer1_outs_valid,
      data_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br46_trueOut_valid,
      trueOut_ready => cond_br46_trueOut_ready,
      falseOut_valid => cond_br46_falseOut_valid,
      falseOut_ready => cond_br46_falseOut_ready
    );

  sink1 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br46_trueOut_valid,
      ins_ready => cond_br46_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br47 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_5,
      condition_valid => fork13_outs_5_valid,
      condition_ready => fork13_outs_5_ready,
      data_valid => buffer2_outs_valid,
      data_ready => buffer2_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br47_trueOut_valid,
      trueOut_ready => cond_br47_trueOut_ready,
      falseOut_valid => cond_br47_falseOut_valid,
      falseOut_ready => cond_br47_falseOut_ready
    );

  sink2 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br47_trueOut_valid,
      ins_ready => cond_br47_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br48 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_4,
      condition_valid => fork13_outs_4_valid,
      condition_ready => fork13_outs_4_ready,
      data_valid => buffer0_outs_valid,
      data_ready => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br48_trueOut_valid,
      trueOut_ready => cond_br48_trueOut_ready,
      falseOut_valid => cond_br48_falseOut_valid,
      falseOut_ready => cond_br48_falseOut_ready
    );

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br48_trueOut_valid,
      ins_ready => cond_br48_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br49 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_3,
      condition_valid => fork13_outs_3_valid,
      condition_ready => fork13_outs_3_ready,
      data_valid => fork2_outs_4_valid,
      data_ready => fork2_outs_4_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br49_trueOut_valid,
      trueOut_ready => cond_br49_trueOut_ready,
      falseOut_valid => cond_br49_falseOut_valid,
      falseOut_ready => cond_br49_falseOut_ready
    );

  sink4 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br49_falseOut_valid,
      ins_ready => cond_br49_falseOut_ready,
      clk => clk,
      rst => rst
    );

  init8 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork13_outs_2,
      ins_valid => fork13_outs_2_valid,
      ins_ready => fork13_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => init8_outs,
      outs_valid => init8_outs_valid,
      outs_ready => init8_outs_ready
    );

  buffer6 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux8_outs_valid,
      ins_ready => mux8_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  mux13 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => init8_outs,
      index_valid => init8_outs_valid,
      index_ready => init8_outs_ready,
      ins_valid(0) => buffer6_outs_valid,
      ins_valid(1) => cond_br49_trueOut_valid,
      ins_ready(0) => buffer6_outs_ready,
      ins_ready(1) => cond_br49_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux13_outs_valid,
      outs_ready => mux13_outs_ready
    );

  buffer8 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux13_outs_valid,
      ins_ready => mux13_outs_ready,
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

  fork2 : entity work.fork_dataless(arch) generic map(5)
    port map(
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_valid(2) => fork2_outs_2_valid,
      outs_valid(3) => fork2_outs_3_valid,
      outs_valid(4) => fork2_outs_4_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready,
      outs_ready(2) => fork2_outs_2_ready,
      outs_ready(3) => fork2_outs_3_ready,
      outs_ready(4) => fork2_outs_4_ready
    );

  unbundle0 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork9_outs_0,
      ins_valid => fork9_outs_0_valid,
      ins_ready => fork9_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle0_outs_0_valid,
      outs_ready => unbundle0_outs_0_ready,
      outs => unbundle0_outs_1
    );

  unbundle1 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle1_outs_0_valid,
      outs_ready => unbundle1_outs_0_ready,
      outs => unbundle1_outs_1
    );

  unbundle2 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle2_outs_0_valid,
      outs_ready => unbundle2_outs_0_ready,
      outs => unbundle2_outs_1
    );

  mux1 : entity work.mux(arch) generic map(2, 8, 1)
    port map(
      index => fork5_outs_1,
      index_valid => fork5_outs_1_valid,
      index_ready => fork5_outs_1_ready,
      ins(0) => extsi12_outs,
      ins(1) => cond_br3_trueOut,
      ins_valid(0) => extsi12_outs_valid,
      ins_valid(1) => cond_br3_trueOut_valid,
      ins_ready(0) => extsi12_outs_ready,
      ins_ready(1) => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer10 : entity work.oehb(arch) generic map(8)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer11 : entity work.tehb(arch) generic map(8)
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

  fork3 : entity work.handshake_fork(arch) generic map(3, 8)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
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

  extsi14 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  extsi15 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => fork3_outs_1,
      ins_valid => fork3_outs_1_valid,
      ins_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  extsi16 : entity work.extsi(arch) generic map(8, 32)
    port map(
      ins => fork3_outs_2,
      ins_valid => fork3_outs_2_valid,
      ins_ready => fork3_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(3, 32)
    port map(
      ins => extsi16_outs,
      ins_valid => extsi16_outs_valid,
      ins_ready => extsi16_outs_ready,
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

  mux2 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => fork5_outs_0,
      index_valid => fork5_outs_0_valid,
      index_ready => fork5_outs_0_ready,
      ins(0) => buffer7_outs,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => buffer7_outs_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => buffer7_outs_ready,
      ins_ready(1) => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  buffer14 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  fork6 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  constant5 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork6_outs_0_valid,
      ctrl_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(4, 2)
    port map(
      ins => constant5_outs,
      ins_valid => constant5_outs_valid,
      ins_ready => constant5_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs(2) => fork7_outs_2,
      outs(3) => fork7_outs_3,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_valid(2) => fork7_outs_2_valid,
      outs_valid(3) => fork7_outs_3_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready,
      outs_ready(2) => fork7_outs_2_ready,
      outs_ready(3) => fork7_outs_3_ready
    );

  extsi17 : entity work.extsi(arch) generic map(2, 9)
    port map(
      ins => fork7_outs_0,
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  extsi18 : entity work.extsi(arch) generic map(2, 9)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => fork7_outs_3,
      ins_valid => fork7_outs_3_valid,
      ins_ready => fork7_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant7 : entity work.handshake_constant_2(arch) generic map(32)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant7_outs,
      outs_valid => constant7_outs_valid,
      outs_ready => constant7_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant15 : entity work.handshake_constant_3(arch) generic map(8)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  extsi19 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant16 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  extsi5 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant16_outs,
      ins_valid => constant16_outs_valid,
      ins_ready => constant16_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi5_outs,
      outs_valid => extsi5_outs_valid,
      outs_ready => extsi5_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork4_outs_0,
      lhs_valid => fork4_outs_0_valid,
      lhs_ready => fork4_outs_0_ready,
      rhs => constant7_outs,
      rhs_valid => constant7_outs_valid,
      rhs_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer0 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle1_outs_0_valid,
      ins_ready => unbundle1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => addi2_result,
      ins_valid(0) => addi2_result_valid,
      ins_valid(1) => fork2_outs_3_valid,
      ins_ready(0) => addi2_result_ready,
      ins_ready(1) => fork2_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  trunci0 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
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

  fork8 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle0_outs_0_valid,
      ins_ready => unbundle0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork4_outs_1,
      ins_valid(0) => fork4_outs_1_valid,
      ins_valid(1) => fork2_outs_2_valid,
      ins_ready(0) => fork4_outs_1_ready,
      ins_ready(1) => fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate1_outs,
      ins_valid => gate1_outs_valid,
      ins_ready => gate1_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  load1 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller1_ldData_1,
      dataFromMem_valid => mem_controller1_ldData_1_valid,
      dataFromMem_ready => mem_controller1_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load1_dataOut,
      ins_valid => load1_dataOut_valid,
      ins_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork8_outs_1,
      lhs_valid => fork8_outs_1_valid,
      lhs_ready => fork8_outs_1_ready,
      rhs => fork9_outs_1,
      rhs_valid => fork9_outs_1_valid,
      rhs_ready => fork9_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  addi3 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi15_outs,
      lhs_valid => extsi15_outs_valid,
      lhs_ready => extsi15_outs_ready,
      rhs => extsi18_outs,
      rhs_valid => extsi18_outs_valid,
      rhs_ready => extsi18_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  extsi20 : entity work.extsi(arch) generic map(9, 32)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  buffer2 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle2_outs_0_valid,
      ins_ready => unbundle2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  gate2 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => extsi20_outs,
      ins_valid(0) => extsi20_outs_valid,
      ins_valid(1) => fork2_outs_1_valid,
      ins_ready(0) => extsi20_outs_ready,
      ins_ready(1) => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => gate2_outs,
      outs_valid => gate2_outs_valid,
      outs_ready => gate2_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate2_outs,
      ins_valid => gate2_outs_valid,
      ins_ready => gate2_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => trunci2_outs,
      addrIn_valid => trunci2_outs_valid,
      addrIn_ready => trunci2_outs_ready,
      dataFromMem => mem_controller1_ldData_2,
      dataFromMem_valid => mem_controller1_ldData_2_valid,
      dataFromMem_ready => mem_controller1_ldData_2_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load2_dataOut,
      ins_valid => load2_dataOut_valid,
      ins_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  buffer15 : entity work.oehb(arch) generic map(32)
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

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer15_outs,
      lhs_valid => buffer15_outs_valid,
      lhs_ready => buffer15_outs_ready,
      rhs => fork10_outs_1,
      rhs_valid => fork10_outs_1_valid,
      rhs_ready => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  buffer16 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork11_outs_1,
      lhs_valid => fork11_outs_1_valid,
      lhs_ready => fork11_outs_1_ready,
      rhs => extsi5_outs,
      rhs_valid => extsi5_outs_valid,
      rhs_ready => extsi5_outs_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer17 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork11_outs_0,
      lhs_valid => fork11_outs_0_valid,
      lhs_ready => fork11_outs_0_ready,
      rhs => buffer17_outs,
      rhs_valid => buffer17_outs_valid,
      rhs_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
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

  gate3 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => fork4_outs_2,
      ins_valid(0) => fork4_outs_2_valid,
      ins_valid(1) => fork2_outs_0_valid,
      ins_ready(0) => fork4_outs_2_ready,
      ins_ready(1) => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => gate3_outs,
      outs_valid => gate3_outs_valid,
      outs_ready => gate3_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(32, 7)
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

  buffer18 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => trunci3_outs,
      addrIn_valid => trunci3_outs_valid,
      addrIn_ready => trunci3_outs_ready,
      dataIn => buffer18_outs,
      dataIn_valid => buffer18_outs_valid,
      dataIn_ready => buffer18_outs_ready,
      doneFromMem_valid => mem_controller0_stDone_0_valid,
      doneFromMem_ready => mem_controller0_stDone_0_ready,
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

  addi4 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi14_outs,
      lhs_valid => extsi14_outs_valid,
      lhs_ready => extsi14_outs_ready,
      rhs => extsi17_outs,
      rhs_valid => extsi17_outs_valid,
      rhs_ready => extsi17_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer19 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 9)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  trunci4 : entity work.trunci(arch) generic map(9, 8)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(9)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => extsi19_outs,
      rhs_valid => extsi19_outs_valid,
      rhs_ready => extsi19_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(10, 1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs(3) => fork13_outs_3,
      outs(4) => fork13_outs_4,
      outs(5) => fork13_outs_5,
      outs(6) => fork13_outs_6,
      outs(7) => fork13_outs_7,
      outs(8) => fork13_outs_8,
      outs(9) => fork13_outs_9,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_valid(3) => fork13_outs_3_valid,
      outs_valid(4) => fork13_outs_4_valid,
      outs_valid(5) => fork13_outs_5_valid,
      outs_valid(6) => fork13_outs_6_valid,
      outs_valid(7) => fork13_outs_7_valid,
      outs_valid(8) => fork13_outs_8_valid,
      outs_valid(9) => fork13_outs_9_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready,
      outs_ready(3) => fork13_outs_3_ready,
      outs_ready(4) => fork13_outs_4_ready,
      outs_ready(5) => fork13_outs_5_ready,
      outs_ready(6) => fork13_outs_6_ready,
      outs_ready(7) => fork13_outs_7_ready,
      outs_ready(8) => fork13_outs_8_ready,
      outs_ready(9) => fork13_outs_9_ready
    );

  cond_br3 : entity work.cond_br(arch) generic map(8)
    port map(
      condition => fork13_outs_0,
      condition_valid => fork13_outs_0_valid,
      condition_ready => fork13_outs_0_ready,
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

  sink5 : entity work.sink(arch) generic map(8)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer12 : entity work.oehb(arch) generic map(3)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer13 : entity work.tehb(arch) generic map(3)
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

  cond_br4 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork13_outs_1,
      condition_valid => fork13_outs_1_valid,
      condition_ready => fork13_outs_1_ready,
      data => buffer13_outs,
      data_valid => buffer13_outs_valid,
      data_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  cond_br5 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_8,
      condition_valid => fork13_outs_8_valid,
      condition_ready => fork13_outs_8_ready,
      data_valid => fork6_outs_1_valid,
      data_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork13_outs_9,
      condition_valid => fork13_outs_9_valid,
      condition_ready => fork13_outs_9_ready,
      data => fork7_outs_2,
      data_valid => fork7_outs_2_valid,
      data_ready => fork7_outs_2_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink6 : entity work.sink(arch) generic map(2)
    port map(
      ins => cond_br6_trueOut,
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi11 : entity work.extsi(arch) generic map(2, 8)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  cond_br50 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork25_outs_7,
      condition_valid => fork25_outs_7_valid,
      condition_ready => fork25_outs_7_ready,
      data_valid => fork18_outs_1_valid,
      data_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br50_trueOut_valid,
      trueOut_ready => cond_br50_trueOut_ready,
      falseOut_valid => cond_br50_falseOut_valid,
      falseOut_ready => cond_br50_falseOut_ready
    );

  sink7 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br50_falseOut_valid,
      ins_ready => cond_br50_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br51 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer39_outs,
      condition_valid => buffer39_outs_valid,
      condition_ready => buffer39_outs_ready,
      data_valid => buffer5_outs_valid,
      data_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br51_trueOut_valid,
      trueOut_ready => cond_br51_trueOut_ready,
      falseOut_valid => cond_br51_falseOut_valid,
      falseOut_ready => cond_br51_falseOut_ready
    );

  buffer39 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork25_outs_6,
      ins_valid => fork25_outs_6_valid,
      ins_ready => fork25_outs_6_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  sink8 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br51_trueOut_valid,
      ins_ready => cond_br51_trueOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br52 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer40_outs,
      condition_valid => buffer40_outs_valid,
      condition_ready => buffer40_outs_ready,
      data_valid => fork16_outs_1_valid,
      data_ready => fork16_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br52_trueOut_valid,
      trueOut_ready => cond_br52_trueOut_ready,
      falseOut_valid => cond_br52_falseOut_valid,
      falseOut_ready => cond_br52_falseOut_ready
    );

  buffer40 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork25_outs_5,
      ins_valid => fork25_outs_5_valid,
      ins_ready => fork25_outs_5_ready,
      clk => clk,
      rst => rst,
      outs => buffer40_outs,
      outs_valid => buffer40_outs_valid,
      outs_ready => buffer40_outs_ready
    );

  sink9 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br52_falseOut_valid,
      ins_ready => cond_br52_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br53 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer41_outs,
      condition_valid => buffer41_outs_valid,
      condition_ready => buffer41_outs_ready,
      data_valid => fork17_outs_1_valid,
      data_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br53_trueOut_valid,
      trueOut_ready => cond_br53_trueOut_ready,
      falseOut_valid => cond_br53_falseOut_valid,
      falseOut_ready => cond_br53_falseOut_ready
    );

  buffer41 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork25_outs_4,
      ins_valid => fork25_outs_4_valid,
      ins_ready => fork25_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => buffer41_outs,
      outs_valid => buffer41_outs_valid,
      outs_ready => buffer41_outs_ready
    );

  sink10 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br53_falseOut_valid,
      ins_ready => cond_br53_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br54 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer42_outs,
      condition_valid => buffer42_outs_valid,
      condition_ready => buffer42_outs_ready,
      data_valid => fork15_outs_1_valid,
      data_ready => fork15_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br54_trueOut_valid,
      trueOut_ready => cond_br54_trueOut_ready,
      falseOut_valid => cond_br54_falseOut_valid,
      falseOut_ready => cond_br54_falseOut_ready
    );

  buffer42 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork25_outs_3,
      ins_valid => fork25_outs_3_valid,
      ins_ready => fork25_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer42_outs,
      outs_valid => buffer42_outs_valid,
      outs_ready => buffer42_outs_ready
    );

  sink11 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br54_falseOut_valid,
      ins_ready => cond_br54_falseOut_ready,
      clk => clk,
      rst => rst
    );

  init16 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork25_outs_2,
      ins_valid => fork25_outs_2_valid,
      ins_ready => fork25_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => init16_outs,
      outs_valid => init16_outs_valid,
      outs_ready => init16_outs_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => init16_outs,
      ins_valid => init16_outs_valid,
      ins_ready => init16_outs_ready,
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

  mux21 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer44_outs,
      index_valid => buffer44_outs_valid,
      index_ready => buffer44_outs_ready,
      ins_valid(0) => cond_br46_falseOut_valid,
      ins_valid(1) => cond_br54_trueOut_valid,
      ins_ready(0) => cond_br46_falseOut_ready,
      ins_ready(1) => cond_br54_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux21_outs_valid,
      outs_ready => mux21_outs_ready
    );

  buffer44 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_3,
      ins_valid => fork14_outs_3_valid,
      ins_ready => fork14_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer44_outs,
      outs_valid => buffer44_outs_valid,
      outs_ready => buffer44_outs_ready
    );

  buffer21 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux21_outs_valid,
      ins_ready => mux21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  buffer22 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  fork15 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  mux22 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer45_outs,
      index_valid => buffer45_outs_valid,
      index_ready => buffer45_outs_ready,
      ins_valid(0) => cond_br48_falseOut_valid,
      ins_valid(1) => cond_br52_trueOut_valid,
      ins_ready(0) => cond_br48_falseOut_ready,
      ins_ready(1) => cond_br52_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux22_outs_valid,
      outs_ready => mux22_outs_ready
    );

  buffer45 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_2,
      ins_valid => fork14_outs_2_valid,
      ins_ready => fork14_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer45_outs,
      outs_valid => buffer45_outs_valid,
      outs_ready => buffer45_outs_ready
    );

  buffer23 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux22_outs_valid,
      ins_ready => mux22_outs_ready,
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

  fork16 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  mux23 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer46_outs,
      index_valid => buffer46_outs_valid,
      index_ready => buffer46_outs_ready,
      ins_valid(0) => cond_br47_falseOut_valid,
      ins_valid(1) => cond_br53_trueOut_valid,
      ins_ready(0) => cond_br47_falseOut_ready,
      ins_ready(1) => cond_br53_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux23_outs_valid,
      outs_ready => mux23_outs_ready
    );

  buffer46 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_1,
      ins_valid => fork14_outs_1_valid,
      ins_ready => fork14_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer46_outs,
      outs_valid => buffer46_outs_valid,
      outs_ready => buffer46_outs_ready
    );

  buffer25 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => mux23_outs_valid,
      ins_ready => mux23_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  buffer26 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  fork17 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork17_outs_0_valid,
      outs_valid(1) => fork17_outs_1_valid,
      outs_ready(0) => fork17_outs_0_ready,
      outs_ready(1) => fork17_outs_1_ready
    );

  buffer20 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br50_trueOut_valid,
      ins_ready => cond_br50_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  mux24 : entity work.mux_dataless(arch) generic map(2, 1)
    port map(
      index => buffer47_outs,
      index_valid => buffer47_outs_valid,
      index_ready => buffer47_outs_ready,
      ins_valid(0) => cond_br45_falseOut_valid,
      ins_valid(1) => buffer20_outs_valid,
      ins_ready(0) => cond_br45_falseOut_ready,
      ins_ready(1) => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => mux24_outs_valid,
      outs_ready => mux24_outs_ready
    );

  buffer47 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork14_outs_0,
      ins_valid => fork14_outs_0_valid,
      ins_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  buffer27 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => mux24_outs_valid,
      ins_ready => mux24_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork18 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  unbundle6 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle6_outs_0_valid,
      outs_ready => unbundle6_outs_0_ready,
      outs => unbundle6_outs_1
    );

  mux3 : entity work.mux(arch) generic map(2, 8, 1)
    port map(
      index => fork21_outs_1,
      index_valid => fork21_outs_1_valid,
      index_ready => fork21_outs_1_ready,
      ins(0) => extsi11_outs,
      ins(1) => cond_br7_trueOut,
      ins_valid(0) => extsi11_outs_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => extsi11_outs_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer28 : entity work.oehb(arch) generic map(8)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  buffer29 : entity work.tehb(arch) generic map(8)
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

  fork19 : entity work.handshake_fork(arch) generic map(2, 8)
    port map(
      ins => buffer29_outs,
      ins_valid => buffer29_outs_valid,
      ins_ready => buffer29_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork19_outs_0,
      outs(1) => fork19_outs_1,
      outs_valid(0) => fork19_outs_0_valid,
      outs_valid(1) => fork19_outs_1_valid,
      outs_ready(0) => fork19_outs_0_ready,
      outs_ready(1) => fork19_outs_1_ready
    );

  extsi21 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => fork19_outs_0,
      ins_valid => fork19_outs_0_valid,
      ins_ready => fork19_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  extsi22 : entity work.extsi(arch) generic map(8, 32)
    port map(
      ins => fork19_outs_1,
      ins_valid => fork19_outs_1_valid,
      ins_ready => fork19_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => extsi22_outs,
      ins_valid => extsi22_outs_valid,
      ins_ready => extsi22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => fork21_outs_0,
      index_valid => fork21_outs_0_valid,
      index_ready => fork21_outs_0_ready,
      ins(0) => cond_br4_falseOut,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => cond_br4_falseOut_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => cond_br4_falseOut_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  control_merge2 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => cond_br5_falseOut_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => cond_br5_falseOut_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge2_outs_valid,
      outs_ready => control_merge2_outs_ready,
      index => control_merge2_index,
      index_valid => control_merge2_index_valid,
      index_ready => control_merge2_index_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  fork22 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork22_outs_0_valid,
      outs_valid(1) => fork22_outs_1_valid,
      outs_ready(0) => fork22_outs_0_ready,
      outs_ready(1) => fork22_outs_1_ready
    );

  constant17 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork22_outs_0_valid,
      ctrl_ready => fork22_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant17_outs,
      outs_valid => constant17_outs_valid,
      outs_ready => constant17_outs_ready
    );

  extsi6 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant17_outs,
      ins_valid => constant17_outs_valid,
      ins_ready => constant17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant18 : entity work.handshake_constant_3(arch) generic map(8)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant18_outs,
      outs_valid => constant18_outs_valid,
      outs_ready => constant18_outs_ready
    );

  extsi23 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => constant18_outs,
      ins_valid => constant18_outs_valid,
      ins_ready => constant18_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant19 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant19_outs,
      outs_valid => constant19_outs_valid,
      outs_ready => constant19_outs_ready
    );

  extsi24 : entity work.extsi(arch) generic map(2, 9)
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

  buffer4 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle6_outs_0_valid,
      ins_ready => unbundle6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  sink12 : entity work.sink_dataless(arch)
    port map(
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst
    );

  gate4 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => buffer53_outs,
      ins_valid(0) => buffer53_outs_valid,
      ins_valid(1) => fork18_outs_0_valid,
      ins_ready(0) => buffer53_outs_ready,
      ins_ready(1) => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => gate4_outs,
      outs_valid => gate4_outs_valid,
      outs_ready => gate4_outs_ready
    );

  buffer53 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer53_outs,
      outs_valid => buffer53_outs_valid,
      outs_ready => buffer53_outs_ready
    );

  trunci5 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate4_outs,
      ins_valid => gate4_outs_valid,
      ins_ready => gate4_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => trunci5_outs,
      addrIn_valid => trunci5_outs_valid,
      addrIn_ready => trunci5_outs_ready,
      dataFromMem => mem_controller0_ldData_0,
      dataFromMem_valid => mem_controller0_ldData_0_valid,
      dataFromMem_ready => mem_controller0_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load3_dataOut,
      ins_valid => load3_dataOut_valid,
      ins_ready => load3_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  buffer5 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => store1_doneOut_valid,
      ins_ready => store1_doneOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  gate5 : entity work.gate(arch) generic map(4, 32)
    port map(
      ins(0) => buffer54_outs,
      ins_valid(0) => buffer54_outs_valid,
      ins_valid(1) => fork17_outs_0_valid,
      ins_valid(2) => fork16_outs_0_valid,
      ins_valid(3) => fork15_outs_0_valid,
      ins_ready(0) => buffer54_outs_ready,
      ins_ready(1) => fork17_outs_0_ready,
      ins_ready(2) => fork16_outs_0_ready,
      ins_ready(3) => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => gate5_outs,
      outs_valid => gate5_outs_valid,
      outs_ready => gate5_outs_ready
    );

  buffer54 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork20_outs_1,
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer54_outs,
      outs_valid => buffer54_outs_valid,
      outs_ready => buffer54_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => gate5_outs,
      ins_valid => gate5_outs_valid,
      ins_ready => gate5_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => trunci6_outs,
      addrIn_valid => trunci6_outs_valid,
      addrIn_ready => trunci6_outs_ready,
      dataIn => fork23_outs_0,
      dataIn_valid => fork23_outs_0_valid,
      dataIn_ready => fork23_outs_0_ready,
      doneFromMem_valid => mem_controller1_stDone_0_valid,
      doneFromMem_ready => mem_controller1_stDone_0_ready,
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

  addi5 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi21_outs,
      lhs_valid => extsi21_outs_valid,
      lhs_ready => extsi21_outs_ready,
      rhs => extsi24_outs,
      rhs_valid => extsi24_outs_valid,
      rhs_ready => extsi24_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer33 : entity work.oehb(arch) generic map(9)
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

  fork24 : entity work.handshake_fork(arch) generic map(2, 9)
    port map(
      ins => buffer33_outs,
      ins_valid => buffer33_outs_valid,
      ins_ready => buffer33_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork24_outs_0,
      outs(1) => fork24_outs_1,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  trunci7 : entity work.trunci(arch) generic map(9, 8)
    port map(
      ins => fork24_outs_0,
      ins_valid => fork24_outs_0_valid,
      ins_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(9)
    port map(
      lhs => fork24_outs_1,
      lhs_valid => fork24_outs_1_valid,
      lhs_ready => fork24_outs_1_ready,
      rhs => extsi23_outs,
      rhs_valid => extsi23_outs_valid,
      rhs_ready => extsi23_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer34 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer34_outs,
      outs_valid => buffer34_outs_valid,
      outs_ready => buffer34_outs_ready
    );

  fork25 : entity work.handshake_fork(arch) generic map(9, 1)
    port map(
      ins => buffer34_outs,
      ins_valid => buffer34_outs_valid,
      ins_ready => buffer34_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs(2) => fork25_outs_2,
      outs(3) => fork25_outs_3,
      outs(4) => fork25_outs_4,
      outs(5) => fork25_outs_5,
      outs(6) => fork25_outs_6,
      outs(7) => fork25_outs_7,
      outs(8) => fork25_outs_8,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_valid(2) => fork25_outs_2_valid,
      outs_valid(3) => fork25_outs_3_valid,
      outs_valid(4) => fork25_outs_4_valid,
      outs_valid(5) => fork25_outs_5_valid,
      outs_valid(6) => fork25_outs_6_valid,
      outs_valid(7) => fork25_outs_7_valid,
      outs_valid(8) => fork25_outs_8_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready,
      outs_ready(2) => fork25_outs_2_ready,
      outs_ready(3) => fork25_outs_3_ready,
      outs_ready(4) => fork25_outs_4_ready,
      outs_ready(5) => fork25_outs_5_ready,
      outs_ready(6) => fork25_outs_6_ready,
      outs_ready(7) => fork25_outs_7_ready,
      outs_ready(8) => fork25_outs_8_ready
    );

  cond_br7 : entity work.cond_br(arch) generic map(8)
    port map(
      condition => fork25_outs_0,
      condition_valid => fork25_outs_0_valid,
      condition_ready => fork25_outs_0_ready,
      data => trunci7_outs,
      data_valid => trunci7_outs_valid,
      data_ready => trunci7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink13 : entity work.sink(arch) generic map(8)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer30 : entity work.oehb(arch) generic map(3)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  buffer31 : entity work.tehb(arch) generic map(3)
    port map(
      ins => buffer30_outs,
      ins_valid => buffer30_outs_valid,
      ins_ready => buffer30_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork25_outs_1,
      condition_valid => fork25_outs_1_valid,
      condition_ready => fork25_outs_1_ready,
      data => buffer31_outs,
      data_valid => buffer31_outs_valid,
      data_ready => buffer31_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  buffer32 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => fork22_outs_1_valid,
      ins_ready => fork22_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  cond_br9 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer60_outs,
      condition_valid => buffer60_outs_valid,
      condition_ready => buffer60_outs_ready,
      data_valid => buffer32_outs_valid,
      data_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  buffer60 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork25_outs_8,
      ins_valid => fork25_outs_8_valid,
      ins_ready => fork25_outs_8_ready,
      clk => clk,
      rst => rst,
      outs => buffer60_outs,
      outs_valid => buffer60_outs_valid,
      outs_ready => buffer60_outs_ready
    );

  cond_br55 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork27_outs_1,
      condition_valid => fork27_outs_1_valid,
      condition_ready => fork27_outs_1_ready,
      data_valid => cond_br51_falseOut_valid,
      data_ready => cond_br51_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br55_trueOut_valid,
      trueOut_ready => cond_br55_trueOut_ready,
      falseOut_valid => cond_br55_falseOut_valid,
      falseOut_ready => cond_br55_falseOut_ready
    );

  sink14 : entity work.sink_dataless(arch)
    port map(
      ins_valid => cond_br55_falseOut_valid,
      ins_ready => cond_br55_falseOut_ready,
      clk => clk,
      rst => rst
    );

  extsi25 : entity work.extsi(arch) generic map(3, 4)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant20 : entity work.handshake_constant_4(arch) generic map(3)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(3, 4)
    port map(
      ins => constant20_outs,
      ins_valid => constant20_outs_valid,
      ins_ready => constant20_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi26_outs,
      outs_valid => extsi26_outs_valid,
      outs_ready => extsi26_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant21 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  extsi27 : entity work.extsi(arch) generic map(2, 4)
    port map(
      ins => constant21_outs,
      ins_valid => constant21_outs_valid,
      ins_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi27_outs,
      outs_valid => extsi27_outs_valid,
      outs_ready => extsi27_outs_ready
    );

  addi6 : entity work.addi(arch) generic map(4)
    port map(
      lhs => extsi25_outs,
      lhs_valid => extsi25_outs_valid,
      lhs_ready => extsi25_outs_ready,
      rhs => extsi27_outs,
      rhs_valid => extsi27_outs_valid,
      rhs_ready => extsi27_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer35 : entity work.oehb(arch) generic map(4)
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

  fork26 : entity work.handshake_fork(arch) generic map(2, 4)
    port map(
      ins => buffer35_outs,
      ins_valid => buffer35_outs_valid,
      ins_ready => buffer35_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork26_outs_0,
      outs(1) => fork26_outs_1,
      outs_valid(0) => fork26_outs_0_valid,
      outs_valid(1) => fork26_outs_1_valid,
      outs_ready(0) => fork26_outs_0_ready,
      outs_ready(1) => fork26_outs_1_ready
    );

  trunci8 : entity work.trunci(arch) generic map(4, 3)
    port map(
      ins => fork26_outs_0,
      ins_valid => fork26_outs_0_valid,
      ins_ready => fork26_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_1(arch) generic map(4)
    port map(
      lhs => fork26_outs_1,
      lhs_valid => fork26_outs_1_valid,
      lhs_ready => fork26_outs_1_ready,
      rhs => extsi26_outs,
      rhs_valid => extsi26_outs_valid,
      rhs_ready => extsi26_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer36 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer36_outs,
      outs_valid => buffer36_outs_valid,
      outs_ready => buffer36_outs_ready
    );

  fork27 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer36_outs,
      ins_valid => buffer36_outs_valid,
      ins_ready => buffer36_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork27_outs_0,
      outs(1) => fork27_outs_1,
      outs(2) => fork27_outs_2,
      outs(3) => fork27_outs_3,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_valid(2) => fork27_outs_2_valid,
      outs_valid(3) => fork27_outs_3_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready,
      outs_ready(2) => fork27_outs_2_ready,
      outs_ready(3) => fork27_outs_3_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork27_outs_0,
      condition_valid => fork27_outs_0_valid,
      condition_ready => fork27_outs_0_ready,
      data => trunci8_outs,
      data_valid => trunci8_outs_valid,
      data_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink16 : entity work.sink(arch) generic map(3)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br11 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork27_outs_3,
      condition_valid => fork27_outs_3_valid,
      condition_ready => fork27_outs_3_ready,
      data_valid => cond_br9_falseOut_valid,
      data_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  fork28 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br11_falseOut_valid,
      ins_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork28_outs_0_valid,
      outs_valid(1) => fork28_outs_1_valid,
      outs_ready(0) => fork28_outs_0_ready,
      outs_ready(1) => fork28_outs_1_ready
    );

end architecture;
