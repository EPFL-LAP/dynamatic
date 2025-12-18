library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mvt_float is
  port (
    A_loadData : in std_logic_vector(31 downto 0);
    x1_loadData : in std_logic_vector(31 downto 0);
    x2_loadData : in std_logic_vector(31 downto 0);
    y1_loadData : in std_logic_vector(31 downto 0);
    y2_loadData : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    x1_start_valid : in std_logic;
    x2_start_valid : in std_logic;
    y1_start_valid : in std_logic;
    y2_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    A_end_ready : in std_logic;
    x1_end_ready : in std_logic;
    x2_end_ready : in std_logic;
    y1_end_ready : in std_logic;
    y2_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    x1_start_ready : out std_logic;
    x2_start_ready : out std_logic;
    y1_start_ready : out std_logic;
    y2_start_ready : out std_logic;
    start_ready : out std_logic;
    A_end_valid : out std_logic;
    x1_end_valid : out std_logic;
    x2_end_valid : out std_logic;
    y1_end_valid : out std_logic;
    y2_end_valid : out std_logic;
    end_valid : out std_logic;
    A_loadEn : out std_logic;
    A_loadAddr : out std_logic_vector(9 downto 0);
    A_storeEn : out std_logic;
    A_storeAddr : out std_logic_vector(9 downto 0);
    A_storeData : out std_logic_vector(31 downto 0);
    x1_loadEn : out std_logic;
    x1_loadAddr : out std_logic_vector(4 downto 0);
    x1_storeEn : out std_logic;
    x1_storeAddr : out std_logic_vector(4 downto 0);
    x1_storeData : out std_logic_vector(31 downto 0);
    x2_loadEn : out std_logic;
    x2_loadAddr : out std_logic_vector(4 downto 0);
    x2_storeEn : out std_logic;
    x2_storeAddr : out std_logic_vector(4 downto 0);
    x2_storeData : out std_logic_vector(31 downto 0);
    y1_loadEn : out std_logic;
    y1_loadAddr : out std_logic_vector(4 downto 0);
    y1_storeEn : out std_logic;
    y1_storeAddr : out std_logic_vector(4 downto 0);
    y1_storeData : out std_logic_vector(31 downto 0);
    y2_loadEn : out std_logic;
    y2_loadAddr : out std_logic_vector(4 downto 0);
    y2_storeEn : out std_logic;
    y2_storeAddr : out std_logic_vector(4 downto 0);
    y2_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of mvt_float is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal fork0_outs_2_valid : std_logic;
  signal fork0_outs_2_ready : std_logic;
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
  signal mem_controller5_stDone_0_valid : std_logic;
  signal mem_controller5_stDone_0_ready : std_logic;
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
  signal mem_controller6_stDone_0_valid : std_logic;
  signal mem_controller6_stDone_0_ready : std_logic;
  signal mem_controller6_memEnd_valid : std_logic;
  signal mem_controller6_memEnd_ready : std_logic;
  signal mem_controller6_loadEn : std_logic;
  signal mem_controller6_loadAddr : std_logic_vector(4 downto 0);
  signal mem_controller6_storeEn : std_logic;
  signal mem_controller6_storeAddr : std_logic_vector(4 downto 0);
  signal mem_controller6_storeData : std_logic_vector(31 downto 0);
  signal mem_controller7_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller7_ldData_0_valid : std_logic;
  signal mem_controller7_ldData_0_ready : std_logic;
  signal mem_controller7_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller7_ldData_1_valid : std_logic;
  signal mem_controller7_ldData_1_ready : std_logic;
  signal mem_controller7_memEnd_valid : std_logic;
  signal mem_controller7_memEnd_ready : std_logic;
  signal mem_controller7_loadEn : std_logic;
  signal mem_controller7_loadAddr : std_logic_vector(9 downto 0);
  signal mem_controller7_storeEn : std_logic;
  signal mem_controller7_storeAddr : std_logic_vector(9 downto 0);
  signal mem_controller7_storeData : std_logic_vector(31 downto 0);
  signal constant14_outs : std_logic_vector(0 downto 0);
  signal constant14_outs_valid : std_logic;
  signal constant14_outs_ready : std_logic;
  signal extsi17_outs : std_logic_vector(5 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal init0_outs : std_logic_vector(0 downto 0);
  signal init0_outs_valid : std_logic;
  signal init0_outs_ready : std_logic;
  signal unbundle0_outs_0_valid : std_logic;
  signal unbundle0_outs_0_ready : std_logic;
  signal unbundle0_outs_1 : std_logic_vector(31 downto 0);
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(5 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal fork1_outs_0 : std_logic_vector(5 downto 0);
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1 : std_logic_vector(5 downto 0);
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal trunci0_outs : std_logic_vector(4 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal control_merge0_outs_valid : std_logic;
  signal control_merge0_outs_ready : std_logic;
  signal control_merge0_index : std_logic_vector(0 downto 0);
  signal control_merge0_index_valid : std_logic;
  signal control_merge0_index_ready : std_logic;
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal constant15_outs : std_logic_vector(0 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(4 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(31 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(31 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal extsi16_outs : std_logic_vector(5 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(5 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(5 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(5 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(5 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(5 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal extsi18_outs : std_logic_vector(9 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(5 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal extsi19_outs : std_logic_vector(6 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(4 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal buffer10_outs : std_logic_vector(5 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(0 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(5 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(5 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(5 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(5 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(5 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal extsi20_outs : std_logic_vector(11 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(5 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(0 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(0 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal fork6_outs_2 : std_logic_vector(0 downto 0);
  signal fork6_outs_2_valid : std_logic;
  signal fork6_outs_2_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(5 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(5 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(5 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal extsi21_outs : std_logic_vector(11 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(5 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal extsi22_outs : std_logic_vector(6 downto 0);
  signal extsi22_outs_valid : std_logic;
  signal extsi22_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant17_outs : std_logic_vector(1 downto 0);
  signal constant17_outs_valid : std_logic;
  signal constant17_outs_ready : std_logic;
  signal extsi23_outs : std_logic_vector(6 downto 0);
  signal extsi23_outs_valid : std_logic;
  signal extsi23_outs_ready : std_logic;
  signal muli0_result : std_logic_vector(11 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal trunci2_outs : std_logic_vector(9 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(9 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
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
  signal mulf0_result : std_logic_vector(31 downto 0);
  signal mulf0_result_valid : std_logic;
  signal mulf0_result_ready : std_logic;
  signal buffer4_outs : std_logic_vector(31 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal addf0_result : std_logic_vector(31 downto 0);
  signal addf0_result_valid : std_logic;
  signal addf0_result_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer9_outs : std_logic_vector(6 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(6 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(6 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal trunci3_outs : std_logic_vector(5 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer12_outs : std_logic_vector(0 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(0 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(0 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal fork9_outs_2 : std_logic_vector(0 downto 0);
  signal fork9_outs_2_valid : std_logic;
  signal fork9_outs_2_ready : std_logic;
  signal fork9_outs_3 : std_logic_vector(0 downto 0);
  signal fork9_outs_3_valid : std_logic;
  signal fork9_outs_3_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(5 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(5 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut : std_logic_vector(31 downto 0);
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut : std_logic_vector(31 downto 0);
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(5 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(5 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(5 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(5 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal extsi24_outs : std_logic_vector(6 downto 0);
  signal extsi24_outs_valid : std_logic;
  signal extsi24_outs_ready : std_logic;
  signal extsi25_outs : std_logic_vector(31 downto 0);
  signal extsi25_outs_valid : std_logic;
  signal extsi25_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(5 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal constant18_outs : std_logic_vector(1 downto 0);
  signal constant18_outs_valid : std_logic;
  signal constant18_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal constant19_outs : std_logic_vector(0 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant20_outs : std_logic_vector(5 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal extsi26_outs : std_logic_vector(6 downto 0);
  signal extsi26_outs_valid : std_logic;
  signal extsi26_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant21_outs : std_logic_vector(1 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi27_outs : std_logic_vector(6 downto 0);
  signal extsi27_outs_valid : std_logic;
  signal extsi27_outs_ready : std_logic;
  signal gate0_outs : std_logic_vector(31 downto 0);
  signal gate0_outs_valid : std_logic;
  signal gate0_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(4 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(4 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal store0_doneOut_valid : std_logic;
  signal store0_doneOut_ready : std_logic;
  signal addi3_result : std_logic_vector(6 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer15_outs : std_logic_vector(6 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(6 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(6 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal trunci5_outs : std_logic_vector(5 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer16_outs : std_logic_vector(0 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
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
  signal cond_br8_trueOut : std_logic_vector(5 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(5 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal cond_br10_trueOut : std_logic_vector(0 downto 0);
  signal cond_br10_trueOut_valid : std_logic;
  signal cond_br10_trueOut_ready : std_logic;
  signal cond_br10_falseOut : std_logic_vector(0 downto 0);
  signal cond_br10_falseOut_valid : std_logic;
  signal cond_br10_falseOut_ready : std_logic;
  signal extsi15_outs : std_logic_vector(5 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal init1_outs : std_logic_vector(0 downto 0);
  signal init1_outs_valid : std_logic;
  signal init1_outs_ready : std_logic;
  signal unbundle2_outs_0_valid : std_logic;
  signal unbundle2_outs_0_ready : std_logic;
  signal unbundle2_outs_1 : std_logic_vector(31 downto 0);
  signal mux4_outs : std_logic_vector(5 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(5 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork14_outs_0 : std_logic_vector(5 downto 0);
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1 : std_logic_vector(5 downto 0);
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal trunci6_outs : std_logic_vector(4 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal control_merge3_outs_valid : std_logic;
  signal control_merge3_outs_ready : std_logic;
  signal control_merge3_index : std_logic_vector(0 downto 0);
  signal control_merge3_index_valid : std_logic;
  signal control_merge3_index_ready : std_logic;
  signal fork15_outs_0_valid : std_logic;
  signal fork15_outs_0_ready : std_logic;
  signal fork15_outs_1_valid : std_logic;
  signal fork15_outs_1_ready : std_logic;
  signal constant22_outs : std_logic_vector(0 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(4 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal fork16_outs_0 : std_logic_vector(31 downto 0);
  signal fork16_outs_0_valid : std_logic;
  signal fork16_outs_0_ready : std_logic;
  signal fork16_outs_1 : std_logic_vector(31 downto 0);
  signal fork16_outs_1_valid : std_logic;
  signal fork16_outs_1_ready : std_logic;
  signal extsi14_outs : std_logic_vector(5 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal mux5_outs : std_logic_vector(5 downto 0);
  signal mux5_outs_valid : std_logic;
  signal mux5_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(5 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal fork17_outs_0 : std_logic_vector(5 downto 0);
  signal fork17_outs_0_valid : std_logic;
  signal fork17_outs_0_ready : std_logic;
  signal fork17_outs_1 : std_logic_vector(5 downto 0);
  signal fork17_outs_1_valid : std_logic;
  signal fork17_outs_1_ready : std_logic;
  signal fork17_outs_2 : std_logic_vector(5 downto 0);
  signal fork17_outs_2_valid : std_logic;
  signal fork17_outs_2_ready : std_logic;
  signal extsi28_outs : std_logic_vector(6 downto 0);
  signal extsi28_outs_valid : std_logic;
  signal extsi28_outs_ready : std_logic;
  signal extsi29_outs : std_logic_vector(11 downto 0);
  signal extsi29_outs_valid : std_logic;
  signal extsi29_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(4 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal mux6_outs : std_logic_vector(31 downto 0);
  signal mux6_outs_valid : std_logic;
  signal mux6_outs_ready : std_logic;
  signal buffer39_outs : std_logic_vector(0 downto 0);
  signal buffer39_outs_valid : std_logic;
  signal buffer39_outs_ready : std_logic;
  signal mux7_outs : std_logic_vector(5 downto 0);
  signal mux7_outs_valid : std_logic;
  signal mux7_outs_ready : std_logic;
  signal buffer21_outs : std_logic_vector(5 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(5 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal fork18_outs_0 : std_logic_vector(5 downto 0);
  signal fork18_outs_0_valid : std_logic;
  signal fork18_outs_0_ready : std_logic;
  signal fork18_outs_1 : std_logic_vector(5 downto 0);
  signal fork18_outs_1_valid : std_logic;
  signal fork18_outs_1_ready : std_logic;
  signal extsi30_outs : std_logic_vector(9 downto 0);
  signal extsi30_outs_valid : std_logic;
  signal extsi30_outs_ready : std_logic;
  signal control_merge4_outs_valid : std_logic;
  signal control_merge4_outs_ready : std_logic;
  signal control_merge4_index : std_logic_vector(0 downto 0);
  signal control_merge4_index_valid : std_logic;
  signal control_merge4_index_ready : std_logic;
  signal fork19_outs_0 : std_logic_vector(0 downto 0);
  signal fork19_outs_0_valid : std_logic;
  signal fork19_outs_0_ready : std_logic;
  signal fork19_outs_1 : std_logic_vector(0 downto 0);
  signal fork19_outs_1_valid : std_logic;
  signal fork19_outs_1_ready : std_logic;
  signal fork19_outs_2 : std_logic_vector(0 downto 0);
  signal fork19_outs_2_valid : std_logic;
  signal fork19_outs_2_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant23_outs : std_logic_vector(5 downto 0);
  signal constant23_outs_valid : std_logic;
  signal constant23_outs_ready : std_logic;
  signal fork20_outs_0 : std_logic_vector(5 downto 0);
  signal fork20_outs_0_valid : std_logic;
  signal fork20_outs_0_ready : std_logic;
  signal fork20_outs_1 : std_logic_vector(5 downto 0);
  signal fork20_outs_1_valid : std_logic;
  signal fork20_outs_1_ready : std_logic;
  signal extsi31_outs : std_logic_vector(6 downto 0);
  signal extsi31_outs_valid : std_logic;
  signal extsi31_outs_ready : std_logic;
  signal extsi32_outs : std_logic_vector(11 downto 0);
  signal extsi32_outs_valid : std_logic;
  signal extsi32_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant24_outs : std_logic_vector(1 downto 0);
  signal constant24_outs_valid : std_logic;
  signal constant24_outs_ready : std_logic;
  signal extsi33_outs : std_logic_vector(6 downto 0);
  signal extsi33_outs_valid : std_logic;
  signal extsi33_outs_ready : std_logic;
  signal muli1_result : std_logic_vector(11 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal trunci8_outs : std_logic_vector(9 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(9 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal load4_addrOut : std_logic_vector(9 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal load5_addrOut : std_logic_vector(4 downto 0);
  signal load5_addrOut_valid : std_logic;
  signal load5_addrOut_ready : std_logic;
  signal load5_dataOut : std_logic_vector(31 downto 0);
  signal load5_dataOut_valid : std_logic;
  signal load5_dataOut_ready : std_logic;
  signal mulf1_result : std_logic_vector(31 downto 0);
  signal mulf1_result_valid : std_logic;
  signal mulf1_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(31 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal addf1_result : std_logic_vector(31 downto 0);
  signal addf1_result_valid : std_logic;
  signal addf1_result_ready : std_logic;
  signal addi4_result : std_logic_vector(6 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer25_outs : std_logic_vector(6 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal fork21_outs_0 : std_logic_vector(6 downto 0);
  signal fork21_outs_0_valid : std_logic;
  signal fork21_outs_0_ready : std_logic;
  signal fork21_outs_1 : std_logic_vector(6 downto 0);
  signal fork21_outs_1_valid : std_logic;
  signal fork21_outs_1_ready : std_logic;
  signal trunci9_outs : std_logic_vector(5 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer26_outs : std_logic_vector(0 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
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
  signal cond_br11_trueOut : std_logic_vector(5 downto 0);
  signal cond_br11_trueOut_valid : std_logic;
  signal cond_br11_trueOut_ready : std_logic;
  signal cond_br11_falseOut : std_logic_vector(5 downto 0);
  signal cond_br11_falseOut_valid : std_logic;
  signal cond_br11_falseOut_ready : std_logic;
  signal cond_br12_trueOut : std_logic_vector(31 downto 0);
  signal cond_br12_trueOut_valid : std_logic;
  signal cond_br12_trueOut_ready : std_logic;
  signal cond_br12_falseOut : std_logic_vector(31 downto 0);
  signal cond_br12_falseOut_valid : std_logic;
  signal cond_br12_falseOut_ready : std_logic;
  signal buffer47_outs : std_logic_vector(0 downto 0);
  signal buffer47_outs_valid : std_logic;
  signal buffer47_outs_ready : std_logic;
  signal cond_br13_trueOut : std_logic_vector(5 downto 0);
  signal cond_br13_trueOut_valid : std_logic;
  signal cond_br13_trueOut_ready : std_logic;
  signal cond_br13_falseOut : std_logic_vector(5 downto 0);
  signal cond_br13_falseOut_valid : std_logic;
  signal cond_br13_falseOut_ready : std_logic;
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal cond_br14_trueOut_valid : std_logic;
  signal cond_br14_trueOut_ready : std_logic;
  signal cond_br14_falseOut_valid : std_logic;
  signal cond_br14_falseOut_ready : std_logic;
  signal fork23_outs_0 : std_logic_vector(5 downto 0);
  signal fork23_outs_0_valid : std_logic;
  signal fork23_outs_0_ready : std_logic;
  signal fork23_outs_1 : std_logic_vector(5 downto 0);
  signal fork23_outs_1_valid : std_logic;
  signal fork23_outs_1_ready : std_logic;
  signal extsi34_outs : std_logic_vector(6 downto 0);
  signal extsi34_outs_valid : std_logic;
  signal extsi34_outs_ready : std_logic;
  signal extsi35_outs : std_logic_vector(31 downto 0);
  signal extsi35_outs_valid : std_logic;
  signal extsi35_outs_ready : std_logic;
  signal buffer52_outs : std_logic_vector(5 downto 0);
  signal buffer52_outs_valid : std_logic;
  signal buffer52_outs_ready : std_logic;
  signal fork24_outs_0_valid : std_logic;
  signal fork24_outs_0_ready : std_logic;
  signal fork24_outs_1_valid : std_logic;
  signal fork24_outs_1_ready : std_logic;
  signal constant25_outs : std_logic_vector(1 downto 0);
  signal constant25_outs_valid : std_logic;
  signal constant25_outs_ready : std_logic;
  signal extsi11_outs : std_logic_vector(31 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant26_outs : std_logic_vector(5 downto 0);
  signal constant26_outs_valid : std_logic;
  signal constant26_outs_ready : std_logic;
  signal extsi36_outs : std_logic_vector(6 downto 0);
  signal extsi36_outs_valid : std_logic;
  signal extsi36_outs_ready : std_logic;
  signal source7_outs_valid : std_logic;
  signal source7_outs_ready : std_logic;
  signal constant27_outs : std_logic_vector(1 downto 0);
  signal constant27_outs_valid : std_logic;
  signal constant27_outs_ready : std_logic;
  signal extsi37_outs : std_logic_vector(6 downto 0);
  signal extsi37_outs_valid : std_logic;
  signal extsi37_outs_ready : std_logic;
  signal gate1_outs : std_logic_vector(31 downto 0);
  signal gate1_outs_valid : std_logic;
  signal gate1_outs_ready : std_logic;
  signal trunci10_outs : std_logic_vector(4 downto 0);
  signal trunci10_outs_valid : std_logic;
  signal trunci10_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(4 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal store1_doneOut_valid : std_logic;
  signal store1_doneOut_ready : std_logic;
  signal addi5_result : std_logic_vector(6 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer27_outs : std_logic_vector(6 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork25_outs_0 : std_logic_vector(6 downto 0);
  signal fork25_outs_0_valid : std_logic;
  signal fork25_outs_0_ready : std_logic;
  signal fork25_outs_1 : std_logic_vector(6 downto 0);
  signal fork25_outs_1_valid : std_logic;
  signal fork25_outs_1_ready : std_logic;
  signal trunci11_outs : std_logic_vector(5 downto 0);
  signal trunci11_outs_valid : std_logic;
  signal trunci11_outs_ready : std_logic;
  signal cmpi3_result : std_logic_vector(0 downto 0);
  signal cmpi3_result_valid : std_logic;
  signal cmpi3_result_ready : std_logic;
  signal buffer28_outs : std_logic_vector(0 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal fork26_outs_0 : std_logic_vector(0 downto 0);
  signal fork26_outs_0_valid : std_logic;
  signal fork26_outs_0_ready : std_logic;
  signal fork26_outs_1 : std_logic_vector(0 downto 0);
  signal fork26_outs_1_valid : std_logic;
  signal fork26_outs_1_ready : std_logic;
  signal fork26_outs_2 : std_logic_vector(0 downto 0);
  signal fork26_outs_2_valid : std_logic;
  signal fork26_outs_2_ready : std_logic;
  signal cond_br15_trueOut : std_logic_vector(5 downto 0);
  signal cond_br15_trueOut_valid : std_logic;
  signal cond_br15_trueOut_ready : std_logic;
  signal cond_br15_falseOut : std_logic_vector(5 downto 0);
  signal cond_br15_falseOut_valid : std_logic;
  signal cond_br15_falseOut_ready : std_logic;
  signal cond_br16_trueOut_valid : std_logic;
  signal cond_br16_trueOut_ready : std_logic;
  signal cond_br16_falseOut_valid : std_logic;
  signal cond_br16_falseOut_ready : std_logic;
  signal fork27_outs_0_valid : std_logic;
  signal fork27_outs_0_ready : std_logic;
  signal fork27_outs_1_valid : std_logic;
  signal fork27_outs_1_ready : std_logic;
  signal fork27_outs_2_valid : std_logic;
  signal fork27_outs_2_ready : std_logic;
  signal fork27_outs_3_valid : std_logic;
  signal fork27_outs_3_ready : std_logic;
  signal fork27_outs_4_valid : std_logic;
  signal fork27_outs_4_ready : std_logic;

begin

  A_end_valid <= mem_controller7_memEnd_valid;
  mem_controller7_memEnd_ready <= A_end_ready;
  x1_end_valid <= mem_controller6_memEnd_valid;
  mem_controller6_memEnd_ready <= x1_end_ready;
  x2_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= x2_end_ready;
  y1_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= y1_end_ready;
  y2_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= y2_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  A_loadEn <= mem_controller7_loadEn;
  A_loadAddr <= mem_controller7_loadAddr;
  A_storeEn <= mem_controller7_storeEn;
  A_storeAddr <= mem_controller7_storeAddr;
  A_storeData <= mem_controller7_storeData;
  x1_loadEn <= mem_controller6_loadEn;
  x1_loadAddr <= mem_controller6_loadAddr;
  x1_storeEn <= mem_controller6_storeEn;
  x1_storeAddr <= mem_controller6_storeAddr;
  x1_storeData <= mem_controller6_storeData;
  x2_loadEn <= mem_controller5_loadEn;
  x2_loadAddr <= mem_controller5_loadAddr;
  x2_storeEn <= mem_controller5_storeEn;
  x2_storeAddr <= mem_controller5_storeAddr;
  x2_storeData <= mem_controller5_storeData;
  y1_loadEn <= mem_controller4_loadEn;
  y1_loadAddr <= mem_controller4_loadAddr;
  y1_storeEn <= mem_controller4_storeEn;
  y1_storeAddr <= mem_controller4_storeAddr;
  y1_storeData <= mem_controller4_storeData;
  y2_loadEn <= mem_controller3_loadEn;
  y2_loadAddr <= mem_controller3_loadAddr;
  y2_storeEn <= mem_controller3_storeEn;
  y2_storeAddr <= mem_controller3_storeAddr;
  y2_storeData <= mem_controller3_storeData;

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

  mem_controller3 : entity work.mem_controller_storeless(arch) generic map(1, 32, 5)
    port map(
      loadData => y2_loadData,
      memStart_valid => y2_start_valid,
      memStart_ready => y2_start_ready,
      ldAddr(0) => load5_addrOut,
      ldAddr_valid(0) => load5_addrOut_valid,
      ldAddr_ready(0) => load5_addrOut_ready,
      ctrlEnd_valid => fork27_outs_4_valid,
      ctrlEnd_ready => fork27_outs_4_ready,
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
      loadData => y1_loadData,
      memStart_valid => y1_start_valid,
      memStart_ready => y1_start_ready,
      ldAddr(0) => load2_addrOut,
      ldAddr_valid(0) => load2_addrOut_valid,
      ldAddr_ready(0) => load2_addrOut_ready,
      ctrlEnd_valid => fork27_outs_3_valid,
      ctrlEnd_ready => fork27_outs_3_ready,
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

  mem_controller5 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 5)
    port map(
      loadData => x2_loadData,
      memStart_valid => x2_start_valid,
      memStart_ready => x2_start_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      ctrl(0) => extsi11_outs,
      ctrl_valid(0) => extsi11_outs_valid,
      ctrl_ready(0) => extsi11_outs_ready,
      stAddr(0) => store1_addrOut,
      stAddr_valid(0) => store1_addrOut_valid,
      stAddr_ready(0) => store1_addrOut_ready,
      stData(0) => store1_dataToMem,
      stData_valid(0) => store1_dataToMem_valid,
      stData_ready(0) => store1_dataToMem_ready,
      ctrlEnd_valid => fork27_outs_2_valid,
      ctrlEnd_ready => fork27_outs_2_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller5_ldData_0,
      ldData_valid(0) => mem_controller5_ldData_0_valid,
      ldData_ready(0) => mem_controller5_ldData_0_ready,
      stDone_valid(0) => mem_controller5_stDone_0_valid,
      stDone_ready(0) => mem_controller5_stDone_0_ready,
      memEnd_valid => mem_controller5_memEnd_valid,
      memEnd_ready => mem_controller5_memEnd_ready,
      loadEn => mem_controller5_loadEn,
      loadAddr => mem_controller5_loadAddr,
      storeEn => mem_controller5_storeEn,
      storeAddr => mem_controller5_storeAddr,
      storeData => mem_controller5_storeData
    );

  mem_controller6 : entity work.mem_controller(arch) generic map(1, 1, 1, 32, 5)
    port map(
      loadData => x1_loadData,
      memStart_valid => x1_start_valid,
      memStart_ready => x1_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ctrl(0) => extsi4_outs,
      ctrl_valid(0) => extsi4_outs_valid,
      ctrl_ready(0) => extsi4_outs_ready,
      stAddr(0) => store0_addrOut,
      stAddr_valid(0) => store0_addrOut_valid,
      stAddr_ready(0) => store0_addrOut_ready,
      stData(0) => store0_dataToMem,
      stData_valid(0) => store0_dataToMem_valid,
      stData_ready(0) => store0_dataToMem_ready,
      ctrlEnd_valid => fork27_outs_1_valid,
      ctrlEnd_ready => fork27_outs_1_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller6_ldData_0,
      ldData_valid(0) => mem_controller6_ldData_0_valid,
      ldData_ready(0) => mem_controller6_ldData_0_ready,
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

  mem_controller7 : entity work.mem_controller_storeless(arch) generic map(2, 32, 10)
    port map(
      loadData => A_loadData,
      memStart_valid => A_start_valid,
      memStart_ready => A_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr(1) => load4_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_valid(1) => load4_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ldAddr_ready(1) => load4_addrOut_ready,
      ctrlEnd_valid => fork27_outs_0_valid,
      ctrlEnd_ready => fork27_outs_0_ready,
      clk => clk,
      rst => rst,
      ldData(0) => mem_controller7_ldData_0,
      ldData(1) => mem_controller7_ldData_1,
      ldData_valid(0) => mem_controller7_ldData_0_valid,
      ldData_valid(1) => mem_controller7_ldData_1_valid,
      ldData_ready(0) => mem_controller7_ldData_0_ready,
      ldData_ready(1) => mem_controller7_ldData_1_ready,
      memEnd_valid => mem_controller7_memEnd_valid,
      memEnd_ready => mem_controller7_memEnd_ready,
      loadEn => mem_controller7_loadEn,
      loadAddr => mem_controller7_loadAddr,
      storeEn => mem_controller7_storeEn,
      storeAddr => mem_controller7_storeAddr,
      storeData => mem_controller7_storeData
    );

  constant14 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork0_outs_0_valid,
      ctrl_ready => fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant14_outs,
      outs_valid => constant14_outs_valid,
      outs_ready => constant14_outs_ready
    );

  extsi17 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant14_outs,
      ins_valid => constant14_outs_valid,
      ins_ready => constant14_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  init0 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork13_outs_1,
      ins_valid => fork13_outs_1_valid,
      ins_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => init0_outs,
      outs_valid => init0_outs_valid,
      outs_ready => init0_outs_ready
    );

  sink0 : entity work.sink(arch) generic map(1)
    port map(
      ins => init0_outs,
      ins_valid => init0_outs_valid,
      ins_ready => init0_outs_ready,
      clk => clk,
      rst => rst
    );

  unbundle0 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle0_outs_0_valid,
      outs_ready => unbundle0_outs_0_ready,
      outs => unbundle0_outs_1
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi17_outs,
      ins(1) => cond_br8_trueOut,
      ins_valid(0) => extsi17_outs_valid,
      ins_valid(1) => cond_br8_trueOut_valid,
      ins_ready(0) => extsi17_outs_ready,
      ins_ready(1) => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer2 : entity work.tehb(arch) generic map(6)
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

  fork1 : entity work.handshake_fork(arch) generic map(2, 6)
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

  trunci0 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork1_outs_0,
      ins_valid => fork1_outs_0_valid,
      ins_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br9_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
      ins_ready(1) => cond_br9_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge0_outs_valid,
      outs_ready => control_merge0_outs_ready,
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready
    );

  fork2 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  constant15 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork2_outs_0_valid,
      ctrl_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  buffer0 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle0_outs_0_valid,
      ins_ready => unbundle0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer0_outs_valid,
      outs_ready => buffer0_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci0_outs,
      addrIn_valid => trunci0_outs_valid,
      addrIn_ready => trunci0_outs_ready,
      dataFromMem => mem_controller6_ldData_0,
      dataFromMem_valid => mem_controller6_ldData_0_valid,
      dataFromMem_ready => mem_controller6_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  extsi16 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork6_outs_1,
      index_valid => fork6_outs_1_valid,
      index_ready => fork6_outs_1_ready,
      ins(0) => extsi16_outs,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => extsi16_outs_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => extsi16_outs_ready,
      ins_ready(1) => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer3 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
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

  extsi18 : entity work.extsi(arch) generic map(6, 10)
    port map(
      ins => buffer8_outs,
      ins_valid => buffer8_outs_valid,
      ins_ready => buffer8_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  buffer8 : entity work.tfifo(arch) generic map(9, 6)
    port map(
      ins => fork4_outs_0,
      ins_valid => fork4_outs_0_valid,
      ins_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  extsi19 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork4_outs_2,
      ins_valid => fork4_outs_2_valid,
      ins_ready => fork4_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => buffer10_outs,
      ins_valid => buffer10_outs_valid,
      ins_ready => buffer10_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  buffer10 : entity work.tfifo(arch) generic map(9, 6)
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

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer11_outs,
      index_valid => buffer11_outs_valid,
      index_ready => buffer11_outs_ready,
      ins(0) => fork3_outs_1,
      ins(1) => cond_br5_trueOut,
      ins_valid(0) => fork3_outs_1_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => fork3_outs_1_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer11 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork6_outs_2,
      ins_valid => fork6_outs_2_valid,
      ins_ready => fork6_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork6_outs_0,
      index_valid => fork6_outs_0_valid,
      index_ready => fork6_outs_0_ready,
      ins(0) => fork1_outs_1,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => fork1_outs_1_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => fork1_outs_1_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer5 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer5_outs,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  buffer6 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer5_outs,
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  extsi20 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  buffer13 : entity work.tfifo(arch) generic map(5, 6)
    port map(
      ins => fork5_outs_1,
      ins_valid => fork5_outs_1_valid,
      ins_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork2_outs_1_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => fork2_outs_1_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge1_outs_valid,
      outs_ready => control_merge1_outs_ready,
      index => control_merge1_index,
      index_valid => control_merge1_index_valid,
      index_ready => control_merge1_index_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
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

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant16 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => constant16_outs,
      ins_valid => constant16_outs_valid,
      ins_ready => constant16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  extsi21 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  buffer14 : entity work.tfifo(arch) generic map(5, 6)
    port map(
      ins => fork7_outs_0,
      ins_valid => fork7_outs_0_valid,
      ins_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  extsi22 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork7_outs_1,
      ins_valid => fork7_outs_1_valid,
      ins_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi22_outs,
      outs_valid => extsi22_outs_valid,
      outs_ready => extsi22_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant17 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant17_outs,
      outs_valid => constant17_outs_valid,
      outs_ready => constant17_outs_ready
    );

  extsi23 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant17_outs,
      ins_valid => constant17_outs_valid,
      ins_ready => constant17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi23_outs,
      outs_valid => extsi23_outs_valid,
      outs_ready => extsi23_outs_ready
    );

  muli0 : entity work.muli(arch) generic map(12)
    port map(
      lhs => extsi20_outs,
      lhs_valid => extsi20_outs_valid,
      lhs_ready => extsi20_outs_ready,
      rhs => extsi21_outs,
      rhs_valid => extsi21_outs_valid,
      rhs_ready => extsi21_outs_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  trunci2 : entity work.trunci(arch) generic map(12, 10)
    port map(
      ins => muli0_result,
      ins_valid => muli0_result_valid,
      ins_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  addi0 : entity work.addi(arch) generic map(10)
    port map(
      lhs => extsi18_outs,
      lhs_valid => extsi18_outs_valid,
      lhs_ready => extsi18_outs_ready,
      rhs => trunci2_outs,
      rhs_valid => trunci2_outs_valid,
      rhs_ready => trunci2_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  load1 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => addi0_result,
      addrIn_valid => addi0_result_valid,
      addrIn_ready => addi0_result_ready,
      dataFromMem => mem_controller7_ldData_0,
      dataFromMem_valid => mem_controller7_ldData_0_valid,
      dataFromMem_ready => mem_controller7_ldData_0_ready,
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
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
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

  mulf0 : entity work.mulf(arch_32_2_875333) generic map(32)
    port map(
      lhs => load1_dataOut,
      lhs_valid => load1_dataOut_valid,
      lhs_ready => load1_dataOut_ready,
      rhs => load2_dataOut,
      rhs_valid => load2_dataOut_valid,
      rhs_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      result => mulf0_result,
      result_valid => mulf0_result_valid,
      result_ready => mulf0_result_ready
    );

  buffer4 : entity work.tehb(arch) generic map(32)
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

  addf0 : entity work.addf(arch_32_2_922000) generic map(32)
    port map(
      lhs => buffer4_outs,
      lhs_valid => buffer4_outs_valid,
      lhs_ready => buffer4_outs_ready,
      rhs => mulf0_result,
      rhs_valid => mulf0_result_valid,
      rhs_ready => mulf0_result_ready,
      clk => clk,
      rst => rst,
      result => addf0_result,
      result_valid => addf0_result_valid,
      result_ready => addf0_result_ready
    );

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi19_outs,
      lhs_valid => extsi19_outs_valid,
      lhs_ready => extsi19_outs_ready,
      rhs => extsi23_outs,
      rhs_valid => extsi23_outs_valid,
      rhs_ready => extsi23_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer9 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(2, 7)
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

  trunci3 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork8_outs_1,
      lhs_valid => fork8_outs_1_valid,
      lhs_ready => fork8_outs_1_ready,
      rhs => extsi22_outs,
      rhs_valid => extsi22_outs_valid,
      rhs_ready => extsi22_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer12 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
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

  cond_br4 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork9_outs_0,
      condition_valid => fork9_outs_0_valid,
      condition_ready => fork9_outs_0_ready,
      data => trunci3_outs,
      data_valid => trunci3_outs_valid,
      data_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br5 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer19_outs,
      condition_valid => buffer19_outs_valid,
      condition_ready => buffer19_outs_ready,
      data => addf0_result,
      data_valid => addf0_result_valid,
      data_ready => addf0_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br5_trueOut,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut => cond_br5_falseOut,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer19 : entity work.tfifo(arch) generic map(15, 1)
    port map(
      ins => fork9_outs_2,
      ins_valid => fork9_outs_2_valid,
      ins_ready => fork9_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork9_outs_1,
      condition_valid => fork9_outs_1_valid,
      condition_ready => fork9_outs_1_ready,
      data => fork5_outs_0,
      data_valid => fork5_outs_0_valid,
      data_ready => fork5_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  buffer7 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  cond_br7 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork9_outs_3,
      condition_valid => fork9_outs_3_valid,
      condition_ready => fork9_outs_3_ready,
      data_valid => buffer7_outs_valid,
      data_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  extsi24 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi24_outs,
      outs_valid => extsi24_outs_valid,
      outs_ready => extsi24_outs_ready
    );

  extsi25 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi25_outs,
      outs_valid => extsi25_outs_valid,
      outs_ready => extsi25_outs_ready
    );

  buffer24 : entity work.tfifo(arch) generic map(1, 6)
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

  fork11 : entity work.fork_dataless(arch) generic map(3)
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

  constant18 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork11_outs_1_valid,
      ctrl_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant18_outs,
      outs_valid => constant18_outs_valid,
      outs_ready => constant18_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant18_outs,
      ins_valid => constant18_outs_valid,
      ins_ready => constant18_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  constant19 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork11_outs_0_valid,
      ctrl_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant19_outs,
      outs_valid => constant19_outs_valid,
      outs_ready => constant19_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant20 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  extsi26 : entity work.extsi(arch) generic map(6, 7)
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

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant21 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant21_outs,
      outs_valid => constant21_outs_valid,
      outs_ready => constant21_outs_ready
    );

  extsi27 : entity work.extsi(arch) generic map(2, 7)
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

  gate0 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => extsi25_outs,
      ins_valid(0) => extsi25_outs_valid,
      ins_valid(1) => buffer0_outs_valid,
      ins_ready(0) => extsi25_outs_ready,
      ins_ready(1) => buffer0_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate0_outs,
      outs_valid => gate0_outs_valid,
      outs_ready => gate0_outs_ready
    );

  trunci4 : entity work.trunci(arch) generic map(32, 5)
    port map(
      ins => gate0_outs,
      ins_valid => gate0_outs_valid,
      ins_ready => gate0_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => trunci4_outs,
      addrIn_valid => trunci4_outs_valid,
      addrIn_ready => trunci4_outs_ready,
      dataIn => cond_br5_falseOut,
      dataIn_valid => cond_br5_falseOut_valid,
      dataIn_ready => cond_br5_falseOut_ready,
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

  sink3 : entity work.sink_dataless(arch)
    port map(
      ins_valid => store0_doneOut_valid,
      ins_ready => store0_doneOut_ready,
      clk => clk,
      rst => rst
    );

  addi3 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi24_outs,
      lhs_valid => extsi24_outs_valid,
      lhs_ready => extsi24_outs_ready,
      rhs => extsi27_outs,
      rhs_valid => extsi27_outs_valid,
      rhs_ready => extsi27_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer15 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  trunci5 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => extsi26_outs,
      rhs_valid => extsi26_outs_valid,
      rhs_ready => extsi26_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer16 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs(3) => fork13_outs_3,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_valid(3) => fork13_outs_3_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready,
      outs_ready(3) => fork13_outs_3_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork13_outs_0,
      condition_valid => fork13_outs_0_valid,
      condition_ready => fork13_outs_0_ready,
      data => trunci5_outs,
      data_valid => trunci5_outs_valid,
      data_ready => trunci5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  sink4 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br9 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_2,
      condition_valid => fork13_outs_2_valid,
      condition_ready => fork13_outs_2_ready,
      data_valid => fork11_outs_2_valid,
      data_ready => fork11_outs_2_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(1)
    port map(
      condition => fork13_outs_3,
      condition_valid => fork13_outs_3_valid,
      condition_ready => fork13_outs_3_ready,
      data => constant19_outs,
      data_valid => constant19_outs_valid,
      data_ready => constant19_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink5 : entity work.sink(arch) generic map(1)
    port map(
      ins => cond_br10_trueOut,
      ins_valid => cond_br10_trueOut_valid,
      ins_ready => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi15 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  init1 : entity work.tehb_init(arch) generic map(1, 0)
    port map(
      ins => fork26_outs_1,
      ins_valid => fork26_outs_1_valid,
      ins_ready => fork26_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => init1_outs,
      outs_valid => init1_outs_valid,
      outs_ready => init1_outs_ready
    );

  sink6 : entity work.sink(arch) generic map(1)
    port map(
      ins => init1_outs,
      ins_valid => init1_outs_valid,
      ins_ready => init1_outs_ready,
      clk => clk,
      rst => rst
    );

  unbundle2 : entity work.unbundle(arch) generic map(32)
    port map(
      ins => fork16_outs_0,
      ins_valid => fork16_outs_0_valid,
      ins_ready => fork16_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => unbundle2_outs_0_valid,
      outs_ready => unbundle2_outs_0_ready,
      outs => unbundle2_outs_1
    );

  mux4 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready,
      ins(0) => extsi15_outs,
      ins(1) => cond_br15_trueOut,
      ins_valid(0) => extsi15_outs_valid,
      ins_valid(1) => cond_br15_trueOut_valid,
      ins_ready(0) => extsi15_outs_ready,
      ins_ready(1) => cond_br15_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux4_outs,
      outs_valid => mux4_outs_valid,
      outs_ready => mux4_outs_ready
    );

  buffer17 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  fork14 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer17_outs,
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork14_outs_0,
      outs(1) => fork14_outs_1,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

  trunci6 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork14_outs_0,
      ins_valid => fork14_outs_0_valid,
      ins_ready => fork14_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  control_merge3 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => cond_br9_falseOut_valid,
      ins_valid(1) => cond_br16_trueOut_valid,
      ins_ready(0) => cond_br9_falseOut_ready,
      ins_ready(1) => cond_br16_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge3_outs_valid,
      outs_ready => control_merge3_outs_ready,
      index => control_merge3_index,
      index_valid => control_merge3_index_valid,
      index_ready => control_merge3_index_ready
    );

  fork15 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => control_merge3_outs_valid,
      ins_ready => control_merge3_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork15_outs_0_valid,
      outs_valid(1) => fork15_outs_1_valid,
      outs_ready(0) => fork15_outs_0_ready,
      outs_ready(1) => fork15_outs_1_ready
    );

  constant22 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork15_outs_0_valid,
      ctrl_ready => fork15_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  buffer1 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => unbundle2_outs_0_valid,
      ins_ready => unbundle2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci6_outs,
      addrIn_valid => trunci6_outs_valid,
      addrIn_ready => trunci6_outs_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  fork16 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load3_dataOut,
      ins_valid => load3_dataOut_valid,
      ins_ready => load3_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork16_outs_0,
      outs(1) => fork16_outs_1,
      outs_valid(0) => fork16_outs_0_valid,
      outs_valid(1) => fork16_outs_1_valid,
      outs_ready(0) => fork16_outs_0_ready,
      outs_ready(1) => fork16_outs_1_ready
    );

  extsi14 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  mux5 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork19_outs_1,
      index_valid => fork19_outs_1_valid,
      index_ready => fork19_outs_1_ready,
      ins(0) => extsi14_outs,
      ins(1) => cond_br11_trueOut,
      ins_valid(0) => extsi14_outs_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => extsi14_outs_ready,
      ins_ready(1) => cond_br11_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux5_outs,
      outs_valid => mux5_outs_valid,
      outs_ready => mux5_outs_ready
    );

  buffer18 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux5_outs,
      ins_valid => mux5_outs_valid,
      ins_ready => mux5_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  fork17 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer18_outs,
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
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

  extsi28 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork17_outs_1,
      ins_valid => fork17_outs_1_valid,
      ins_ready => fork17_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi28_outs,
      outs_valid => extsi28_outs_valid,
      outs_ready => extsi28_outs_ready
    );

  extsi29 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => fork17_outs_2,
      ins_valid => fork17_outs_2_valid,
      ins_ready => fork17_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi29_outs,
      outs_valid => extsi29_outs_valid,
      outs_ready => extsi29_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork17_outs_0,
      ins_valid => fork17_outs_0_valid,
      ins_ready => fork17_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  mux6 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer39_outs,
      index_valid => buffer39_outs_valid,
      index_ready => buffer39_outs_ready,
      ins(0) => fork16_outs_1,
      ins(1) => cond_br12_trueOut,
      ins_valid(0) => fork16_outs_1_valid,
      ins_valid(1) => cond_br12_trueOut_valid,
      ins_ready(0) => fork16_outs_1_ready,
      ins_ready(1) => cond_br12_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux6_outs,
      outs_valid => mux6_outs_valid,
      outs_ready => mux6_outs_ready
    );

  buffer39 : entity work.tfifo(arch) generic map(2, 1)
    port map(
      ins => fork19_outs_2,
      ins_valid => fork19_outs_2_valid,
      ins_ready => fork19_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer39_outs,
      outs_valid => buffer39_outs_valid,
      outs_ready => buffer39_outs_ready
    );

  mux7 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork19_outs_0,
      index_valid => fork19_outs_0_valid,
      index_ready => fork19_outs_0_ready,
      ins(0) => fork14_outs_1,
      ins(1) => cond_br13_trueOut,
      ins_valid(0) => fork14_outs_1_valid,
      ins_valid(1) => cond_br13_trueOut_valid,
      ins_ready(0) => fork14_outs_1_ready,
      ins_ready(1) => cond_br13_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux7_outs,
      outs_valid => mux7_outs_valid,
      outs_ready => mux7_outs_ready
    );

  buffer21 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux7_outs,
      ins_valid => mux7_outs_valid,
      ins_ready => mux7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  buffer22 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  fork18 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer22_outs,
      ins_valid => buffer22_outs_valid,
      ins_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork18_outs_0,
      outs(1) => fork18_outs_1,
      outs_valid(0) => fork18_outs_0_valid,
      outs_valid(1) => fork18_outs_1_valid,
      outs_ready(0) => fork18_outs_0_ready,
      outs_ready(1) => fork18_outs_1_ready
    );

  extsi30 : entity work.extsi(arch) generic map(6, 10)
    port map(
      ins => fork18_outs_0,
      ins_valid => fork18_outs_0_valid,
      ins_ready => fork18_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi30_outs,
      outs_valid => extsi30_outs_valid,
      outs_ready => extsi30_outs_ready
    );

  control_merge4 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork15_outs_1_valid,
      ins_valid(1) => cond_br14_trueOut_valid,
      ins_ready(0) => fork15_outs_1_ready,
      ins_ready(1) => cond_br14_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => control_merge4_outs_valid,
      outs_ready => control_merge4_outs_ready,
      index => control_merge4_index,
      index_valid => control_merge4_index_valid,
      index_ready => control_merge4_index_ready
    );

  fork19 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => control_merge4_index,
      ins_valid => control_merge4_index_valid,
      ins_ready => control_merge4_index_ready,
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

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant23 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant23_outs,
      outs_valid => constant23_outs_valid,
      outs_ready => constant23_outs_ready
    );

  fork20 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => constant23_outs,
      ins_valid => constant23_outs_valid,
      ins_ready => constant23_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork20_outs_0,
      outs(1) => fork20_outs_1,
      outs_valid(0) => fork20_outs_0_valid,
      outs_valid(1) => fork20_outs_1_valid,
      outs_ready(0) => fork20_outs_0_ready,
      outs_ready(1) => fork20_outs_1_ready
    );

  extsi31 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork20_outs_0,
      ins_valid => fork20_outs_0_valid,
      ins_ready => fork20_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi31_outs,
      outs_valid => extsi31_outs_valid,
      outs_ready => extsi31_outs_ready
    );

  extsi32 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => fork20_outs_1,
      ins_valid => fork20_outs_1_valid,
      ins_ready => fork20_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi32_outs,
      outs_valid => extsi32_outs_valid,
      outs_ready => extsi32_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant24 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant24_outs,
      outs_valid => constant24_outs_valid,
      outs_ready => constant24_outs_ready
    );

  extsi33 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant24_outs,
      ins_valid => constant24_outs_valid,
      ins_ready => constant24_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi33_outs,
      outs_valid => extsi33_outs_valid,
      outs_ready => extsi33_outs_ready
    );

  muli1 : entity work.muli(arch) generic map(12)
    port map(
      lhs => extsi29_outs,
      lhs_valid => extsi29_outs_valid,
      lhs_ready => extsi29_outs_ready,
      rhs => extsi32_outs,
      rhs_valid => extsi32_outs_valid,
      rhs_ready => extsi32_outs_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  trunci8 : entity work.trunci(arch) generic map(12, 10)
    port map(
      ins => muli1_result,
      ins_valid => muli1_result_valid,
      ins_ready => muli1_result_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(10)
    port map(
      lhs => extsi30_outs,
      lhs_valid => extsi30_outs_valid,
      lhs_ready => extsi30_outs_ready,
      rhs => trunci8_outs,
      rhs_valid => trunci8_outs_valid,
      rhs_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  load4 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => addi1_result,
      addrIn_valid => addi1_result_valid,
      addrIn_ready => addi1_result_ready,
      dataFromMem => mem_controller7_ldData_1,
      dataFromMem_valid => mem_controller7_ldData_1_valid,
      dataFromMem_ready => mem_controller7_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  load5 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci7_outs,
      addrIn_valid => trunci7_outs_valid,
      addrIn_ready => trunci7_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load5_addrOut,
      addrOut_valid => load5_addrOut_valid,
      addrOut_ready => load5_addrOut_ready,
      dataOut => load5_dataOut,
      dataOut_valid => load5_dataOut_valid,
      dataOut_ready => load5_dataOut_ready
    );

  mulf1 : entity work.mulf(arch_32_2_875333) generic map(32)
    port map(
      lhs => load4_dataOut,
      lhs_valid => load4_dataOut_valid,
      lhs_ready => load4_dataOut_ready,
      rhs => load5_dataOut,
      rhs_valid => load5_dataOut_valid,
      rhs_ready => load5_dataOut_ready,
      clk => clk,
      rst => rst,
      result => mulf1_result,
      result_valid => mulf1_result_valid,
      result_ready => mulf1_result_ready
    );

  buffer20 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux6_outs,
      ins_valid => mux6_outs_valid,
      ins_ready => mux6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  addf1 : entity work.addf(arch_32_2_922000) generic map(32)
    port map(
      lhs => buffer20_outs,
      lhs_valid => buffer20_outs_valid,
      lhs_ready => buffer20_outs_ready,
      rhs => mulf1_result,
      rhs_valid => mulf1_result_valid,
      rhs_ready => mulf1_result_ready,
      clk => clk,
      rst => rst,
      result => addf1_result,
      result_valid => addf1_result_valid,
      result_ready => addf1_result_ready
    );

  addi4 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi28_outs,
      lhs_valid => extsi28_outs_valid,
      lhs_ready => extsi28_outs_ready,
      rhs => extsi33_outs,
      rhs_valid => extsi33_outs_valid,
      rhs_ready => extsi33_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer25 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  fork21 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork21_outs_0,
      outs(1) => fork21_outs_1,
      outs_valid(0) => fork21_outs_0_valid,
      outs_valid(1) => fork21_outs_1_valid,
      outs_ready(0) => fork21_outs_0_ready,
      outs_ready(1) => fork21_outs_1_ready
    );

  trunci9 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork21_outs_0,
      ins_valid => fork21_outs_0_valid,
      ins_ready => fork21_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork21_outs_1,
      lhs_valid => fork21_outs_1_valid,
      lhs_ready => fork21_outs_1_ready,
      rhs => extsi31_outs,
      rhs_valid => extsi31_outs_valid,
      rhs_ready => extsi31_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi2_result,
      result_valid => cmpi2_result_valid,
      result_ready => cmpi2_result_ready
    );

  buffer26 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi2_result,
      ins_valid => cmpi2_result_valid,
      ins_ready => cmpi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  fork22 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
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

  cond_br11 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork22_outs_0,
      condition_valid => fork22_outs_0_valid,
      condition_ready => fork22_outs_0_ready,
      data => trunci9_outs,
      data_valid => trunci9_outs_valid,
      data_ready => trunci9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br11_trueOut,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut => cond_br11_falseOut,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  sink7 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br11_falseOut,
      ins_valid => cond_br11_falseOut_valid,
      ins_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br12 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer47_outs,
      condition_valid => buffer47_outs_valid,
      condition_ready => buffer47_outs_ready,
      data => addf1_result,
      data_valid => addf1_result_valid,
      data_ready => addf1_result_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br12_trueOut,
      trueOut_valid => cond_br12_trueOut_valid,
      trueOut_ready => cond_br12_trueOut_ready,
      falseOut => cond_br12_falseOut,
      falseOut_valid => cond_br12_falseOut_valid,
      falseOut_ready => cond_br12_falseOut_ready
    );

  buffer47 : entity work.tfifo(arch) generic map(3, 1)
    port map(
      ins => fork22_outs_2,
      ins_valid => fork22_outs_2_valid,
      ins_ready => fork22_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer47_outs,
      outs_valid => buffer47_outs_valid,
      outs_ready => buffer47_outs_ready
    );

  cond_br13 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork22_outs_1,
      condition_valid => fork22_outs_1_valid,
      condition_ready => fork22_outs_1_ready,
      data => fork18_outs_1,
      data_valid => fork18_outs_1_valid,
      data_ready => fork18_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br13_trueOut,
      trueOut_valid => cond_br13_trueOut_valid,
      trueOut_ready => cond_br13_trueOut_ready,
      falseOut => cond_br13_falseOut,
      falseOut_valid => cond_br13_falseOut_valid,
      falseOut_ready => cond_br13_falseOut_ready
    );

  buffer23 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => control_merge4_outs_valid,
      ins_ready => control_merge4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  cond_br14 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork22_outs_3,
      condition_valid => fork22_outs_3_valid,
      condition_ready => fork22_outs_3_ready,
      data_valid => buffer23_outs_valid,
      data_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br14_trueOut_valid,
      trueOut_ready => cond_br14_trueOut_ready,
      falseOut_valid => cond_br14_falseOut_valid,
      falseOut_ready => cond_br14_falseOut_ready
    );

  fork23 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => cond_br13_falseOut,
      ins_valid => cond_br13_falseOut_valid,
      ins_ready => cond_br13_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork23_outs_0,
      outs(1) => fork23_outs_1,
      outs_valid(0) => fork23_outs_0_valid,
      outs_valid(1) => fork23_outs_1_valid,
      outs_ready(0) => fork23_outs_0_ready,
      outs_ready(1) => fork23_outs_1_ready
    );

  extsi34 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork23_outs_0,
      ins_valid => fork23_outs_0_valid,
      ins_ready => fork23_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi34_outs,
      outs_valid => extsi34_outs_valid,
      outs_ready => extsi34_outs_ready
    );

  extsi35 : entity work.extsi(arch) generic map(6, 32)
    port map(
      ins => buffer52_outs,
      ins_valid => buffer52_outs_valid,
      ins_ready => buffer52_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi35_outs,
      outs_valid => extsi35_outs_valid,
      outs_ready => extsi35_outs_ready
    );

  buffer52 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork23_outs_1,
      ins_valid => fork23_outs_1_valid,
      ins_ready => fork23_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer52_outs,
      outs_valid => buffer52_outs_valid,
      outs_ready => buffer52_outs_ready
    );

  fork24 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br14_falseOut_valid,
      ins_ready => cond_br14_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork24_outs_0_valid,
      outs_valid(1) => fork24_outs_1_valid,
      outs_ready(0) => fork24_outs_0_ready,
      outs_ready(1) => fork24_outs_1_ready
    );

  constant25 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork24_outs_0_valid,
      ctrl_ready => fork24_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant25_outs,
      outs_valid => constant25_outs_valid,
      outs_ready => constant25_outs_ready
    );

  extsi11 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant25_outs,
      ins_valid => constant25_outs_valid,
      ins_ready => constant25_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant26 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant26_outs,
      outs_valid => constant26_outs_valid,
      outs_ready => constant26_outs_ready
    );

  extsi36 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant26_outs,
      ins_valid => constant26_outs_valid,
      ins_ready => constant26_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi36_outs,
      outs_valid => extsi36_outs_valid,
      outs_ready => extsi36_outs_ready
    );

  source7 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source7_outs_valid,
      outs_ready => source7_outs_ready
    );

  constant27 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source7_outs_valid,
      ctrl_ready => source7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant27_outs,
      outs_valid => constant27_outs_valid,
      outs_ready => constant27_outs_ready
    );

  extsi37 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant27_outs,
      ins_valid => constant27_outs_valid,
      ins_ready => constant27_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi37_outs,
      outs_valid => extsi37_outs_valid,
      outs_ready => extsi37_outs_ready
    );

  gate1 : entity work.gate(arch) generic map(2, 32)
    port map(
      ins(0) => extsi35_outs,
      ins_valid(0) => extsi35_outs_valid,
      ins_valid(1) => buffer1_outs_valid,
      ins_ready(0) => extsi35_outs_ready,
      ins_ready(1) => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs => gate1_outs,
      outs_valid => gate1_outs_valid,
      outs_ready => gate1_outs_ready
    );

  trunci10 : entity work.trunci(arch) generic map(32, 5)
    port map(
      ins => gate1_outs,
      ins_valid => gate1_outs_valid,
      ins_ready => gate1_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci10_outs,
      outs_valid => trunci10_outs_valid,
      outs_ready => trunci10_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => trunci10_outs,
      addrIn_valid => trunci10_outs_valid,
      addrIn_ready => trunci10_outs_ready,
      dataIn => cond_br12_falseOut,
      dataIn_valid => cond_br12_falseOut_valid,
      dataIn_ready => cond_br12_falseOut_ready,
      doneFromMem_valid => mem_controller5_stDone_0_valid,
      doneFromMem_ready => mem_controller5_stDone_0_ready,
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

  sink9 : entity work.sink_dataless(arch)
    port map(
      ins_valid => store1_doneOut_valid,
      ins_ready => store1_doneOut_ready,
      clk => clk,
      rst => rst
    );

  addi5 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi34_outs,
      lhs_valid => extsi34_outs_valid,
      lhs_ready => extsi34_outs_ready,
      rhs => extsi37_outs,
      rhs_valid => extsi37_outs_valid,
      rhs_ready => extsi37_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer27 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork25 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork25_outs_0,
      outs(1) => fork25_outs_1,
      outs_valid(0) => fork25_outs_0_valid,
      outs_valid(1) => fork25_outs_1_valid,
      outs_ready(0) => fork25_outs_0_ready,
      outs_ready(1) => fork25_outs_1_ready
    );

  trunci11 : entity work.trunci(arch) generic map(7, 6)
    port map(
      ins => fork25_outs_0,
      ins_valid => fork25_outs_0_valid,
      ins_ready => fork25_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci11_outs,
      outs_valid => trunci11_outs_valid,
      outs_ready => trunci11_outs_ready
    );

  cmpi3 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork25_outs_1,
      lhs_valid => fork25_outs_1_valid,
      lhs_ready => fork25_outs_1_ready,
      rhs => extsi36_outs,
      rhs_valid => extsi36_outs_valid,
      rhs_ready => extsi36_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi3_result,
      result_valid => cmpi3_result_valid,
      result_ready => cmpi3_result_ready
    );

  buffer28 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi3_result,
      ins_valid => cmpi3_result_valid,
      ins_ready => cmpi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  fork26 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer28_outs,
      ins_valid => buffer28_outs_valid,
      ins_ready => buffer28_outs_ready,
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

  cond_br15 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork26_outs_0,
      condition_valid => fork26_outs_0_valid,
      condition_ready => fork26_outs_0_ready,
      data => trunci11_outs,
      data_valid => trunci11_outs_valid,
      data_ready => trunci11_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br15_trueOut,
      trueOut_valid => cond_br15_trueOut_valid,
      trueOut_ready => cond_br15_trueOut_ready,
      falseOut => cond_br15_falseOut,
      falseOut_valid => cond_br15_falseOut_valid,
      falseOut_ready => cond_br15_falseOut_ready
    );

  sink10 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br15_falseOut,
      ins_valid => cond_br15_falseOut_valid,
      ins_ready => cond_br15_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br16 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork26_outs_2,
      condition_valid => fork26_outs_2_valid,
      condition_ready => fork26_outs_2_ready,
      data_valid => fork24_outs_1_valid,
      data_ready => fork24_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br16_trueOut_valid,
      trueOut_ready => cond_br16_trueOut_ready,
      falseOut_valid => cond_br16_falseOut_valid,
      falseOut_ready => cond_br16_falseOut_ready
    );

  fork27 : entity work.fork_dataless(arch) generic map(5)
    port map(
      ins_valid => cond_br16_falseOut_valid,
      ins_ready => cond_br16_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork27_outs_0_valid,
      outs_valid(1) => fork27_outs_1_valid,
      outs_valid(2) => fork27_outs_2_valid,
      outs_valid(3) => fork27_outs_3_valid,
      outs_valid(4) => fork27_outs_4_valid,
      outs_ready(0) => fork27_outs_0_ready,
      outs_ready(1) => fork27_outs_1_ready,
      outs_ready(2) => fork27_outs_2_ready,
      outs_ready(3) => fork27_outs_3_ready,
      outs_ready(4) => fork27_outs_4_ready
    );

end architecture;
