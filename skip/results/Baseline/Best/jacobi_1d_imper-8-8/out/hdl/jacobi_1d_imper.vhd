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
  signal lsq2_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq2_ldData_0_valid : std_logic;
  signal lsq2_ldData_0_ready : std_logic;
  signal lsq2_memEnd_valid : std_logic;
  signal lsq2_memEnd_ready : std_logic;
  signal lsq2_loadEn : std_logic;
  signal lsq2_loadAddr : std_logic_vector(6 downto 0);
  signal lsq2_storeEn : std_logic;
  signal lsq2_storeAddr : std_logic_vector(6 downto 0);
  signal lsq2_storeData : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0_valid : std_logic;
  signal lsq3_ldData_0_ready : std_logic;
  signal lsq3_ldData_1 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_1_valid : std_logic;
  signal lsq3_ldData_1_ready : std_logic;
  signal lsq3_ldData_2 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_2_valid : std_logic;
  signal lsq3_ldData_2_ready : std_logic;
  signal lsq3_memEnd_valid : std_logic;
  signal lsq3_memEnd_ready : std_logic;
  signal lsq3_loadEn : std_logic;
  signal lsq3_loadAddr : std_logic_vector(6 downto 0);
  signal lsq3_storeEn : std_logic;
  signal lsq3_storeAddr : std_logic_vector(6 downto 0);
  signal lsq3_storeData : std_logic_vector(31 downto 0);
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi11_outs : std_logic_vector(2 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
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
  signal constant1_outs : std_logic_vector(1 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal extsi10_outs : std_logic_vector(7 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(2 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(7 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(7 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(7 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(7 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2 : std_logic_vector(7 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_3 : std_logic_vector(7 downto 0);
  signal fork2_outs_3_valid : std_logic;
  signal fork2_outs_3_ready : std_logic;
  signal fork2_outs_4 : std_logic_vector(7 downto 0);
  signal fork2_outs_4_valid : std_logic;
  signal fork2_outs_4_ready : std_logic;
  signal trunci0_outs : std_logic_vector(6 downto 0);
  signal trunci0_outs_valid : std_logic;
  signal trunci0_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(6 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal extsi12_outs : std_logic_vector(8 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(6 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(6 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(2 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal control_merge1_outs_valid : std_logic;
  signal control_merge1_outs_ready : std_logic;
  signal control_merge1_index : std_logic_vector(0 downto 0);
  signal control_merge1_index_valid : std_logic;
  signal control_merge1_index_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal lazy_fork0_outs_2_valid : std_logic;
  signal lazy_fork0_outs_2_ready : std_logic;
  signal lazy_fork0_outs_3_valid : std_logic;
  signal lazy_fork0_outs_3_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant7_outs : std_logic_vector(31 downto 0);
  signal constant7_outs_valid : std_logic;
  signal constant7_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(6 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant2_outs : std_logic_vector(7 downto 0);
  signal constant2_outs_valid : std_logic;
  signal constant2_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(8 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal constant4_outs : std_logic_vector(1 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal fork4_outs_0 : std_logic_vector(1 downto 0);
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1 : std_logic_vector(1 downto 0);
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal fork4_outs_2 : std_logic_vector(1 downto 0);
  signal fork4_outs_2_valid : std_logic;
  signal fork4_outs_2_ready : std_logic;
  signal extsi14_outs : std_logic_vector(6 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal extsi15_outs : std_logic_vector(8 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant5_outs : std_logic_vector(1 downto 0);
  signal constant5_outs_valid : std_logic;
  signal constant5_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(6 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal buffer8_outs : std_logic_vector(6 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(6 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal buffer3_outs : std_logic_vector(6 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal load1_addrOut : std_logic_vector(6 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal addi3_result : std_logic_vector(6 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer10_outs : std_logic_vector(6 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(6 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal buffer9_outs : std_logic_vector(31 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal buffer11_outs : std_logic_vector(31 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(31 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(31 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal shli0_result : std_logic_vector(31 downto 0);
  signal shli0_result_valid : std_logic;
  signal shli0_result_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal addi7_result : std_logic_vector(31 downto 0);
  signal addi7_result_valid : std_logic;
  signal addi7_result_ready : std_logic;
  signal buffer2_outs : std_logic_vector(6 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(6 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal addi4_result : std_logic_vector(8 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer14_outs : std_logic_vector(8 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(8 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(8 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal trunci5_outs : std_logic_vector(7 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer15_outs : std_logic_vector(0 downto 0);
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
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
  signal cond_br3_trueOut : std_logic_vector(7 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(7 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer4_outs : std_logic_vector(2 downto 0);
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs : std_logic_vector(2 downto 0);
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(2 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(2 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
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
  signal extsi9_outs : std_logic_vector(7 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(7 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer16_outs : std_logic_vector(7 downto 0);
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(7 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(7 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2 : std_logic_vector(7 downto 0);
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal extsi16_outs : std_logic_vector(8 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal trunci6_outs : std_logic_vector(6 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(6 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal mux4_outs : std_logic_vector(2 downto 0);
  signal mux4_outs_valid : std_logic;
  signal mux4_outs_ready : std_logic;
  signal control_merge2_outs_valid : std_logic;
  signal control_merge2_outs_ready : std_logic;
  signal control_merge2_index : std_logic_vector(0 downto 0);
  signal control_merge2_index_valid : std_logic;
  signal control_merge2_index_ready : std_logic;
  signal fork9_outs_0 : std_logic_vector(0 downto 0);
  signal fork9_outs_0_valid : std_logic;
  signal fork9_outs_0_ready : std_logic;
  signal fork9_outs_1 : std_logic_vector(0 downto 0);
  signal fork9_outs_1_valid : std_logic;
  signal fork9_outs_1_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal lazy_fork1_outs_2_valid : std_logic;
  signal lazy_fork1_outs_2_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant15_outs : std_logic_vector(7 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal extsi17_outs : std_logic_vector(8 downto 0);
  signal extsi17_outs_valid : std_logic;
  signal extsi17_outs_ready : std_logic;
  signal source4_outs_valid : std_logic;
  signal source4_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(1 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal extsi18_outs : std_logic_vector(8 downto 0);
  signal extsi18_outs_valid : std_logic;
  signal extsi18_outs_ready : std_logic;
  signal buffer18_outs : std_logic_vector(6 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal load3_addrOut : std_logic_vector(6 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal buffer17_outs : std_logic_vector(6 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal buffer22_outs : std_logic_vector(31 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(6 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal addi5_result : std_logic_vector(8 downto 0);
  signal addi5_result_valid : std_logic;
  signal addi5_result_ready : std_logic;
  signal buffer23_outs : std_logic_vector(8 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(8 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(8 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal trunci8_outs : std_logic_vector(7 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer24_outs : std_logic_vector(0 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(0 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(0 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal fork11_outs_2 : std_logic_vector(0 downto 0);
  signal fork11_outs_2_valid : std_logic;
  signal fork11_outs_2_ready : std_logic;
  signal cond_br7_trueOut : std_logic_vector(7 downto 0);
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut : std_logic_vector(7 downto 0);
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer19_outs : std_logic_vector(2 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal buffer20_outs : std_logic_vector(2 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(2 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(2 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal cond_br9_trueOut_valid : std_logic;
  signal cond_br9_trueOut_ready : std_logic;
  signal cond_br9_falseOut_valid : std_logic;
  signal cond_br9_falseOut_ready : std_logic;
  signal extsi19_outs : std_logic_vector(3 downto 0);
  signal extsi19_outs_valid : std_logic;
  signal extsi19_outs_ready : std_logic;
  signal source5_outs_valid : std_logic;
  signal source5_outs_ready : std_logic;
  signal constant17_outs : std_logic_vector(2 downto 0);
  signal constant17_outs_valid : std_logic;
  signal constant17_outs_ready : std_logic;
  signal extsi20_outs : std_logic_vector(3 downto 0);
  signal extsi20_outs_valid : std_logic;
  signal extsi20_outs_ready : std_logic;
  signal source6_outs_valid : std_logic;
  signal source6_outs_ready : std_logic;
  signal constant18_outs : std_logic_vector(1 downto 0);
  signal constant18_outs_valid : std_logic;
  signal constant18_outs_ready : std_logic;
  signal extsi21_outs : std_logic_vector(3 downto 0);
  signal extsi21_outs_valid : std_logic;
  signal extsi21_outs_ready : std_logic;
  signal addi6_result : std_logic_vector(3 downto 0);
  signal addi6_result_valid : std_logic;
  signal addi6_result_ready : std_logic;
  signal buffer25_outs : std_logic_vector(3 downto 0);
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(3 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(3 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal trunci9_outs : std_logic_vector(2 downto 0);
  signal trunci9_outs_valid : std_logic;
  signal trunci9_outs_ready : std_logic;
  signal cmpi2_result : std_logic_vector(0 downto 0);
  signal cmpi2_result_valid : std_logic;
  signal cmpi2_result_ready : std_logic;
  signal buffer26_outs : std_logic_vector(0 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
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
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;

begin

  A_end_valid <= lsq3_memEnd_valid;
  lsq3_memEnd_ready <= A_end_ready;
  B_end_valid <= lsq2_memEnd_valid;
  lsq2_memEnd_ready <= B_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  A_loadEn <= lsq3_loadEn;
  A_loadAddr <= lsq3_loadAddr;
  A_storeEn <= lsq3_storeEn;
  A_storeAddr <= lsq3_storeAddr;
  A_storeData <= lsq3_storeData;
  B_loadEn <= lsq2_loadEn;
  B_loadAddr <= lsq2_loadAddr;
  B_storeEn <= lsq2_storeEn;
  B_storeAddr <= lsq2_storeAddr;
  B_storeData <= lsq2_storeData;

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

  lsq2 : entity work.handshake_lsq_lsq2(arch)
    port map(
      io_loadData => B_loadData,
      io_memStart_valid => B_start_valid,
      io_memStart_ready => B_start_ready,
      io_ctrl_0_valid => lazy_fork0_outs_0_valid,
      io_ctrl_0_ready => lazy_fork0_outs_0_ready,
      io_stAddr_0_bits => store0_addrOut,
      io_stAddr_0_valid => store0_addrOut_valid,
      io_stAddr_0_ready => store0_addrOut_ready,
      io_stData_0_bits => store0_dataToMem,
      io_stData_0_valid => store0_dataToMem_valid,
      io_stData_0_ready => store0_dataToMem_ready,
      io_ctrl_1_valid => lazy_fork1_outs_0_valid,
      io_ctrl_1_ready => lazy_fork1_outs_0_ready,
      io_ldAddr_0_bits => load3_addrOut,
      io_ldAddr_0_valid => load3_addrOut_valid,
      io_ldAddr_0_ready => load3_addrOut_ready,
      io_ctrlEnd_valid => fork14_outs_1_valid,
      io_ctrlEnd_ready => fork14_outs_1_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq2_ldData_0,
      io_ldData_0_valid => lsq2_ldData_0_valid,
      io_ldData_0_ready => lsq2_ldData_0_ready,
      io_memEnd_valid => lsq2_memEnd_valid,
      io_memEnd_ready => lsq2_memEnd_ready,
      io_loadEn => lsq2_loadEn,
      io_loadAddr => lsq2_loadAddr,
      io_storeEn => lsq2_storeEn,
      io_storeAddr => lsq2_storeAddr,
      io_storeData => lsq2_storeData
    );

  lsq3 : entity work.handshake_lsq_lsq3(arch)
    port map(
      io_loadData => A_loadData,
      io_memStart_valid => A_start_valid,
      io_memStart_ready => A_start_ready,
      io_ctrl_0_valid => lazy_fork0_outs_2_valid,
      io_ctrl_0_ready => lazy_fork0_outs_2_ready,
      io_ldAddr_0_bits => load0_addrOut,
      io_ldAddr_0_valid => load0_addrOut_valid,
      io_ldAddr_0_ready => load0_addrOut_ready,
      io_ldAddr_1_bits => load1_addrOut,
      io_ldAddr_1_valid => load1_addrOut_valid,
      io_ldAddr_1_ready => load1_addrOut_ready,
      io_ldAddr_2_bits => load2_addrOut,
      io_ldAddr_2_valid => load2_addrOut_valid,
      io_ldAddr_2_ready => load2_addrOut_ready,
      io_ctrl_1_valid => lazy_fork1_outs_2_valid,
      io_ctrl_1_ready => lazy_fork1_outs_2_ready,
      io_stAddr_0_bits => store1_addrOut,
      io_stAddr_0_valid => store1_addrOut_valid,
      io_stAddr_0_ready => store1_addrOut_ready,
      io_stData_0_bits => store1_dataToMem,
      io_stData_0_valid => store1_dataToMem_valid,
      io_stData_0_ready => store1_dataToMem_ready,
      io_ctrlEnd_valid => fork14_outs_0_valid,
      io_ctrlEnd_ready => fork14_outs_0_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq3_ldData_0,
      io_ldData_0_valid => lsq3_ldData_0_valid,
      io_ldData_0_ready => lsq3_ldData_0_ready,
      io_ldData_1_bits => lsq3_ldData_1,
      io_ldData_1_valid => lsq3_ldData_1_valid,
      io_ldData_1_ready => lsq3_ldData_1_ready,
      io_ldData_2_bits => lsq3_ldData_2,
      io_ldData_2_valid => lsq3_ldData_2_valid,
      io_ldData_2_ready => lsq3_ldData_2_ready,
      io_memEnd_valid => lsq3_memEnd_valid,
      io_memEnd_ready => lsq3_memEnd_ready,
      io_loadEn => lsq3_loadEn,
      io_loadAddr => lsq3_loadAddr,
      io_storeEn => lsq3_storeEn,
      io_storeAddr => lsq3_storeAddr,
      io_storeData => lsq3_storeData
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

  extsi11 : entity work.extsi(arch) generic map(1, 3)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi11_outs,
      ins(1) => cond_br10_trueOut,
      ins_valid(0) => extsi11_outs_valid,
      ins_valid(1) => cond_br10_trueOut_valid,
      ins_ready(0) => extsi11_outs_ready,
      ins_ready(1) => cond_br10_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  control_merge0 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => fork0_outs_2_valid,
      ins_valid(1) => cond_br11_trueOut_valid,
      ins_ready(0) => fork0_outs_2_ready,
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

  constant1 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => fork1_outs_0_valid,
      ctrl_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  extsi10 : entity work.extsi(arch) generic map(2, 8)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  buffer0 : entity work.tehb(arch) generic map(3)
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

  mux1 : entity work.mux(arch) generic map(2, 8, 1)
    port map(
      index => fork3_outs_1,
      index_valid => fork3_outs_1_valid,
      index_ready => fork3_outs_1_ready,
      ins(0) => extsi10_outs,
      ins(1) => cond_br3_trueOut,
      ins_valid(0) => extsi10_outs_valid,
      ins_valid(1) => cond_br3_trueOut_valid,
      ins_ready(0) => extsi10_outs_ready,
      ins_ready(1) => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer1 : entity work.tehb(arch) generic map(8)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(5, 8)
    port map(
      ins => buffer1_outs,
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs(2) => fork2_outs_2,
      outs(3) => fork2_outs_3,
      outs(4) => fork2_outs_4,
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

  trunci0 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork2_outs_0,
      ins_valid => fork2_outs_0_valid,
      ins_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci0_outs,
      outs_valid => trunci0_outs_valid,
      outs_ready => trunci0_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork2_outs_1,
      ins_valid => fork2_outs_1_valid,
      ins_ready => fork2_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci1_outs,
      outs_valid => trunci1_outs_valid,
      outs_ready => trunci1_outs_ready
    );

  extsi12 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => fork2_outs_4,
      ins_valid => fork2_outs_4_valid,
      ins_ready => fork2_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  trunci2 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork2_outs_2,
      ins_valid => fork2_outs_2_valid,
      ins_ready => fork2_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => trunci2_outs,
      outs_valid => trunci2_outs_valid,
      outs_ready => trunci2_outs_ready
    );

  trunci3 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork2_outs_3,
      ins_valid => fork2_outs_3_valid,
      ins_ready => fork2_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => trunci3_outs,
      outs_valid => trunci3_outs_valid,
      outs_ready => trunci3_outs_ready
    );

  mux2 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => fork3_outs_0,
      index_valid => fork3_outs_0_valid,
      index_ready => fork3_outs_0_ready,
      ins(0) => buffer0_outs,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => buffer0_outs_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => buffer0_outs_ready,
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

  fork3 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge1_index,
      ins_valid => control_merge1_index_valid,
      ins_ready => control_merge1_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(4)
    port map(
      ins_valid => control_merge1_outs_valid,
      ins_ready => control_merge1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_valid(2) => lazy_fork0_outs_2_valid,
      outs_valid(3) => lazy_fork0_outs_3_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready,
      outs_ready(2) => lazy_fork0_outs_2_ready,
      outs_ready(3) => lazy_fork0_outs_3_ready
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

  trunci4 : entity work.trunci(arch) generic map(32, 7)
    port map(
      ins => constant7_outs,
      ins_valid => constant7_outs_valid,
      ins_ready => constant7_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant2 : entity work.handshake_constant_3(arch) generic map(8)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant2_outs,
      outs_valid => constant2_outs_valid,
      outs_ready => constant2_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(8, 9)
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

  buffer7 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => lazy_fork0_outs_3_valid,
      ins_ready => lazy_fork0_outs_3_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  constant4 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => buffer7_outs_valid,
      ctrl_ready => buffer7_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  fork4 : entity work.handshake_fork(arch) generic map(3, 2)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
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

  extsi14 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => fork4_outs_0,
      ins_valid => fork4_outs_0_valid,
      ins_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  extsi15 : entity work.extsi(arch) generic map(2, 9)
    port map(
      ins => fork4_outs_1,
      ins_valid => fork4_outs_1_valid,
      ins_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant5 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant5_outs,
      outs_valid => constant5_outs_valid,
      outs_ready => constant5_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(2, 32)
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

  addi2 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci0_outs,
      lhs_valid => trunci0_outs_valid,
      lhs_ready => trunci0_outs_ready,
      rhs => trunci4_outs,
      rhs_valid => trunci4_outs_valid,
      rhs_ready => trunci4_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  buffer8 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi2_result,
      ins_valid => addi2_result_valid,
      ins_ready => addi2_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer8_outs,
      addrIn_valid => buffer8_outs_valid,
      addrIn_ready => buffer8_outs_ready,
      dataFromMem => lsq3_ldData_0,
      dataFromMem_valid => lsq3_ldData_0_valid,
      dataFromMem_ready => lsq3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  buffer3 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => trunci3_outs,
      ins_valid => trunci3_outs_valid,
      ins_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  load1 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer3_outs,
      addrIn_valid => buffer3_outs_valid,
      addrIn_ready => buffer3_outs_ready,
      dataFromMem => lsq3_ldData_1,
      dataFromMem_valid => lsq3_ldData_1_valid,
      dataFromMem_ready => lsq3_ldData_1_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load0_dataOut,
      lhs_valid => load0_dataOut_valid,
      lhs_ready => load0_dataOut_ready,
      rhs => load1_dataOut,
      rhs_valid => load1_dataOut_valid,
      rhs_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  addi3 : entity work.addi(arch) generic map(7)
    port map(
      lhs => trunci1_outs,
      lhs_valid => trunci1_outs_valid,
      lhs_ready => trunci1_outs_ready,
      rhs => extsi14_outs,
      rhs_valid => extsi14_outs_valid,
      rhs_ready => extsi14_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer10 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer10_outs,
      addrIn_valid => buffer10_outs_valid,
      addrIn_ready => buffer10_outs_ready,
      dataFromMem => lsq3_ldData_2,
      dataFromMem_valid => lsq3_ldData_2_valid,
      dataFromMem_ready => lsq3_ldData_2_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  buffer9 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer9_outs,
      lhs_valid => buffer9_outs_valid,
      lhs_ready => buffer9_outs_ready,
      rhs => load2_dataOut,
      rhs_valid => load2_dataOut_valid,
      rhs_ready => load2_dataOut_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  buffer11 : entity work.oehb(arch) generic map(32)
    port map(
      ins => addi1_result,
      ins_valid => addi1_result_valid,
      ins_ready => addi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  shli0 : entity work.shli(arch) generic map(32)
    port map(
      lhs => fork5_outs_1,
      lhs_valid => fork5_outs_1_valid,
      lhs_ready => fork5_outs_1_ready,
      rhs => extsi4_outs,
      rhs_valid => extsi4_outs_valid,
      rhs_ready => extsi4_outs_ready,
      clk => clk,
      rst => rst,
      result => shli0_result,
      result_valid => shli0_result_valid,
      result_ready => shli0_result_ready
    );

  buffer12 : entity work.oehb(arch) generic map(32)
    port map(
      ins => shli0_result,
      ins_valid => shli0_result_valid,
      ins_ready => shli0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  addi7 : entity work.addi(arch) generic map(32)
    port map(
      lhs => fork5_outs_0,
      lhs_valid => fork5_outs_0_valid,
      lhs_ready => fork5_outs_0_ready,
      rhs => buffer12_outs,
      rhs_valid => buffer12_outs_valid,
      rhs_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      result => addi7_result,
      result_valid => addi7_result_valid,
      result_ready => addi7_result_ready
    );

  buffer2 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => trunci2_outs,
      ins_valid => trunci2_outs_valid,
      ins_ready => trunci2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  buffer13 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => addi7_result,
      ins_valid => addi7_result_valid,
      ins_ready => addi7_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer2_outs,
      addrIn_valid => buffer2_outs_valid,
      addrIn_ready => buffer2_outs_ready,
      dataIn => buffer13_outs,
      dataIn_valid => buffer13_outs_valid,
      dataIn_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  addi4 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi12_outs,
      lhs_valid => extsi12_outs_valid,
      lhs_ready => extsi12_outs_ready,
      rhs => extsi15_outs,
      rhs_valid => extsi15_outs_valid,
      rhs_ready => extsi15_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer14 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 9)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  trunci5 : entity work.trunci(arch) generic map(9, 8)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(9)
    port map(
      lhs => fork6_outs_1,
      lhs_valid => fork6_outs_1_valid,
      lhs_ready => fork6_outs_1_ready,
      rhs => extsi13_outs,
      rhs_valid => extsi13_outs_valid,
      rhs_ready => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer15 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer15_outs,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer15_outs,
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
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

  cond_br3 : entity work.cond_br(arch) generic map(8)
    port map(
      condition => fork7_outs_0,
      condition_valid => fork7_outs_0_valid,
      condition_ready => fork7_outs_0_ready,
      data => trunci5_outs,
      data_valid => trunci5_outs_valid,
      data_ready => trunci5_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(8)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer4 : entity work.oehb(arch) generic map(3)
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

  buffer5 : entity work.tehb(arch) generic map(3)
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

  cond_br4 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork7_outs_1,
      condition_valid => fork7_outs_1_valid,
      condition_ready => fork7_outs_1_ready,
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

  buffer6 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_1_valid,
      ins_ready => lazy_fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  cond_br5 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork7_outs_2,
      condition_valid => fork7_outs_2_valid,
      condition_ready => fork7_outs_2_ready,
      data_valid => buffer6_outs_valid,
      data_ready => buffer6_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(2)
    port map(
      condition => fork7_outs_3,
      condition_valid => fork7_outs_3_valid,
      condition_ready => fork7_outs_3_ready,
      data => fork4_outs_2,
      data_valid => fork4_outs_2_valid,
      data_ready => fork4_outs_2_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  sink1 : entity work.sink(arch) generic map(2)
    port map(
      ins => cond_br6_trueOut,
      ins_valid => cond_br6_trueOut_valid,
      ins_ready => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst
    );

  extsi9 : entity work.extsi(arch) generic map(2, 8)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 8, 1)
    port map(
      index => fork9_outs_1,
      index_valid => fork9_outs_1_valid,
      index_ready => fork9_outs_1_ready,
      ins(0) => extsi9_outs,
      ins(1) => cond_br7_trueOut,
      ins_valid(0) => extsi9_outs_valid,
      ins_valid(1) => cond_br7_trueOut_valid,
      ins_ready(0) => extsi9_outs_ready,
      ins_ready(1) => cond_br7_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer16 : entity work.tehb(arch) generic map(8)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer16_outs,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(3, 8)
    port map(
      ins => buffer16_outs,
      ins_valid => buffer16_outs_valid,
      ins_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs(2) => fork8_outs_2,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready
    );

  extsi16 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => fork8_outs_2,
      ins_valid => fork8_outs_2_valid,
      ins_ready => fork8_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  trunci6 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork8_outs_0,
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci6_outs,
      outs_valid => trunci6_outs_valid,
      outs_ready => trunci6_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(8, 7)
    port map(
      ins => fork8_outs_1,
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  mux4 : entity work.mux(arch) generic map(2, 3, 1)
    port map(
      index => fork9_outs_0,
      index_valid => fork9_outs_0_valid,
      index_ready => fork9_outs_0_ready,
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

  fork9 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => control_merge2_index,
      ins_valid => control_merge2_index_valid,
      ins_ready => control_merge2_index_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork9_outs_0,
      outs(1) => fork9_outs_1,
      outs_valid(0) => fork9_outs_0_valid,
      outs_valid(1) => fork9_outs_1_valid,
      outs_ready(0) => fork9_outs_0_ready,
      outs_ready(1) => fork9_outs_1_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => control_merge2_outs_valid,
      ins_ready => control_merge2_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork1_outs_0_valid,
      outs_valid(1) => lazy_fork1_outs_1_valid,
      outs_valid(2) => lazy_fork1_outs_2_valid,
      outs_ready(0) => lazy_fork1_outs_0_ready,
      outs_ready(1) => lazy_fork1_outs_1_ready,
      outs_ready(2) => lazy_fork1_outs_2_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant15 : entity work.handshake_constant_3(arch) generic map(8)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  extsi17 : entity work.extsi(arch) generic map(8, 9)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi17_outs,
      outs_valid => extsi17_outs_valid,
      outs_ready => extsi17_outs_ready
    );

  source4 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source4_outs_valid,
      outs_ready => source4_outs_ready
    );

  constant16 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source4_outs_valid,
      ctrl_ready => source4_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  extsi18 : entity work.extsi(arch) generic map(2, 9)
    port map(
      ins => constant16_outs,
      ins_valid => constant16_outs_valid,
      ins_ready => constant16_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi18_outs,
      outs_valid => extsi18_outs_valid,
      outs_ready => extsi18_outs_ready
    );

  buffer18 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => trunci7_outs,
      ins_valid => trunci7_outs_valid,
      ins_ready => trunci7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer18_outs,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  load3 : entity work.load(arch) generic map(32, 7)
    port map(
      addrIn => buffer18_outs,
      addrIn_valid => buffer18_outs_valid,
      addrIn_ready => buffer18_outs_ready,
      dataFromMem => lsq2_ldData_0,
      dataFromMem_valid => lsq2_ldData_0_valid,
      dataFromMem_ready => lsq2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  buffer17 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => trunci6_outs,
      ins_valid => trunci6_outs_valid,
      ins_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  buffer22 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => load3_dataOut,
      ins_valid => load3_dataOut_valid,
      ins_ready => load3_dataOut_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 7)
    port map(
      addrIn => buffer17_outs,
      addrIn_valid => buffer17_outs_valid,
      addrIn_ready => buffer17_outs_ready,
      dataIn => buffer22_outs,
      dataIn_valid => buffer22_outs_valid,
      dataIn_ready => buffer22_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  addi5 : entity work.addi(arch) generic map(9)
    port map(
      lhs => extsi16_outs,
      lhs_valid => extsi16_outs_valid,
      lhs_ready => extsi16_outs_ready,
      rhs => extsi18_outs,
      rhs_valid => extsi18_outs_valid,
      rhs_ready => extsi18_outs_ready,
      clk => clk,
      rst => rst,
      result => addi5_result,
      result_valid => addi5_result_valid,
      result_ready => addi5_result_ready
    );

  buffer23 : entity work.oehb(arch) generic map(9)
    port map(
      ins => addi5_result,
      ins_valid => addi5_result_valid,
      ins_ready => addi5_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 9)
    port map(
      ins => buffer23_outs,
      ins_valid => buffer23_outs_valid,
      ins_ready => buffer23_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  trunci8 : entity work.trunci(arch) generic map(9, 8)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci8_outs,
      outs_valid => trunci8_outs_valid,
      outs_ready => trunci8_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(9)
    port map(
      lhs => fork10_outs_1,
      lhs_valid => fork10_outs_1_valid,
      lhs_ready => fork10_outs_1_ready,
      rhs => extsi17_outs,
      rhs_valid => extsi17_outs_valid,
      rhs_ready => extsi17_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer24 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer24_outs,
      ins_valid => buffer24_outs_valid,
      ins_ready => buffer24_outs_ready,
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

  cond_br7 : entity work.cond_br(arch) generic map(8)
    port map(
      condition => fork11_outs_0,
      condition_valid => fork11_outs_0_valid,
      condition_ready => fork11_outs_0_ready,
      data => trunci8_outs,
      data_valid => trunci8_outs_valid,
      data_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br7_trueOut,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut => cond_br7_falseOut,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  sink2 : entity work.sink(arch) generic map(8)
    port map(
      ins => cond_br7_falseOut,
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer19 : entity work.oehb(arch) generic map(3)
    port map(
      ins => mux4_outs,
      ins_valid => mux4_outs_valid,
      ins_ready => mux4_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer19_outs,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  buffer20 : entity work.tehb(arch) generic map(3)
    port map(
      ins => buffer19_outs,
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork11_outs_1,
      condition_valid => fork11_outs_1_valid,
      condition_ready => fork11_outs_1_ready,
      data => buffer20_outs,
      data_valid => buffer20_outs_valid,
      data_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br8_trueOut,
      trueOut_valid => cond_br8_trueOut_valid,
      trueOut_ready => cond_br8_trueOut_ready,
      falseOut => cond_br8_falseOut,
      falseOut_valid => cond_br8_falseOut_valid,
      falseOut_ready => cond_br8_falseOut_ready
    );

  buffer21 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_1_valid,
      ins_ready => lazy_fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  cond_br9 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork11_outs_2,
      condition_valid => fork11_outs_2_valid,
      condition_ready => fork11_outs_2_ready,
      data_valid => buffer21_outs_valid,
      data_ready => buffer21_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br9_trueOut_valid,
      trueOut_ready => cond_br9_trueOut_ready,
      falseOut_valid => cond_br9_falseOut_valid,
      falseOut_ready => cond_br9_falseOut_ready
    );

  extsi19 : entity work.extsi(arch) generic map(3, 4)
    port map(
      ins => cond_br8_falseOut,
      ins_valid => cond_br8_falseOut_valid,
      ins_ready => cond_br8_falseOut_ready,
      clk => clk,
      rst => rst,
      outs => extsi19_outs,
      outs_valid => extsi19_outs_valid,
      outs_ready => extsi19_outs_ready
    );

  source5 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source5_outs_valid,
      outs_ready => source5_outs_ready
    );

  constant17 : entity work.handshake_constant_4(arch) generic map(3)
    port map(
      ctrl_valid => source5_outs_valid,
      ctrl_ready => source5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant17_outs,
      outs_valid => constant17_outs_valid,
      outs_ready => constant17_outs_ready
    );

  extsi20 : entity work.extsi(arch) generic map(3, 4)
    port map(
      ins => constant17_outs,
      ins_valid => constant17_outs_valid,
      ins_ready => constant17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi20_outs,
      outs_valid => extsi20_outs_valid,
      outs_ready => extsi20_outs_ready
    );

  source6 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source6_outs_valid,
      outs_ready => source6_outs_ready
    );

  constant18 : entity work.handshake_constant_1(arch) generic map(2)
    port map(
      ctrl_valid => source6_outs_valid,
      ctrl_ready => source6_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant18_outs,
      outs_valid => constant18_outs_valid,
      outs_ready => constant18_outs_ready
    );

  extsi21 : entity work.extsi(arch) generic map(2, 4)
    port map(
      ins => constant18_outs,
      ins_valid => constant18_outs_valid,
      ins_ready => constant18_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi21_outs,
      outs_valid => extsi21_outs_valid,
      outs_ready => extsi21_outs_ready
    );

  addi6 : entity work.addi(arch) generic map(4)
    port map(
      lhs => extsi19_outs,
      lhs_valid => extsi19_outs_valid,
      lhs_ready => extsi19_outs_ready,
      rhs => extsi21_outs,
      rhs_valid => extsi21_outs_valid,
      rhs_ready => extsi21_outs_ready,
      clk => clk,
      rst => rst,
      result => addi6_result,
      result_valid => addi6_result_valid,
      result_ready => addi6_result_ready
    );

  buffer25 : entity work.oehb(arch) generic map(4)
    port map(
      ins => addi6_result,
      ins_valid => addi6_result_valid,
      ins_ready => addi6_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer25_outs,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 4)
    port map(
      ins => buffer25_outs,
      ins_valid => buffer25_outs_valid,
      ins_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  trunci9 : entity work.trunci(arch) generic map(4, 3)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci9_outs,
      outs_valid => trunci9_outs_valid,
      outs_ready => trunci9_outs_ready
    );

  cmpi2 : entity work.handshake_cmpi_1(arch) generic map(4)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => extsi20_outs,
      rhs_valid => extsi20_outs_valid,
      rhs_ready => extsi20_outs_ready,
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

  fork13 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => buffer26_outs,
      ins_valid => buffer26_outs_valid,
      ins_ready => buffer26_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready
    );

  cond_br10 : entity work.cond_br(arch) generic map(3)
    port map(
      condition => fork13_outs_0,
      condition_valid => fork13_outs_0_valid,
      condition_ready => fork13_outs_0_ready,
      data => trunci9_outs,
      data_valid => trunci9_outs_valid,
      data_ready => trunci9_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br10_trueOut,
      trueOut_valid => cond_br10_trueOut_valid,
      trueOut_ready => cond_br10_trueOut_ready,
      falseOut => cond_br10_falseOut,
      falseOut_valid => cond_br10_falseOut_valid,
      falseOut_ready => cond_br10_falseOut_ready
    );

  sink4 : entity work.sink(arch) generic map(3)
    port map(
      ins => cond_br10_falseOut,
      ins_valid => cond_br10_falseOut_valid,
      ins_ready => cond_br10_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br11 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork13_outs_1,
      condition_valid => fork13_outs_1_valid,
      condition_ready => fork13_outs_1_ready,
      data_valid => cond_br9_falseOut_valid,
      data_ready => cond_br9_falseOut_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br11_trueOut_valid,
      trueOut_ready => cond_br11_trueOut_ready,
      falseOut_valid => cond_br11_falseOut_valid,
      falseOut_ready => cond_br11_falseOut_ready
    );

  fork14 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br11_falseOut_valid,
      ins_ready => cond_br11_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready
    );

end architecture;
