library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bicg is
  port (
    a_loadData : in std_logic_vector(31 downto 0);
    s_loadData : in std_logic_vector(31 downto 0);
    q_loadData : in std_logic_vector(31 downto 0);
    p_loadData : in std_logic_vector(31 downto 0);
    r_loadData : in std_logic_vector(31 downto 0);
    a_start_valid : in std_logic;
    s_start_valid : in std_logic;
    q_start_valid : in std_logic;
    p_start_valid : in std_logic;
    r_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    a_end_ready : in std_logic;
    s_end_ready : in std_logic;
    q_end_ready : in std_logic;
    p_end_ready : in std_logic;
    r_end_ready : in std_logic;
    end_ready : in std_logic;
    a_start_ready : out std_logic;
    s_start_ready : out std_logic;
    q_start_ready : out std_logic;
    p_start_ready : out std_logic;
    r_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    a_end_valid : out std_logic;
    s_end_valid : out std_logic;
    q_end_valid : out std_logic;
    p_end_valid : out std_logic;
    r_end_valid : out std_logic;
    end_valid : out std_logic;
    a_loadEn : out std_logic;
    a_loadAddr : out std_logic_vector(9 downto 0);
    a_storeEn : out std_logic;
    a_storeAddr : out std_logic_vector(9 downto 0);
    a_storeData : out std_logic_vector(31 downto 0);
    s_loadEn : out std_logic;
    s_loadAddr : out std_logic_vector(4 downto 0);
    s_storeEn : out std_logic;
    s_storeAddr : out std_logic_vector(4 downto 0);
    s_storeData : out std_logic_vector(31 downto 0);
    q_loadEn : out std_logic;
    q_loadAddr : out std_logic_vector(4 downto 0);
    q_storeEn : out std_logic;
    q_storeAddr : out std_logic_vector(4 downto 0);
    q_storeData : out std_logic_vector(31 downto 0);
    p_loadEn : out std_logic;
    p_loadAddr : out std_logic_vector(4 downto 0);
    p_storeEn : out std_logic;
    p_storeAddr : out std_logic_vector(4 downto 0);
    p_storeData : out std_logic_vector(31 downto 0);
    r_loadEn : out std_logic;
    r_loadAddr : out std_logic_vector(4 downto 0);
    r_storeEn : out std_logic;
    r_storeAddr : out std_logic_vector(4 downto 0);
    r_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of bicg is

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
  signal lsq2_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq2_ldData_0_valid : std_logic;
  signal lsq2_ldData_0_ready : std_logic;
  signal lsq2_memEnd_valid : std_logic;
  signal lsq2_memEnd_ready : std_logic;
  signal lsq2_loadEn : std_logic;
  signal lsq2_loadAddr : std_logic_vector(4 downto 0);
  signal lsq2_storeEn : std_logic;
  signal lsq2_storeAddr : std_logic_vector(4 downto 0);
  signal lsq2_storeData : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq3_ldData_0_valid : std_logic;
  signal lsq3_ldData_0_ready : std_logic;
  signal lsq3_memEnd_valid : std_logic;
  signal lsq3_memEnd_ready : std_logic;
  signal lsq3_loadEn : std_logic;
  signal lsq3_loadAddr : std_logic_vector(4 downto 0);
  signal lsq3_storeEn : std_logic;
  signal lsq3_storeAddr : std_logic_vector(4 downto 0);
  signal lsq3_storeData : std_logic_vector(31 downto 0);
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
  signal constant0_outs : std_logic_vector(0 downto 0);
  signal constant0_outs_valid : std_logic;
  signal constant0_outs_ready : std_logic;
  signal extsi7_outs : std_logic_vector(5 downto 0);
  signal extsi7_outs_valid : std_logic;
  signal extsi7_outs_ready : std_logic;
  signal mux0_outs : std_logic_vector(5 downto 0);
  signal mux0_outs_valid : std_logic;
  signal mux0_outs_ready : std_logic;
  signal buffer0_outs : std_logic_vector(5 downto 0);
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
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
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal lazy_fork0_outs_2_valid : std_logic;
  signal lazy_fork0_outs_2_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal constant1_outs : std_logic_vector(0 downto 0);
  signal constant1_outs_valid : std_logic;
  signal constant1_outs_ready : std_logic;
  signal buffer2_outs : std_logic_vector(4 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(4 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal extsi6_outs : std_logic_vector(5 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal buffer1_outs : std_logic_vector(5 downto 0);
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal mux1_outs : std_logic_vector(5 downto 0);
  signal mux1_outs_valid : std_logic;
  signal mux1_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(5 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(5 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(5 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal fork2_outs_2 : std_logic_vector(5 downto 0);
  signal fork2_outs_2_valid : std_logic;
  signal fork2_outs_2_ready : std_logic;
  signal fork2_outs_3 : std_logic_vector(5 downto 0);
  signal fork2_outs_3_valid : std_logic;
  signal fork2_outs_3_ready : std_logic;
  signal fork2_outs_4 : std_logic_vector(5 downto 0);
  signal fork2_outs_4_valid : std_logic;
  signal fork2_outs_4_ready : std_logic;
  signal extsi8_outs : std_logic_vector(9 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(5 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal extsi9_outs : std_logic_vector(6 downto 0);
  signal extsi9_outs_valid : std_logic;
  signal extsi9_outs_ready : std_logic;
  signal trunci1_outs : std_logic_vector(4 downto 0);
  signal trunci1_outs_valid : std_logic;
  signal trunci1_outs_ready : std_logic;
  signal trunci2_outs : std_logic_vector(4 downto 0);
  signal trunci2_outs_valid : std_logic;
  signal trunci2_outs_ready : std_logic;
  signal trunci3_outs : std_logic_vector(4 downto 0);
  signal trunci3_outs_valid : std_logic;
  signal trunci3_outs_ready : std_logic;
  signal mux2_outs : std_logic_vector(31 downto 0);
  signal mux2_outs_valid : std_logic;
  signal mux2_outs_ready : std_logic;
  signal buffer8_outs : std_logic_vector(0 downto 0);
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal mux3_outs : std_logic_vector(5 downto 0);
  signal mux3_outs_valid : std_logic;
  signal mux3_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(5 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer14_outs : std_logic_vector(5 downto 0);
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(5 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(5 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal fork3_outs_2 : std_logic_vector(5 downto 0);
  signal fork3_outs_2_valid : std_logic;
  signal fork3_outs_2_ready : std_logic;
  signal extsi10_outs : std_logic_vector(11 downto 0);
  signal extsi10_outs_valid : std_logic;
  signal extsi10_outs_ready : std_logic;
  signal trunci4_outs : std_logic_vector(4 downto 0);
  signal trunci4_outs_valid : std_logic;
  signal trunci4_outs_ready : std_logic;
  signal buffer11_outs : std_logic_vector(5 downto 0);
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
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
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant8_outs : std_logic_vector(5 downto 0);
  signal constant8_outs_valid : std_logic;
  signal constant8_outs_ready : std_logic;
  signal fork6_outs_0 : std_logic_vector(5 downto 0);
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1 : std_logic_vector(5 downto 0);
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal extsi11_outs : std_logic_vector(11 downto 0);
  signal extsi11_outs_valid : std_logic;
  signal extsi11_outs_ready : std_logic;
  signal extsi12_outs : std_logic_vector(6 downto 0);
  signal extsi12_outs_valid : std_logic;
  signal extsi12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(5 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant9_outs : std_logic_vector(1 downto 0);
  signal constant9_outs_valid : std_logic;
  signal constant9_outs_ready : std_logic;
  signal extsi13_outs : std_logic_vector(6 downto 0);
  signal extsi13_outs_valid : std_logic;
  signal extsi13_outs_ready : std_logic;
  signal muli2_result : std_logic_vector(11 downto 0);
  signal muli2_result_valid : std_logic;
  signal muli2_result_ready : std_logic;
  signal trunci5_outs : std_logic_vector(9 downto 0);
  signal trunci5_outs_valid : std_logic;
  signal trunci5_outs_ready : std_logic;
  signal addi2_result : std_logic_vector(9 downto 0);
  signal addi2_result_valid : std_logic;
  signal addi2_result_ready : std_logic;
  signal load1_addrOut : std_logic_vector(9 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal fork7_outs_0 : std_logic_vector(31 downto 0);
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1 : std_logic_vector(31 downto 0);
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal buffer9_outs : std_logic_vector(4 downto 0);
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(4 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;
  signal load3_addrOut : std_logic_vector(4 downto 0);
  signal load3_addrOut_valid : std_logic;
  signal load3_addrOut_ready : std_logic;
  signal load3_dataOut : std_logic_vector(31 downto 0);
  signal load3_dataOut_valid : std_logic;
  signal load3_dataOut_ready : std_logic;
  signal muli0_result : std_logic_vector(31 downto 0);
  signal muli0_result_valid : std_logic;
  signal muli0_result_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal buffer7_outs : std_logic_vector(4 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal buffer17_outs : std_logic_vector(31 downto 0);
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal store0_addrOut : std_logic_vector(4 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal load4_addrOut : std_logic_vector(4 downto 0);
  signal load4_addrOut_valid : std_logic;
  signal load4_addrOut_ready : std_logic;
  signal load4_dataOut : std_logic_vector(31 downto 0);
  signal load4_dataOut_valid : std_logic;
  signal load4_dataOut_ready : std_logic;
  signal muli1_result : std_logic_vector(31 downto 0);
  signal muli1_result_valid : std_logic;
  signal muli1_result_ready : std_logic;
  signal buffer10_outs : std_logic_vector(31 downto 0);
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal addi1_result : std_logic_vector(31 downto 0);
  signal addi1_result_valid : std_logic;
  signal addi1_result_ready : std_logic;
  signal addi3_result : std_logic_vector(6 downto 0);
  signal addi3_result_valid : std_logic;
  signal addi3_result_ready : std_logic;
  signal buffer20_outs : std_logic_vector(6 downto 0);
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal fork8_outs_0 : std_logic_vector(6 downto 0);
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1 : std_logic_vector(6 downto 0);
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal trunci6_outs : std_logic_vector(5 downto 0);
  signal trunci6_outs_valid : std_logic;
  signal trunci6_outs_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal buffer21_outs : std_logic_vector(0 downto 0);
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
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
  signal cond_br2_trueOut : std_logic_vector(5 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(5 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal buffer18_outs : std_logic_vector(31 downto 0);
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal cond_br3_trueOut : std_logic_vector(31 downto 0);
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut : std_logic_vector(31 downto 0);
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer19_outs : std_logic_vector(0 downto 0);
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(5 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(5 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer22_outs : std_logic_vector(0 downto 0);
  signal buffer22_outs_valid : std_logic;
  signal buffer22_outs_ready : std_logic;
  signal fork10_outs_0 : std_logic_vector(5 downto 0);
  signal fork10_outs_0_valid : std_logic;
  signal fork10_outs_0_ready : std_logic;
  signal fork10_outs_1 : std_logic_vector(5 downto 0);
  signal fork10_outs_1_valid : std_logic;
  signal fork10_outs_1_ready : std_logic;
  signal extsi14_outs : std_logic_vector(6 downto 0);
  signal extsi14_outs_valid : std_logic;
  signal extsi14_outs_ready : std_logic;
  signal trunci7_outs : std_logic_vector(4 downto 0);
  signal trunci7_outs_valid : std_logic;
  signal trunci7_outs_ready : std_logic;
  signal fork11_outs_0 : std_logic_vector(31 downto 0);
  signal fork11_outs_0_valid : std_logic;
  signal fork11_outs_0_ready : std_logic;
  signal fork11_outs_1 : std_logic_vector(31 downto 0);
  signal fork11_outs_1_valid : std_logic;
  signal fork11_outs_1_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(5 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal extsi15_outs : std_logic_vector(6 downto 0);
  signal extsi15_outs_valid : std_logic;
  signal extsi15_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant11_outs : std_logic_vector(1 downto 0);
  signal constant11_outs_valid : std_logic;
  signal constant11_outs_ready : std_logic;
  signal extsi16_outs : std_logic_vector(6 downto 0);
  signal extsi16_outs_valid : std_logic;
  signal extsi16_outs_ready : std_logic;
  signal buffer23_outs : std_logic_vector(4 downto 0);
  signal buffer23_outs_valid : std_logic;
  signal buffer23_outs_ready : std_logic;
  signal buffer24_outs : std_logic_vector(31 downto 0);
  signal buffer24_outs_valid : std_logic;
  signal buffer24_outs_ready : std_logic;
  signal store1_addrOut : std_logic_vector(4 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal addi4_result : std_logic_vector(6 downto 0);
  signal addi4_result_valid : std_logic;
  signal addi4_result_ready : std_logic;
  signal buffer27_outs : std_logic_vector(6 downto 0);
  signal buffer27_outs_valid : std_logic;
  signal buffer27_outs_ready : std_logic;
  signal fork12_outs_0 : std_logic_vector(6 downto 0);
  signal fork12_outs_0_valid : std_logic;
  signal fork12_outs_0_ready : std_logic;
  signal fork12_outs_1 : std_logic_vector(6 downto 0);
  signal fork12_outs_1_valid : std_logic;
  signal fork12_outs_1_ready : std_logic;
  signal trunci8_outs : std_logic_vector(5 downto 0);
  signal trunci8_outs_valid : std_logic;
  signal trunci8_outs_ready : std_logic;
  signal buffer26_outs : std_logic_vector(6 downto 0);
  signal buffer26_outs_valid : std_logic;
  signal buffer26_outs_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal buffer32_outs : std_logic_vector(0 downto 0);
  signal buffer32_outs_valid : std_logic;
  signal buffer32_outs_ready : std_logic;
  signal fork13_outs_0 : std_logic_vector(0 downto 0);
  signal fork13_outs_0_valid : std_logic;
  signal fork13_outs_0_ready : std_logic;
  signal fork13_outs_1 : std_logic_vector(0 downto 0);
  signal fork13_outs_1_valid : std_logic;
  signal fork13_outs_1_ready : std_logic;
  signal fork13_outs_2 : std_logic_vector(0 downto 0);
  signal fork13_outs_2_valid : std_logic;
  signal fork13_outs_2_ready : std_logic;
  signal cond_br6_trueOut : std_logic_vector(5 downto 0);
  signal cond_br6_trueOut_valid : std_logic;
  signal cond_br6_trueOut_ready : std_logic;
  signal cond_br6_falseOut : std_logic_vector(5 downto 0);
  signal cond_br6_falseOut_valid : std_logic;
  signal cond_br6_falseOut_ready : std_logic;
  signal buffer28_outs : std_logic_vector(0 downto 0);
  signal buffer28_outs_valid : std_logic;
  signal buffer28_outs_ready : std_logic;
  signal buffer25_outs_valid : std_logic;
  signal buffer25_outs_ready : std_logic;
  signal cond_br7_trueOut_valid : std_logic;
  signal cond_br7_trueOut_ready : std_logic;
  signal cond_br7_falseOut_valid : std_logic;
  signal cond_br7_falseOut_ready : std_logic;
  signal buffer29_outs : std_logic_vector(0 downto 0);
  signal buffer29_outs_valid : std_logic;
  signal buffer29_outs_ready : std_logic;
  signal cond_br8_trueOut : std_logic_vector(31 downto 0);
  signal cond_br8_trueOut_valid : std_logic;
  signal cond_br8_trueOut_ready : std_logic;
  signal cond_br8_falseOut : std_logic_vector(31 downto 0);
  signal cond_br8_falseOut_valid : std_logic;
  signal cond_br8_falseOut_ready : std_logic;
  signal buffer30_outs : std_logic_vector(0 downto 0);
  signal buffer30_outs_valid : std_logic;
  signal buffer30_outs_ready : std_logic;
  signal buffer31_outs : std_logic_vector(31 downto 0);
  signal buffer31_outs_valid : std_logic;
  signal buffer31_outs_ready : std_logic;
  signal fork14_outs_0_valid : std_logic;
  signal fork14_outs_0_ready : std_logic;
  signal fork14_outs_1_valid : std_logic;
  signal fork14_outs_1_ready : std_logic;
  signal fork14_outs_2_valid : std_logic;
  signal fork14_outs_2_ready : std_logic;
  signal fork14_outs_3_valid : std_logic;
  signal fork14_outs_3_ready : std_logic;
  signal fork14_outs_4_valid : std_logic;
  signal fork14_outs_4_ready : std_logic;

begin

  out0 <= cond_br8_falseOut;
  out0_valid <= cond_br8_falseOut_valid;
  cond_br8_falseOut_ready <= out0_ready;
  a_end_valid <= mem_controller5_memEnd_valid;
  mem_controller5_memEnd_ready <= a_end_ready;
  s_end_valid <= lsq3_memEnd_valid;
  lsq3_memEnd_ready <= s_end_ready;
  q_end_valid <= lsq2_memEnd_valid;
  lsq2_memEnd_ready <= q_end_ready;
  p_end_valid <= mem_controller4_memEnd_valid;
  mem_controller4_memEnd_ready <= p_end_ready;
  r_end_valid <= mem_controller3_memEnd_valid;
  mem_controller3_memEnd_ready <= r_end_ready;
  end_valid <= fork0_outs_1_valid;
  fork0_outs_1_ready <= end_ready;
  a_loadEn <= mem_controller5_loadEn;
  a_loadAddr <= mem_controller5_loadAddr;
  a_storeEn <= mem_controller5_storeEn;
  a_storeAddr <= mem_controller5_storeAddr;
  a_storeData <= mem_controller5_storeData;
  s_loadEn <= lsq3_loadEn;
  s_loadAddr <= lsq3_loadAddr;
  s_storeEn <= lsq3_storeEn;
  s_storeAddr <= lsq3_storeAddr;
  s_storeData <= lsq3_storeData;
  q_loadEn <= lsq2_loadEn;
  q_loadAddr <= lsq2_loadAddr;
  q_storeEn <= lsq2_storeEn;
  q_storeAddr <= lsq2_storeAddr;
  q_storeData <= lsq2_storeData;
  p_loadEn <= mem_controller4_loadEn;
  p_loadAddr <= mem_controller4_loadAddr;
  p_storeEn <= mem_controller4_storeEn;
  p_storeAddr <= mem_controller4_storeAddr;
  p_storeData <= mem_controller4_storeData;
  r_loadEn <= mem_controller3_loadEn;
  r_loadAddr <= mem_controller3_loadAddr;
  r_storeEn <= mem_controller3_storeEn;
  r_storeAddr <= mem_controller3_storeAddr;
  r_storeData <= mem_controller3_storeData;

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
      loadData => r_loadData,
      memStart_valid => r_start_valid,
      memStart_ready => r_start_ready,
      ldAddr(0) => load3_addrOut,
      ldAddr_valid(0) => load3_addrOut_valid,
      ldAddr_ready(0) => load3_addrOut_ready,
      ctrlEnd_valid => fork14_outs_4_valid,
      ctrlEnd_ready => fork14_outs_4_ready,
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
      loadData => p_loadData,
      memStart_valid => p_start_valid,
      memStart_ready => p_start_ready,
      ldAddr(0) => load4_addrOut,
      ldAddr_valid(0) => load4_addrOut_valid,
      ldAddr_ready(0) => load4_addrOut_ready,
      ctrlEnd_valid => fork14_outs_3_valid,
      ctrlEnd_ready => fork14_outs_3_ready,
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

  lsq2 : entity work.handshake_lsq_lsq2(arch)
    port map(
      io_loadData => q_loadData,
      io_memStart_valid => q_start_valid,
      io_memStart_ready => q_start_ready,
      io_ctrl_0_valid => lazy_fork0_outs_0_valid,
      io_ctrl_0_ready => lazy_fork0_outs_0_ready,
      io_ldAddr_0_bits => load0_addrOut,
      io_ldAddr_0_valid => load0_addrOut_valid,
      io_ldAddr_0_ready => load0_addrOut_ready,
      io_ctrl_1_valid => lazy_fork1_outs_0_valid,
      io_ctrl_1_ready => lazy_fork1_outs_0_ready,
      io_stAddr_0_bits => store1_addrOut,
      io_stAddr_0_valid => store1_addrOut_valid,
      io_stAddr_0_ready => store1_addrOut_ready,
      io_stData_0_bits => store1_dataToMem,
      io_stData_0_valid => store1_dataToMem_valid,
      io_stData_0_ready => store1_dataToMem_ready,
      io_ctrlEnd_valid => fork14_outs_2_valid,
      io_ctrlEnd_ready => fork14_outs_2_ready,
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
      io_loadData => s_loadData,
      io_memStart_valid => s_start_valid,
      io_memStart_ready => s_start_ready,
      io_ctrl_0_valid => fork5_outs_0_valid,
      io_ctrl_0_ready => fork5_outs_0_ready,
      io_ldAddr_0_bits => load2_addrOut,
      io_ldAddr_0_valid => load2_addrOut_valid,
      io_ldAddr_0_ready => load2_addrOut_ready,
      io_stAddr_0_bits => store0_addrOut,
      io_stAddr_0_valid => store0_addrOut_valid,
      io_stAddr_0_ready => store0_addrOut_ready,
      io_stData_0_bits => store0_dataToMem,
      io_stData_0_valid => store0_dataToMem_valid,
      io_stData_0_ready => store0_dataToMem_ready,
      io_ctrlEnd_valid => fork14_outs_1_valid,
      io_ctrlEnd_ready => fork14_outs_1_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq3_ldData_0,
      io_ldData_0_valid => lsq3_ldData_0_valid,
      io_ldData_0_ready => lsq3_ldData_0_ready,
      io_memEnd_valid => lsq3_memEnd_valid,
      io_memEnd_ready => lsq3_memEnd_ready,
      io_loadEn => lsq3_loadEn,
      io_loadAddr => lsq3_loadAddr,
      io_storeEn => lsq3_storeEn,
      io_storeAddr => lsq3_storeAddr,
      io_storeData => lsq3_storeData
    );

  mem_controller5 : entity work.mem_controller_storeless(arch) generic map(1, 32, 10)
    port map(
      loadData => a_loadData,
      memStart_valid => a_start_valid,
      memStart_ready => a_start_ready,
      ldAddr(0) => load1_addrOut,
      ldAddr_valid(0) => load1_addrOut_valid,
      ldAddr_ready(0) => load1_addrOut_ready,
      ctrlEnd_valid => fork14_outs_0_valid,
      ctrlEnd_ready => fork14_outs_0_ready,
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

  extsi7 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant0_outs,
      ins_valid => constant0_outs_valid,
      ins_ready => constant0_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi7_outs,
      outs_valid => extsi7_outs_valid,
      outs_ready => extsi7_outs_ready
    );

  mux0 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => control_merge0_index,
      index_valid => control_merge0_index_valid,
      index_ready => control_merge0_index_ready,
      ins(0) => extsi7_outs,
      ins(1) => cond_br6_trueOut,
      ins_valid(0) => extsi7_outs_valid,
      ins_valid(1) => cond_br6_trueOut_valid,
      ins_ready(0) => extsi7_outs_ready,
      ins_ready(1) => cond_br6_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux0_outs,
      outs_valid => mux0_outs_valid,
      outs_ready => mux0_outs_ready
    );

  buffer0 : entity work.tehb(arch) generic map(6)
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

  fork1 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => buffer0_outs,
      ins_valid => buffer0_outs_valid,
      ins_ready => buffer0_outs_ready,
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

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => control_merge0_outs_valid,
      ins_ready => control_merge0_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_valid(2) => lazy_fork0_outs_2_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready,
      outs_ready(2) => lazy_fork0_outs_2_ready
    );

  buffer5 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => lazy_fork0_outs_2_valid,
      ins_ready => lazy_fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  constant1 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => buffer5_outs_valid,
      ctrl_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant1_outs,
      outs_valid => constant1_outs_valid,
      outs_ready => constant1_outs_ready
    );

  buffer2 : entity work.tfifo(arch) generic map(1, 5)
    port map(
      ins => trunci0_outs,
      ins_valid => trunci0_outs_valid,
      ins_ready => trunci0_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer2_outs,
      outs_valid => buffer2_outs_valid,
      outs_ready => buffer2_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => buffer2_outs,
      addrIn_valid => buffer2_outs_valid,
      addrIn_ready => buffer2_outs_ready,
      dataFromMem => lsq2_ldData_0,
      dataFromMem_valid => lsq2_ldData_0_valid,
      dataFromMem_ready => lsq2_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load0_addrOut,
      addrOut_valid => load0_addrOut_valid,
      addrOut_ready => load0_addrOut_ready,
      dataOut => load0_dataOut,
      dataOut_valid => load0_dataOut_valid,
      dataOut_ready => load0_dataOut_ready
    );

  extsi6 : entity work.extsi(arch) generic map(1, 6)
    port map(
      ins => constant1_outs,
      ins_valid => constant1_outs_valid,
      ins_ready => constant1_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  buffer1 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork1_outs_1,
      ins_valid => fork1_outs_1_valid,
      ins_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer1_outs,
      outs_valid => buffer1_outs_valid,
      outs_ready => buffer1_outs_ready
    );

  buffer4 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_1_valid,
      ins_ready => lazy_fork0_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  mux1 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork4_outs_1,
      index_valid => fork4_outs_1_valid,
      index_ready => fork4_outs_1_ready,
      ins(0) => extsi6_outs,
      ins(1) => cond_br2_trueOut,
      ins_valid(0) => extsi6_outs_valid,
      ins_valid(1) => cond_br2_trueOut_valid,
      ins_ready(0) => extsi6_outs_ready,
      ins_ready(1) => cond_br2_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux1_outs,
      outs_valid => mux1_outs_valid,
      outs_ready => mux1_outs_ready
    );

  buffer6 : entity work.tehb(arch) generic map(6)
    port map(
      ins => mux1_outs,
      ins_valid => mux1_outs_valid,
      ins_ready => mux1_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer6_outs,
      outs_valid => buffer6_outs_valid,
      outs_ready => buffer6_outs_ready
    );

  fork2 : entity work.handshake_fork(arch) generic map(5, 6)
    port map(
      ins => buffer6_outs,
      ins_valid => buffer6_outs_valid,
      ins_ready => buffer6_outs_ready,
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

  extsi8 : entity work.extsi(arch) generic map(6, 10)
    port map(
      ins => buffer3_outs,
      ins_valid => buffer3_outs_valid,
      ins_ready => buffer3_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  buffer3 : entity work.tfifo(arch) generic map(2, 6)
    port map(
      ins => fork2_outs_0,
      ins_valid => fork2_outs_0_valid,
      ins_ready => fork2_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer3_outs,
      outs_valid => buffer3_outs_valid,
      outs_ready => buffer3_outs_ready
    );

  extsi9 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork2_outs_4,
      ins_valid => fork2_outs_4_valid,
      ins_ready => fork2_outs_4_ready,
      clk => clk,
      rst => rst,
      outs => extsi9_outs,
      outs_valid => extsi9_outs_valid,
      outs_ready => extsi9_outs_ready
    );

  trunci1 : entity work.trunci(arch) generic map(6, 5)
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

  trunci2 : entity work.trunci(arch) generic map(6, 5)
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

  trunci3 : entity work.trunci(arch) generic map(6, 5)
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

  mux2 : entity work.mux(arch) generic map(2, 32, 1)
    port map(
      index => buffer8_outs,
      index_valid => buffer8_outs_valid,
      index_ready => buffer8_outs_ready,
      ins(0) => load0_dataOut,
      ins(1) => cond_br3_trueOut,
      ins_valid(0) => load0_dataOut_valid,
      ins_valid(1) => cond_br3_trueOut_valid,
      ins_ready(0) => load0_dataOut_ready,
      ins_ready(1) => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux2_outs,
      outs_valid => mux2_outs_valid,
      outs_ready => mux2_outs_ready
    );

  buffer8 : entity work.tfifo(arch) generic map(4, 1)
    port map(
      ins => fork4_outs_2,
      ins_valid => fork4_outs_2_valid,
      ins_ready => fork4_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer8_outs,
      outs_valid => buffer8_outs_valid,
      outs_ready => buffer8_outs_ready
    );

  mux3 : entity work.mux(arch) generic map(2, 6, 1)
    port map(
      index => fork4_outs_0,
      index_valid => fork4_outs_0_valid,
      index_ready => fork4_outs_0_ready,
      ins(0) => buffer1_outs,
      ins(1) => cond_br4_trueOut,
      ins_valid(0) => buffer1_outs_valid,
      ins_valid(1) => cond_br4_trueOut_valid,
      ins_ready(0) => buffer1_outs_ready,
      ins_ready(1) => cond_br4_trueOut_ready,
      clk => clk,
      rst => rst,
      outs => mux3_outs,
      outs_valid => mux3_outs_valid,
      outs_ready => mux3_outs_ready
    );

  buffer12 : entity work.oehb(arch) generic map(6)
    port map(
      ins => mux3_outs,
      ins_valid => mux3_outs_valid,
      ins_ready => mux3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer12_outs,
      outs_valid => buffer12_outs_valid,
      outs_ready => buffer12_outs_ready
    );

  buffer14 : entity work.tehb(arch) generic map(6)
    port map(
      ins => buffer12_outs,
      ins_valid => buffer12_outs_valid,
      ins_ready => buffer12_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer14_outs,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(3, 6)
    port map(
      ins => buffer14_outs,
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
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

  extsi10 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => fork3_outs_2,
      ins_valid => fork3_outs_2_valid,
      ins_ready => fork3_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => extsi10_outs,
      outs_valid => extsi10_outs_valid,
      outs_ready => extsi10_outs_ready
    );

  trunci4 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => buffer11_outs,
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs => trunci4_outs,
      outs_valid => trunci4_outs_valid,
      outs_ready => trunci4_outs_ready
    );

  buffer11 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork3_outs_0,
      ins_valid => fork3_outs_0_valid,
      ins_ready => fork3_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer11_outs,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  control_merge1 : entity work.control_merge_dataless(arch) generic map(2, 1)
    port map(
      ins_valid(0) => buffer4_outs_valid,
      ins_valid(1) => cond_br5_trueOut_valid,
      ins_ready(0) => buffer4_outs_ready,
      ins_ready(1) => cond_br5_trueOut_ready,
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

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant8 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant8_outs,
      outs_valid => constant8_outs_valid,
      outs_ready => constant8_outs_ready
    );

  fork6 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => constant8_outs,
      ins_valid => constant8_outs_valid,
      ins_ready => constant8_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork6_outs_0,
      outs(1) => fork6_outs_1,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  extsi11 : entity work.extsi(arch) generic map(6, 12)
    port map(
      ins => fork6_outs_0,
      ins_valid => fork6_outs_0_valid,
      ins_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => extsi11_outs,
      outs_valid => extsi11_outs_valid,
      outs_ready => extsi11_outs_ready
    );

  extsi12 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => buffer13_outs,
      ins_valid => buffer13_outs_valid,
      ins_ready => buffer13_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi12_outs,
      outs_valid => extsi12_outs_valid,
      outs_ready => extsi12_outs_ready
    );

  buffer13 : entity work.tfifo(arch) generic map(1, 6)
    port map(
      ins => fork6_outs_1,
      ins_valid => fork6_outs_1_valid,
      ins_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer13_outs,
      outs_valid => buffer13_outs_valid,
      outs_ready => buffer13_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant9 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant9_outs,
      outs_valid => constant9_outs_valid,
      outs_ready => constant9_outs_ready
    );

  extsi13 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant9_outs,
      ins_valid => constant9_outs_valid,
      ins_ready => constant9_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi13_outs,
      outs_valid => extsi13_outs_valid,
      outs_ready => extsi13_outs_ready
    );

  muli2 : entity work.muli(arch) generic map(12)
    port map(
      lhs => extsi10_outs,
      lhs_valid => extsi10_outs_valid,
      lhs_ready => extsi10_outs_ready,
      rhs => extsi11_outs,
      rhs_valid => extsi11_outs_valid,
      rhs_ready => extsi11_outs_ready,
      clk => clk,
      rst => rst,
      result => muli2_result,
      result_valid => muli2_result_valid,
      result_ready => muli2_result_ready
    );

  trunci5 : entity work.trunci(arch) generic map(12, 10)
    port map(
      ins => muli2_result,
      ins_valid => muli2_result_valid,
      ins_ready => muli2_result_ready,
      clk => clk,
      rst => rst,
      outs => trunci5_outs,
      outs_valid => trunci5_outs_valid,
      outs_ready => trunci5_outs_ready
    );

  addi2 : entity work.addi(arch) generic map(10)
    port map(
      lhs => extsi8_outs,
      lhs_valid => extsi8_outs_valid,
      lhs_ready => extsi8_outs_ready,
      rhs => trunci5_outs,
      rhs_valid => trunci5_outs_valid,
      rhs_ready => trunci5_outs_ready,
      clk => clk,
      rst => rst,
      result => addi2_result,
      result_valid => addi2_result_valid,
      result_ready => addi2_result_ready
    );

  load1 : entity work.load(arch) generic map(32, 10)
    port map(
      addrIn => addi2_result,
      addrIn_valid => addi2_result_valid,
      addrIn_ready => addi2_result_ready,
      dataFromMem => mem_controller5_ldData_0,
      dataFromMem_valid => mem_controller5_ldData_0_valid,
      dataFromMem_ready => mem_controller5_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load1_addrOut,
      addrOut_valid => load1_addrOut_valid,
      addrOut_ready => load1_addrOut_ready,
      dataOut => load1_dataOut,
      dataOut_valid => load1_dataOut_valid,
      dataOut_ready => load1_dataOut_ready
    );

  fork7 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load1_dataOut,
      ins_valid => load1_dataOut_valid,
      ins_ready => load1_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork7_outs_0,
      outs(1) => fork7_outs_1,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  buffer9 : entity work.tfifo(arch) generic map(1, 5)
    port map(
      ins => trunci3_outs,
      ins_valid => trunci3_outs_valid,
      ins_ready => trunci3_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer9_outs,
      outs_valid => buffer9_outs_valid,
      outs_ready => buffer9_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => buffer9_outs,
      addrIn_valid => buffer9_outs_valid,
      addrIn_ready => buffer9_outs_ready,
      dataFromMem => lsq3_ldData_0,
      dataFromMem_valid => lsq3_ldData_0_valid,
      dataFromMem_ready => lsq3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

  load3 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci4_outs,
      addrIn_valid => trunci4_outs_valid,
      addrIn_ready => trunci4_outs_ready,
      dataFromMem => mem_controller3_ldData_0,
      dataFromMem_valid => mem_controller3_ldData_0_valid,
      dataFromMem_ready => mem_controller3_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load3_addrOut,
      addrOut_valid => load3_addrOut_valid,
      addrOut_ready => load3_addrOut_ready,
      dataOut => load3_dataOut,
      dataOut_valid => load3_dataOut_valid,
      dataOut_ready => load3_dataOut_ready
    );

  muli0 : entity work.muli(arch) generic map(32)
    port map(
      lhs => load3_dataOut,
      lhs_valid => load3_dataOut_valid,
      lhs_ready => load3_dataOut_ready,
      rhs => fork7_outs_0,
      rhs_valid => fork7_outs_0_valid,
      rhs_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      result => muli0_result,
      result_valid => muli0_result_valid,
      result_ready => muli0_result_ready
    );

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => load2_dataOut,
      lhs_valid => load2_dataOut_valid,
      lhs_ready => load2_dataOut_ready,
      rhs => muli0_result,
      rhs_valid => muli0_result_valid,
      rhs_ready => muli0_result_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  buffer7 : entity work.tfifo(arch) generic map(1, 5)
    port map(
      ins => trunci2_outs,
      ins_valid => trunci2_outs_valid,
      ins_ready => trunci2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer7_outs,
      outs_valid => buffer7_outs_valid,
      outs_ready => buffer7_outs_ready
    );

  buffer17 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => addi0_result,
      ins_valid => addi0_result_valid,
      ins_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer17_outs,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  store0 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => buffer7_outs,
      addrIn_valid => buffer7_outs_valid,
      addrIn_ready => buffer7_outs_ready,
      dataIn => buffer17_outs,
      dataIn_valid => buffer17_outs_valid,
      dataIn_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  load4 : entity work.load(arch) generic map(32, 5)
    port map(
      addrIn => trunci1_outs,
      addrIn_valid => trunci1_outs_valid,
      addrIn_ready => trunci1_outs_ready,
      dataFromMem => mem_controller4_ldData_0,
      dataFromMem_valid => mem_controller4_ldData_0_valid,
      dataFromMem_ready => mem_controller4_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load4_addrOut,
      addrOut_valid => load4_addrOut_valid,
      addrOut_ready => load4_addrOut_ready,
      dataOut => load4_dataOut,
      dataOut_valid => load4_dataOut_valid,
      dataOut_ready => load4_dataOut_ready
    );

  muli1 : entity work.muli(arch) generic map(32)
    port map(
      lhs => fork7_outs_1,
      lhs_valid => fork7_outs_1_valid,
      lhs_ready => fork7_outs_1_ready,
      rhs => load4_dataOut,
      rhs_valid => load4_dataOut_valid,
      rhs_ready => load4_dataOut_ready,
      clk => clk,
      rst => rst,
      result => muli1_result,
      result_valid => muli1_result_valid,
      result_ready => muli1_result_ready
    );

  buffer10 : entity work.tehb(arch) generic map(32)
    port map(
      ins => mux2_outs,
      ins_valid => mux2_outs_valid,
      ins_ready => mux2_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer10_outs,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  addi1 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer10_outs,
      lhs_valid => buffer10_outs_valid,
      lhs_ready => buffer10_outs_ready,
      rhs => muli1_result,
      rhs_valid => muli1_result_valid,
      rhs_ready => muli1_result_ready,
      clk => clk,
      rst => rst,
      result => addi1_result,
      result_valid => addi1_result_valid,
      result_ready => addi1_result_ready
    );

  addi3 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi9_outs,
      lhs_valid => extsi9_outs_valid,
      lhs_ready => extsi9_outs_ready,
      rhs => extsi13_outs,
      rhs_valid => extsi13_outs_valid,
      rhs_ready => extsi13_outs_ready,
      clk => clk,
      rst => rst,
      result => addi3_result,
      result_valid => addi3_result_valid,
      result_ready => addi3_result_ready
    );

  buffer20 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi3_result,
      ins_valid => addi3_result_valid,
      ins_ready => addi3_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer20_outs,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  fork8 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer20_outs,
      ins_valid => buffer20_outs_valid,
      ins_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork8_outs_0,
      outs(1) => fork8_outs_1,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready
    );

  trunci6 : entity work.trunci(arch) generic map(7, 6)
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

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork8_outs_1,
      lhs_valid => fork8_outs_1_valid,
      lhs_ready => fork8_outs_1_ready,
      rhs => extsi12_outs,
      rhs_valid => extsi12_outs_valid,
      rhs_ready => extsi12_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  buffer21 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer21_outs,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  fork9 : entity work.handshake_fork(arch) generic map(4, 1)
    port map(
      ins => buffer21_outs,
      ins_valid => buffer21_outs_valid,
      ins_ready => buffer21_outs_ready,
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

  cond_br2 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork9_outs_0,
      condition_valid => fork9_outs_0_valid,
      condition_ready => fork9_outs_0_ready,
      data => trunci6_outs,
      data_valid => trunci6_outs_valid,
      data_ready => trunci6_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br2_trueOut,
      trueOut_valid => cond_br2_trueOut_valid,
      trueOut_ready => cond_br2_trueOut_ready,
      falseOut => cond_br2_falseOut,
      falseOut_valid => cond_br2_falseOut_valid,
      falseOut_ready => cond_br2_falseOut_ready
    );

  sink0 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer18 : entity work.oehb(arch) generic map(32)
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

  cond_br3 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer19_outs,
      condition_valid => buffer19_outs_valid,
      condition_ready => buffer19_outs_ready,
      data => buffer18_outs,
      data_valid => buffer18_outs_valid,
      data_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br3_trueOut,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut => cond_br3_falseOut,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer19 : entity work.tfifo(arch) generic map(4, 1)
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

  cond_br4 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => fork9_outs_1,
      condition_valid => fork9_outs_1_valid,
      condition_ready => fork9_outs_1_ready,
      data => fork3_outs_1,
      data_valid => fork3_outs_1_valid,
      data_ready => fork3_outs_1_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br4_trueOut,
      trueOut_valid => cond_br4_trueOut_valid,
      trueOut_ready => cond_br4_trueOut_ready,
      falseOut => cond_br4_falseOut,
      falseOut_valid => cond_br4_falseOut_valid,
      falseOut_ready => cond_br4_falseOut_ready
    );

  buffer15 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => fork5_outs_1_valid,
      ins_ready => fork5_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  buffer16 : entity work.tfifo_dataless(arch) generic map(1)
    port map(
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  cond_br5 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer22_outs,
      condition_valid => buffer22_outs_valid,
      condition_ready => buffer22_outs_ready,
      data_valid => buffer16_outs_valid,
      data_ready => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer22 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork9_outs_3,
      ins_valid => fork9_outs_3_valid,
      ins_ready => fork9_outs_3_ready,
      clk => clk,
      rst => rst,
      outs => buffer22_outs,
      outs_valid => buffer22_outs_valid,
      outs_ready => buffer22_outs_ready
    );

  fork10 : entity work.handshake_fork(arch) generic map(2, 6)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork10_outs_0,
      outs(1) => fork10_outs_1,
      outs_valid(0) => fork10_outs_0_valid,
      outs_valid(1) => fork10_outs_1_valid,
      outs_ready(0) => fork10_outs_0_ready,
      outs_ready(1) => fork10_outs_1_ready
    );

  extsi14 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => fork10_outs_1,
      ins_valid => fork10_outs_1_valid,
      ins_ready => fork10_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => extsi14_outs,
      outs_valid => extsi14_outs_valid,
      outs_ready => extsi14_outs_ready
    );

  trunci7 : entity work.trunci(arch) generic map(6, 5)
    port map(
      ins => fork10_outs_0,
      ins_valid => fork10_outs_0_valid,
      ins_ready => fork10_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => trunci7_outs,
      outs_valid => trunci7_outs_valid,
      outs_ready => trunci7_outs_ready
    );

  fork11 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => cond_br3_falseOut,
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork11_outs_0,
      outs(1) => fork11_outs_1,
      outs_valid(0) => fork11_outs_0_valid,
      outs_valid(1) => fork11_outs_1_valid,
      outs_ready(0) => fork11_outs_0_ready,
      outs_ready(1) => fork11_outs_1_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(2)
    port map(
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
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

  constant10 : entity work.handshake_constant_1(arch) generic map(6)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  extsi15 : entity work.extsi(arch) generic map(6, 7)
    port map(
      ins => constant10_outs,
      ins_valid => constant10_outs_valid,
      ins_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi15_outs,
      outs_valid => extsi15_outs_valid,
      outs_ready => extsi15_outs_ready
    );

  source3 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source3_outs_valid,
      outs_ready => source3_outs_ready
    );

  constant11 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => source3_outs_valid,
      ctrl_ready => source3_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant11_outs,
      outs_valid => constant11_outs_valid,
      outs_ready => constant11_outs_ready
    );

  extsi16 : entity work.extsi(arch) generic map(2, 7)
    port map(
      ins => constant11_outs,
      ins_valid => constant11_outs_valid,
      ins_ready => constant11_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi16_outs,
      outs_valid => extsi16_outs_valid,
      outs_ready => extsi16_outs_ready
    );

  buffer23 : entity work.tfifo(arch) generic map(1, 5)
    port map(
      ins => trunci7_outs,
      ins_valid => trunci7_outs_valid,
      ins_ready => trunci7_outs_ready,
      clk => clk,
      rst => rst,
      outs => buffer23_outs,
      outs_valid => buffer23_outs_valid,
      outs_ready => buffer23_outs_ready
    );

  buffer24 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork11_outs_0,
      ins_valid => fork11_outs_0_valid,
      ins_ready => fork11_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer24_outs,
      outs_valid => buffer24_outs_valid,
      outs_ready => buffer24_outs_ready
    );

  store1 : entity work.store(arch) generic map(32, 5)
    port map(
      addrIn => buffer23_outs,
      addrIn_valid => buffer23_outs_valid,
      addrIn_ready => buffer23_outs_ready,
      dataIn => buffer24_outs,
      dataIn_valid => buffer24_outs_valid,
      dataIn_ready => buffer24_outs_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  addi4 : entity work.addi(arch) generic map(7)
    port map(
      lhs => extsi14_outs,
      lhs_valid => extsi14_outs_valid,
      lhs_ready => extsi14_outs_ready,
      rhs => extsi16_outs,
      rhs_valid => extsi16_outs_valid,
      rhs_ready => extsi16_outs_ready,
      clk => clk,
      rst => rst,
      result => addi4_result,
      result_valid => addi4_result_valid,
      result_ready => addi4_result_ready
    );

  buffer27 : entity work.oehb(arch) generic map(7)
    port map(
      ins => addi4_result,
      ins_valid => addi4_result_valid,
      ins_ready => addi4_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer27_outs,
      outs_valid => buffer27_outs_valid,
      outs_ready => buffer27_outs_ready
    );

  fork12 : entity work.handshake_fork(arch) generic map(2, 7)
    port map(
      ins => buffer27_outs,
      ins_valid => buffer27_outs_valid,
      ins_ready => buffer27_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork12_outs_0,
      outs(1) => fork12_outs_1,
      outs_valid(0) => fork12_outs_0_valid,
      outs_valid(1) => fork12_outs_1_valid,
      outs_ready(0) => fork12_outs_0_ready,
      outs_ready(1) => fork12_outs_1_ready
    );

  trunci8 : entity work.trunci(arch) generic map(7, 6)
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

  buffer26 : entity work.tfifo(arch) generic map(1, 7)
    port map(
      ins => fork12_outs_0,
      ins_valid => fork12_outs_0_valid,
      ins_ready => fork12_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer26_outs,
      outs_valid => buffer26_outs_valid,
      outs_ready => buffer26_outs_ready
    );

  cmpi1 : entity work.handshake_cmpi_0(arch) generic map(7)
    port map(
      lhs => fork12_outs_1,
      lhs_valid => fork12_outs_1_valid,
      lhs_ready => fork12_outs_1_ready,
      rhs => extsi15_outs,
      rhs_valid => extsi15_outs_valid,
      rhs_ready => extsi15_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  buffer32 : entity work.oehb(arch) generic map(1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs => buffer32_outs,
      outs_valid => buffer32_outs_valid,
      outs_ready => buffer32_outs_ready
    );

  fork13 : entity work.handshake_fork(arch) generic map(3, 1)
    port map(
      ins => buffer32_outs,
      ins_valid => buffer32_outs_valid,
      ins_ready => buffer32_outs_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork13_outs_0,
      outs(1) => fork13_outs_1,
      outs(2) => fork13_outs_2,
      outs_valid(0) => fork13_outs_0_valid,
      outs_valid(1) => fork13_outs_1_valid,
      outs_valid(2) => fork13_outs_2_valid,
      outs_ready(0) => fork13_outs_0_ready,
      outs_ready(1) => fork13_outs_1_ready,
      outs_ready(2) => fork13_outs_2_ready
    );

  cond_br6 : entity work.cond_br(arch) generic map(6)
    port map(
      condition => buffer28_outs,
      condition_valid => buffer28_outs_valid,
      condition_ready => buffer28_outs_ready,
      data => trunci8_outs,
      data_valid => trunci8_outs_valid,
      data_ready => trunci8_outs_ready,
      clk => clk,
      rst => rst,
      trueOut => cond_br6_trueOut,
      trueOut_valid => cond_br6_trueOut_valid,
      trueOut_ready => cond_br6_trueOut_ready,
      falseOut => cond_br6_falseOut,
      falseOut_valid => cond_br6_falseOut_valid,
      falseOut_ready => cond_br6_falseOut_ready
    );

  buffer28 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork13_outs_0,
      ins_valid => fork13_outs_0_valid,
      ins_ready => fork13_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => buffer28_outs,
      outs_valid => buffer28_outs_valid,
      outs_ready => buffer28_outs_ready
    );

  sink2 : entity work.sink(arch) generic map(6)
    port map(
      ins => cond_br6_falseOut,
      ins_valid => cond_br6_falseOut_valid,
      ins_ready => cond_br6_falseOut_ready,
      clk => clk,
      rst => rst
    );

  buffer25 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_1_valid,
      ins_ready => lazy_fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer25_outs_valid,
      outs_ready => buffer25_outs_ready
    );

  cond_br7 : entity work.cond_br_dataless(arch)
    port map(
      condition => buffer29_outs,
      condition_valid => buffer29_outs_valid,
      condition_ready => buffer29_outs_ready,
      data_valid => buffer25_outs_valid,
      data_ready => buffer25_outs_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br7_trueOut_valid,
      trueOut_ready => cond_br7_trueOut_ready,
      falseOut_valid => cond_br7_falseOut_valid,
      falseOut_ready => cond_br7_falseOut_ready
    );

  buffer29 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork13_outs_1,
      ins_valid => fork13_outs_1_valid,
      ins_ready => fork13_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer29_outs,
      outs_valid => buffer29_outs_valid,
      outs_ready => buffer29_outs_ready
    );

  cond_br8 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => buffer30_outs,
      condition_valid => buffer30_outs_valid,
      condition_ready => buffer30_outs_ready,
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

  buffer30 : entity work.tfifo(arch) generic map(1, 1)
    port map(
      ins => fork13_outs_2,
      ins_valid => fork13_outs_2_valid,
      ins_ready => fork13_outs_2_ready,
      clk => clk,
      rst => rst,
      outs => buffer30_outs,
      outs_valid => buffer30_outs_valid,
      outs_ready => buffer30_outs_ready
    );

  buffer31 : entity work.tfifo(arch) generic map(1, 32)
    port map(
      ins => fork11_outs_1,
      ins_valid => fork11_outs_1_valid,
      ins_ready => fork11_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => buffer31_outs,
      outs_valid => buffer31_outs_valid,
      outs_ready => buffer31_outs_ready
    );

  sink3 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br8_trueOut,
      ins_valid => cond_br8_trueOut_valid,
      ins_ready => cond_br8_trueOut_ready,
      clk => clk,
      rst => rst
    );

  fork14 : entity work.fork_dataless(arch) generic map(5)
    port map(
      ins_valid => cond_br7_falseOut_valid,
      ins_ready => cond_br7_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork14_outs_0_valid,
      outs_valid(1) => fork14_outs_1_valid,
      outs_valid(2) => fork14_outs_2_valid,
      outs_valid(3) => fork14_outs_3_valid,
      outs_valid(4) => fork14_outs_4_valid,
      outs_ready(0) => fork14_outs_0_ready,
      outs_ready(1) => fork14_outs_1_ready,
      outs_ready(2) => fork14_outs_2_ready,
      outs_ready(3) => fork14_outs_3_ready,
      outs_ready(4) => fork14_outs_4_ready
    );

end architecture;
