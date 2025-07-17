library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity iterative_sqrt is
  port (
    A_loadData : in std_logic_vector(31 downto 0);
    A_start_valid : in std_logic;
    start_valid : in std_logic;
    clk : in std_logic;
    rst : in std_logic;
    out0_ready : in std_logic;
    A_end_ready : in std_logic;
    end_ready : in std_logic;
    A_start_ready : out std_logic;
    start_ready : out std_logic;
    out0 : out std_logic_vector(31 downto 0);
    out0_valid : out std_logic;
    A_end_valid : out std_logic;
    end_valid : out std_logic;
    A_loadEn : out std_logic;
    A_loadAddr : out std_logic_vector(3 downto 0);
    A_storeEn : out std_logic;
    A_storeAddr : out std_logic_vector(3 downto 0);
    A_storeData : out std_logic_vector(31 downto 0)
  );
end entity;

architecture behavioral of iterative_sqrt is

  signal fork0_outs_0_valid : std_logic;
  signal fork0_outs_0_ready : std_logic;
  signal fork0_outs_1_valid : std_logic;
  signal fork0_outs_1_ready : std_logic;
  signal buffer21_outs_valid : std_logic;
  signal buffer21_outs_ready : std_logic;
  signal mem_controller1_ldData_0 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_0_valid : std_logic;
  signal mem_controller1_ldData_0_ready : std_logic;
  signal mem_controller1_ldData_1 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_1_valid : std_logic;
  signal mem_controller1_ldData_1_ready : std_logic;
  signal mem_controller1_ldData_2 : std_logic_vector(31 downto 0);
  signal mem_controller1_ldData_2_valid : std_logic;
  signal mem_controller1_ldData_2_ready : std_logic;
  signal mem_controller1_memEnd_valid : std_logic;
  signal mem_controller1_memEnd_ready : std_logic;
  signal mem_controller1_loadEn : std_logic;
  signal mem_controller1_loadAddr : std_logic_vector(3 downto 0);
  signal mem_controller1_storeEn : std_logic;
  signal mem_controller1_storeAddr : std_logic_vector(3 downto 0);
  signal mem_controller1_storeData : std_logic_vector(31 downto 0);
  signal lsq1_ldData_0 : std_logic_vector(31 downto 0);
  signal lsq1_ldData_0_valid : std_logic;
  signal lsq1_ldData_0_ready : std_logic;
  signal lsq1_ldAddrToMC : std_logic_vector(3 downto 0);
  signal lsq1_ldAddrToMC_valid : std_logic;
  signal lsq1_ldAddrToMC_ready : std_logic;
  signal lsq1_stAddrToMC : std_logic_vector(3 downto 0);
  signal lsq1_stAddrToMC_valid : std_logic;
  signal lsq1_stAddrToMC_ready : std_logic;
  signal lsq1_stDataToMC : std_logic_vector(31 downto 0);
  signal lsq1_stDataToMC_valid : std_logic;
  signal lsq1_stDataToMC_ready : std_logic;
  signal merge3_outs_valid : std_logic;
  signal merge3_outs_ready : std_logic;
  signal buffer0_outs_valid : std_logic;
  signal buffer0_outs_ready : std_logic;
  signal buffer1_outs_valid : std_logic;
  signal buffer1_outs_ready : std_logic;
  signal fork1_outs_0_valid : std_logic;
  signal fork1_outs_0_ready : std_logic;
  signal fork1_outs_1_valid : std_logic;
  signal fork1_outs_1_ready : std_logic;
  signal constant4_outs : std_logic_vector(0 downto 0);
  signal constant4_outs_valid : std_logic;
  signal constant4_outs_ready : std_logic;
  signal extui0_outs : std_logic_vector(3 downto 0);
  signal extui0_outs_valid : std_logic;
  signal extui0_outs_ready : std_logic;
  signal source0_outs_valid : std_logic;
  signal source0_outs_ready : std_logic;
  signal constant14_outs : std_logic_vector(4 downto 0);
  signal constant14_outs_valid : std_logic;
  signal constant14_outs_ready : std_logic;
  signal extsi1_outs : std_logic_vector(31 downto 0);
  signal extsi1_outs_valid : std_logic;
  signal extsi1_outs_ready : std_logic;
  signal load0_addrOut : std_logic_vector(3 downto 0);
  signal load0_addrOut_valid : std_logic;
  signal load0_addrOut_ready : std_logic;
  signal load0_dataOut : std_logic_vector(31 downto 0);
  signal load0_dataOut_valid : std_logic;
  signal load0_dataOut_ready : std_logic;
  signal fork2_outs_0 : std_logic_vector(31 downto 0);
  signal fork2_outs_0_valid : std_logic;
  signal fork2_outs_0_ready : std_logic;
  signal fork2_outs_1 : std_logic_vector(31 downto 0);
  signal fork2_outs_1_valid : std_logic;
  signal fork2_outs_1_ready : std_logic;
  signal cmpi0_result : std_logic_vector(0 downto 0);
  signal cmpi0_result_valid : std_logic;
  signal cmpi0_result_ready : std_logic;
  signal fork3_outs_0 : std_logic_vector(0 downto 0);
  signal fork3_outs_0_valid : std_logic;
  signal fork3_outs_0_ready : std_logic;
  signal fork3_outs_1 : std_logic_vector(0 downto 0);
  signal fork3_outs_1_valid : std_logic;
  signal fork3_outs_1_ready : std_logic;
  signal cond_br2_trueOut : std_logic_vector(31 downto 0);
  signal cond_br2_trueOut_valid : std_logic;
  signal cond_br2_trueOut_ready : std_logic;
  signal cond_br2_falseOut : std_logic_vector(31 downto 0);
  signal cond_br2_falseOut_valid : std_logic;
  signal cond_br2_falseOut_ready : std_logic;
  signal cond_br3_trueOut_valid : std_logic;
  signal cond_br3_trueOut_ready : std_logic;
  signal cond_br3_falseOut_valid : std_logic;
  signal cond_br3_falseOut_ready : std_logic;
  signal buffer4_outs_valid : std_logic;
  signal buffer4_outs_ready : std_logic;
  signal buffer5_outs_valid : std_logic;
  signal buffer5_outs_ready : std_logic;
  signal fork4_outs_0_valid : std_logic;
  signal fork4_outs_0_ready : std_logic;
  signal fork4_outs_1_valid : std_logic;
  signal fork4_outs_1_ready : std_logic;
  signal constant15_outs : std_logic_vector(1 downto 0);
  signal constant15_outs_valid : std_logic;
  signal constant15_outs_ready : std_logic;
  signal extui1_outs : std_logic_vector(3 downto 0);
  signal extui1_outs_valid : std_logic;
  signal extui1_outs_ready : std_logic;
  signal source1_outs_valid : std_logic;
  signal source1_outs_ready : std_logic;
  signal constant16_outs : std_logic_vector(4 downto 0);
  signal constant16_outs_valid : std_logic;
  signal constant16_outs_ready : std_logic;
  signal extsi3_outs : std_logic_vector(31 downto 0);
  signal extsi3_outs_valid : std_logic;
  signal extsi3_outs_ready : std_logic;
  signal load1_addrOut : std_logic_vector(3 downto 0);
  signal load1_addrOut_valid : std_logic;
  signal load1_addrOut_ready : std_logic;
  signal load1_dataOut : std_logic_vector(31 downto 0);
  signal load1_dataOut_valid : std_logic;
  signal load1_dataOut_ready : std_logic;
  signal cmpi1_result : std_logic_vector(0 downto 0);
  signal cmpi1_result_valid : std_logic;
  signal cmpi1_result_ready : std_logic;
  signal fork5_outs_0 : std_logic_vector(0 downto 0);
  signal fork5_outs_0_valid : std_logic;
  signal fork5_outs_0_ready : std_logic;
  signal fork5_outs_1 : std_logic_vector(0 downto 0);
  signal fork5_outs_1_valid : std_logic;
  signal fork5_outs_1_ready : std_logic;
  signal buffer2_outs : std_logic_vector(31 downto 0);
  signal buffer2_outs_valid : std_logic;
  signal buffer2_outs_ready : std_logic;
  signal buffer3_outs : std_logic_vector(31 downto 0);
  signal buffer3_outs_valid : std_logic;
  signal buffer3_outs_ready : std_logic;
  signal cond_br4_trueOut : std_logic_vector(31 downto 0);
  signal cond_br4_trueOut_valid : std_logic;
  signal cond_br4_trueOut_ready : std_logic;
  signal cond_br4_falseOut : std_logic_vector(31 downto 0);
  signal cond_br4_falseOut_valid : std_logic;
  signal cond_br4_falseOut_ready : std_logic;
  signal cond_br5_trueOut_valid : std_logic;
  signal cond_br5_trueOut_ready : std_logic;
  signal cond_br5_falseOut_valid : std_logic;
  signal cond_br5_falseOut_ready : std_logic;
  signal buffer8_outs_valid : std_logic;
  signal buffer8_outs_ready : std_logic;
  signal buffer9_outs_valid : std_logic;
  signal buffer9_outs_ready : std_logic;
  signal lazy_fork0_outs_0_valid : std_logic;
  signal lazy_fork0_outs_0_ready : std_logic;
  signal lazy_fork0_outs_1_valid : std_logic;
  signal lazy_fork0_outs_1_ready : std_logic;
  signal lazy_fork0_outs_2_valid : std_logic;
  signal lazy_fork0_outs_2_ready : std_logic;
  signal buffer11_outs_valid : std_logic;
  signal buffer11_outs_ready : std_logic;
  signal fork6_outs_0_valid : std_logic;
  signal fork6_outs_0_ready : std_logic;
  signal fork6_outs_1_valid : std_logic;
  signal fork6_outs_1_ready : std_logic;
  signal constant17_outs : std_logic_vector(1 downto 0);
  signal constant17_outs_valid : std_logic;
  signal constant17_outs_ready : std_logic;
  signal extsi4_outs : std_logic_vector(31 downto 0);
  signal extsi4_outs_valid : std_logic;
  signal extsi4_outs_ready : std_logic;
  signal constant18_outs : std_logic_vector(0 downto 0);
  signal constant18_outs_valid : std_logic;
  signal constant18_outs_ready : std_logic;
  signal extui2_outs : std_logic_vector(3 downto 0);
  signal extui2_outs_valid : std_logic;
  signal extui2_outs_ready : std_logic;
  signal source2_outs_valid : std_logic;
  signal source2_outs_ready : std_logic;
  signal constant10_outs : std_logic_vector(31 downto 0);
  signal constant10_outs_valid : std_logic;
  signal constant10_outs_ready : std_logic;
  signal buffer6_outs : std_logic_vector(31 downto 0);
  signal buffer6_outs_valid : std_logic;
  signal buffer6_outs_ready : std_logic;
  signal buffer7_outs : std_logic_vector(31 downto 0);
  signal buffer7_outs_valid : std_logic;
  signal buffer7_outs_ready : std_logic;
  signal addi0_result : std_logic_vector(31 downto 0);
  signal addi0_result_valid : std_logic;
  signal addi0_result_ready : std_logic;
  signal store0_addrOut : std_logic_vector(3 downto 0);
  signal store0_addrOut_valid : std_logic;
  signal store0_addrOut_ready : std_logic;
  signal store0_dataToMem : std_logic_vector(31 downto 0);
  signal store0_dataToMem_valid : std_logic;
  signal store0_dataToMem_ready : std_logic;
  signal buffer10_outs_valid : std_logic;
  signal buffer10_outs_ready : std_logic;
  signal buffer14_outs_valid : std_logic;
  signal buffer14_outs_ready : std_logic;
  signal buffer15_outs_valid : std_logic;
  signal buffer15_outs_ready : std_logic;
  signal lazy_fork1_outs_0_valid : std_logic;
  signal lazy_fork1_outs_0_ready : std_logic;
  signal lazy_fork1_outs_1_valid : std_logic;
  signal lazy_fork1_outs_1_ready : std_logic;
  signal lazy_fork1_outs_2_valid : std_logic;
  signal lazy_fork1_outs_2_ready : std_logic;
  signal buffer17_outs_valid : std_logic;
  signal buffer17_outs_ready : std_logic;
  signal fork7_outs_0_valid : std_logic;
  signal fork7_outs_0_ready : std_logic;
  signal fork7_outs_1_valid : std_logic;
  signal fork7_outs_1_ready : std_logic;
  signal constant19_outs : std_logic_vector(1 downto 0);
  signal constant19_outs_valid : std_logic;
  signal constant19_outs_ready : std_logic;
  signal extsi6_outs : std_logic_vector(31 downto 0);
  signal extsi6_outs_valid : std_logic;
  signal extsi6_outs_ready : std_logic;
  signal constant20_outs : std_logic_vector(0 downto 0);
  signal constant20_outs_valid : std_logic;
  signal constant20_outs_ready : std_logic;
  signal extui3_outs : std_logic_vector(3 downto 0);
  signal extui3_outs_valid : std_logic;
  signal extui3_outs_ready : std_logic;
  signal source3_outs_valid : std_logic;
  signal source3_outs_ready : std_logic;
  signal constant21_outs : std_logic_vector(1 downto 0);
  signal constant21_outs_valid : std_logic;
  signal constant21_outs_ready : std_logic;
  signal extsi8_outs : std_logic_vector(31 downto 0);
  signal extsi8_outs_valid : std_logic;
  signal extsi8_outs_ready : std_logic;
  signal buffer12_outs : std_logic_vector(31 downto 0);
  signal buffer12_outs_valid : std_logic;
  signal buffer12_outs_ready : std_logic;
  signal buffer13_outs : std_logic_vector(31 downto 0);
  signal buffer13_outs_valid : std_logic;
  signal buffer13_outs_ready : std_logic;
  signal shrsi0_result : std_logic_vector(31 downto 0);
  signal shrsi0_result_valid : std_logic;
  signal shrsi0_result_ready : std_logic;
  signal store1_addrOut : std_logic_vector(3 downto 0);
  signal store1_addrOut_valid : std_logic;
  signal store1_addrOut_ready : std_logic;
  signal store1_dataToMem : std_logic_vector(31 downto 0);
  signal store1_dataToMem_valid : std_logic;
  signal store1_dataToMem_ready : std_logic;
  signal buffer16_outs_valid : std_logic;
  signal buffer16_outs_ready : std_logic;
  signal buffer18_outs_valid : std_logic;
  signal buffer18_outs_ready : std_logic;
  signal buffer19_outs_valid : std_logic;
  signal buffer19_outs_ready : std_logic;
  signal fork8_outs_0_valid : std_logic;
  signal fork8_outs_0_ready : std_logic;
  signal fork8_outs_1_valid : std_logic;
  signal fork8_outs_1_ready : std_logic;
  signal fork8_outs_2_valid : std_logic;
  signal fork8_outs_2_ready : std_logic;
  signal buffer20_outs_valid : std_logic;
  signal buffer20_outs_ready : std_logic;
  signal constant22_outs : std_logic_vector(0 downto 0);
  signal constant22_outs_valid : std_logic;
  signal constant22_outs_ready : std_logic;
  signal extui4_outs : std_logic_vector(3 downto 0);
  signal extui4_outs_valid : std_logic;
  signal extui4_outs_ready : std_logic;
  signal load2_addrOut : std_logic_vector(3 downto 0);
  signal load2_addrOut_valid : std_logic;
  signal load2_addrOut_ready : std_logic;
  signal load2_dataOut : std_logic_vector(31 downto 0);
  signal load2_dataOut_valid : std_logic;
  signal load2_dataOut_ready : std_logic;

begin

  out0 <= load2_dataOut;
  out0_valid <= load2_dataOut_valid;
  load2_dataOut_ready <= out0_ready;
  A_end_valid <= mem_controller1_memEnd_valid;
  mem_controller1_memEnd_ready <= A_end_ready;
  end_valid <= fork0_outs_0_valid;
  fork0_outs_0_ready <= end_ready;
  A_loadEn <= mem_controller1_loadEn;
  A_loadAddr <= mem_controller1_loadAddr;
  A_storeEn <= mem_controller1_storeEn;
  A_storeAddr <= mem_controller1_storeAddr;
  A_storeData <= mem_controller1_storeData;

  fork0 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => start_valid,
      ins_ready => start_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork0_outs_0_valid,
      outs_valid(1) => fork0_outs_1_valid,
      outs_ready(0) => fork0_outs_0_ready,
      outs_ready(1) => fork0_outs_1_ready
    );

  buffer21 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => fork8_outs_1_valid,
      ins_ready => fork8_outs_1_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer21_outs_valid,
      outs_ready => buffer21_outs_ready
    );

  mem_controller1 : entity work.mem_controller(arch) generic map(2, 3, 1, 32, 4)
    port map(
      loadData => A_loadData,
      memStart_valid => A_start_valid,
      memStart_ready => A_start_ready,
      ldAddr(0) => load0_addrOut,
      ldAddr(1) => load1_addrOut,
      ldAddr(2) => lsq1_ldAddrToMC,
      ldAddr_valid(0) => load0_addrOut_valid,
      ldAddr_valid(1) => load1_addrOut_valid,
      ldAddr_valid(2) => lsq1_ldAddrToMC_valid,
      ldAddr_ready(0) => load0_addrOut_ready,
      ldAddr_ready(1) => load1_addrOut_ready,
      ldAddr_ready(2) => lsq1_ldAddrToMC_ready,
      ctrl(0) => extsi6_outs,
      ctrl(1) => extsi4_outs,
      ctrl_valid(0) => extsi6_outs_valid,
      ctrl_valid(1) => extsi4_outs_valid,
      ctrl_ready(0) => extsi6_outs_ready,
      ctrl_ready(1) => extsi4_outs_ready,
      stAddr(0) => lsq1_stAddrToMC,
      stAddr_valid(0) => lsq1_stAddrToMC_valid,
      stAddr_ready(0) => lsq1_stAddrToMC_ready,
      stData(0) => lsq1_stDataToMC,
      stData_valid(0) => lsq1_stDataToMC_valid,
      stData_ready(0) => lsq1_stDataToMC_ready,
      ctrlEnd_valid => buffer21_outs_valid,
      ctrlEnd_ready => buffer21_outs_ready,
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
      memEnd_valid => mem_controller1_memEnd_valid,
      memEnd_ready => mem_controller1_memEnd_ready,
      loadEn => mem_controller1_loadEn,
      loadAddr => mem_controller1_loadAddr,
      storeEn => mem_controller1_storeEn,
      storeAddr => mem_controller1_storeAddr,
      storeData => mem_controller1_storeData
    );

  lsq1 : entity work.handshake_lsq_lsq1(arch)
    port map(
      io_ctrl_0_valid => lazy_fork0_outs_1_valid,
      io_ctrl_0_ready => lazy_fork0_outs_1_ready,
      io_stAddr_0_bits => store0_addrOut,
      io_stAddr_0_valid => store0_addrOut_valid,
      io_stAddr_0_ready => store0_addrOut_ready,
      io_stData_0_bits => store0_dataToMem,
      io_stData_0_valid => store0_dataToMem_valid,
      io_stData_0_ready => store0_dataToMem_ready,
      io_ctrl_1_valid => lazy_fork1_outs_1_valid,
      io_ctrl_1_ready => lazy_fork1_outs_1_ready,
      io_stAddr_1_bits => store1_addrOut,
      io_stAddr_1_valid => store1_addrOut_valid,
      io_stAddr_1_ready => store1_addrOut_ready,
      io_stData_1_bits => store1_dataToMem,
      io_stData_1_valid => store1_dataToMem_valid,
      io_stData_1_ready => store1_dataToMem_ready,
      io_ctrl_2_valid => fork8_outs_2_valid,
      io_ctrl_2_ready => fork8_outs_2_ready,
      io_ldAddr_0_bits => load2_addrOut,
      io_ldAddr_0_valid => load2_addrOut_valid,
      io_ldAddr_0_ready => load2_addrOut_ready,
      io_ldDataFromMC_bits => mem_controller1_ldData_2,
      io_ldDataFromMC_valid => mem_controller1_ldData_2_valid,
      io_ldDataFromMC_ready => mem_controller1_ldData_2_ready,
      clock => clk,
      reset => rst,
      io_ldData_0_bits => lsq1_ldData_0,
      io_ldData_0_valid => lsq1_ldData_0_valid,
      io_ldData_0_ready => lsq1_ldData_0_ready,
      io_ldAddrToMC_bits => lsq1_ldAddrToMC,
      io_ldAddrToMC_valid => lsq1_ldAddrToMC_valid,
      io_ldAddrToMC_ready => lsq1_ldAddrToMC_ready,
      io_stAddrToMC_bits => lsq1_stAddrToMC,
      io_stAddrToMC_valid => lsq1_stAddrToMC_valid,
      io_stAddrToMC_ready => lsq1_stAddrToMC_ready,
      io_stDataToMC_bits => lsq1_stDataToMC,
      io_stDataToMC_valid => lsq1_stDataToMC_valid,
      io_stDataToMC_ready => lsq1_stDataToMC_ready
    );

  merge3 : entity work.merge_dataless(arch) generic map(3)
    port map(
      ins_valid(0) => fork0_outs_1_valid,
      ins_valid(1) => buffer10_outs_valid,
      ins_valid(2) => buffer16_outs_valid,
      ins_ready(0) => fork0_outs_1_ready,
      ins_ready(1) => buffer10_outs_ready,
      ins_ready(2) => buffer16_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => merge3_outs_valid,
      outs_ready => merge3_outs_ready
    );

  buffer0 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => merge3_outs_valid,
      ins_ready => merge3_outs_ready,
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

  fork1 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer1_outs_valid,
      ins_ready => buffer1_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork1_outs_0_valid,
      outs_valid(1) => fork1_outs_1_valid,
      outs_ready(0) => fork1_outs_0_ready,
      outs_ready(1) => fork1_outs_1_ready
    );

  constant4 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork1_outs_1_valid,
      ctrl_ready => fork1_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant4_outs,
      outs_valid => constant4_outs_valid,
      outs_ready => constant4_outs_ready
    );

  extui0 : entity work.extui(arch) generic map(1, 4)
    port map(
      ins => constant4_outs,
      ins_valid => constant4_outs_valid,
      ins_ready => constant4_outs_ready,
      clk => clk,
      rst => rst,
      outs => extui0_outs,
      outs_valid => extui0_outs_valid,
      outs_ready => extui0_outs_ready
    );

  source0 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source0_outs_valid,
      outs_ready => source0_outs_ready
    );

  constant14 : entity work.handshake_constant_1(arch) generic map(5)
    port map(
      ctrl_valid => source0_outs_valid,
      ctrl_ready => source0_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant14_outs,
      outs_valid => constant14_outs_valid,
      outs_ready => constant14_outs_ready
    );

  extsi1 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => constant14_outs,
      ins_valid => constant14_outs_valid,
      ins_ready => constant14_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi1_outs,
      outs_valid => extsi1_outs_valid,
      outs_ready => extsi1_outs_ready
    );

  load0 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => extui0_outs,
      addrIn_valid => extui0_outs_valid,
      addrIn_ready => extui0_outs_ready,
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

  fork2 : entity work.handshake_fork(arch) generic map(2, 32)
    port map(
      ins => load0_dataOut,
      ins_valid => load0_dataOut_valid,
      ins_ready => load0_dataOut_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork2_outs_0,
      outs(1) => fork2_outs_1,
      outs_valid(0) => fork2_outs_0_valid,
      outs_valid(1) => fork2_outs_1_valid,
      outs_ready(0) => fork2_outs_0_ready,
      outs_ready(1) => fork2_outs_1_ready
    );

  cmpi0 : entity work.handshake_cmpi_0(arch) generic map(32)
    port map(
      lhs => fork2_outs_1,
      lhs_valid => fork2_outs_1_valid,
      lhs_ready => fork2_outs_1_ready,
      rhs => extsi1_outs,
      rhs_valid => extsi1_outs_valid,
      rhs_ready => extsi1_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi0_result,
      result_valid => cmpi0_result_valid,
      result_ready => cmpi0_result_ready
    );

  fork3 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi0_result,
      ins_valid => cmpi0_result_valid,
      ins_ready => cmpi0_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork3_outs_0,
      outs(1) => fork3_outs_1,
      outs_valid(0) => fork3_outs_0_valid,
      outs_valid(1) => fork3_outs_1_valid,
      outs_ready(0) => fork3_outs_0_ready,
      outs_ready(1) => fork3_outs_1_ready
    );

  cond_br2 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork3_outs_1,
      condition_valid => fork3_outs_1_valid,
      condition_ready => fork3_outs_1_ready,
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

  sink1 : entity work.sink(arch) generic map(32)
    port map(
      ins => cond_br2_falseOut,
      ins_valid => cond_br2_falseOut_valid,
      ins_ready => cond_br2_falseOut_ready,
      clk => clk,
      rst => rst
    );

  cond_br3 : entity work.cond_br_dataless(arch)
    port map(
      condition => fork3_outs_0,
      condition_valid => fork3_outs_0_valid,
      condition_ready => fork3_outs_0_ready,
      data_valid => fork1_outs_0_valid,
      data_ready => fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br3_trueOut_valid,
      trueOut_ready => cond_br3_trueOut_ready,
      falseOut_valid => cond_br3_falseOut_valid,
      falseOut_ready => cond_br3_falseOut_ready
    );

  buffer4 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br3_trueOut_valid,
      ins_ready => cond_br3_trueOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer4_outs_valid,
      outs_ready => buffer4_outs_ready
    );

  buffer5 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer4_outs_valid,
      ins_ready => buffer4_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer5_outs_valid,
      outs_ready => buffer5_outs_ready
    );

  fork4 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer5_outs_valid,
      ins_ready => buffer5_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork4_outs_0_valid,
      outs_valid(1) => fork4_outs_1_valid,
      outs_ready(0) => fork4_outs_0_ready,
      outs_ready(1) => fork4_outs_1_ready
    );

  constant15 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork4_outs_1_valid,
      ctrl_ready => fork4_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant15_outs,
      outs_valid => constant15_outs_valid,
      outs_ready => constant15_outs_ready
    );

  extui1 : entity work.extui(arch) generic map(2, 4)
    port map(
      ins => constant15_outs,
      ins_valid => constant15_outs_valid,
      ins_ready => constant15_outs_ready,
      clk => clk,
      rst => rst,
      outs => extui1_outs,
      outs_valid => extui1_outs_valid,
      outs_ready => extui1_outs_ready
    );

  source1 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source1_outs_valid,
      outs_ready => source1_outs_ready
    );

  constant16 : entity work.handshake_constant_1(arch) generic map(5)
    port map(
      ctrl_valid => source1_outs_valid,
      ctrl_ready => source1_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant16_outs,
      outs_valid => constant16_outs_valid,
      outs_ready => constant16_outs_ready
    );

  extsi3 : entity work.extsi(arch) generic map(5, 32)
    port map(
      ins => constant16_outs,
      ins_valid => constant16_outs_valid,
      ins_ready => constant16_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi3_outs,
      outs_valid => extsi3_outs_valid,
      outs_ready => extsi3_outs_ready
    );

  load1 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => extui1_outs,
      addrIn_valid => extui1_outs_valid,
      addrIn_ready => extui1_outs_ready,
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

  cmpi1 : entity work.handshake_cmpi_1(arch) generic map(32)
    port map(
      lhs => load1_dataOut,
      lhs_valid => load1_dataOut_valid,
      lhs_ready => load1_dataOut_ready,
      rhs => extsi3_outs,
      rhs_valid => extsi3_outs_valid,
      rhs_ready => extsi3_outs_ready,
      clk => clk,
      rst => rst,
      result => cmpi1_result,
      result_valid => cmpi1_result_valid,
      result_ready => cmpi1_result_ready
    );

  fork5 : entity work.handshake_fork(arch) generic map(2, 1)
    port map(
      ins => cmpi1_result,
      ins_valid => cmpi1_result_valid,
      ins_ready => cmpi1_result_ready,
      clk => clk,
      rst => rst,
      outs(0) => fork5_outs_0,
      outs(1) => fork5_outs_1,
      outs_valid(0) => fork5_outs_0_valid,
      outs_valid(1) => fork5_outs_1_valid,
      outs_ready(0) => fork5_outs_0_ready,
      outs_ready(1) => fork5_outs_1_ready
    );

  buffer2 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br2_trueOut,
      ins_valid => cond_br2_trueOut_valid,
      ins_ready => cond_br2_trueOut_ready,
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

  cond_br4 : entity work.cond_br(arch) generic map(32)
    port map(
      condition => fork5_outs_1,
      condition_valid => fork5_outs_1_valid,
      condition_ready => fork5_outs_1_ready,
      data => buffer3_outs,
      data_valid => buffer3_outs_valid,
      data_ready => buffer3_outs_ready,
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
      condition => fork5_outs_0,
      condition_valid => fork5_outs_0_valid,
      condition_ready => fork5_outs_0_ready,
      data_valid => fork4_outs_0_valid,
      data_ready => fork4_outs_0_ready,
      clk => clk,
      rst => rst,
      trueOut_valid => cond_br5_trueOut_valid,
      trueOut_ready => cond_br5_trueOut_ready,
      falseOut_valid => cond_br5_falseOut_valid,
      falseOut_ready => cond_br5_falseOut_ready
    );

  buffer8 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br5_trueOut_valid,
      ins_ready => cond_br5_trueOut_ready,
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

  lazy_fork0 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => buffer9_outs_valid,
      ins_ready => buffer9_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork0_outs_0_valid,
      outs_valid(1) => lazy_fork0_outs_1_valid,
      outs_valid(2) => lazy_fork0_outs_2_valid,
      outs_ready(0) => lazy_fork0_outs_0_ready,
      outs_ready(1) => lazy_fork0_outs_1_ready,
      outs_ready(2) => lazy_fork0_outs_2_ready
    );

  buffer11 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_2_valid,
      ins_ready => lazy_fork0_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer11_outs_valid,
      outs_ready => buffer11_outs_ready
    );

  fork6 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer11_outs_valid,
      ins_ready => buffer11_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork6_outs_0_valid,
      outs_valid(1) => fork6_outs_1_valid,
      outs_ready(0) => fork6_outs_0_ready,
      outs_ready(1) => fork6_outs_1_ready
    );

  constant17 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork6_outs_1_valid,
      ctrl_ready => fork6_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant17_outs,
      outs_valid => constant17_outs_valid,
      outs_ready => constant17_outs_ready
    );

  extsi4 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant17_outs,
      ins_valid => constant17_outs_valid,
      ins_ready => constant17_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi4_outs,
      outs_valid => extsi4_outs_valid,
      outs_ready => extsi4_outs_ready
    );

  constant18 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork6_outs_0_valid,
      ctrl_ready => fork6_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant18_outs,
      outs_valid => constant18_outs_valid,
      outs_ready => constant18_outs_ready
    );

  extui2 : entity work.extui(arch) generic map(1, 4)
    port map(
      ins => constant18_outs,
      ins_valid => constant18_outs_valid,
      ins_ready => constant18_outs_ready,
      clk => clk,
      rst => rst,
      outs => extui2_outs,
      outs_valid => extui2_outs_valid,
      outs_ready => extui2_outs_ready
    );

  source2 : entity work.source(arch)
    port map(
      clk => clk,
      rst => rst,
      outs_valid => source2_outs_valid,
      outs_ready => source2_outs_ready
    );

  constant10 : entity work.handshake_constant_3(arch) generic map(32)
    port map(
      ctrl_valid => source2_outs_valid,
      ctrl_ready => source2_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant10_outs,
      outs_valid => constant10_outs_valid,
      outs_ready => constant10_outs_ready
    );

  buffer6 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br4_trueOut,
      ins_valid => cond_br4_trueOut_valid,
      ins_ready => cond_br4_trueOut_ready,
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

  addi0 : entity work.addi(arch) generic map(32)
    port map(
      lhs => buffer7_outs,
      lhs_valid => buffer7_outs_valid,
      lhs_ready => buffer7_outs_ready,
      rhs => constant10_outs,
      rhs_valid => constant10_outs_valid,
      rhs_ready => constant10_outs_ready,
      clk => clk,
      rst => rst,
      result => addi0_result,
      result_valid => addi0_result_valid,
      result_ready => addi0_result_ready
    );

  store0 : entity work.store(arch) generic map(32, 4)
    port map(
      addrIn => extui2_outs,
      addrIn_valid => extui2_outs_valid,
      addrIn_ready => extui2_outs_ready,
      dataIn => addi0_result,
      dataIn_valid => addi0_result_valid,
      dataIn_ready => addi0_result_ready,
      clk => clk,
      rst => rst,
      addrOut => store0_addrOut,
      addrOut_valid => store0_addrOut_valid,
      addrOut_ready => store0_addrOut_ready,
      dataToMem => store0_dataToMem,
      dataToMem_valid => store0_dataToMem_valid,
      dataToMem_ready => store0_dataToMem_ready
    );

  buffer10 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork0_outs_0_valid,
      ins_ready => lazy_fork0_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer10_outs_valid,
      outs_ready => buffer10_outs_ready
    );

  buffer14 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br5_falseOut_valid,
      ins_ready => cond_br5_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer14_outs_valid,
      outs_ready => buffer14_outs_ready
    );

  buffer15 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer14_outs_valid,
      ins_ready => buffer14_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer15_outs_valid,
      outs_ready => buffer15_outs_ready
    );

  lazy_fork1 : entity work.lazy_fork_dataless(arch) generic map(3)
    port map(
      ins_valid => buffer15_outs_valid,
      ins_ready => buffer15_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => lazy_fork1_outs_0_valid,
      outs_valid(1) => lazy_fork1_outs_1_valid,
      outs_valid(2) => lazy_fork1_outs_2_valid,
      outs_ready(0) => lazy_fork1_outs_0_ready,
      outs_ready(1) => lazy_fork1_outs_1_ready,
      outs_ready(2) => lazy_fork1_outs_2_ready
    );

  buffer17 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_2_valid,
      ins_ready => lazy_fork1_outs_2_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer17_outs_valid,
      outs_ready => buffer17_outs_ready
    );

  fork7 : entity work.fork_dataless(arch) generic map(2)
    port map(
      ins_valid => buffer17_outs_valid,
      ins_ready => buffer17_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork7_outs_0_valid,
      outs_valid(1) => fork7_outs_1_valid,
      outs_ready(0) => fork7_outs_0_ready,
      outs_ready(1) => fork7_outs_1_ready
    );

  constant19 : entity work.handshake_constant_2(arch) generic map(2)
    port map(
      ctrl_valid => fork7_outs_1_valid,
      ctrl_ready => fork7_outs_1_ready,
      clk => clk,
      rst => rst,
      outs => constant19_outs,
      outs_valid => constant19_outs_valid,
      outs_ready => constant19_outs_ready
    );

  extsi6 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant19_outs,
      ins_valid => constant19_outs_valid,
      ins_ready => constant19_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi6_outs,
      outs_valid => extsi6_outs_valid,
      outs_ready => extsi6_outs_ready
    );

  constant20 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => fork7_outs_0_valid,
      ctrl_ready => fork7_outs_0_ready,
      clk => clk,
      rst => rst,
      outs => constant20_outs,
      outs_valid => constant20_outs_valid,
      outs_ready => constant20_outs_ready
    );

  extui3 : entity work.extui(arch) generic map(1, 4)
    port map(
      ins => constant20_outs,
      ins_valid => constant20_outs_valid,
      ins_ready => constant20_outs_ready,
      clk => clk,
      rst => rst,
      outs => extui3_outs,
      outs_valid => extui3_outs_valid,
      outs_ready => extui3_outs_ready
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

  extsi8 : entity work.extsi(arch) generic map(2, 32)
    port map(
      ins => constant21_outs,
      ins_valid => constant21_outs_valid,
      ins_ready => constant21_outs_ready,
      clk => clk,
      rst => rst,
      outs => extsi8_outs,
      outs_valid => extsi8_outs_valid,
      outs_ready => extsi8_outs_ready
    );

  buffer12 : entity work.oehb(arch) generic map(32)
    port map(
      ins => cond_br4_falseOut,
      ins_valid => cond_br4_falseOut_valid,
      ins_ready => cond_br4_falseOut_ready,
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

  shrsi0 : entity work.shrsi(arch) generic map(32)
    port map(
      lhs => buffer13_outs,
      lhs_valid => buffer13_outs_valid,
      lhs_ready => buffer13_outs_ready,
      rhs => extsi8_outs,
      rhs_valid => extsi8_outs_valid,
      rhs_ready => extsi8_outs_ready,
      clk => clk,
      rst => rst,
      result => shrsi0_result,
      result_valid => shrsi0_result_valid,
      result_ready => shrsi0_result_ready
    );

  store1 : entity work.store(arch) generic map(32, 4)
    port map(
      addrIn => extui3_outs,
      addrIn_valid => extui3_outs_valid,
      addrIn_ready => extui3_outs_ready,
      dataIn => shrsi0_result,
      dataIn_valid => shrsi0_result_valid,
      dataIn_ready => shrsi0_result_ready,
      clk => clk,
      rst => rst,
      addrOut => store1_addrOut,
      addrOut_valid => store1_addrOut_valid,
      addrOut_ready => store1_addrOut_ready,
      dataToMem => store1_dataToMem,
      dataToMem_valid => store1_dataToMem_valid,
      dataToMem_ready => store1_dataToMem_ready
    );

  buffer16 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => lazy_fork1_outs_0_valid,
      ins_ready => lazy_fork1_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer16_outs_valid,
      outs_ready => buffer16_outs_ready
    );

  buffer18 : entity work.oehb_dataless(arch)
    port map(
      ins_valid => cond_br3_falseOut_valid,
      ins_ready => cond_br3_falseOut_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer18_outs_valid,
      outs_ready => buffer18_outs_ready
    );

  buffer19 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => buffer18_outs_valid,
      ins_ready => buffer18_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer19_outs_valid,
      outs_ready => buffer19_outs_ready
    );

  fork8 : entity work.fork_dataless(arch) generic map(3)
    port map(
      ins_valid => buffer19_outs_valid,
      ins_ready => buffer19_outs_ready,
      clk => clk,
      rst => rst,
      outs_valid(0) => fork8_outs_0_valid,
      outs_valid(1) => fork8_outs_1_valid,
      outs_valid(2) => fork8_outs_2_valid,
      outs_ready(0) => fork8_outs_0_ready,
      outs_ready(1) => fork8_outs_1_ready,
      outs_ready(2) => fork8_outs_2_ready
    );

  buffer20 : entity work.tehb_dataless(arch)
    port map(
      ins_valid => fork8_outs_0_valid,
      ins_ready => fork8_outs_0_ready,
      clk => clk,
      rst => rst,
      outs_valid => buffer20_outs_valid,
      outs_ready => buffer20_outs_ready
    );

  constant22 : entity work.handshake_constant_0(arch) generic map(1)
    port map(
      ctrl_valid => buffer20_outs_valid,
      ctrl_ready => buffer20_outs_ready,
      clk => clk,
      rst => rst,
      outs => constant22_outs,
      outs_valid => constant22_outs_valid,
      outs_ready => constant22_outs_ready
    );

  extui4 : entity work.extui(arch) generic map(1, 4)
    port map(
      ins => constant22_outs,
      ins_valid => constant22_outs_valid,
      ins_ready => constant22_outs_ready,
      clk => clk,
      rst => rst,
      outs => extui4_outs,
      outs_valid => extui4_outs_valid,
      outs_ready => extui4_outs_ready
    );

  load2 : entity work.load(arch) generic map(32, 4)
    port map(
      addrIn => extui4_outs,
      addrIn_valid => extui4_outs_valid,
      addrIn_ready => extui4_outs_ready,
      dataFromMem => lsq1_ldData_0,
      dataFromMem_valid => lsq1_ldData_0_valid,
      dataFromMem_ready => lsq1_ldData_0_ready,
      clk => clk,
      rst => rst,
      addrOut => load2_addrOut,
      addrOut_valid => load2_addrOut_valid,
      addrOut_ready => load2_addrOut_ready,
      dataOut => load2_dataOut,
      dataOut_valid => load2_dataOut_valid,
      dataOut_ready => load2_dataOut_ready
    );

end architecture;
