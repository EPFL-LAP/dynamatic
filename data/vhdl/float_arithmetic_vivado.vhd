----------------------------------------------------------------------- 
-- float add, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fadd_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fadd_op is

    -- Interface to Vivado component
    component array_RAM_fadd_32bkb is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 10;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
    begin 
  

        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(8)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    
        array_RAM_fadd_32ns_32ns_32_10_full_dsp_1_U1 :  component array_RAM_fadd_32bkb
        port map (
            clk   => clk,
            reset => rst,
            ce    => oehb_ready,
            din0  => dataInArray(0),
            din1  => dataInArray(1),
            dout  => dataOutArray(0));
    
end architecture;


----------------------------------------------------------------------- 
-- float sub, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fsub_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fsub_op is

    -- Interface to Vivado component
    component array_RAM_fsub_32bkb is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 10;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(8)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    array_RAM_fsub_32ns_32ns_32_10_full_dsp_1_U1 :  component array_RAM_fsub_32bkb
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready,
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

----------------------------------------------------------------------- 
-- float mul, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fmul_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fmul_op is

    -- Interface to Vivado component
    component array_RAM_fmul_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 6;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

        signal join_valid : STD_LOGIC;

    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
        join: entity work.join(arch) generic map(2)
        port map( pValidArray,  
                oehb_ready,                        
                join_valid,                  
                readyArray);   

        buff: entity work.delay_buffer(arch) generic map(4)
        port map(clk,
                rst,
                join_valid,
                oehb_ready,
                buff_valid);

        oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
                port map (
                --inputspValidArray
                    clk => clk, 
                    rst => rst, 
                    pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
                    nReadyArray(0) => nReadyArray(0),    
                    validArray(0) => validArray(0), 
                --outputs
                    readyArray(0) => oehb_ready,   
                    dataInArray(0) => oehb_datain,
                    dataOutArray(0) => oehb_dataOut
                );

    array_RAM_fmul_32ns_32ns_32_6_max_dsp_1_U1 :  component array_RAM_fmul_32cud
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready, 
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

----------------------------------------------------------------------- 
-- float division, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fdiv_op is
Generic (
 INPUTS: integer; OUTPUTS: integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port (
  clk : IN STD_LOGIC;
  rst : IN STD_LOGIC;
  pValidArray : IN std_logic_vector(1 downto 0);
  nReadyArray : in std_logic_vector(0 downto 0);
  validArray : out std_logic_vector(0 downto 0);
  readyArray : OUT std_logic_vector(1 downto 0);
  dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
  dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0));
end entity;

architecture arch of fdiv_op is

    -- Interface to Vivado component
    component array_RAM_fdiv_32ns_32ns_32_30_1 is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 30;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 32
        );
        port (
            clk   : in  std_logic;
            reset : in  std_logic;
            ce    : in  std_logic;
            din0  : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1  : in  std_logic_vector(din1_WIDTH-1 downto 0);
            dout  : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    signal buff_valid, oehb_valid, oehb_ready : STD_LOGIC;
    signal oehb_dataOut, oehb_datain : std_logic_vector(0 downto 0);
    
begin 
    
    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                      oehb_ready,     --nready                    
                      join_valid,         --valid          
                      readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
        generic map(28)
        port map(clk,
                 rst,
                 join_valid,
                 oehb_ready,
                 buff_valid);

    oehb: entity work.OEHB(arch) generic map (1, 1, 1, 1)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => buff_valid, -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => validArray(0), 
        --outputs
            readyArray(0) => oehb_ready,   
            dataInArray(0) => oehb_datain,
            dataOutArray(0) => oehb_dataOut
        );

    array_RAM_fdiv_32ns_32ns_32_30_1_U1 :  component array_RAM_fdiv_32ns_32ns_32_30_1
    port map (
        clk   => clk,
        reset => rst,
        ce    => oehb_ready,
        din0  => dataInArray(0),
        din1  => dataInArray(1),
        dout  => dataOutArray(0));

end architecture;

-----------------------------------------------------------------------
-- fcmp oeq, version 0.0
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;

use work.customTypes.all;

entity fcmp_oeq_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_oeq_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00001";

begin 
    
    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');

    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0)
    );

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

-----------------------------------------------------------------------
-- fcmp ogt, version 0.0
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_ogt_op is
Generic (
INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
);
port(
    clk, rst : in std_logic; 
    dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
    dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
    pValidArray : in std_logic_vector(1 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    validArray : out std_logic_vector(0 downto 0);
    readyArray : out std_logic_vector(1 downto 0));
end entity;

architecture arch of fcmp_ogt_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00010";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

-----------------------------------------------------------------------
-- fcmp oge, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_oge_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
end entity;
    
architecture arch of fcmp_oge_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00011";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));


end architecture;

-----------------------------------------------------------------------
-- fcmp olt, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_olt_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
end entity;
    
architecture arch of fcmp_olt_op is

    --Interface to vivado component
    component array_RAM_fcmp_32cud is
        generic (
            ID         : integer := 1;
            NUM_STAGE  : integer := 2;
            din0_WIDTH : integer := 32;
            din1_WIDTH : integer := 32;
            dout_WIDTH : integer := 1
        );
        port (
            clk    : in  std_logic;
            reset  : in  std_logic;
            ce     : in  std_logic;
            din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
            din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
            opcode : in  std_logic_vector(4 downto 0);
            dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
        );
    end component;

    signal join_valid : STD_LOGIC;
    constant alu_opcode : std_logic_vector(4 downto 0) := "00100";

begin 

    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

end architecture;

-----------------------------------------------------------------------
-- fcmp ole, version 0.0
-- TODO
-----------------------------------------------------------------------

Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;
entity fcmp_ole_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
    end entity;
    
    architecture arch of fcmp_ole_op is
    
        --Interface to vivado component
        component array_RAM_fcmp_32cud is
            generic (
                ID         : integer := 1;
                NUM_STAGE  : integer := 2;
                din0_WIDTH : integer := 32;
                din1_WIDTH : integer := 32;
                dout_WIDTH : integer := 1
            );
            port (
                clk    : in  std_logic;
                reset  : in  std_logic;
                ce     : in  std_logic;
                din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
                din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
                opcode : in  std_logic_vector(4 downto 0);
                dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
            );
        end component;
    
        signal join_valid : STD_LOGIC;
        constant alu_opcode : std_logic_vector(4 downto 0) := "00101";
    
    begin 
    
    --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

    
    end architecture;

-----------------------------------------------------------------------
-- fcmp one, version 0.0
-- TODO
-----------------------------------------------------------------------
Library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;
use work.customTypes.all;

entity fcmp_one_op is
    Generic (
    INPUTS:integer; OUTPUTS:integer; DATA_SIZE_IN: integer; DATA_SIZE_OUT: integer
    );
    port(
        clk, rst : in std_logic; 
        dataInArray : in data_array (1 downto 0)(DATA_SIZE_IN-1 downto 0); 
        dataOutArray : out data_array (0 downto 0)(DATA_SIZE_OUT-1 downto 0);      
        pValidArray : in std_logic_vector(1 downto 0);
        nReadyArray : in std_logic_vector(0 downto 0);
        validArray : out std_logic_vector(0 downto 0);
        readyArray : out std_logic_vector(1 downto 0));
    end entity;
    
    architecture arch of fcmp_one_op is
    
        --Interface to vivado component
        component array_RAM_fcmp_32cud is
            generic (
                ID         : integer := 1;
                NUM_STAGE  : integer := 2;
                din0_WIDTH : integer := 32;
                din1_WIDTH : integer := 32;
                dout_WIDTH : integer := 1
            );
            port (
                clk    : in  std_logic;
                reset  : in  std_logic;
                ce     : in  std_logic;
                din0   : in  std_logic_vector(din0_WIDTH-1 downto 0);
                din1   : in  std_logic_vector(din1_WIDTH-1 downto 0);
                opcode : in  std_logic_vector(4 downto 0);
                dout   : out std_logic_vector(dout_WIDTH-1 downto 0)
            );
        end component;
    
        signal join_valid : STD_LOGIC;
        constant alu_opcode : std_logic_vector(4 downto 0) := "00110";
    
    begin 
    
        --TODO check with lana
    dataOutArray(0)(DATA_SIZE_OUT - 1 downto 1) <= (others => '0');


    array_RAM_fcmp_32ns_32ns_1_2_1_u1 : component array_RAM_fcmp_32cud 
    generic map (
        ID => 1,
        NUM_STAGE => 2,
        din0_WIDTH => 32,
        din1_WIDTH => 32,
        dout_WIDTH => 1)
    port map (
        clk => clk,
        reset => rst,
        din0 => dataInArray(0),
        din1 => dataInArray(1),
        ce => nReadyArray(0),
        opcode => alu_opcode,
        dout(0) => dataOutArray(0)(0));

    join_write_temp:   entity work.join(arch) generic map(2)
            port map( pValidArray,  --pValidArray
                nReadyArray(0),     --nready                    
                      join_valid,         --valid          
                readyArray);   --readyarray 

    buff: entity work.delay_buffer(arch) 
    generic map(1)
    port map(clk,
             rst,
             join_valid,
             nReadyArray(0),
             validArray(0));

    
    end architecture;
