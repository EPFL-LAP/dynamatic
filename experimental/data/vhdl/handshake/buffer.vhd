--------------------------------------------------------------  TEHB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity TEHB is 
    generic(
        BITWIDTH : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        pValidArray   : in  std_logic_vector(0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(0 downto 0));
end TEHB;

architecture arch of TEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg: std_logic_vector(BITWIDTH-1 downto 0);
begin
    
    process(clk, rst) is

          begin
           if (rst = '1') then
                full_reg <= '0';
              
            elsif (rising_edge(clk)) then
                full_reg <= validArray(0) and not nReadyArray(0);                    
            
            end if;
    end process; 

    process(clk, rst) is

          begin
           if (rst = '1') then
                data_reg <= (others => '0');
              
            elsif (rising_edge(clk)) then
                if (reg_en) then
                    data_reg<= dataInArray(0);  
                end if;                  
            
            end if;
    end process;

    process (mux_sel, data_reg, dataInArray) is
        begin
            if (mux_sel = '1') then
                dataOutArray(0) <= data_reg;
            else
                dataOutArray(0) <= dataInArray(0);
            end if;


    end process;


    validArray(0) <= pValidArray(0) or full_reg;    
    readyArray(0) <= not full_reg;
    reg_en <= readyArray(0) and pValidArray(0) and not nReadyArray(0);
    mux_sel <= full_reg;


end arch;

--------------------------------------------------------------  OEHB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity OEHB is 
    generic(
        BITWIDTH : integer
    );
port (
        clk, rst      : in  std_logic;
        dataInArray   : in  data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        dataOutArray  : out data_array(0 downto 0)(BITWIDTH - 1 downto 0);
        pValidArray   : in  std_logic_vector(0 downto 0);
        nReadyArray   : in  std_logic_vector(0 downto 0);
        validArray    : out std_logic_vector(0 downto 0);
        readyArray    : out std_logic_vector(0 downto 0));
end OEHB;

architecture arch of OEHB is
    signal full_reg, reg_en, mux_sel : std_logic;
    signal data_reg: std_logic_vector(BITWIDTH-1 downto 0);
begin
    
    process(clk, rst) is

          begin
           if (rst = '1') then
                validArray(0) <= '0';
              
            elsif (rising_edge(clk)) then
                validArray(0) <=  pValidArray(0) or not readyArray(0);                   
            
            end if;
    end process; 

    process(clk, rst) is

          begin
           if (rst = '1') then
                data_reg <= (others => '0');
              
            elsif (rising_edge(clk)) then
                if (reg_en) then
                    data_reg<= dataInArray(0);  
                end if;                  
            
            end if;
    end process;


    readyArray(0) <= not validArray(0) or nReadyArray(0);
    reg_en <= readyArray(0) and pValidArray(0);
    dataOutArray(0) <= data_reg;

end arch;

--------------------------------------------------------------  EB
---------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
USE work.customTypes.all;

entity buffer is
Generic (
    BITWIDTH: integer 
);
port(
    clk, rst : in std_logic;  
    dataInArray : in data_array (0 downto 0)(BITWIDTH-1 downto 0);
    dataOutArray : out data_array (0 downto 0)(BITWIDTH-1 downto 0);
    ReadyArray : out std_logic_vector(0 downto 0);
    ValidArray : out std_logic_vector(0 downto 0);
    nReadyArray : in std_logic_vector(0 downto 0);
    pValidArray : in std_logic_vector(0 downto 0));
end buffer;
------------------------------------------------------------------------ 
-- elastic buffer 
------------------------------------------------------------------------ 
architecture arch of buffer is
    
    signal tehb1_valid, tehb1_ready : STD_LOGIC;
    signal oehb1_valid, oehb1_ready : STD_LOGIC;
    signal tehb1_dataOut, oehb1_dataOut : std_logic_vector(BITWIDTH-1 downto 0);

    
begin

tehb1: entity work.TEHB(arch) generic map (BITWIDTH)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => pValidArray(0), -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => oehb1_ready,    
            validArray(0) => tehb1_valid, 
        --outputs
            readyArray(0) => tehb1_ready,   
            dataInArray(0) => dataInArray(0),
            dataOutArray(0) => tehb1_dataOut
        );

oehb1: entity work.OEHB(arch) generic map (BITWIDTH)
        port map (
        --inputspValidArray
            clk => clk, 
            rst => rst, 
            pValidArray(0)  => tehb1_valid, -- real or speculatef condition (determined by merge1)
            nReadyArray(0) => nReadyArray(0),    
            validArray(0) => oehb1_valid, 
        --outputs
            readyArray(0) => oehb1_ready,   
            dataInArray(0) =>tehb1_dataOut,
            dataOutArray(0) => oehb1_dataOut
        );

dataOutArray(0) <= oehb1_dataOut;
ValidArray(0) <= oehb1_valid;
ReadyArray(0) <= tehb1_ready;
    
end arch;