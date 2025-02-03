library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity spec_save_commit is
  generic (
    DATA_TYPE : integer;
    FIFO_DEPTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- inputs
    ins : in std_logic_vector(DATA_TYPE - 1 downto 0);
    ins_valid : in std_logic;
    ins_spec_tag : in std_logic;
    ctrl : in std_logic_vector(2 downto 0); -- 000:pass, 001:kill, 010:resend, 011:kill-pass, 100:no_cmp
    ctrl_valid : in std_logic;
    ctrl_spec_tag : in std_logic; -- not used
    outs_ready : in std_logic;
    -- outputs
    outs : out std_logic_vector(DATA_TYPE - 1 downto 0);
    outs_valid : out std_logic;
    outs_spec_tag : out std_logic;
    ins_ready : out std_logic;
    ctrl_ready : out std_logic
  );
end spec_save_commit;

architecture arch of spec_save_commit is

    signal HeadEn   : std_logic := '0';
    signal TailEn  : std_logic := '0';
    signal CurrEn  : std_logic := '0';

    signal CurrHeadEqual : std_logic;

    signal PassEn  : std_logic := '0';
    signal KillEn  : std_logic := '0';
    signal ResendEn  : std_logic := '0';
    signal NoCmpEn : std_logic := '0';

    signal Tail : natural range 0 to FIFO_DEPTH - 1;
    signal Head : natural range 0 to FIFO_DEPTH - 1;
    signal Curr : natural range 0 to FIFO_DEPTH - 1;

    signal CurrEmpty    : std_logic;
    signal Empty    : std_logic;
    signal Full : std_logic;

    type FIFO_Memory is array (0 to FIFO_DEPTH - 1) of STD_LOGIC_VECTOR (DATA_TYPE+1 -1 downto 0);
    signal Memory : FIFO_Memory;

    signal specdataInArray  : data_array(0 downto 0)(DATA_TYPE+1 - 1 downto 0);

    signal bypass : std_logic;

begin
    specdataInArray(0) <= ins_spec_tag & ins;

    ins_ready <= not Full;
    outs_valid <= (PassEn and (not CurrEmpty or ins_valid)) or (ResendEn and not Empty);
    --validArray(0) <= (PassEn and not CurrEmpty) or (ResendEn and not Empty);

    CurrHeadEqual <= '1' when Curr = Head else '0';
    TailEn <= not Full and ins_valid;
    HeadEn <= not Empty and ((outs_ready and ResendEn) or (ctrl_ready and KillEn));
    CurrEn <= ( (not CurrEmpty or ins_valid) and (outs_ready and PassEn) ) or
              (not Empty and (outs_ready and NoCmpEn) ) or
              (CurrHeadEqual and ctrl_ready and KillEn);

    bypass <= ins_valid and CurrEmpty;
-------------------
-- comb process for control en
en_proc : process (ins_valid, ctrl_valid, ctrl)
    begin
        PassEn <= '0';
        KillEn <= '0';
        ResendEn <= '0';
        NoCmpEn <= '0';

        if ctrl_valid = '1' and ctrl = "000" then
            PassEn <= '1';
        elsif ctrl_valid = '1' and ctrl = "001" then
            KillEn <= '1';
        elsif ctrl_valid = '1' and ctrl = "010" then
            ResendEn <= '1';
        elsif ctrl_valid = '1' and ctrl = "011" then
            PassEn <= '1';
            KillEn <= '1';
        elsif ctrl_valid = '1' and ctrl = "100" then
            ResendEn <= '1';
            NoCmpEn <= '1';
        end if;
    end process;

-------------------------------------------
-- comb process for control ready
ready_proc : process (PassEn, KillEn, ResendEn, CurrEmpty, Empty, outs_ready, ctrl_valid, ins_valid)
    begin
        -- Note: PassEn and KillEn can be simultaneously '1'
        -- In that case, PassEn is prioritized
        if PassEn = '1' then
            ctrl_ready <= (not CurrEmpty or ins_valid) and outs_ready;
            --ctrl_ready <= not CurrEmpty and outs_ready;
        elsif ResendEn = '1' then
            ctrl_ready <= not Empty and outs_ready;
        elsif KillEn = '1' then
            ctrl_ready <= not Empty;
        else
            ctrl_ready <= '0';
        end if;
    end process;
-------------------------------------------
-- comb process for output data
output_proc : process (PassEn, Memory, Curr, bypass, specdataInArray, Head)
    begin
        if PassEn = '1' then
            if bypass = '1' then
                outs <=  specdataInArray(0)(DATA_TYPE - 1 downto 0);
                outs_spec_tag <= specdataInArray(0)(DATA_TYPE+1 - 1);
            else
                outs <=  Memory(Curr)(DATA_TYPE - 1 downto 0);
                outs_spec_tag <= Memory(Curr)(DATA_TYPE+1 - 1);
            end if;
        else
            outs <=  Memory(Head)(DATA_TYPE - 1 downto 0);
            outs_spec_tag <= '0';
        end if;
    end process;


----------------------------------------------------------------

-- Sequential Process

----------------------------------------------------------------

-------------------------------------------
-- process for writing data
fifo_proc : process (clk)

     begin
        if rising_edge(clk) then
          if rst = '1' then

          else

            if (TailEn = '1' ) then
                -- Write Data to Memory
                Memory(Tail) <= specdataInArray(0);

            end if;

          end if;
        end if;
    end process;



-------------------------------------------
-- process for updating tail
TailUpdate_proc : process (clk)

      begin
        if rising_edge(clk) then

            if rst = '1' then
               Tail <= 0;
            else

                if (TailEn = '1') then

                    Tail  <= (Tail + 1) mod FIFO_DEPTH;

                end if;

            end if;
        end if;
    end process;

-------------------------------------------
-- process for updating head
HeadUpdate_proc : process (clk)

  begin
  if rising_edge(clk) then

    if rst = '1' then
       Head <= 0;
    else

        if (HeadEn = '1') then

            Head  <= (Head + 1) mod FIFO_DEPTH;

        end if;

    end if;
  end if;
end process;

-------------------------------------------
-- process for updating curr
CurrUpdate_proc : process (clk)

  begin
  if rising_edge(clk) then

    if rst = '1' then
       Curr <= 0;
    else

        if (CurrEn = '1') then

            Curr  <= (Curr + 1) mod FIFO_DEPTH;

        end if;

    end if;
  end if;
end process;

-------------------------------------------
-- process for updating full
FullUpdate_proc : process (clk)

  begin
  if rising_edge(clk) then

    if rst = '1' then
       Full <= '0';
    else

        -- if only filling but not emptying
        if (TailEn = '1') and (HeadEn = '0') then

            -- if new tail index will reach head index
            if ((Tail +1) mod FIFO_DEPTH = Head) then

                Full  <= '1';

            end if;
        -- if only emptying but not filling
        elsif (TailEn = '0') and (HeadEn = '1') then
                Full <= '0';
        -- otherwise, nothing is happening or simultaneous read and write

        end if;

    end if;
  end if;
end process;

 -------------------------------------------
-- process for updating empty
EmptyUpdate_proc : process (clk)

  begin
  if rising_edge(clk) then

    if rst = '1' then
       Empty <= '1';
    else
        -- if only emptying but not filling
        if (TailEn = '0') and (HeadEn = '1') then

            -- if new head index will reach tail index
            if ((Head +1) mod FIFO_DEPTH = Tail) then

                Empty  <= '1';

            end if;
        -- if only filling but not emptying
        elsif (TailEn = '1') and (HeadEn = '0') then
                Empty <= '0';
       -- otherwise, nothing is happening or simultaneous read and write

        end if;

    end if;
  end if;
end process;

 -------------------------------------------
-- process for updating curr empty
CurrEmptyUpdate_proc : process (clk)

  begin
  if rising_edge(clk) then

    if rst = '1' then
       CurrEmpty <= '1';
    else
        -- if only emptying but not filling
        if (TailEn = '0') and (CurrEn = '1') then

            -- if new head index will reach tail index
            if ((Curr +1) mod FIFO_DEPTH = Tail) then

                CurrEmpty  <= '1';

            end if;
        -- if only filling but not emptying
        elsif (TailEn = '1') and (CurrEn = '0') then
                CurrEmpty <= '0';
       -- otherwise, nothing is happening or simultaneous read and write

        end if;

    end if;
  end if;
end process;

end architecture;
