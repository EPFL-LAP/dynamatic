module single_argument (
    clk,
    rst,
    ce0,
    we0,
    din0,
    dout0,
    dout0_ready,
    dout0_valid,
    done
);


//------------------------Local signal-------------------
parameter TV_IN = "";
parameter TV_OUT = "";
parameter DATA_WIDTH = 32;

// Input and Output
input clk;
input rst;
input ce0;
input we0;
input [DATA_WIDTH-1:0] din0;
output reg [DATA_WIDTH -1 : 0] dout0;
input dout0_ready;
output reg dout0_valid;
input done;

// Inner signals
reg tokenEmitted;
reg [DATA_WIDTH-1:0] mem;
reg memReady;

initial begin
    mem = '0;
    memReady = 1'b0;
end

//------------------------Task and function--------------
task read_token;
    input integer fp;
    output string token;
    integer ret;
    begin
        token = "";
        ret = 0;
        ret = $fscanf(fp,"%s",token);
    end
endtask


//------------------------Read array-------------------
initial begin : file_to_mem
    integer fp;
    string token;
    integer ret;
    reg [ DATA_WIDTH - 1 : 0 ] mem_tmp;
    int transaction_num;

    if (TV_IN != "") begin
        wait (!rst);
        transaction_num = 0;

        fp = $fopen(TV_IN, "r");
        if (fp == 0) begin
            $fatal("ERROR: Could not open file %s", TV_IN);
        end

        // [[[runtime]]]
        read_token(fp, token);
        if (token != "[[[runtime]]]") begin
          $fatal("ERROR: Simulation failed.");
        end

        // Parse transactions
        read_token(fp, token);
          while (token != "[[[/runtime]]]") begin
            if (token != "[[transaction]]") begin
              $display("ERROR: Simulation using HLS TB failed.");
              $finish;
            end

            // discard transaction number
            read_token(fp, token);

            // wait for done
            @(posedge clk);
            wait (done);

            // read data
            read_token(fp, token);
            ret = $sscanf(token, "%x", mem_tmp);
        
            
            mem = mem_tmp;
            memReady = 1'b1;

            if (ret != 1) begin
              $display("Failed to parse token!");
              $finish;
            end

            read_token(fp,token);
            wait (!done);

            // [[/transaction]]
            if(token != "[[/transaction]]") begin
              $display("ERROR: Simulation using HLS TB failed.");
              $finish;
            end
            read_token(fp, token);

            wait (!done);
            transaction_num++;
        end

        $fclose(fp);
    end
end


// Read data from array to RTL
always @ (posedge clk or posedge rst) begin
    if(rst) begin
        tokenEmitted <= 1'b0;
        dout0 <= {DATA_WIDTH{1'b0}};
        dout0_valid <= 1'b0;
    end else begin
	    if(!tokenEmitted && memReady) begin
            tokenEmitted <= 1'b1;
	        dout0 <= mem;
            dout0_valid <= 1'b1;
        end else begin
            dout0_valid <= dout0_valid & ~dout0_ready;
        end
    end
end

//------------------------Write array-------------------

// Write data from RTL to array
always @(posedge clk) begin
    if ((we0 == 1) && (ce0 == 1)) begin
        mem <= din0;
    end
end

// Write data from array to file
initial begin : mem_to_file
    integer fp;
    int transaction_num;
    string token;

    if (TV_OUT != "") begin
        wait (!rst);
        transaction_num = 0;
        // skip first done
        while (!done) @(posedge clk);
        wait (!done);

        while(1) begin
            while (!done) @(posedge clk);

            fp = $fopen(TV_OUT, "a");
            if (fp == 0) begin
                $fatal("ERROR: Could not open file %s", TV_OUT);
            end
            $fdisplay(fp, "[[transaction]] %d", transaction_num);
            $fdisplay(fp,"0x%x",mem);
            $fdisplay(fp, "[[/transaction]]");
            transaction_num++;
            $fclose(fp);
            
            wait (!done);
        end
    end
end

endmodule
