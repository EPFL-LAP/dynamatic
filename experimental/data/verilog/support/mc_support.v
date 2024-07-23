
//
//  read_address_mux Module
//

module read_address_mux #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_WIDTH = 32
)(
  input  [ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_WIDTH - 1 : 0] addr_in,
  output [ADDR_WIDTH - 1 : 0] addr_out 
);
  integer i;
  reg [ADDR_WIDTH - 1 : 0] addr_out_var;

  always @(*) begin
    addr_out_var = {ADDR_WIDTH{1'b0}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i * ADDR_WIDTH +: ADDR_WIDTH];
      end
    end
  end

  assign addr_out = addr_out_var;
  
endmodule

//
//  read_address_ready Module
//

module read_address_ready #(
  parameter ARBITER_SIZE = 1
)(
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;
  
endmodule

//
//  read_data_signals Module
//

module read_data_signals #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_WIDTH = 32
)(
  input  rst,
  input  clk,
  input  [ARBITER_SIZE - 1 : 0] sel,
  input  [DATA_WIDTH - 1 : 0] read_data,
  output reg [ARBITER_SIZE * DATA_WIDTH - 1 :0] out_data,
  output reg [ARBITER_SIZE - 1 : 0] valid,
  input  [ARBITER_SIZE - 1 : 0] nReady
);
  reg [ARBITER_SIZE - 1 : 0] sel_prev = 0;
  reg [ARBITER_SIZE * DATA_WIDTH  - 1 : 0] out_reg = 0;

  integer i;

  always @(posedge clk, posedge rst) begin
		if(rst)begin
			valid <= 0;
			sel_prev <= 0;
		end 
		else begin
			sel_prev <= sel;
			for(i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
				if(sel[i])
					valid[i] <= 1;
				else if(nReady[i])
					valid[i] <= 0;
		end
	end

  always @(posedge clk) begin
		for(i = 0; i <= ARBITER_SIZE - 1; i = i + 1)
			if(sel_prev[i])
				out_reg[i * DATA_WIDTH +: DATA_WIDTH] <= read_data;
	end

  always @(*) begin
		for(i = 0; i <= ARBITER_SIZE - 1; i = i + 1)begin
			if(sel_prev[i])
				out_data[i * DATA_WIDTH +: DATA_WIDTH] = read_data;
			else
				out_data[i * DATA_WIDTH +: DATA_WIDTH] = out_reg[i * DATA_WIDTH +: DATA_WIDTH];
		end
	end
  
endmodule

//
//  read_priority Module
//

module read_priority #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE - 1 : 0] req,          
  input  [ARBITER_SIZE - 1 : 0] data_ready,   
  output reg [ARBITER_SIZE - 1 : 0] priority_out 
);
  reg prio_req = 0;
	integer i;

  always@(req, data_ready) begin
		priority_out[0] = req[0] & data_ready[0];
		prio_req = req[0] & data_ready[0];
		
		for(i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
			priority_out[i] = ~prio_req & req[i] & data_ready[i];
			prio_req = prio_req | (req[i] & data_ready[i]);
		end
	end
endmodule

module read_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_WIDTH = 32,
  parameter DATA_WIDTH = 32
) (
  input  clk,
  input  rst,
  // Interface to previous
  input  [ARBITER_SIZE - 1 : 0] pValid,
  output [ARBITER_SIZE - 1 : 0] ready,
  input  [ARBITER_SIZE * ADDR_WIDTH - 1 : 0] address_in,
  // Interface to next
  input  [ARBITER_SIZE - 1 : 0] nReady,
  output [ARBITER_SIZE - 1 : 0] valid,
  output [ARBITER_SIZE * DATA_WIDTH - 1 : 0] data_out,
  // Interface to memory
  output read_enable,
  output [ADDR_WIDTH - 1 : 0] read_address,
  input  [DATA_WIDTH - 1 : 0] data_from_memory
);

  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Instance of read_priority
  read_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) priority (
    .req          (pValid     ),
    .data_ready   (nReady     ),
    .priority_out (priorityOut)
  );

  // Instance of read_address_mux
  read_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE), 
    .ADDR_WIDTH  (ADDR_WIDTH  )
  ) addressing (
    .sel      (priorityOut ),
    .addr_in  (address_in  ),
    .addr_out (read_address)
  );

  // Instance of read_address_ready
  read_address_ready #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) adderssReady (
    .sel    (priorityOut),
    .nReady (nReady     ),
    .ready  (ready      )
  );

  // Instance of read_data_signals
  read_data_signals #(
    .ARBITER_SIZE(ARBITER_SIZE), 
    .DATA_WIDTH(DATA_WIDTH)
  ) data_signals_inst (
    .rst       (rst             ),
    .clk       (clk             ),
    .sel       (priorityOut     ),
    .read_data (data_from_memory),
    .out_data  (data_out        ),
    .valid     (valid           ),
    .nReady    (nReady          )
  );

  assign read_enable = | priorityOut;
  
endmodule


//
//  write_address_mux Module
//

module write_address_mux #(
  parameter ARBITER_SIZE = 1,
  parameter ADDR_WIDTH = 32
)(
  input  [ARBITER_SIZE - 1 : 0] sel,
  input  [ARBITER_SIZE * ADDR_WIDTH - 1 : 0] addr_in,
  output [ADDR_WIDTH - 1 : 0] addr_out 
);
  integer i;
  reg [ADDR_WIDTH - 1 : 0] addr_out_var;

  always @(*) begin
    //! May need to chaneg the following line to remove the assignment to 0
    addr_out_var = {ADDR_WIDTH{1'b0}};

    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      if (sel[i]) begin
        addr_out_var = addr_in[i * ADDR_WIDTH +: ADDR_WIDTH];
      end
    end
  end

  assign addr_out = addr_out_var;
  
endmodule

//
//  write_address_mux Module
//

module write_address_ready #(
  parameter ARBITER_SIZE = 1
)(
  input  [ARBITER_SIZE-1:0] sel,
  input  [ARBITER_SIZE-1:0] nReady,
  output [ARBITER_SIZE-1:0] ready
);
  assign ready = nReady & sel;
  
endmodule

//
//  write_data_signals Module
//

module write_data_signals #(
  parameter ARBITER_SIZE = 1,
  parameter DATA_WIDTH = 32
)(
  input  clk,
  input  rst,
	input  [ARBITER_SIZE - 1 : 0] sel,
	output [DATA_WIDTH - 1 : 0] write_data,
	input  [ARBITER_SIZE * DATA_WIDTH  - 1 : 0] in_data,
	output reg [ARBITER_SIZE - 1 : 0] valid
);
  integer i;
  reg [DATA_WIDTH - 1 : 0] data_out_var;

  always @(*) begin
    data_out_var = {DATA_WIDTH{1'b0}};
    for (i = 0; i < ARBITER_SIZE; i = i + 1) begin
      data_out_var = in_data[i * DATA_WIDTH +: DATA_WIDTH];
    end
  end

  always @(posedge clk, posedge rst) begin
    if (rst) begin
      valid <= 0;
    end else begin
      valid <= sel;
    end
  end

endmodule

//
//  write_priority Module
//

module write_priority #(
  parameter ARBITER_SIZE = 1
) (
  input  [ARBITER_SIZE - 1 : 0] req,          
  input  [ARBITER_SIZE - 1 : 0] data_ready,   
  output reg [ARBITER_SIZE - 1 : 0] priority_out 
);
  reg prio_req = 0;
	integer i;

  always@(req, data_ready) begin
		priority_out[0] = req[0] & data_ready[0];
		prio_req = req[0] & data_ready[0];
		
		for(i = 1; i <= ARBITER_SIZE - 1; i = i + 1) begin
			priority_out[i] = ~prio_req & req[i] & data_ready[i];
			prio_req = prio_req | (req[i] & data_ready[i]);
		end
	end

endmodule

//
//  write_memory_arbiter Module
//

module write_memory_arbiter #(
  parameter ARBITER_SIZE = 2,
  parameter ADDR_WIDTH = 32,
  parameter DATA_WIDTH = 32
) (
  input  rst,
  input  clk,
  // Interface to previous
  input  [ARBITER_SIZE - 1 : 0] pValid,  
  output [ARBITER_SIZE - 1 : 0] ready,   
  input  [ADDR_WIDTH * ARBITER_SIZE - 1 : 0] address_in,  
  input  [DATA_WIDTH * ARBITER_SIZE - 1 : 0] data_in,
  // Interface to next    
  input  [ARBITER_SIZE - 1 : 0] nReady,  
  output [ARBITER_SIZE - 1 : 0] valid,
  // Interface to memory   
  output write_enable,               
  output enable,                     
  output [ADDR_WIDTH - 1 : 0] write_address,  
  output [DATA_WIDTH - 1 : 0] data_to_memory  
);
  wire [ARBITER_SIZE - 1 : 0] priorityOut;

  // Priority handling
  write_priority #(
    .ARBITER_SIZE(ARBITER_SIZE)
  ) priority (
    .req          (pValid     ),
    .data_ready   (nReady     ),
    .priority_out (priorityOut)
  );

  // Address multiplexing
  write_address_mux #(
    .ARBITER_SIZE(ARBITER_SIZE),
    .ADDR_WIDTH  (ADDR_WIDTH  ) 
  ) addressing (
    .sel      (priorityOut  ),
    .addr_in  (address_in   ),
    .addr_out (write_address)
  );

    // Ready signal handling
    write_address_ready #(
      .ARBITER_SIZE(ARBITER_SIZE)
    ) addressReady (
      .sel    (priorityOut),
      .nReady (nReady     ),
      .ready  (ready      )
    );

    // Data handling
    write_data_signals #(
        .ARBITER_SIZE (ARBITER_SIZE),
        .DATA_WIDTH   (DATA_WIDTH  )
    ) data_signals_inst (
        .rst        (rst           ),
        .clk        (clk           ),
        .sel        (priorityOut   ),
        .in_data    (data_in       ),
        .write_data (data_to_memory),
        .valid      (valid         )
    );

    assign write_enable = | priorityOut;
    assign enable = | priorityOut;
    
endmodule
