/**************************************************************************/
// MODULE: Cfu
// FILE NAME: Cfu.v
// VERSRION: 1.0
// DATE: JAN 08, 2024
// AUTHOR: Kuan-Wei Chen, NYCU IEE
// CODE TYPE: RTL or Behavioral Level (Verilog)
// DESCRIPTION: 2024 FALL AAML / Final Project
// MODIFICATION HISTORY:
// Date                 Description
// 
/**************************************************************************/
module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output              rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
);

//==============================================//
//       parameter & integer declaration        //
//==============================================//
// Parameters
localparam DIM_M = 'd4;
localparam DIM_N = 'd4;

// Op codes
localparam OP_0  = 7'd0;  // Reset
localparam OP_1  = 7'd1;  // Write Global Buffer A
localparam OP_2  = 7'd2;  // Write Global Buffer B
localparam OP_3  = 7'd3;  // Start
localparam OP_4  = 7'd4;  // Respond out_valid
localparam OP_5  = 7'd5;  // Read C_buffer1[127:96]
localparam OP_6  = 7'd6;  // Read C_buffer1[95:64];
localparam OP_7  = 7'd7;  // Read C_buffer1[63:32];
localparam OP_8  = 7'd8;  // Read C_buffer1[31:0];
localparam OP_9  = 7'd9;  // Read C_buffer2[127:96]
localparam OP_10 = 7'd10; // Read C_buffer2[95:64];
localparam OP_11 = 7'd11; // Read C_buffer2[63:32]; 
localparam OP_12 = 7'd12; // Read C_buffer2[31:0]; 
localparam OP_13 = 7'd13; // Read C_buffer3[127:96]; 
localparam OP_14 = 7'd14; // Read C_buffer3[95:64];
localparam OP_15 = 7'd15; // Read C_buffer3[63:32]; 
localparam OP_16 = 7'd16; // Read C_buffer3[31:0];
localparam OP_17 = 7'd17; // Read C_buffer4[127:96]
localparam OP_18 = 7'd18; // Read C_buffer4[95:64];
localparam OP_19 = 7'd19; // Read C_buffer4[63:32]; 
localparam OP_20 = 7'd20; // Read C_buffer4[31:0];
localparam OP_21 = 7'd21; // Invalid, Set Dimension K

//==============================================//
//           reg & wire declaration             //
//==============================================//
reg  TPU_start;
wire  TPU_in_valid;

reg  [9:0] DIM_K_ff;
wire [9:0] DIM_K;

wire rst_n;

wire A_write;
wire B_write;
wire start;

wire [6:0] op_code;

wire         A_wr_en;
wire [15:0]  A_index;
wire [31:0]  A_data_in;
wire [31:0]  A_data_out;

wire         B_wr_en;
wire [15:0]  B_index;
wire [31:0]  B_data_in;
wire [31:0]  B_data_out;

wire         TPU_A_wr_en;
wire [15:0]  TPU_A_index;
wire [31:0]  TPU_A_data_in;
wire [31:0]  TPU_A_data_out;

wire         TPU_B_wr_en;
wire [15:0]  TPU_B_index;
wire [31:0]  TPU_B_data_in;
wire [31:0]  TPU_B_data_out;

wire [127:0] TPU_C_buffer1;
wire [127:0] TPU_C_buffer2;
wire [127:0] TPU_C_buffer3;
wire [127:0] TPU_C_buffer4;

wire TPU_busy;


//==============================================//
//                Global Buffer                 //
//==============================================//
global_buffer #(.ADDR_BITS(10), .DATA_BITS(32), .DEPTH(576)) gbuff_A(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
);

global_buffer #(.ADDR_BITS(10), .DATA_BITS(32), .DEPTH(576)) gbuff_B(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out)
);


//==============================================//
//                    TPU                       //
//==============================================//
TPU My_TPU(
.clk            (clk),     
.rst_n          (rst_n),     
.in_valid       (TPU_in_valid),
.start          (TPU_start),      
.K              (DIM_K), 
.M              (DIM_M), 
.N              (DIM_N),
.busy           (TPU_busy),     
.A_wr_en        (TPU_A_wr_en),         
.A_index        (TPU_A_index),         
.A_data_in      (TPU_A_data_in),         
.A_data_out     (TPU_A_data_out),         
.B_wr_en        (TPU_B_wr_en),         
.B_index        (TPU_B_index),         
.B_data_in      (TPU_B_data_in),         
.B_data_out     (TPU_B_data_out),
.C_buffer1      (TPU_C_buffer1),       
.C_buffer2      (TPU_C_buffer2),       
.C_buffer3      (TPU_C_buffer3),      
.C_buffer4      (TPU_C_buffer4)
);

//==============================================//
//                  design                      //
//==============================================//
assign rst_n = ~reset;

assign rsp_valid = cmd_valid;
assign cmd_ready = rsp_ready;

assign op_code = cmd_payload_function_id[9:3];

assign TPU_A_data_out = A_data_out;
assign TPU_B_data_out = B_data_out;

assign      A_write = cmd_valid & cmd_ready & op_code == OP_1;
assign      B_write = cmd_valid & cmd_ready & op_code == OP_2;
assign TPU_in_valid = cmd_valid & cmd_ready & op_code == OP_21;
assign        start = cmd_valid & cmd_ready & op_code == OP_3;


assign A_wr_en   = (A_write) ? 'b1 : TPU_A_wr_en;
assign A_index   = (A_write) ? cmd_payload_inputs_0 : TPU_A_index;
assign A_data_in = (A_write) ? cmd_payload_inputs_1 : TPU_A_data_in;

assign B_wr_en   = (B_write) ? 'b1 : TPU_B_wr_en;
assign B_index   = (B_write) ? cmd_payload_inputs_0 : TPU_B_index;
assign B_data_in = (B_write) ? cmd_payload_inputs_1 : TPU_B_data_in;

assign DIM_K     = (TPU_in_valid) ? cmd_payload_inputs_0 : DIM_K_ff;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)             DIM_K_ff <= 'b0;
    else if (TPU_in_valid) DIM_K_ff <= cmd_payload_inputs_0;;
end

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)      TPU_start <= 'b0;
    else if (start) TPU_start <= 'b1;
    else            TPU_start <= 'b0;
end


always @(*) begin
    if(cmd_valid & cmd_ready) begin
        if(rsp_valid & rsp_ready) begin
            case(op_code)
                OP_4:     rsp_payload_outputs_0 = TPU_busy;

                OP_5:     rsp_payload_outputs_0 = TPU_C_buffer1[127:96];
                OP_6:     rsp_payload_outputs_0 = TPU_C_buffer1[95:64];
                OP_7:     rsp_payload_outputs_0 = TPU_C_buffer1[63:32];
                OP_8:     rsp_payload_outputs_0 = TPU_C_buffer1[31:0];

                OP_9:     rsp_payload_outputs_0 = TPU_C_buffer2[127:96];
                OP_10:    rsp_payload_outputs_0 = TPU_C_buffer2[95:64];
                OP_11:    rsp_payload_outputs_0 = TPU_C_buffer2[63:32];
                OP_12:    rsp_payload_outputs_0 = TPU_C_buffer2[31:0];

                OP_13:    rsp_payload_outputs_0 = TPU_C_buffer3[127:96];
                OP_14:    rsp_payload_outputs_0 = TPU_C_buffer3[95:64];
                OP_15:    rsp_payload_outputs_0 = TPU_C_buffer3[63:32];
                OP_16:    rsp_payload_outputs_0 = TPU_C_buffer3[31:0];

                OP_17:    rsp_payload_outputs_0 = TPU_C_buffer4[127:96];
                OP_18:    rsp_payload_outputs_0 = TPU_C_buffer4[95:64];
                OP_19:    rsp_payload_outputs_0 = TPU_C_buffer4[63:32];
                OP_20:    rsp_payload_outputs_0 = TPU_C_buffer4[31:0];

                default: rsp_payload_outputs_0 = 'd0;
            endcase
        end
        else rsp_payload_outputs_0 = 'd0;
    end
    else     rsp_payload_outputs_0 = 'd0;
end

endmodule



// ===========================================
// Global Buffer
// ===========================================
module global_buffer #(parameter ADDR_BITS=16, parameter DATA_BITS=8, parameter DEPTH=25)(clk, rst_n, wr_en, index, data_in, data_out);
input clk;
input rst_n;
input wr_en; // 1: write, 0:read
input      [ADDR_BITS-1:0] index;
input      [DATA_BITS-1:0] data_in;
output reg [DATA_BITS-1:0] data_out;

integer i;

//parameter DEPTH = 2**ADDR_BITS;

reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

always @ (posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        for(i=0; i<(DEPTH); i=i+1)
            gbuff[i] <= 'd0;
    end
    else begin
        if(wr_en) gbuff[index] <= data_in;
        else          data_out <= gbuff[index];
    end
  end
endmodule


// ===========================================
// Tensor Process Unit
// ===========================================
module TPU(
    clk,
    rst_n,

    in_valid,
    start,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_buffer1,
    C_buffer2,
    C_buffer3,
    C_buffer4
);

input clk;
input rst_n;
input in_valid;
input start;
input [9:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output reg  [127:0]   C_buffer1;
output reg  [127:0]   C_buffer2;
output reg  [127:0]   C_buffer3;
output reg  [127:0]   C_buffer4;

// Implement your design here
//==============================================//
//       parameter & integer declaration        //
//==============================================//
// bit Width
localparam SA_SIZE        =   4;
localparam DIM_WIDTH      =   8;
localparam DATA_WIDTH     =   8;
localparam PE_OUT_WIDTH   =  32;

// dimension
localparam DIM_M = 'd4;
localparam DIM_N = 'd4;

// states
localparam S_IDLE       = 2'd0;
localparam S_CAL        = 2'd1;
localparam S_WRITE_BACK = 2'd2;

// control signals
localparam C_CLR        = 1'b0;
localparam C_CAL        = 1'b1;

integer i, j;

//==============================================//
//           reg & wire declaration             //
//==============================================//
wire [7:0]   A_net  [0:SA_SIZE-1];
wire [7:0]   B_net  [0:SA_SIZE-1];
wire [32:0]  C_net  [0:SA_SIZE-1];

wire [DATA_WIDTH-1:0]   PE_out_1 [0:SA_SIZE-1][0:SA_SIZE-1];
wire [DATA_WIDTH-1:0]   PE_out_2 [0:SA_SIZE-1][0:SA_SIZE-1];
wire [PE_OUT_WIDTH-1:0] PE_p_sum [0:SA_SIZE-1][0:SA_SIZE-1];

// K, M, N
reg [9:0] K_ff;
reg [DIM_WIDTH-1:0] M_ff, N_ff;


// Buffer A
reg [DATA_WIDTH-1:0] A_buf0;
reg [DATA_WIDTH-1:0] A_buf1 [0:1];
reg [DATA_WIDTH-1:0] A_buf2 [0:2];
reg [DATA_WIDTH-1:0] A_buf3 [0:3];

// Buffer B
reg [DATA_WIDTH-1:0] B_buf0;
reg [DATA_WIDTH-1:0] B_buf1 [0:1];
reg [DATA_WIDTH-1:0] B_buf2 [0:2];
reg [DATA_WIDTH-1:0] B_buf3 [0:3];

// Buffer C
reg signed [PE_OUT_WIDTH-1:0] C_buf [0:SA_SIZE-1][0:SA_SIZE-1];

// State
reg [1:0] cs, ns;

// PE Countrol
reg ctrl;

// Counter
reg [9:0]  cal_cnt;
reg [1:0]  wr_cnt;

// cal_cycles
wire [9:0] cal_cycles;

// A_addr, B_addr
reg [15:0] A_addr, B_addr;

//==============================================//
//                 submodule                    //
//==============================================//
// PE Control
always @(*) begin
    case(cs)
        S_CAL:   ctrl = C_CAL;
        default: ctrl = C_CLR;
    endcase
end

PE PE00(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(A_buf0),         .in_2(B_buf0),         .out_1(PE_out_1[0][0]), .out_2(PE_out_2[0][0]), .p_sum(PE_p_sum[0][0]));
PE PE01(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[0][0]), .in_2(B_buf1[1]),      .out_1(PE_out_1[0][1]), .out_2(PE_out_2[0][1]), .p_sum(PE_p_sum[0][1]));
PE PE02(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[0][1]), .in_2(B_buf2[2]),      .out_1(PE_out_1[0][2]), .out_2(PE_out_2[0][2]), .p_sum(PE_p_sum[0][2]));
PE PE03(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[0][2]), .in_2(B_buf3[3]),      .out_1(PE_out_1[0][3]), .out_2(PE_out_2[0][3]), .p_sum(PE_p_sum[0][3]));

PE PE10(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(A_buf1[1]),      .in_2(PE_out_2[0][0]), .out_1(PE_out_1[1][0]), .out_2(PE_out_2[1][0]), .p_sum(PE_p_sum[1][0]));
PE PE11(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[1][0]), .in_2(PE_out_2[0][1]), .out_1(PE_out_1[1][1]), .out_2(PE_out_2[1][1]), .p_sum(PE_p_sum[1][1]));
PE PE12(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[1][1]), .in_2(PE_out_2[0][2]), .out_1(PE_out_1[1][2]), .out_2(PE_out_2[1][2]), .p_sum(PE_p_sum[1][2]));
PE PE13(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[1][2]), .in_2(PE_out_2[0][3]), .out_1(PE_out_1[1][3]), .out_2(PE_out_2[1][3]), .p_sum(PE_p_sum[1][3]));

PE PE20(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(A_buf2[2]),      .in_2(PE_out_2[1][0]), .out_1(PE_out_1[2][0]), .out_2(PE_out_2[2][0]), .p_sum(PE_p_sum[2][0]));
PE PE21(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[2][0]), .in_2(PE_out_2[1][1]), .out_1(PE_out_1[2][1]), .out_2(PE_out_2[2][1]), .p_sum(PE_p_sum[2][1]));
PE PE22(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[2][1]), .in_2(PE_out_2[1][2]), .out_1(PE_out_1[2][2]), .out_2(PE_out_2[2][2]), .p_sum(PE_p_sum[2][2]));
PE PE23(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[2][2]), .in_2(PE_out_2[1][3]), .out_1(PE_out_1[2][3]), .out_2(PE_out_2[2][3]), .p_sum(PE_p_sum[2][3]));

PE PE30(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(A_buf3[3]),      .in_2(PE_out_2[2][0]), .out_1(PE_out_1[3][0]), .out_2(PE_out_2[3][0]), .p_sum(PE_p_sum[3][0]));
PE PE31(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[3][0]), .in_2(PE_out_2[2][1]), .out_1(PE_out_1[3][1]), .out_2(PE_out_2[3][1]), .p_sum(PE_p_sum[3][1]));
PE PE32(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[3][1]), .in_2(PE_out_2[2][2]), .out_1(PE_out_1[3][2]), .out_2(PE_out_2[3][2]), .p_sum(PE_p_sum[3][2]));
PE PE33(.clk(clk), .rst_n(rst_n), .ctrl(ctrl), .in_1(PE_out_1[3][2]), .in_2(PE_out_2[2][3]), .out_1(PE_out_1[3][3]), .out_2(PE_out_2[3][3]), .p_sum(PE_p_sum[3][3]));

//==============================================//
//                  design                      //
//==============================================//
// Current State
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) cs <= S_IDLE;
    else       cs <= ns;
end

// Next State
always @(*) begin
    case(cs)
        S_IDLE:               ns =                 (start) ?        S_CAL : S_IDLE;
        S_CAL:                ns = (cal_cnt == cal_cycles) ? S_WRITE_BACK : S_CAL;
        S_WRITE_BACK:         ns =         (wr_cnt == 'd3) ?       S_IDLE : S_WRITE_BACK;
        default:              ns = S_IDLE;
    endcase
end

// cal_cycles
assign cal_cycles = K_ff + 'd8;

// K, M, N
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        K_ff <= 'd0;
        M_ff <= 'd0;
        N_ff <= 'd0;
    end
    else begin
        if(in_valid) begin
            K_ff <= K;
            M_ff <= M;
            N_ff <= N;
        end
    end
end

// cal_cnt
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) cal_cnt <= 'd0;
    else begin
        case(cs)
            S_CAL:   cal_cnt <= cal_cnt + 'd1;
            default: cal_cnt <= 'd0;
        endcase
    end
end

// wr_cnt
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) wr_cnt <= 'd0;
    else begin
        case(cs)
            S_WRITE_BACK: wr_cnt <= wr_cnt + 'd1;
            default:      wr_cnt <= 'd0;
        endcase
    end
end

/* SRAM Control */
// A_addr
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) A_addr <= 'd0;
    else begin
        case(cs)
            S_IDLE:  A_addr <= (start) ? A_addr + 'd1 : 'd0;
            S_CAL:   A_addr <= (cal_cnt > K_ff) ? A_addr : A_addr + 'd1;
        endcase
    end
end

// B_addr
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) B_addr <= 'd0;
    else begin
        case(cs)
            S_IDLE:  B_addr <= (start) ? B_addr + 'd1 : 'd0;
            S_CAL:   B_addr <= (cal_cnt > K_ff) ? B_addr : B_addr + 'd1;
        endcase
    end
end

// ===========================================
// Read SRAM A, B
// ===========================================
// SRAM A control
assign A_wr_en   = 'd0;
assign A_data_in = 'd0;
assign A_index   = A_addr;

// SRAM B control
assign B_wr_en   = 'd0;
assign B_data_in = 'd0;
assign B_index   = B_addr;


// A_net
assign A_net[0] = A_data_out[31:24];
assign A_net[1] = A_data_out[23:16];
assign A_net[2] = A_data_out[15:8];
assign A_net[3] = A_data_out[7:0];

// B_net
assign B_net[0] = B_data_out[31:24];
assign B_net[1] = B_data_out[23:16];
assign B_net[2] = B_data_out[15:8];
assign B_net[3] = B_data_out[7:0];


// A Buffer
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
       A_buf0 <= 'd0;
       for(i = 0; i < 2; i = i + 1) A_buf1[i] <= 'd0;
       for(i = 0; i < 3; i = i + 1) A_buf2[i] <= 'd0;
       for(i = 0; i < 4; i = i + 1) A_buf3[i] <= 'd0;
    end
    else begin
        case(cs)
            S_IDLE: begin
                A_buf0 <= 'd0;
               for(i = 0; i < 2; i = i + 1) A_buf1[i] <= 'd0;
               for(i = 0; i < 3; i = i + 1) A_buf2[i] <= 'd0;
               for(i = 0; i < 4; i = i + 1) A_buf3[i] <= 'd0;
            end
            S_CAL: begin
                if(cal_cnt < K_ff) begin
                    A_buf0    <= A_net[0];
                    A_buf1[0] <= A_net[1];
                    A_buf2[0] <= A_net[2];
                    A_buf3[0] <= A_net[3];
                end
                else begin
                    A_buf0    <= 'd0;
                    A_buf1[0] <= 'd0;
                    A_buf2[0] <= 'd0;
                    A_buf3[0] <= 'd0;
                end

                A_buf1[1] <= A_buf1[0]; 
                
                A_buf2[1] <= A_buf2[0];
                A_buf2[2] <= A_buf2[1];

                A_buf3[1] <= A_buf3[0];
                A_buf3[2] <= A_buf3[1];
                A_buf3[3] <= A_buf3[2]; 
            end

        endcase
    end
end

// B Buffer
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
       B_buf0 <= 'd0;
       for(i = 0; i < 2; i = i + 1) B_buf1[i] <= 'd0;
       for(i = 0; i < 3; i = i + 1) B_buf2[i] <= 'd0;
       for(i = 0; i < 4; i = i + 1) B_buf3[i] <= 'd0;
    end
    else begin
        case(cs)
            S_IDLE: begin
                B_buf0 <= 'd0;
               for(i = 0; i < 2; i = i + 1) B_buf1[i] <= 'd0;
               for(i = 0; i < 3; i = i + 1) B_buf2[i] <= 'd0;
               for(i = 0; i < 4; i = i + 1) B_buf3[i] <= 'd0;
            end
            S_CAL: begin
                if(cal_cnt < K_ff) begin
                    B_buf0    <= B_net[0];
                    B_buf1[0] <= B_net[1];
                    B_buf2[0] <= B_net[2];
                    B_buf3[0] <= B_net[3];
                end
                else begin
                    B_buf0    <= 'd0;
                    B_buf1[0] <= 'd0;
                    B_buf2[0] <= 'd0;
                    B_buf3[0] <= 'd0;
                end

                B_buf1[1] <= B_buf1[0]; 
                
                B_buf2[1] <= B_buf2[0];
                B_buf2[2] <= B_buf2[1];

                B_buf3[1] <= B_buf3[0];
                B_buf3[2] <= B_buf3[1];
                B_buf3[3] <= B_buf3[2]; 
            end
        endcase
    end
end

// C Buffer
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        for(i = 0; i < SA_SIZE; i = i + 1)
            for(j = 0; j < SA_SIZE; j = j + 1)
                C_buf[i][j] <= 'd0;
    end
    else begin
        case(cs)
            S_IDLE: begin
                for(i = 0; i < SA_SIZE; i = i + 1)
                    for(j = 0; j < SA_SIZE; j = j + 1)
                        C_buf[i][j] <= 'd0;
            end
            S_CAL: begin
                for(i = 0; i < SA_SIZE; i = i + 1)
                    for(j = 0; j < SA_SIZE; j = j + 1)
                        C_buf[i][j] <= PE_p_sum[i][j];
            end
        endcase
    end
end


// C Buffer
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        C_buffer1 <= 'd0;
        C_buffer2 <= 'd0;
        C_buffer3 <= 'd0;
        C_buffer4 <= 'd0;
    end
    else begin
        case(cs)
            S_CAL: begin
                C_buffer1 <= 'd0;
                C_buffer2 <= 'd0;
                C_buffer3 <= 'd0;
                C_buffer4 <= 'd0;
            end
            S_WRITE_BACK: begin
                C_buffer1 <= {C_buf[0][0], C_buf[0][1], C_buf[0][2], C_buf[0][3]};
                C_buffer2 <= {C_buf[1][0], C_buf[1][1], C_buf[1][2], C_buf[1][3]};
                C_buffer3 <= {C_buf[2][0], C_buf[2][1], C_buf[2][2], C_buf[2][3]};
                C_buffer4 <= {C_buf[3][0], C_buf[3][1], C_buf[3][2], C_buf[3][3]};
            end
        endcase
    end
end

/* busy */
always @(posedge clk or negedge rst_n) begin
    if(!rst_n)            busy <= 'b1;
    else if(start)        busy <= 'b1;
    else if(cs == S_IDLE) busy <= 'b0;
end

endmodule


// ===========================================
// Processing Element
// ===========================================
module PE #(parameter PE_IN_WIDTH=8, parameter PE_OUT_WIDTH=32)(
    clk,
    rst_n,
    ctrl,
    in_1,
    in_2,
    out_1,
    out_2,
    p_sum
);

input  clk;
input  rst_n;
input  ctrl;
input  signed [PE_IN_WIDTH-1:0]  in_1;
input  signed [PE_IN_WIDTH-1:0]  in_2;
output signed [PE_IN_WIDTH-1:0]  out_1;
output signed [PE_IN_WIDTH-1:0]  out_2;
output signed [PE_OUT_WIDTH-1:0] p_sum;

reg signed [PE_IN_WIDTH-1:0]  in_1_ff, in_2_ff;
reg signed [PE_OUT_WIDTH-1:0] p_sum_ff;

// input offset
localparam InputOffset = $signed(9'd128);

// control signals
localparam C_CLR  = 2'd0;
localparam C_CAL  = 2'd1;

assign out_1 = in_1_ff;
assign out_2 = in_2_ff;
assign p_sum = p_sum_ff;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        in_1_ff <= 'd0;
        in_2_ff <= 'd0;
    end
    else begin
        case(ctrl)
            C_CAL: begin
                in_1_ff <= in_1;
                in_2_ff <= in_2;
            end
            default: begin
                in_1_ff <= 'd0;
                in_2_ff <= 'd0;
            end
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)  p_sum_ff <= 'd0;
    else begin
        case(ctrl)
            C_CAL:   p_sum_ff <= (in_1_ff + InputOffset) * in_2_ff + p_sum_ff;
            default: p_sum_ff <= 'd0;
        endcase
    end
end

endmodule
