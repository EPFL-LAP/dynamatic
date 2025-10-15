
dynamatic=$(realpath ../../../../..)

cd "$dynamatic"


OUTPUT_DIR=$dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/hdl

# Clean up any existing generated files
echo "Cleaning up existing generated files..."
for file in one_slot_break_dvr.v shift_reg_break_dv.v tfifo_dataless.v tfifo.v; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        rm "$OUTPUT_DIR/$file"
        echo "Removed existing $file"
    fi
done

echo "Generating Verilog modules..."

python $dynamatic/experimental/tools/unit-generators/verilog/verilog-unit-generator.py -n one_slot_break_dvr -o $OUTPUT_DIR/one_slot_break_dvr.v -t one_slot_break_dvr -p data_type=32
python $dynamatic/experimental/tools/unit-generators/verilog/verilog-unit-generator.py -n shift_reg_break_dv -o $OUTPUT_DIR/shift_reg_break_dv.v -t shift_reg_break_dv -p data_type=32 num_slots=2
python $dynamatic/experimental/tools/unit-generators/verilog/verilog-unit-generator.py -n dataless_tfifo -o $OUTPUT_DIR/tfifo_dataless.v -t dataless_tfifo -p data_type=32 num_slots=2
python $dynamatic/experimental/tools/unit-generators/verilog/verilog-unit-generator.py -n tfifo -o $OUTPUT_DIR/tfifo.v -t tfifo -p data_type=32 num_slots=2

$dynamatic/bin/dynamatic << EOF
set-dynamatic-path .
set-src $dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/minimal.c
simulate
EOF


# Check if the generated Verilog files were created successfully
failed=false
echo "Checking generated Verilog files:"
for file in one_slot_break_dvr.v shift_reg_break_dv.v tfifo_dataless.v tfifo.v; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        echo "✓ $file generated successfully"
    else
        echo "✗ $file NOT found"
        failed=true
    fi
done

# Search in simulation report for compilation success
REPORT_FILE=$OUTPUT_DIR/../sim/report.txt
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo "Checking compilation results in $REPORT_FILE:"
    
    # Check each generated file for successful compilation
    for module in tfifo.v tfifo_dataless.v one_slot_break_dvr.v shift_reg_break_dv.v; do
        if grep -q "# Compile of $module was successful." "$REPORT_FILE"; then
            echo "✓ $module compiled successfully"
        else
            echo "✗ $module compilation not found or failed"
            failed=true
        fi
    done
else
    echo "Report file not found: $REPORT_FILE"
    failed=true
fi

echo ""
echo "Generated files are located in: $OUTPUT_DIR"
echo "Simulation report is located in: $REPORT_FILE"

if [ "$failed" = true ]; then
    echo "FAILED"
else
    echo "SUCCESS"
fi