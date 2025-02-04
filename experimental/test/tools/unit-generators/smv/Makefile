GENPATH ?= "../../../../tools/unit-generators/smv/"
VERBOSE ?= 0

ifeq ($(VERBOSE), 0)
	OUT = 1> /dev/null
else
	OUT = 
endif


test: handshake arith

arith:

handshake: br buffer cond_br constant control_merge fork join lazy_fork load merge mux sink source store

br:
	@echo "Generating br..."
	@python $(GENPATH)smv-unit-generator.py -n test_br_dataless -t br -p data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_br -t br -p data_type="\"!handshake.channel<i32>\"" $(OUT)

buffer:
	@echo "Generating buffer..."
	@python $(GENPATH)smv-unit-generator.py -n test_tehb_dataless -t buffer -p slots=1 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_tehb -t buffer -p slots=1 data_type="\"!handshake.channel<ui16>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_tfifo_dataless -t buffer -p slots=5 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_tfifo -t buffer -p slots=5 data_type="\"!handshake.channel<ui16>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_oehb_dataless -t buffer -p slots=1 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_oehb -t buffer -p slots=1 data_type="\"!handshake.channel<ui16>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_ofifo_dataless -t buffer -p slots=5 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_ofifo -t buffer -p slots=5 data_type="\"!handshake.channel<ui16>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)

cond_br:
	@echo "Generating cond_br..."
	@python $(GENPATH)smv-unit-generator.py -n test_cond_br_dataless -t cond_br -p data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_cond_br -t cond_br -p data_type="\"!handshake.channel<i32>\"" $(OUT)	

constant:
	@echo "Generating constant..."
	@python $(GENPATH)smv-unit-generator.py -n test_constant -t constant -p value=42 data_type="\"!handshake.control<>\"" $(OUT)

control_merge:
	@echo "Generating control_merge..."
	@python $(GENPATH)smv-unit-generator.py -n test_control_merge_dataless -t control_merge -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_control_merge -t control_merge -p size=2 data_type="\"!handshake.channel<i32>\"" $(OUT)	

fork:
	@echo "Generating fork..."
	@python $(GENPATH)smv-unit-generator.py -n test_fork_dataless -t fork -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_fork -t fork -p size=2 data_type="\"!handshake.channel<i32>\"" $(OUT)	

join:
	@echo "Generating join..."
	@python $(GENPATH)smv-unit-generator.py -n test_join -t join -p size=3 $(OUT)

lazy_fork:
	@echo "Generating lazy_fork..."
	@python $(GENPATH)smv-unit-generator.py -n test_lazy_fork_dataless -t lazy_fork -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_lazy_fork -t lazy_fork -p size=2 data_type="\"!handshake.channel<i32>\"" $(OUT)	

load:
	@echo "Generating load..."
	@python $(GENPATH)smv-unit-generator.py -n test_load -t load -p data_type="\"!handshake.channel<i16>\"" addr_type="\"!handshake.channel<i32>\"" $(OUT)

merge:
	@echo "Generating merge..."
	@python $(GENPATH)smv-unit-generator.py -n test_merge -t merge -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_merge -t merge -p size=2 data_type="\"!handshake.channel<i32>\"" $(OUT)	

mux:
	@echo "Generating mux..."
	@python $(GENPATH)smv-unit-generator.py -n test_mux -t mux -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_mux -t mux -p size=2 data_type="\"!handshake.channel<i32>\"" $(OUT)	

sink:
	@echo "Generating sink..."
	@python $(GENPATH)smv-unit-generator.py -n test_sink -t sink -p data_type="\"!handshake.control<>\"" $(OUT)
	@python $(GENPATH)smv-unit-generator.py -n test_sink -t sink -p data_type="\"!handshake.channel<i32>\"" $(OUT)	

source:
	@echo "Generating source..."
	@python $(GENPATH)smv-unit-generator.py -n test_source -t source $(OUT)

store:
	@echo "Generating store..."
	@python $(GENPATH)smv-unit-generator.py -n test_store -t store -p data_type="\"!handshake.channel<i16>\"" addr_type="\"!handshake.channel<i32>\"" $(OUT)
