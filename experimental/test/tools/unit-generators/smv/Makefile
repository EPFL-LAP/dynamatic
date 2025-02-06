GENPATH ?= "../../../../tools/unit-generators/smv/"
NUXMVPAT ?= "../../../../../../nuXmv-2.0.0-Linux/bin/"
NUXMV ?= 1

ifeq ($(NUXMV), 0)
	OUT = 1> /dev/null
else
	OUT = 1> ./tmp/module.smv
endif

define run_nuxmv
if [ "$(NUXMV)" = "1" ]; then \
	python testbench-generator.py $(1) > ./tmp/main.smv; \
	$(NUXMVPAT)nuXmv -source prove.cmd 1> /dev/null; \
fi
endef


test: handshake arith

arith:

handshake: br buffer cond_br constant control_merge fork join lazy_fork load merge mux select sink source store

br:
	@echo "Testing br..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t br -p data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 2)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t br -p data_type="\"!handshake.channel<i1>\"" $(OUT)
	@$(call run_nuxmv, 3)

buffer:
	@echo "Testing buffer..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@$(call run_nuxmv, 2)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type="\"!handshake.channel<i1>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@$(call run_nuxmv, 3)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@$(call run_nuxmv, 2)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type="\"!handshake.channel<i1>\"" timing="\"#handshake.timing< {R: 1}>\"" $(OUT)
	@$(call run_nuxmv, 3)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@$(call run_nuxmv, 2)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=1 data_type="\"!handshake.channel<i1>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@$(call run_nuxmv, 3)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type="\"!handshake.control<>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@$(call run_nuxmv, 2)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t buffer -p slots=5 data_type="\"!handshake.channel<i1>\"" timing="\"#handshake.timing< {R: 0}>\"" $(OUT)
	@$(call run_nuxmv, 3)

cond_br:
	@echo "Testing cond_br..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t cond_br -p data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 5)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t cond_br -p data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 6)

constant:
	@echo "Testing constant..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t constant -p value=42 data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 2)

control_merge:
	@echo "Testing control_merge..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t control_merge -p size=2 index_type="\"!handshake.channel<i1>\"" data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 4)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t control_merge -p size=2 index_type="\"!handshake.channel<i1>\"" data_type="\"!handshake.channel<i1>\"" $(OUT)
	@$(call run_nuxmv, 6)

fork:
	@echo "Testing fork..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t fork -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 5)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t fork -p size=2 data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 4)

join:
	@echo "Testing join..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t join -p size=3 $(OUT)
	@$(call run_nuxmv, 4)

lazy_fork:
	@echo "Testing lazy_fork..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t lazy_fork -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 5)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t lazy_fork -p size=2 data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 4)

load:
	@echo "Testing load..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t load -p data_type="\"!handshake.channel<i1>\"" addr_type="\"!handshake.channel<i1>\"" $(OUT)
	@$(call run_nuxmv, 6)

merge:
	@echo "Testing merge..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t merge -p size=4 data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 5)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t merge -p size=2 data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 5)

mux:
	@echo "Testing mux..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t mux -p size=2 select_type="\"!handshake.channel<i1>\"" data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 5)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t mux -p size=2 select_type="\"!handshake.channel<i1>\"" data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 7)

select:
	@echo "Testing select..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t select -p data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 7)

sink:
	@echo "Testing sink..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t sink -p data_type="\"!handshake.control<>\"" $(OUT)
	@$(call run_nuxmv, 1)
	@python $(GENPATH)smv-unit-generator.py -n test_module -t sink -p data_type="\"!handshake.channel<i1>\"" $(OUT)	
	@$(call run_nuxmv, 2)

source:
	@echo "Testing source..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t source $(OUT)
	@$(call run_nuxmv, 1)

store:
	@echo "Testing store..."
	@python $(GENPATH)smv-unit-generator.py -n test_module -t store -p data_type="\"!handshake.channel<i1>\"" addr_type="\"!handshake.channel<i1>\"" $(OUT)
	@$(call run_nuxmv, 6)


clean:
	rm tmp/*