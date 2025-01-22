
set verbose_level 0;
set pp_list cpp;
set counter_examples 0;
set dynamic_reorder 1;
set on_failure_script_quits;
set reorder_method sift;
set enable_sexp2bdd_caching 0;
set bdd_static_order_heuristics basic;
set cone_of_influence;
set use_coi_size_sorting 1;
read_model -i tmp/main.smv;
flatten_hierarchy;
encode_variables;
build_flat_model;
build_model -f;
quit
