F_MODEL=$1
F_MAIN=$2
F_NUXMV_PROP=$3
F_NUXMV_CMD=$4


echo "set verbose_level 0;
set pp_list cpp;
set counter_examples 0;
set dynamic_reorder 1;
set on_failure_script_quits;
set reorder_method sift;
set enable_sexp2bdd_caching 0;
set bdd_static_order_heuristics basic;
set cone_of_influence;
set use_coi_size_sorting 1;
read_model -i $F_MAIN;
flatten_hierarchy;
encode_variables;
build_flat_model;
build_model -f;
check_invar -s forward;
check_ctlspec;
show_property -o $F_NUXMV_PROP;
time;
quit" > $F_NUXMV_CMD

file=$F_MODEL
total=$(( $(awk '/-- properties/ {found=1; next} found' "$file" | wc -l) + 27))
bar_length=30
i=0

echo "[INFO] Running nuXmv" >&2
nuXmv -source $F_NUXMV_CMD | while IFS= read -r line; do
    ((i++))
    percent=$(( i * 100 / total ))
    filled=$(( i * bar_length / total ))
    bar=$(printf "%-${filled}s" "#" | tr ' ' '#')
    max_line_length=$(( $(tput cols) - bar_length - 20 ))
    display_line="${line:0:max_line_length}"
    printf "\r\033[K[%-${bar_length}s] %3d%% %s" "$bar" "$percent" "$display_line" >&2
done
echo >&2