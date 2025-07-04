REWRITE_NAME=$1

cd "$(dirname "$0")"

./visualize.sh "elastic_miter_${REWRITE_NAME}_lhs_${REWRITE_NAME}_rhs" "out/${REWRITE_NAME}/miter"
