#include "LSQNoDepTypeSystem.h"

dynamatic::gen::TransferFnArray<dynamatic::ast::ArrayReadExpression> dynamatic::
    gen::detail::LSQNoDepTypeSystemInner::getArrayReadExpressionTransferFns() {
  return {
      copyFromInput<ast::ArrayReadExpression>(),
      TransferFn<ast::ArrayReadExpression, INPUT_DEPENDENCY>(
          [](LSQNoDepContext context) {
            context.inArrayReadIndexExpression = true;
            return context;
          }),
      copyInputToOutput<ast::ArrayReadExpression>(),
  };
}

dynamatic::gen::TransferFnArray<dynamatic::ast::ArrayAssignmentStatement>
dynamatic::gen::detail::LSQNoDepTypeSystemInner::
    getArrayAssignmentStatementTransferFns() {
  return {
      /*array parameter=*/copyFromInput<ast::ArrayAssignmentStatement>(),
      /*index=*/copyFromInput<ast::ArrayAssignmentStatement>(),
      /*value=*/
      TransferFn<ast::ArrayAssignmentStatement,
                 ast::ArrayAssignmentStatement::ARRAY,
                 ast::ArrayAssignmentStatement::INDEX>(
          [](const LSQNoDepContext &, const ast::ArrayParameter &parameter,
             const LSQNoDepContext &, const ast::Expression &index) {
            // Construct the context such as to force the value expression to
            // use this specific parameter and index.
            return LSQNoDepContext{&parameter,
                                   // Successful cast guaranteed by every other
                                   // kind of expression being discarded.
                                   &llvm::cast<ast::Variable>(index)};
          }),
      copyInputToOutput<ast::ArrayAssignmentStatement>(),
  };
}
