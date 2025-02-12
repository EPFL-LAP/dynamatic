#include <llvm/ADT/StringSet.h>
#include <string>

namespace dynamatic::experimental {

llvm::StringSet<> getStateSet(const std::string &filename);
int getStates(const std::string &infFile, const std::string &finFile);

} // namespace dynamatic::experimental