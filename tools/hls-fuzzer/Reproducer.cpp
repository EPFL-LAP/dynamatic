#include "Reproducer.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace dynamatic;

llvm::Error dynamatic::writeReproducer(const std::filesystem::path &file,
                                       const ReproducerInfo &info) {
  llvm::json::Array arguments;
  for (const std::string &argument : info.arguments)
    arguments.push_back(argument);
  llvm::json::Value reproducer = llvm::json::Object{
      {"seed", info.seed},
      {"arguments", std::move(arguments)},
  };

  return llvm::writeToOutput(file.string(), [&](llvm::raw_ostream &os) {
    llvm::json::OStream(os, /*IndentSize=*/2).value(reproducer);
    os << '\n';
    return llvm::Error::success();
  });
}

std::optional<ReproducerInfo>
dynamatic::readReproducer(const std::filesystem::path &file) {
  auto fail = [&](llvm::Error error) -> std::optional<ReproducerInfo> {
    llvm::errs() << "Failed to read '" << file.string()
                 << "': " << llvm::toString(std::move(error)) << '\n';
    return std::nullopt;
  };

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(file.string());
  if (std::error_code error = buffer.getError())
    return fail(llvm::errorCodeToError(error));

  llvm::Expected<llvm::json::Value> value =
      llvm::json::parse((*buffer)->getBuffer());
  if (!value)
    return fail(value.takeError());

  ReproducerInfo info;
  std::uint64_t seed = 0;
  llvm::json::Path::Root root;
  llvm::json::ObjectMapper mapper(*value, root);
  if (!mapper || !mapper.map("seed", seed) ||
      !mapper.map("arguments", info.arguments))
    return fail(root.getError());

  info.seed = static_cast<std::uint32_t>(seed);
  return info;
}
