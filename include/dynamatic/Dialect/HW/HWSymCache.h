//===- HWSymCache.h - Declare Symbol Cache ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This file declares a Symbol Cache specialized for HW instances.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_DIALECT_HW_HW_SYM_CACHE_H
#define DYNAMATIC_DIALECT_HW_HW_SYM_CACHE_H

#include "dynamatic/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace dynamatic {
namespace hw {

/// Base symbol cache class to allow for cache lookup through a pointer to some
/// abstract cache. A symbol cache stores lookup tables to make manipulating and
/// working with the IR more efficient.
class SymbolCacheBase {
public:
  virtual ~SymbolCacheBase();

  /// Defines 'op' as associated with the 'symbol' in the cache.
  virtual void addDefinition(mlir::Attribute symbol, mlir::Operation *op) = 0;

  /// Adds the symbol-defining 'op' to the cache.
  void addSymbol(mlir::SymbolOpInterface op) {
    addDefinition(op.getNameAttr(), op);
  }

  /// Populate the symbol cache with all symbol-defining operations within the
  /// 'top' operation.
  void addDefinitions(mlir::Operation *top);

  /// Lookup a definition for 'symbol' in the cache.
  virtual mlir::Operation *getDefinition(mlir::Attribute symbol) const = 0;

  /// Lookup a definition for 'symbol' in the cache.
  mlir::Operation *getDefinition(mlir::FlatSymbolRefAttr symbol) const {
    return getDefinition(symbol.getAttr());
  }

  /// Iterator support through a pointer to some abstract cache.
  /// The implementing cache must provide an iterator that carries values on the
  /// form of <mlir::Attribute, mlir::Operation*>.
  using CacheItem = std::pair<mlir::Attribute, mlir::Operation *>;
  struct CacheIteratorImpl {
    virtual ~CacheIteratorImpl() = default;
    virtual void operator++() = 0;
    virtual CacheItem operator*() = 0;
    virtual bool operator==(CacheIteratorImpl *other) = 0;
  };

  struct Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          CacheItem> {
    Iterator(std::unique_ptr<CacheIteratorImpl> &&impl)
        : impl(std::move(impl)) {}
    CacheItem operator*() const { return **impl; }
    using llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                     CacheItem>::operator++;
    bool operator==(const Iterator &other) const {
      return *impl == other.impl.get();
    }
    void operator++() { impl->operator++(); }

  private:
    std::unique_ptr<CacheIteratorImpl> impl;
  };
  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
};

/// Default symbol cache implementation; stores associations between names
/// (StringAttr's) to mlir::Operation's.
/// Adding/getting definitions from the symbol cache is not
/// thread safe. If this is required, synchronizing cache acccess should be
/// ensured by the caller.
class SymbolCache : public SymbolCacheBase {
public:
  /// In the building phase, add symbols.
  void addDefinition(mlir::Attribute key, mlir::Operation *op) override {
    symbolCache.try_emplace(key, op);
  }

  // Pull in getDefinition(mlir::FlatSymbolRefAttr symbol)
  using SymbolCacheBase::getDefinition;
  mlir::Operation *getDefinition(mlir::Attribute attr) const override {
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    return it->second;
  }

protected:
  /// This stores a lookup table from symbol attribute to the operation
  /// that defines it.
  llvm::DenseMap<mlir::Attribute, mlir::Operation *> symbolCache;

private:
  /// Iterator support: A simple mapping between decltype(symbolCache)::iterator
  /// to SymbolCacheBase::Iterator.
  using Iterator = decltype(symbolCache)::iterator;
  struct SymbolCacheIteratorImpl : public CacheIteratorImpl {
    SymbolCacheIteratorImpl(Iterator it) : it(it) {}
    CacheItem operator*() override { return {it->getFirst(), it->getSecond()}; }
    void operator++() override { it++; }
    bool operator==(CacheIteratorImpl *other) override {
      return it == static_cast<SymbolCacheIteratorImpl *>(other)->it;
    }
    Iterator it;
  };

public:
  SymbolCacheBase::Iterator begin() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.begin()));
  }
  SymbolCacheBase::Iterator end() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<SymbolCacheIteratorImpl>(symbolCache.end()));
  }
};

/// This stores lookup tables to make manipulating and working with the IR more
/// efficient.  There are two phases to this object: the "building" phase in
/// which it is "write only" and then the "using" phase which is read-only (and
/// thus can be used by multiple threads).  The  "freeze" method transitions
/// between the two states.
class HWSymbolCache : public SymbolCacheBase {
public:
  class Item {
  public:
    Item(mlir::Operation *op) : op(op), port(~0ULL) {}
    Item(mlir::Operation *op, size_t port) : op(op), port(port) {}
    bool hasPort() const { return port != ~0ULL; }
    size_t getPort() const { return port; }
    mlir::Operation *getOp() const { return op; }

  private:
    mlir::Operation *op;
    size_t port;
  };

  // Add inner names, which might be ports
  void addDefinition(mlir::StringAttr modSymbol, mlir::StringAttr name,
                     mlir::Operation *op, size_t port = ~0ULL) {
    auto key = InnerRefAttr::get(modSymbol, name);
    symbolCache.try_emplace(key, op, port);
  }

  void addDefinition(mlir::Attribute key, mlir::Operation *op) override {
    assert(!isFrozen && "cannot mutate a frozen cache");
    symbolCache.try_emplace(key, op);
  }

  // Pull in getDefinition(mlir::FlatSymbolRefAttr symbol)
  using SymbolCacheBase::getDefinition;
  mlir::Operation *getDefinition(mlir::Attribute attr) const override {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    assert(!it->second.hasPort() && "Module names should never be ports");
    return it->second.getOp();
  }

  HWSymbolCache::Item getInnerDefinition(mlir::StringAttr modSymbol,
                                         mlir::StringAttr name) const {
    return lookupInner(InnerRefAttr::get(modSymbol, name));
  }

  HWSymbolCache::Item getInnerDefinition(InnerRefAttr inner) const {
    return lookupInner(inner);
  }

  /// Mark the cache as frozen, which allows it to be shared across threads.
  void freeze() { isFrozen = true; }

private:
  Item lookupInner(InnerRefAttr attr) const {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    return it == symbolCache.end() ? Item{nullptr, ~0ULL} : it->second;
  }

  bool isFrozen = false;

  /// This stores a lookup table from symbol attribute to the item
  /// that defines it.
  llvm::DenseMap<mlir::Attribute, Item> symbolCache;

private:
  // Iterator support. Map from Item's to their inner operations.
  using Iterator = decltype(symbolCache)::iterator;
  struct HwSymbolCacheIteratorImpl : public CacheIteratorImpl {
    HwSymbolCacheIteratorImpl(Iterator it) : it(it) {}
    CacheItem operator*() override {
      return {it->getFirst(), it->getSecond().getOp()};
    }
    void operator++() override { it++; }
    bool operator==(CacheIteratorImpl *other) override {
      return it == static_cast<HwSymbolCacheIteratorImpl *>(other)->it;
    }
    Iterator it;
  };

public:
  SymbolCacheBase::Iterator begin() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<HwSymbolCacheIteratorImpl>(symbolCache.begin()));
  }
  SymbolCacheBase::Iterator end() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<HwSymbolCacheIteratorImpl>(symbolCache.end()));
  }
};

} // namespace hw
} // namespace dynamatic

#endif // DYNAMATIC_DIALECT_HW_HW_SYM_CACHE_H
