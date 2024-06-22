//===- ControlDependenceAnalysis.h - Control dependence analyis utilities ----------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares some utility functions useful to analyzing the control dependencies between basic blocks of the CFG.
// The control dependendies are calculated using the algorithm from the following paper
//   J.Ferrante, K.J. Ottenstein, and J. D. Warren, "The Program Dependence Graph and its Use in Optimizations", ACM Trans. Program. Lang. Syst., vol. 9, pp. 319-349, 1987.
//
//===----------------------------------------------------------------------===//


#include "dynamatic/Analysis/ControlDependenceAnalysis.h"

using namespace mlir;
using namespace llvm;
using namespace dynamatic;

void ControlDependenceAnalysis::identifyAllControlDeps(mlir::func::FuncOp &funcOp) {
  Region &funcReg = funcOp.getRegion();

  // Initialize the control_deps_map by creating an entry for every block constituting the region
  for (Block &block : funcReg.getBlocks()) {
    SmallVector<mlir::Block*, 4> deps;
    control_deps_map.insert(std::make_pair(&block, deps));
  }

  if(funcReg.hasOneBlock()) {
    return;  // functions made up of 1 basic block do not have any dependencies
  }

  // Get information about post-dominance
  PostDominanceInfo postDomInfo;
  // Get the post-dominance tree
  llvm::DominatorTreeBase<Block, true> &postDomTree =
      postDomInfo.getDomTree(&funcReg);

  // Loop over the control flow edges connnecting the different blocks of this region
  for (Block& block : funcReg.getBlocks()) {
    for (Block* block_succ : block.getSuccessors()) {
      if (!postDomInfo.properlyPostDominates(block_succ, &block)) {
          Block* least_common_anc = postDomInfo.findNearestCommonDominator(block_succ, &block);

          if(least_common_anc == &block) {  
            // All nodes in the post-dominator tree on the path from the "block" to "block_succ" (including both "block" and "block_succ") should be control dependent on "block" 

            // easy case of block and block_succ
            control_deps_map[&block].push_back(&block);  // loop case
            control_deps_map[block_succ].push_back(&block);

            // traverse the tree to get all nodes between "least_common_anc" and "block_succ"
            llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4> traversed_nodes; 
            traversePostDomTree(&block, block_succ, &funcReg, &postDomTree, &traversed_nodes);

            //printPostDomTreeTraversal(traversed_nodes);
            
            for(size_t i = 0; i < traversed_nodes.size(); i++) {
              for(size_t j = 0; j < traversed_nodes[i].size(); j++) {
                // for every node in every path, add block to its control dependencies
                Block* b = traversed_nodes[i][j]->getBlock();

                if(b == &block || b == block_succ)
                  continue;  // skip the start and the end nodes because we have taken care of them above

                if(std::find(control_deps_map[b].begin(), control_deps_map[b].end(), &block) == control_deps_map[b].end()) {
                  llvm::errs() << "\n 1 Adding ";
                  block.printAsOperand(llvm::errs());
                  llvm::errs() << " to the control deps of ";
                  b->printAsOperand(llvm::errs());

                  control_deps_map[b].push_back(&block);

                }
              }
            }

          } else {
            // All nodes in the post-dominator tree on the path from the "least_common_anc" to "block_succ" (including "block_succ") should be control dependent on "block"
            // easy case of block_succ
            control_deps_map[block_succ].push_back(&block);  

            // traverse the tree to get all nodes between "least_common_anc" and "block_succ"
            llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4> traversed_nodes; 
            traversePostDomTree(least_common_anc, block_succ, &funcReg, &postDomTree, &traversed_nodes);

            // printPostDomTreeTraversal(traversed_nodes);

            for(size_t i = 0; i < traversed_nodes.size(); i++) {
              for(size_t j = 0; j < traversed_nodes[i].size(); j++) {
                // for every node in every path, add block to its control dependencies
                Block* b = traversed_nodes[i][j]->getBlock();
                if(b == &block || b == block_succ || b == least_common_anc)
                  continue;  // skip the start and the end nodes because we have taken care of them above
                if(std::find(control_deps_map[b].begin(), control_deps_map[b].end(), &block) == control_deps_map[b].end()) {
                  control_deps_map[b].push_back(&block);
                }
              }
            }

          }
      }
    }
  }

}

void ControlDependenceAnalysis::traversePostDomTree(mlir::Block *start_block, mlir::Block *end_block, Region *funcReg, llvm::DominatorTreeBase<mlir::Block, true> *postDomTree, llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4>*traversed_nodes) {

  DominanceInfoNode *start_node = postDomTree->getNode(start_block);
  DominanceInfoNode *end_node = postDomTree->getNode(end_block);

  llvm::DenseMap<DominanceInfoNode*, bool> is_visited;
  for(Block &block : funcReg->getBlocks()) {
    is_visited[postDomTree->getNode(&block)] = false;
  }

  llvm::SmallVector<mlir::DominanceInfoNode*, 4> path;
  path.resize(funcReg->getBlocks().size());
  int path_index = 0;

  traversePostDomTreeUtil(start_node, end_node, is_visited, path, path_index, traversed_nodes);

}

// DFS to return all nodes in the path between the start_node and end_node (not including start_node and end_node) in the postDom tree 
void ControlDependenceAnalysis::traversePostDomTreeUtil(DominanceInfoNode *start_node, DominanceInfoNode *end_node, llvm::DenseMap<DominanceInfoNode*, bool> is_visited, llvm::SmallVector<mlir::DominanceInfoNode*, 4> path, int path_index, llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4>*traversed_nodes) {
  is_visited[start_node] = true;
  path[path_index] = start_node;
  path_index++;

  // if start is same as end, we have completed one path so push it to traversed_nodes
  if(start_node == end_node) {
     // slice of the path from its beginning until the path_index
    llvm::SmallVector<mlir::DominanceInfoNode*, 4> actual_path;
    for(int i = 0; i < path_index; i++) {
      actual_path.push_back(path[i]);
    }
    traversed_nodes->push_back(actual_path);

  } else{
     // loop over the children of start_node
    for( DominanceInfoNode::iterator iter = start_node->begin(); iter < start_node->end(); iter++ ) {
      if(!is_visited[*iter]) {
        traversePostDomTreeUtil(*iter, end_node, is_visited, path, path_index, traversed_nodes);
      } 
    }
  }
 
  // remove this node from path and mark it as unvisited
  path_index--;
  is_visited[start_node] = false;
}

void ControlDependenceAnalysis::returnControlDeps(mlir::Block* block, llvm::SmallVector<mlir::Block*, 4>& returned_control_deps) {
  returned_control_deps = control_deps_map[block];
}

void ControlDependenceAnalysis::printBlocksDeps(mlir::func::FuncOp &funcOp) {
  Region &funcReg = funcOp.getRegion();
  llvm::errs() << "\n*********************************\n\n";
  for (Block &block : funcReg.getBlocks()) {
    block.printAsOperand(llvm::errs());
    llvm::errs() << " is control dependent on: ";
    for(size_t i = 0; i < control_deps_map[&block].size(); i++) {
      control_deps_map[&block][i]->printAsOperand(llvm::errs());
      llvm::errs() << ", ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n*********************************\n";
}

void dynamatic::printPostDomTreeTraversal(llvm::SmallVector<llvm::SmallVector<mlir::DominanceInfoNode*, 4> , 4>traversed_nodes) {
  llvm::errs() << "\n*********************************\n";
  llvm::errs() << "\nPrinting one graph traversal:\n";
  llvm::errs() << "We have a total of " << traversed_nodes.size() << " paths\n";
  for(size_t i = 0; i < traversed_nodes.size(); i++) {
      llvm::errs() << "Printing the details of path " << i << " ";
      for(size_t j = 0; j < traversed_nodes[i].size(); j++) {
        Block* b = traversed_nodes[i][j]->getBlock();
        b->printAsOperand(llvm::errs());
        llvm::errs() << " -> ";
      }
      llvm::errs() << "\n";
  }
  llvm::errs() << "\n*********************************\n";
}
