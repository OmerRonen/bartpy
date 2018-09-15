from abc import abstractclassmethod
from typing import Callable

import numpy as np

from bartpy.errors import NoSplittableVariableException, NoPrunableNodeException
from bartpy.mutation import TreeMutation, PruneMutation, GrowMutation, ChangeMutation
from bartpy.node import sample_split_node
from bartpy.tree import LeafNode, Tree, random_prunable_decision_node, random_splittable_leaf_node


class TreeMutationProposer:

    def __init__(self, tree_structure: Tree):
        self.tree_structure = tree_structure

    @abstractclassmethod
    def proposal(self) -> TreeMutation:
        raise NotImplementedError()


class GrowTreeMutationProposer(TreeMutationProposer):

    def __init__(self, tree_structure: Tree):
        super().__init__(tree_structure)

    def proposal(self) -> TreeMutation:
        node = random_splittable_leaf_node(self.tree_structure)
        updated_node = sample_split_node(node)
        return GrowMutation(node, updated_node)


class PruneTreeMutationProposer(TreeMutationProposer):

    def __init__(self, tree_structure: Tree):
        super().__init__(tree_structure)

    def proposal(self) -> TreeMutation:
        node = random_prunable_decision_node(self.tree_structure)
        updated_node = LeafNode(node.split, depth=node.depth)
        return PruneMutation(node, updated_node)


class ChangeTreeMutationProposer(TreeMutationProposer):

    def __init__(self, tree_structure: Tree):
        super().__init__(tree_structure)

    def proposal(self) -> TreeMutation:
        node = random_prunable_decision_node(self.tree_structure)
        leaf_node = LeafNode(node.split, depth=node.depth)
        updated_split_node = sample_split_node(leaf_node)
        return ChangeMutation(node, updated_split_node)


class Proposer:

    def __init__(self, p_grow: float, p_prune: float, p_change: float):
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.p_change = p_change

    def sample_mutation_method(self) -> Callable[[Tree], TreeMutationProposer]:
        method = np.random.choice([ChangeTreeMutationProposer, GrowTreeMutationProposer, PruneTreeMutationProposer], p=[self.p_change, self.p_grow, self.p_prune])
        return method

    def propose(self, tree_structure: Tree) -> TreeMutation:
        method = self.sample_mutation_method()
        try:
            return method(tree_structure).proposal()
        except NoSplittableVariableException:
            return self.propose(tree_structure)
        except NoPrunableNodeException:
            return self.propose(tree_structure)


if __name__ == "__main__":
    from bartpy.data import Data
    import pandas as pd
    from bartpy.sigma import Sigma
    from bartpy.model import Model

    data = Data(pd.DataFrame({"b": [1, 2, 3]}), pd.Series([1, 2, 3]), normalize=True)
    sigma = Sigma(1., 2.)
    model = Model(data, sigma)

    change = Proposer(0, 0, 1).propose(model.trees[0])
    print(change.existing_node.split)
    print(change.updated_node.split)
