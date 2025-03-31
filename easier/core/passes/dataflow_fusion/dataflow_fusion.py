from typing import List, Sequence, Tuple

from torch import fx, nn

import easier as esr
from easier.core.passes.utils import EasierInterpreter
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_users

from .node_group import NodeGroup, get_node_group, set_node_group, \
    fuse_node_groups
from .fusion_rules import NodeRule, GroupRule, \
    FuseGetAttrNodeToOtherNode, FuseOtherNodeToSelectNode, \
    FuseReduceNodeToOtherNode, FuseReplicaNodeToOtherNode, \
    FuseReduceGroupToReduceGroup, FuseReduceGroupAndReplicaGroup, \
    FuseReduceGroupAndOtherGroup


class NodeGrouper(EasierInterpreter):
    def __init__(
        self,
        modules: Sequence[esr.Module],
        graphs: Sequence[fx.Graph],
        node_rules: List[NodeRule] = [
            FuseGetAttrNodeToOtherNode(),
            FuseOtherNodeToSelectNode(),
            FuseReduceNodeToOtherNode(),
            FuseReplicaNodeToOtherNode()],
        group_rules: List[GroupRule] = [
            FuseReduceGroupAndOtherGroup(),
            FuseReduceGroupToReduceGroup(),
            FuseReduceGroupAndReplicaGroup()]
    ) -> None:
        super().__init__(modules, graphs)
        self.node_rules = node_rules
        self.group_rules = group_rules

    def _check_rules(self, ng1: NodeGroup, ng2: NodeGroup) -> bool:
        for rule in self.group_rules:
            if not rule.check(ng1, ng2):
                return False

        for n1 in ng1.nodes:
            for n2 in ng2.nodes:
                if n2 in n1.users:
                    for rule in self.node_rules:
                        if not rule.check(n1, n2):
                            return False

        return True

    def _create_node_group(self, node: fx.Node) -> NodeGroup:
        ng1 = NodeGroup(self.current_module, [node])
        set_node_group(node, ng1)

        for user in node.users:
            if get_node_group(user):
                ng2 = get_node_group(user)
            else:
                ng2 = self._create_node_group(user)

            if ng1 is not ng2 and self._check_rules(ng1, ng2):
                fuse_node_groups(ng1, ng2)

        return ng1

    def for_each_node(self) -> None:
        if not get_node_group(self.current_node):
            self._create_node_group(self.current_node)


def fuse_dataflow(
    modules: List[nn.Module],
    graphs: List[fx.Graph]
) -> Tuple[List[nn.Module], List[fx.Graph]]:
    esr.logger.info('Fusion pass is running.')
    NodeGrouper(modules, graphs).run()
    esr.logger.info('Fusion pass has completed.')

    return modules, graphs
