from abc import ABC, abstractmethod

from torch import fx

from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_users

from .node_group import NodeGroup, NodeType, GroupType, get_node_group, \
    get_node_type, get_submod


class NodeRule(ABC):
    @abstractmethod
    def check(self, n1: fx.Node, n2: fx.Node) -> bool:
        pass


class FuseGetAttrNodeToOtherNode(NodeRule):
    def check(self, n1: fx.Node, n2: fx.Node) -> bool:
        root = get_node_group(n1).root
        return False if get_node_type(root, n1) is NodeType.GetAttr else True


class FuseOtherNodeToSelectNode(NodeRule):
    def check(self, n1: fx.Node, n2: fx.Node) -> bool:
        root = get_node_group(n2).root
        return False if get_node_type(root, n2) is NodeType.Select else True


class FuseReduceNodeToOtherNode(NodeRule):
    def check(self, n1: fx.Node, n2: fx.Node) -> bool:
        root = get_node_group(n1).root
        return False if get_node_type(root, n1) is NodeType.Reduce else True


class FuseReplicaNodeToOtherNode(NodeRule):
    def check(self, n1: fx.Node, n2: fx.Node) -> bool:
        root = get_node_group(n1).root
        return False if get_node_type(root, n1) is NodeType.Replica else True


class GroupRule(ABC):
    @abstractmethod
    def check(self, ng1: NodeGroup, n2: NodeGroup) -> bool:
        pass


class FuseReduceGroupToReduceGroup(GroupRule):
    def check(self, ng1: NodeGroup, ng2: NodeGroup) -> bool:
        if ng1.type() in [GroupType.Reduce, GroupType.SelectReduce] and \
                ng2.type() in [GroupType.Reduce, GroupType.SelectReduce]:
            r1 = get_submod(ng1.root, ng1.get_node_with_type(NodeType.Reduce))
            r2 = get_submod(ng2.root, ng2.get_node_with_type(NodeType.Reduce))
            return False if r1 is not r2 else True
        else:
            return True


class FuseReduceGroupAndReplicaGroup(GroupRule):
    def check(self, ng1: fx.Node, ng2: fx.Node) -> bool:
        ng1_type = ng1.type()
        ng2_type = ng2.type()
        if (
            ng1_type in [GroupType.Reduce, GroupType.SelectReduce] and
            ng2_type is GroupType.Replica
        ) or (
            ng2_type in [GroupType.Reduce, GroupType.SelectReduce] and
            ng1_type is GroupType.Replica
        ):
            return False
        else:
            return True


class FuseReduceGroupAndOtherGroup(GroupRule):
    def check(self, ng1: fx.Node, ng2: fx.Node) -> bool:
        ng1_type = ng1.type()
        ng2_type = ng2.type()

        if ng1_type in [GroupType.Reduce, GroupType.SelectReduce]:
            reduce_node = ng1.get_node_with_type(NodeType.Reduce)
            for usr in get_data_dependency_users(reduce_node):
                if usr in ng2.nodes:
                    return False

        if ng2_type in [GroupType.Reduce, GroupType.SelectReduce]:
            reduce_node = ng2.get_node_with_type(NodeType.Reduce)
            for usr in get_data_dependency_users(reduce_node):
                if usr in ng1.nodes:
                    return False

        return True
