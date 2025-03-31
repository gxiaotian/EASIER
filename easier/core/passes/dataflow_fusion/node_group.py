from enum import Enum, auto
from typing import cast, Union, List, Sequence, Set

from torch import fx, nn

import easier as esr
from easier.core.utils import EasierJitException
from easier.core.passes.utils import FX, OrderedSet
from easier.core.passes.metadata_propagation import get_node_meta, Role
from ..data_dependency_analysis import get_data_dependency_users


KEY__FUSION_NODE_GROUP = 'easier_fusion_node_group'


class NodeType(Enum):
    Select = auto()
    Reduce = auto()
    Map = auto()
    GetAttr = auto()
    Replica = auto()


class GroupType(Enum):
    Select = auto()
    Reduce = auto()
    Map = auto()
    SelectReduce = auto()
    Replica = auto()


class NodeGroup:
    def __init__(self, root: nn.Module, nodes: Sequence[fx.Node]) -> None:
        self.root = root
        self.nodes = OrderedSet(nodes)

    def __repr__(self) -> str:
        return str([n for n in self.nodes])

    def __eq__(self, other: 'NodeGroup') -> bool:
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        ret = 0
        for node in self.nodes:
            ret += hash(node)
        return ret

    def __ls__(self, other: 'NodeGroup') -> bool:
        if _has_path(other, self):
            return False
        else:
            return True

    def __gt__(self, other: 'NodeGroup') -> bool:
        if _has_path(self, other):
            return False
        else:
            return True

    def type(self) -> GroupType:
        res = GroupType.Map
        for node in self.nodes:
            res = _infer_group_type(res, get_node_type(self.root, node))
        return res

    # TODO: not neccessary, can be replaced with get_reduce_node
    def get_node_with_type(self, ntype: NodeType) -> Union[fx.Node, bool]:
        for node in self.nodes:
            if get_node_type(self.root, node) is ntype:
                return node
        return False

    def all_select_submods(self) -> OrderedSet[fx.Node]:
        ret = OrderedSet()
        for node in self.nodes:
            if get_node_type(self.root, node) is NodeType.Select:
                ret.add(get_submod(self.root, node))
        return ret

    def all_select_nodes(self) -> OrderedSet[fx.Node]:
        ret = OrderedSet()
        for node in self.nodes:
            if get_node_type(self.root, node) is NodeType.Select:
                ret.add(node)
        return ret

    def all_reduce_submods(self) -> OrderedSet[fx.Node]:
        ret = OrderedSet()
        for node in self.nodes:
            if get_node_type(self.root, node) is NodeType.Reduce:
                ret.add(get_submod(self.root, node))
        return ret

    def all_reduce_nodes(self) -> OrderedSet[fx.Node]:
        ret = OrderedSet()
        for node in self.nodes:
            if get_node_type(self.root, node) is NodeType.Reduce:
                ret.add(node)
        return ret

    def all_input_nodes(self) -> OrderedSet[fx.Node]:
        inputs = OrderedSet()
        for node in self.nodes:
            for arg in node.all_input_nodes:
                if arg not in self.nodes:
                    inputs.add(arg)
        return inputs

    def all_output_nodes(self) -> OrderedSet[fx.Node]:
        outputs = OrderedSet()
        for node in self.nodes:
            for user in node.users:
                if user not in self.nodes:
                    outputs.add(node)
        return outputs


def _infer_group_type(gt: GroupType, nt: NodeType) -> GroupType:
    if gt is GroupType.Map:
        if nt is NodeType.Select:
            return GroupType.Select
        elif nt is NodeType.Reduce:
            return GroupType.Reduce
        elif nt is NodeType.Replica:
            return GroupType.Replica
        else:
            return GroupType.Map

    elif gt is GroupType.Select:
        if nt is NodeType.Reduce:
            return GroupType.SelectReduce
        elif nt is NodeType.Replica:
            return GroupType.Replica
        else:
            return GroupType.Select

    elif gt is GroupType.Reduce:
        if nt is NodeType.Select:
            return GroupType.SelectReduce
        elif nt is NodeType.Replica:
            raise EasierJitException(
                'Reduce node and Replica node are in the same NodeGroup.')
        else:
            return GroupType.Reduce

    elif gt is GroupType.SelectReduce:
        if nt is NodeType.Replica:
            raise EasierJitException(
                'Reduce node and Replica node are in the same NodeGroup.')
        else:
            return GroupType.SelectReduce

    elif gt is GroupType.Replica:
        return GroupType.Replica

    else:
        raise EasierJitException(f'Unkown GroupType {gt}')


def _has_path(
        n1: Union[NodeGroup, fx.Node], n2: Union[NodeGroup, fx.Node]) -> bool:
    if n1 == n2:
        return True
    else:
        for usr in _get_downstream(n1):
            if _has_path(usr, n2):
                return True
        return False


def _get_downstream(
        input: Union[NodeGroup, fx.Node]) -> Set[Union[NodeGroup, fx.Node]]:
    ret = set()
    if isinstance(input, fx.Node):
        for usr in set(list(input.users) + get_data_dependency_users(input)):
            if get_node_group(usr):
                ret.add(get_node_group(usr))
            else:
                ret.add(usr)

    else:
        for node in input.nodes:
            for usr in set(list(node.users) + get_data_dependency_users(node)):
                if usr not in input.nodes:
                    if get_node_group(usr):
                        ret.add(get_node_group(usr))
                    else:
                        ret.add(usr)
    return ret


def get_submod(root: nn.Module, node: fx.Node) -> nn.Module:
    return root.get_submodule(cast(str, node.target))


def get_node_type(root: nn.Module, node: fx.Node) -> NodeType:
    if node.op is FX.GET_ATTR:
        return NodeType.GetAttr

    elif node.op is FX.CALL_MODULE:
        submod = get_submod(root, node)
        if isinstance(submod, esr.Reducer):
            return NodeType.Reduce
        elif isinstance(submod, esr.Selector):
            return NodeType.Select
        else:
            raise EasierJitException(f'Unsupported CALL_MODULE {submod}')

    elif get_node_meta(node).role is Role.REPLICA:
        return NodeType.Replica

    else:
        return NodeType.Map


def sort_node_groups(groups: List[NodeGroup]):
    if len(groups) <= 1:
        return groups

    pivot = groups[len(groups) // 2]
    left = []
    right = []
    middle = []
    for group in groups:
        if group == pivot:
            middle.append(group)
        elif group > pivot:
            right.append(group)
        elif group < pivot:
            left.append(group)

    return sort_node_groups(left) + middle + sort_node_groups(right)


def fuse_node_groups(ng1: NodeGroup, ng2: NodeGroup) -> None:
    ng1_nodes = OrderedSet(ng1.nodes)
    ng2_nodes = OrderedSet(ng2.nodes)
    ng1.nodes.extend(ng2.nodes)
    ng2.nodes.extend(ng1.nodes)

    for usr in _get_downstream(ng1):
        if _has_path(usr, ng1):
            ng1.nodes = ng1_nodes
            ng2.nodes = ng2_nodes
            return False

    for node in ng1.nodes:
        set_node_group(node, ng1)

    return True


def get_node_group(node: fx.Node):
    return node.meta.get(KEY__FUSION_NODE_GROUP, None)


def set_node_group(node: fx.Node, ng: NodeGroup):
    node.meta[KEY__FUSION_NODE_GROUP] = ng
