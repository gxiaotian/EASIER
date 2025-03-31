import operator
from typing import Any, Tuple
from functools import reduce

import torch
from torch import fx

import easier as esr
from easier.core.passes.code_generation.utils import emit_scalar_type
from easier.core.utils import EasierJitException
from easier.core.passes.metadata_propagation import get_node_meta


_op_dict = {}


def get_op_emitter(op):
    ret = _op_dict.get(op, None)
    if not ret:
        raise EasierJitException(f'Unsupported Operator {op}.')
    else:
        return ret


def register_op(op):
    def func(emitter):
        _op_dict[op] = emitter
    return func


class Op:
    def __init__(self, node: fx.Node) -> None:
        self.node = node
        self.args = [arg for arg in node.args]
        shape = get_node_meta(node).shape[1:]
        self.shape = shape if shape else (1,)
        self.extent = reduce(lambda x, y: x * y, self.shape, 1)
        self.dtype = emit_scalar_type(get_node_meta(node).dtype)
        self.code = []

    def emit(self):
        raise NotImplementedError


class ElementWiseOp(Op):
    def __init__(self, node: fx.Node):
        super().__init__(node)

    def emit(self):
        for i in range(self.extent):
            self.code.append(self._for_each_element(i))
        return self.code

    def _for_each_element(self, i) -> str:
        raise NotImplementedError


class BinaryElementWiseOp(ElementWiseOp):
    def __init__(self, node: fx.Node):
        super().__init__(node)

    def _for_each_element(self, i) -> str:
        arg0 = f'{self.args[0].name}[{i}]' \
            if isinstance(self.args[0], fx.Node) else f'{self.args[0]}'
        arg1 = f'{self.args[1].name}[{i}]' \
            if isinstance(self.args[1], fx.Node) else f'{self.args[1]}'

        return f'{self.node.name}[{i}] = {arg0} {self._operator()} {arg1};'

    def _operator(self):
        raise NotImplementedError


@register_op(operator.mul)
class MulOp(BinaryElementWiseOp):
    def _operator(self):
        return '*'


@register_op(operator.add)
class AddOp(BinaryElementWiseOp):
    def _operator(self):
        return '+'


@register_op(operator.sub)
class SubOp(BinaryElementWiseOp):
    def _operator(self):
        return '-'


@register_op(operator.truediv)
class DivOp(BinaryElementWiseOp):
    def _operator(self):
        return '/'


@register_op(operator.setitem)
class SetitemOp(Op):
    def __init__(self, node: fx.Node) -> Any:
        super().__init__(node)

    def emit(self):
        # do not support slice for now
        if isinstance(self.args[1], Tuple):
            offsets = torch.tensor(list(range(0, self.extent))).reshape(
                self.shape)[self.args[1][1:]].flatten()
        else:
            offsets = list(range(self.extent))

        for i, offset in enumerate(offsets):
            self.code.append(self._for_each_element(i, offset))

        return self.code

    def _for_each_element(self, i, offset):
        return f'{self.args[0].name}[{offset}] = {self.args[2].name}[{i}];'


@register_op(operator.getitem)
class GetitemOp(Op):
    def __init__(self, node: fx.Node) -> None:
        super().__init__(node)

    def emit(self):
        if isinstance(self.args[1], Tuple):
            shape = get_node_meta(self.args[0]).shape[1:]
            extent = reduce(lambda x, y: x * y, shape, 1)
            offsets = torch.tensor(list(range(0, extent))).reshape(shape)[
                self.args[1][1:]].flatten()
        else:
            offsets = list(range(self.extent))

        for i, offset in enumerate(offsets):
            self.code.append(self._for_each_element(i, offset))

        return self.code

    def _for_each_element(self, i, offset):
        return f'{self.node.name}[{i}] = {self.args[0].name}[{offset}];'


@register_op(esr.Selector)
class SelectOp(ElementWiseOp):
    def __init__(self, node: fx.Node) -> None:
        super().__init__(node)

        # NOTE: this implementation of selector requires continuous memory.
        # Halo exchanger should make sure this requirement is met

    def _for_each_element(self, i) -> str:
        return f'{self.node.name}[{i}] =' \
               f' {self.args[0].name}_input[*{self.node.name}_idx' \
               f' * {self.extent} + {i}];'


@register_op(esr.Reducer)
class ReduceOp(ElementWiseOp):
    def __init__(self, node: fx.Node, fixup=False) -> None:
        super().__init__(node)
        self.fixup = fixup

    def _for_each_element(self, i) -> str:
        if self.fixup:
            return ''
        else:
            return f'{self.node.name}_running_total[{i}] +=' \
                   f' {self.args[0].name}[{i}];'


@register_op(esr.sum)
class SumOp(ElementWiseOp):
    def __init__(self, node: fx.Node, fixup=False) -> None:
        super().__init__(node)
        self.fixup = fixup

    def _for_each_element(self, i) -> str:
        if self.fixup:
            return None
        else:
            return f'{self.node.name}_running_total[{i}] +=' \
                   f' {self.args[0].name}[{i}];'


@register_op(esr.norm)
class NormOp(ElementWiseOp):
    def __init__(self, node: fx.Node, fixup=False) -> None:
        super().__init__(node)
        self.fixup = fixup
        if len(node.args) < 2:
            self.p = 2
        else:
            self.p = node.args[1]

    def _for_each_element(self, i) -> str:
        if self.fixup:
            return f'{self.node.name}_input[{i}] =' \
                   f' pow({self.node.name}_input[{i}], 1. / {self.p});'
        else:
            return f'{self.node.name}_running_total[{i}] +=' \
                   f' pow({self.args[0].name}[{i}], {self.p});'
