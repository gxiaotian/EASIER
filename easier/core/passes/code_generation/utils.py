from typing import List, Union
from functools import reduce

import torch

# import easier as esr
from easier.core.utils import EasierJitException
from easier.core.passes.utils import FX
# from easier.core.passes.metadata_propagation import ScalarType, get_node_meta
from easier.core.passes.dataflow_fusion.node_group import \
    get_submod, get_node_group


def emit_scalar_type(T: ScalarType):
    if T.is_floating_point:
        if T.precision == 32:
            ret = 'float'
        elif T.precision == 64:
            ret = 'double'
        else:
            raise EasierJitException('Unsupported ScalarType.')

    elif T.is_integer:
        if T.precision == 1:
            ret = 'bool'
        elif T.precision == 8:
            ret = 'int8_t'
        elif T.precision == 16:
            ret = 'int16_t'
        elif T.precision == 32:
            ret = 'int32_t'
        elif T.precision == 64:
            ret = 'int64_t'
        else:
            raise EasierJitException('Unsupported ScalarType.')

    else:
        raise EasierJitException('Unsupported ScalarType.')

    return ret


def emit_torch_type(T: torch.dtype):
    if T == torch.float32:
        ret = 'float'
    elif T == torch.float64:
        ret = 'double'
    elif T == torch.bool:
        ret = 'bool'
    elif T == torch.int8:
        ret = 'int8_t'
    elif T == torch.int16:
        ret = 'int16_t'
    elif T == torch.int32:
        ret = 'int32_t'
    elif T == torch.int64:
        ret = 'int64_t'
    else:
        raise EasierJitException('Unsupported Pytorch Type.')

    return ret


def get_node_return(node):
    dtype = emit_scalar_type(get_node_meta(node).dtype)
    shape = get_node_meta(node).shape
    extent = reduce(lambda x, y: x * y, shape[1:], 1)
    return dtype, shape, extent


def get_idx_dtype(node):
    assert node.op == FX.CALL_MODULE
    root = get_node_group(node).root
    return emit_torch_type(get_submod(root, node).idx.dtype)


class CodeBlock:
    def __init__(
        self,
        heads: List[str] = None,
        tails: List[str] = None,
        items: List[Union[str, 'CodeBlock']] = None
    ) -> None:
        self._heads = heads if heads else []
        self._tails = tails if tails else []
        self._items = items if items else []

    def add_item(self, item: Union[str, 'CodeBlock']):
        self._items.append(item)

    def add_head(self, head: str):
        self._heads.append(head)

    def add_tail(self, tail: str):
        self._tails.append(tail)

    def emit(self):
        ret = ''
        for head in self._heads:
            ret += head + '\n'

        for item in self._items:
            if isinstance(item, str):
                ret += '  ' + item + '\n'
            elif isinstance(item, CodeBlock):
                for line in item.emit().split('\n'):
                    ret += '  ' + line + '\n'
            else:
                raise EasierJitException(
                    f'Unsupported body item {item.__class__}')

        for tail in self._tails:
            ret += tail + '\n'

        return ret[:-1]


class OmpScope:
    def __init__(self):
        pass

