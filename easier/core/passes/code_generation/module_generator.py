import os
from re import L
from tempfile import NamedTemporaryFile as ntf
from typing import Dict, List
from collections import OrderedDict

import torch
from torch import nn, fx
from torch.utils.cpp_extension import load
from jinja2 import Environment, FileSystemLoader

import easier as esr
from easier.core.utils import EasierJitException
from easier.core.passes.utils import FX
from ..metadata_propagation import \
    get_node_meta, convert_scalar_type_to_torch_dtype
from ..dataflow_fusion.node_group import NodeGroup, GroupType, \
    NodeType, get_submod, get_node_type
from .op_emitter import get_op_emitter
from .utils import emit_torch_type, get_node_return
from easier.core.passes.utils import tree_map


class FusedModule(nn.Module):
    def __init__(self, name: str, ng: NodeGroup):
        super().__init__()
        self.input_tensors: List[List] = []

    def __call__(self, *args):
        # get the dtype and shape of the input tensors at JIT time
        for i, arg in enumerate(args):
            # first time to call the fused module
            if len(self.input_tensors[i]) == 1:
                self.input_tensors[i].extend([arg.dtype, arg.shape])
            # check if the dtype and shape are consistent
            elif self.input_tensors[i][1] != arg.dtype or \
                    self.input_tensors[i][2] != arg.shape:
                self.input_tensors[i].extend([arg.dtype, arg.shape])
            # call the compiled fused module
            else:
                super().__call__(*args, *self.outputs,
                                 *self.idx_tensors, self.row_end_offsets)
                return self.outputs

        super().__call__(
            *args, *self.outputs, *self.idx_tensors, self.row_end_offsets)

        return self.outputs

    def collect_tensor_args(self, ng: NodeGroup):
        # collect tensor arguments for the fused kernel
        self.outputs = []

        for node in ng.all_input_nodes():
            # TODO extent is not necessary
            # dtype and shape will be obtained at JIT time
            self.input_tensors.append([node.name])

        for node in ng.all_output_nodes():
            meta = get_node_meta(node)
            shape = meta.shape
            dtype = convert_scalar_type_to_torch_dtype(meta.dtype)
            outputs.append(torch.zeros(shape, dtype=dtype))

            dtype, shape, extent = get_node_return(node)
            self.tensor_args[node.name] = str(extent)

            if get_node_type(group.root, node) is NodeType.Replica:
                dtype, shape, extent = get_node_return(node)
                self.replica_outputs[node.name] = str(extent)



class OmpReduceModule(FusedModule):
    def __init__(self, ngroup: NodeGroup):
        super().__init__(ngroup)


template_folder = os.path.dirname(__file__) + '/templates'
env = Environment(loader=FileSystemLoader(template_folder))


class Adapter(nn.Module):
    def __init__(self, idx_tensors, outputs, row_end_offsets):
        super().__init__()
        self.idx_tensors = idx_tensors
        self.outputs = outputs
        self.row_end_offsets = row_end_offsets

    def __call__(self, *args):
        super().__call__(
            *args, *self.outputs, *self.idx_tensors, self.row_end_offsets)
        return self.outputs

    def forward(self):
        pass


class KernelEmitter:
    def __init__(self, name: str, group: NodeGroup):
        self.name = name
        self.group = group

        self.tensor_args = OrderedDict()
        self.index_args = []

        self.offset_type = emit_torch_type(torch.int64)
        self.value_type = emit_torch_type(torch.double)

        self.num_threads = int(os.getenv('OMP_NUM_THREADS', 1))
        self.num_rows = 0
        self.num_nonzeros = 0

        self.nodes = OrderedDict()

        self.reducers = OrderedDict()
        self.inplace_reducers = OrderedDict()

        self.replica_inputs = OrderedDict()
        self.replica_outputs = OrderedDict()

        self.post_proc = OrderedDict()
        self.compute = []

        idx_tensors = []
        outputs = []
        row_end_offsets = torch.tensor([0])

        if group.type() is GroupType.Map:
            node = group.get_node_with_type(NodeType.Map)
            self.num_rows = get_node_meta(node).shape[0]

        elif group.type() is GroupType.Select:
            node = group.get_node_with_type(NodeType.Select)
            self.num_rows = get_node_meta(node).shape[0]

        elif group.type() is GroupType.Replica:
            node = group.get_node_with_type(NodeType.Replica)
            self.num_rows = get_node_meta(node.args[0]).shape[0]

        elif group.type() in [GroupType.SelectReduce, GroupType.Reduce]:
            node = group.get_node_with_type(NodeType.Reduce)
            self.num_rows = get_node_meta(node).shape[0]
            self.num_nonzeros = get_node_meta(node.args[0]).shape[0]

        else:
            raise EasierJitException()

        for node in group.all_input_nodes():
            dtype, shape, extent = get_node_return(node)
            self.tensor_args[node.name] = str(extent)

            if get_node_type(group.root, node) is NodeType.Replica:
                self.replica_inputs[node.name] = str(extent)

        for node in group.all_output_nodes():
            meta = get_node_meta(node)
            shape = meta.shape
            dtype = convert_scalar_type_to_torch_dtype(meta.dtype)
            outputs.append(torch.zeros(shape, dtype=dtype))

            dtype, shape, extent = get_node_return(node)
            self.tensor_args[node.name] = str(extent)

            if get_node_type(group.root, node) is NodeType.Replica:
                dtype, shape, extent = get_node_return(node)
                self.replica_outputs[node.name] = str(extent)

        for node in group.nodes:
            dtype, shape, extent = get_node_return(node)
            self.nodes[node.name] = str(extent)
            # NOTE the value type is assumed compatible
            self.value_type = dtype

            if get_node_type(group.root, node) is NodeType.Select:
                submod = get_submod(group.root, node)
                idx_tensors.append(submod.idx)
                # NOTE the index type is assumed compatible
                self.offset_type = emit_torch_type(submod.idx.dtype)
                self.index_args.append(node)
                if isinstance(submod, esr.Selector):
                    for line in get_op_emitter(esr.Selector)(node).emit():
                        self.compute.append(line)

            elif get_node_type(group.root, node) is NodeType.Reduce:
                reducer = get_submod(group.root, node)
                # NOTE the index tensor is assumed already sorted
                row_end_offsets = torch.cumsum(torch.bincount(
                    reducer.idx, minlength=reducer.n), dim=0).to(torch.int32)

                dtype, shape, extent = get_node_return(node)
                self.reducers[node.name] = str(extent)
                if 'out' in node.kwargs:
                    self.inplace_reducers[node.name] = node.kwargs['out'].name

                for line in get_op_emitter(esr.Reducer)(node).emit():
                    self.compute.append(line)

            if node.op is FX.CALL_FUNCTION:
                for line in get_op_emitter(node.target)(node).emit():
                    self.compute.append(line)

                if node.target in [esr.sum, esr.norm]:
                    self.post_proc[node.name] = []
                    for line in get_op_emitter(node.target)(node, True).emit():
                        if line:
                            self.post_proc[node.name].append(line)

        self.adapter = Adapter(idx_tensors, outputs, row_end_offsets)

    def emit_adapter(self):
        return self.adapter

    def emit_code(self):
        if self.group.type() in [GroupType.Map, GroupType.Select]:
            template = env.get_template('cpu_map.cpp.j2')
        elif self.group.type() in [GroupType.Reduce, GroupType.SelectReduce]:
            template = env.get_template('cpu_reduce.cpp.j2')
        elif self.group.type() in [GroupType.Replica]:
            template = env.get_template('cpu_replica.cpp.j2')

        code = template.render(
            name=self.name,

            tensor_args=self.tensor_args,
            index_args=self.index_args,

            OffsetT=self.offset_type,
            ValueT=self.value_type,

            num_threads=self.num_threads,
            num_rows=self.num_rows,
            num_nonzeros=self.num_nonzeros,

            reducers=self.reducers,
            inplace_reducers=self.inplace_reducers,

            replica_inputs=self.replica_inputs,
            replica_outputs=self.replica_outputs,

            nodes=self.nodes,

            post_proc=self.post_proc,
            compute=self.compute
        )

        return code


class ModuleGenerator:
    def __init__(self, backend):
        self.backend = backend
        self.kernels: Dict[str, NodeGroup] = {}
        self.modules: Dict[str, nn.Module] = {}

    def add_node_group(self, name, group):
        if self.backend == 'torch':
            return self._torch_add_node_group(name, group)

        elif self.backend == 'cpu':
            return self._cpu_add_node_group(name, group)

    def load(self):
        if self.backend == 'torch':
            return

        elif self.backend == 'cpu':
            return self._cpu_load()

    def _torch_add_node_group(self, name: str, group: NodeGroup):
        node_dict = {}
        new_graph = fx.Graph()
        for arg in group.all_input_nodes():
            node = new_graph.create_node('placeholder', arg.name)
            node_dict[arg.name] = node

        # When not accessing via `.all_input_nodes`, operation like `stack`
        # may have nested input e.g.
        # Node{target=torch.stack, args=([a1,a2,...], -1)}
        def _try_get_node_dict(arg):
            if isinstance(arg, fx.Node):
                return node_dict[arg.name]
            else:
                return arg

        for node in group.nodes:
            args = []
            for arg in node.args:
                args.append(tree_map(arg, _try_get_node_dict))

            kwargs = {}
            for key, value in node.kwargs.items():
                kwargs[key] = tree_map(value, _try_get_node_dict)

            new_node = new_graph.create_node(
                node.op, node.target, tuple(args), kwargs, node.name)
            node_dict[new_node.name] = new_node

        output = []
        for node in group.all_output_nodes():
            output.append(node_dict[node.name])

        new_graph.create_node(
            'output', 'output', args=(output,), name='output')
        gm = fx.GraphModule(group.root, new_graph)
        self.modules[name] = gm

        return gm, gm.graph

    def _cpu_add_node_group(self, name: str, group: NodeGroup):
        kernel = KernelEmitter(name, group)
        self.kernels[name] = kernel
        self.modules[name] = kernel.emit_adapter()
        return self.modules[name], kernel.emit_code()

    def _cpu_load(self) -> None:
        header = os.path.join(
            os.path.dirname(__file__), 'headers/cpu_backend.hpp')
        code = f'#include <{header}>\n'

        for kernel in self.kernels.values():
            code += f'{kernel.emit_code()}'

        code += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
        for name in self.kernels.keys():
            code += f'  m.def("{name}", &{name});\n'
        code += '}\n'

        with ntf(mode="w", suffix=".cpp", delete=True) as f:
            f.write(code)
            f.flush()

            use_t_num = 4
            ext = load(
                name="easier_cpu_extension",
                sources=[f.name],
                extra_cflags=[
                    "-O3",
                    "-D", f"THREAD_NUM_D={use_t_num}",
                    # f"-I/tmp/easier/support/cpu/{postfix}",
                    "-fopenmp"]
            )

        for name, module in self.modules.items():
            module.forward = getattr(ext, name)
