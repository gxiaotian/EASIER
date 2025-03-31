import operator
from typing import List, Sequence, Literal, Tuple

import torch
from torch import nn, fx

import easier as esr
from easier.core.utils import EasierJitException
from easier.core.passes.utils import OrderedSet, EasierInterpreter, \
    tree_map, FX
from easier.core.passes.dataflow_fusion.node_group import \
    NodeGroup, get_node_group, sort_node_groups
from .module_generator import ModuleGenerator


class CodeGenerator(EasierInterpreter):
    def __init__(
        self,
        modules: Sequence[nn.Module],
        graphs: Sequence[fx.Graph],
        backend: Literal['troch', 'cpu', 'gpu'],
    ) -> None:
        super().__init__(modules, graphs)
        self.module_generator = ModuleGenerator(backend)

    def _collect_node_groups(self, graph: fx.Graph) -> OrderedSet[NodeGroup]:
        groups = OrderedSet()
        for node in graph.nodes:
            if node.op != FX.OUTPUT:
                groups.add(get_node_group(node))
        return groups

    def for_each_graph(self, graph_idx: int) -> None:
        node_dict = {}

        esr.logger.debug(
            f'Original fx.Graph for nn.Module {self.current_module}: \n' +
            self.current_graph.__str__())

        def _try_get_node(arg):
            if isinstance(arg, fx.Node):
                if arg.name in node_dict.keys():
                    return node_dict[arg.name]
                else:
                    raise EasierJitException(
                        f'A required argument node {arg.name} is not '
                        'in the node dictionary, probably caused by '
                        'incorrect topological order')
            else:
                return arg

        groups = sort_node_groups(
            list(self._collect_node_groups(self.current_graph)))

        module_idx = 0
        new_graph = fx.Graph()
        for group in groups:
            if len(group.nodes) == 1:
                node = list(group.nodes)[0]

                args = []
                for arg in node.args:
                    args.append(tree_map(arg, _try_get_node))

                kwargs = {}
                for key, value in node.kwargs.items():
                    kwargs[key] = tree_map(value, _try_get_node)

                new_node = new_graph.create_node(
                    node.op, node.target, tuple(args), kwargs, node.name)
                node_dict[new_node.name] = new_node
            else:
                name = f'easier_{graph_idx}_{module_idx}'
                module_idx += 1

                esr.logger.debug('Kernel ' + name)
                module, code = \
                    self.module_generator.add_node_group(name, group)
                esr.logger.debug(code)

                args = []
                for arg in group.all_input_nodes():
                    args.append(tree_map(arg, _try_get_node))

                self.current_module.add_module(name, module)
                new_node = new_graph.create_node(
                    "call_module", name, tuple(args), name=name)

                for i, output in enumerate(group.all_output_nodes()):
                    new_output = new_graph.create_node(
                        "call_function", operator.getitem,
                        (new_node, i), name=output.name)
                    node_dict[new_output.name] = new_output

        self.graphs[graph_idx] = new_graph
        esr.logger.debug(
            f'Compiled fx.Graph for nn.Module {self.current_module}: \n' +
            new_graph.__str__())

    def run(self) -> None:
        super().run()
        self.module_generator.load()


def generate_code(
    modules: List[nn.Module],
    graphs:  List[fx.Graph],
    backend: Literal['troch', 'cpu', 'gpu'],
) -> Tuple[List[nn.Module], List[fx.Graph]]:

    esr.logger.info('Code generation pass is running.')
    CodeGenerator(modules, graphs, backend).run()
    esr.logger.info('Code generation pass has completed.')

    return modules, graphs
