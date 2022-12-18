import warnings
from collections import OrderedDict, namedtuple
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime

try:
    import tensorrt as trt
except Exception:
    trt = None
import torch
from numpy import ndarray
from torch import Tensor

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class BackendWrapper:

    def __init__(
            self,
            weight: Union[str, Path],
            device: Optional[Union[str, int, torch.device]] = None) -> None:
        weight = Path(weight) if isinstance(weight, str) else weight
        assert weight.exists() and weight.suffix in ('.onnx', '.engine',
                                                     '.plan')
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.weight = weight
        self.device = device
        self.__build_model()
        self.__init_runtime()
        self.__warm_up(10)

    def __build_model(self) -> None:
        model_info = dict()
        num_input = num_output = 0
        names = []
        is_dynamic = False
        if self.weight.suffix == '.onnx':
            model_info['backend'] = 'ONNXRuntime'
            providers = ['CPUExecutionProvider']
            if 'cuda' in self.device.type:
                providers.insert(0, 'CUDAExecutionProvider')
            model = onnxruntime.InferenceSession(
                str(self.weight), providers=providers)
            for i, tensor in enumerate(model.get_inputs()):
                model_info[tensor.name] = dict(
                    shape=tensor.shape, dtype=tensor.type)
                num_input += 1
                names.append(tensor.name)
                is_dynamic |= any(
                    map(lambda x: isinstance(x, str), tensor.shape))
            for i, tensor in enumerate(model.get_outputs()):
                model_info[tensor.name] = dict(
                    shape=tensor.shape, dtype=tensor.type)
                num_output += 1
                names.append(tensor.name)
        else:
            model_info['backend'] = 'TensorRT'
            logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(logger, namespace='')
            with trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(
                    self.weight.read_bytes())
            profile_shape = []
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                shape = tuple(model.get_binding_shape(i))
                dtype = trt.nptype(model.get_binding_dtype(i))
                is_dynamic |= (-1 in shape)
                if model.binding_is_input(i):
                    num_input += 1
                    profile_shape.append(model.get_profile_shape(i, 0))
                else:
                    num_output += 1
                model_info[name] = dict(shape=shape, dtype=dtype)
                names.append(name)
            model_info['profile_shape'] = profile_shape

        self.num_input = num_input
        self.num_output = num_output
        self.names = names
        self.is_dynamic = is_dynamic
        self.model = model
        self.model_info = model_info

    def __init_runtime(self) -> None:
        bindings = OrderedDict()
        Binding = namedtuple('Binding',
                             ('name', 'dtype', 'shape', 'data', 'ptr'))
        if self.model_info['backend'] == 'TensorRT':
            context = self.model.create_execution_context()
            for name in self.names:
                shape, dtype = self.model_info[name].values()
                if self.is_dynamic:
                    cpu_tensor, gpu_tensor, ptr = None, None, None
                else:
                    cpu_tensor = np.empty(shape, dtype=np.dtype(dtype))
                    gpu_tensor = torch.from_numpy(cpu_tensor).to(self.device)
                    ptr = int(gpu_tensor.data_ptr())
                bindings[name] = Binding(name, dtype, shape, gpu_tensor, ptr)
        else:
            output_names = []
            for i, name in enumerate(self.names):
                if i >= self.num_input:
                    output_names.append(name)
                shape, dtype = self.model_info[name].values()
                bindings[name] = Binding(name, dtype, shape, None, None)
            context = partial(self.model.run, output_names)
        self.addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        self.bindings = bindings
        self.context = context

    def __infer(
            self, inputs: List[Union[ndarray,
                                     Tensor]]) -> List[Union[ndarray, Tensor]]:
        assert len(inputs) == self.num_input
        if self.model_info['backend'] == 'TensorRT':
            outputs = []
            for i, (name, gpu_input) in enumerate(
                    zip(self.names[:self.num_input], inputs)):
                if self.is_dynamic:
                    self.context.set_binding_shape(i, gpu_input.shape)
                self.addrs[name] = gpu_input.data_ptr()

            for i, name in enumerate(self.names[self.num_input:]):
                i += self.num_input
                if self.is_dynamic:
                    shape = tuple(self.context.get_binding_shape(i))
                    dtype = self.bindings[name].dtype
                    cpu_tensor = np.empty(shape, dtype=np.dtype(dtype))
                    out = torch.from_numpy(cpu_tensor).to(self.device)
                    self.addrs[name] = out.data_ptr()
                else:
                    out = self.bindings[name].data
                outputs.append(out)
            assert self.context.execute_v2(list(
                self.addrs.values())), 'Infer fault'
        else:
            input_feed = {
                name: inputs[i]
                for i, name in enumerate(self.names[:self.num_input])
            }
            outputs = self.context(input_feed)
        return outputs

    def __warm_up(self, n=10) -> None:
        for _ in range(n):
            _tmp = []
            if self.model_info['backend'] == 'TensorRT':
                for i, name in enumerate(self.names[:self.num_input]):
                    if self.is_dynamic:
                        shape = self.model_info['profile_shape'][i][1]
                        dtype = self.bindings[name].dtype
                        cpu_tensor = np.empty(shape, dtype=np.dtype(dtype))
                        _tmp.append(
                            torch.from_numpy(cpu_tensor).to(self.device))
                    else:
                        _tmp.append(self.bindings[name].data)
            else:
                print('Please warm up ONNXRuntime model by yourself')
                print("So this model doesn't warm up")
                return
            _ = self.__infer(_tmp)

    def __call__(
            self, inputs: Union[List, Tensor,
                                ndarray]) -> List[Union[Tensor, ndarray]]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = self.__infer(inputs)
        return outputs
