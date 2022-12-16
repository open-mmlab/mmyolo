import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

try:
    import tensorrt as trt
except Exception:
    trt = None
import torch

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


class BackendWrapper(torch.nn.Module):

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]):
        super().__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=device)
        self.__init_engine()
        self.__init_bindings()

    def __init_engine(self):
        logger = trt.Logger(trt.Logger.ERROR)
        self.log = partial(logger.log, trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace='')
        self.logger = logger
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]

        num_inputs = sum(
            [model.binding_is_input(i) for i in range(model.num_bindings)])

        self.is_dynamic = -1 in model.get_binding_shape(0)

        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]
        self.num_inputs = num_inputs
        self.num_outputs = len(names) - num_inputs
        self.num_bindings = len(names)

    def __init_bindings(self):
        input_dtypes = []
        input_shapes = []
        output_dtypes = []
        output_shapes = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = self.torch_dtype_from_trt(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            input_dtypes.append(dtype)
            input_shapes.append(shape)
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = self.torch_dtype_from_trt(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            output_dtypes.append(dtype)
            output_shapes.append(shape)
        self.input_dtypes = input_dtypes
        self.input_shapes = input_shapes
        self.output_dtypes = output_dtypes
        self.output_shapes = output_shapes
        if not self.is_dynamic:
            self.output_tensor = [
                torch.empty(shape, dtype=dtype, device=self.device)
                for (shape, dtype) in zip(output_shapes, output_dtypes)
            ]

    def forward(self, *inputs):

        assert len(inputs) == self.num_inputs

        contiguous_inputs: List[torch.Tensor] = [
            i.contiguous() for i in inputs
        ]
        bindings: List[int] = [0] * self.num_bindings

        for i in range(self.num_inputs):
            bindings[i] = contiguous_inputs[i].data_ptr()
            self.context.set_binding_shape(i,
                                           tuple(contiguous_inputs[i].shape))

        # create output tensors
        outputs: List[torch.Tensor] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                output = torch.empty(
                    size=shape,
                    dtype=self.output_dtypes[i],
                    device=self.device)

            else:
                output = self.output_tensor[i]
            outputs.append(output)
            bindings[j] = output.data_ptr()

        self.context.execute_async_v2(bindings, self.stream.cuda_stream)

        return tuple(outputs)

    @staticmethod
    def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
        """Convert TensorRT data types to PyTorch data types.

        Args:
            dtype (TRTDataType): A TensorRT data type.
        Returns:
            The equivalent PyTorch data type.
        """
        if dtype == trt.int8:
            return torch.int8
        elif trt.__version__ >= '7.0' and dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError(f'{dtype} is not supported by torch')
