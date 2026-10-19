"""Reading and fixing a `.tflite` flatbuffer without running it (operators, tensors, quantisation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_edge_litert import schema_py_generated as schema


def _enum_names(enum_class: Any) -> dict[int, str]:
    return {
        value: name for name, value in vars(enum_class).items() if isinstance(value, int) and not name.startswith("_")
    }


_BUILTIN_NAMES = _enum_names(schema.BuiltinOperator)
_TYPE_NAMES = _enum_names(schema.TensorType)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scale: float | None = None
    zero_point: int | None = None


@dataclass(frozen=True)
class ModelInfo:
    operators: tuple[str, ...]  # every operator, in graph order
    custom_operators: tuple[str, ...]
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    size_bytes: int

    @property
    def operator_set(self) -> set[str]:
        return set(self.operators)

    @property
    def flex_operators(self) -> tuple[str, ...]:
        """Flex (TensorFlow) ops are custom ops whose name starts with `Flex`."""
        return tuple(name for name in self.custom_operators if name.startswith("Flex"))


def load_model(data: bytes) -> Any:
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))


def _tensor_info(tensor: Any) -> TensorInfo:
    name = tensor.name.decode() if isinstance(tensor.name, bytes) else str(tensor.name)
    quantization = tensor.quantization
    scale = zero_point = None
    if quantization is not None and quantization.scale is not None and len(quantization.scale):
        scale = float(quantization.scale[0])
        zero_point = int(quantization.zeroPoint[0]) if quantization.zeroPoint is not None else 0
    return TensorInfo(
        name, tuple(int(d) for d in tensor.shape), _TYPE_NAMES.get(tensor.type, str(tensor.type)), scale, zero_point
    )


def inspect_bytes(data: bytes) -> ModelInfo:
    model = load_model(data)
    subgraph = model.subgraphs[0]
    codes: list[tuple[str, bool]] = []
    for code in model.operatorCodes:
        builtin = max(int(code.builtinCode), int(code.deprecatedBuiltinCode))
        if builtin == schema.BuiltinOperator.CUSTOM:
            custom = code.customCode.decode() if isinstance(code.customCode, bytes) else str(code.customCode)
            codes.append((custom, True))
        else:
            codes.append((_BUILTIN_NAMES.get(builtin, f"BUILTIN_{builtin}"), False))
    operators = tuple(codes[op.opcodeIndex][0] for op in subgraph.operators)
    customs = tuple(sorted({name for name, is_custom in codes if is_custom}))
    return ModelInfo(
        operators,
        customs,
        tuple(_tensor_info(subgraph.tensors[i]) for i in subgraph.inputs),
        tuple(_tensor_info(subgraph.tensors[i]) for i in subgraph.outputs),
        len(data),
    )


def inspect_file(path: str | Path) -> ModelInfo:
    return inspect_bytes(Path(path).read_bytes())


def rename_io(data: bytes, input_name: str, output_name: str) -> bytes:
    """Gives the single input and output tensors (and the signature entries) the contract names."""
    import flatbuffers

    model = load_model(data)
    subgraph = model.subgraphs[0]
    old_input = subgraph.tensors[subgraph.inputs[0]].name
    old_output = subgraph.tensors[subgraph.outputs[0]].name
    subgraph.tensors[subgraph.inputs[0]].name = input_name.encode()
    subgraph.tensors[subgraph.outputs[0]].name = output_name.encode()
    for signature in model.signatureDefs or []:
        for entry in signature.inputs or []:
            if entry.tensorIndex == subgraph.inputs[0]:
                entry.name = input_name.encode()
        for entry in signature.outputs or []:
            if entry.tensorIndex == subgraph.outputs[0]:
                entry.name = output_name.encode()
    del old_input, old_output
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    return bytes(builder.Output())
