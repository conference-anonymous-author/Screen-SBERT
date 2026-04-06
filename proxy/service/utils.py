"""Utility functions for Triton client inference."""

from __future__ import annotations

import threading
from typing import Any, Dict

import numpy as np
import tritonclient.grpc as grpcclient

_THREAD_LOCAL = threading.local()


def _get_triton_client(triton_grpc_url: str) -> grpcclient.InferenceServerClient:
    clients = getattr(_THREAD_LOCAL, "triton_clients", None)
    if clients is None:
        clients = {}
        _THREAD_LOCAL.triton_clients = clients

    client = clients.get(triton_grpc_url)
    if client is None:
        client = grpcclient.InferenceServerClient(url=triton_grpc_url)
        clients[triton_grpc_url] = client
    return client


def run_triton_client(
    triton_grpc_url: str,
    model_name: str,
    input_dict: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    input_names = input_dict["input_names"]
    input_sizes = input_dict["input_sizes"]
    input_data = input_dict["input_data"]
    input_types = input_dict["input_types"]
    output_names = input_dict["output_names"]

    triton_client = _get_triton_client(triton_grpc_url)

    triton_inputs = []
    for name, size, data, dtype in zip(input_names, input_sizes, input_data, input_types):
        infer_input = grpcclient.InferInput(name, size, dtype)
        infer_input.set_data_from_numpy(data)
        triton_inputs.append(infer_input)

    triton_outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

    try:
        results = triton_client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )
    except Exception:
        clients = getattr(_THREAD_LOCAL, "triton_clients", None)
        if isinstance(clients, dict):
            clients.pop(triton_grpc_url, None)
        triton_client = _get_triton_client(triton_grpc_url)
        results = triton_client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )

    numpy_outputs: Dict[str, np.ndarray] = {}
    for name in output_names:
        output = results.as_numpy(name)
        assert output is not None
        numpy_outputs[name] = output

    return numpy_outputs
