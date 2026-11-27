"""Build one device-controlled graph for a complete implicit mesh simulation."""

import ctypes as ct
from time import perf_counter
import cupy as cp
import numpy as np
from .conditional_graph import Graph


def run(model, step):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Expected a positive finite time step")
    started = perf_counter()
    module = model.module
    size = cp.empty(1, cp.int32)
    module.get_function("mesh_graph_size")((1,), (1,), (size,))
    state = cp.zeros(int(size.get()[0]), cp.uint8)
    model.graph_state = state
    graph = Graph()
    steps = graph.condition()
    newton = graph.condition()
    support_blocks = min(model.blocks, 8)
    functions = {}

    def launch(g, name, grid=1, block=1, args=()):
        f = functions.setdefault(name, module.get_function("mesh_graph_" + name))
        g.kernel(f, (grid,), (block,), (model.scene, state, *args))

    def force_pipeline(g):
        launch(g, "force_init")
        for pair, law in enumerate(model.laws):
            fresh = graph.condition()
            launch(g, "pair", args=(ct.c_int(pair), ct.c_uint64(fresh)))
            body = g.conditional(fresh)
            source = model.meshes[int(law["b"])]
            launch(
                body, "broad", grid=(int(source.leaves.size) + 127) // 128, block=128
            )
            launch(body, "filter", grid=128, block=128)
            launch(body, "narrow", grid=64, block=128)
            launch(body, "place", grid=128, block=128)
            launch(body, "confirm", grid=128, block=32)
            launch(body, "group", block=128)
            launch(body, "support", grid=support_blocks, block=128)
            launch(body, "collect")
            launch(g, "accumulate", args=(ct.c_int(pair),))
        launch(g, "linearize")

    try:
        launch(
            graph,
            "init",
            args=(
                ct.c_int(model.steps),
                ct.c_double(step),
                ct.c_uint64(steps),
                ct.c_uint64(newton),
                ct.c_uint64(0),
                ct.c_int(support_blocks),
            ),
        )
        force_pipeline(graph)
        launch(graph, "initial_finish")
        stepping = graph.conditional(steps, repeat=True)
        launch(stepping, "prepare_step")
        iteration = stepping.conditional(newton, repeat=True)
        force_pipeline(iteration)
        launch(iteration, "newton")
        launch(stepping, "finish_step")
        graph.instantiate()
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        preparation = perf_counter() - started
        start = perf_counter()
        graph.launch(stream)
        stream.synchronize()
        elapsed = perf_counter() - start
        status = model.status()
        counts = cp.empty(16)
        module.get_function("mesh_graph_telemetry")((1,), (1,), (state, counts))
        values = counts.get()
        model.integration_stats = dict(
            accepted_substeps_by_level=values[:9].astype(int).tolist(),
            geometry_retries_by_status={"26": int(values[9]), "27": int(values[10])},
            newton_rejections=int(values[11]),
            first_geometry_failure=(values[12:].tolist() if values[12] else None),
        )
        if status[0] or abs(status[1] - model.steps * step) > 1e-8:
            raise RuntimeError(("Incomplete graph mesh dynamics", status.tolist()))
        return model.output.get(), dict(
            online_s=elapsed,
            setup_s=model.setup_s + preparation,
            graph_setup_s=preparation,
            status=status.tolist(),
            integration=model.integration_stats,
            execution="device conditional CUDA graph",
            p99_step_s=float(np.percentile(np.diff(model.finished.get()) * 1e-9, 99)),
            kernels={name: f.attributes for name, f in functions.items()},
        )
    finally:
        graph.close()
