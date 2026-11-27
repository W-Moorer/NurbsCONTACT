"""Small CUDA Driver API adapter for device-controlled IF and WHILE graphs."""

import ctypes as ct
import os

P = ct.c_void_p
U = ct.c_uint
Q = ct.c_uint64


class Kernel(ct.Structure):
    _fields_ = [
        ("func", P),
        ("gridX", U),
        ("gridY", U),
        ("gridZ", U),
        ("blockX", U),
        ("blockY", U),
        ("blockZ", U),
        ("shared", U),
        ("parameters", ct.POINTER(P)),
        ("extra", P),
        ("kern", P),
        ("context", P),
    ]


class Condition(ct.Structure):
    _fields_ = [
        ("handle", Q),
        ("kind", ct.c_int),
        ("size", U),
        ("graphs", ct.POINTER(P)),
        ("context", P),
    ]


class Payload(ct.Union):
    _fields_ = [
        ("reserved", ct.c_longlong * 29),
        ("kernel", Kernel),
        ("conditional", Condition),
    ]


class Parameters(ct.Structure):
    _fields_ = [
        ("kind", ct.c_int),
        ("reserved0", ct.c_int * 3),
        ("payload", Payload),
        ("reserved2", ct.c_longlong),
    ]


class Driver:
    def __init__(self):
        self.dll = (
            ct.WinDLL("nvcuda.dll") if os.name == "nt" else ct.CDLL("libcuda.so.1")
        )
        declarations = {
            "cuCtxGetCurrent": [ct.POINTER(P)],
            "cuGraphCreate": [ct.POINTER(P), U],
            "cuGraphDestroy": [P],
            "cuGraphExecDestroy": [P],
            "cuGraphConditionalHandleCreate": [ct.POINTER(Q), P, P, U, U],
            "cuGraphAddNode": [
                ct.POINTER(P),
                P,
                ct.POINTER(P),
                ct.c_size_t,
                ct.POINTER(Parameters),
            ],
            "cuGraphInstantiateWithFlags": [ct.POINTER(P), P, Q],
            "cuGraphLaunch": [P, P],
            "cuGetErrorName": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for name, args in declarations.items():
            f = getattr(self.dll, name)
            f.argtypes = args
            f.restype = ct.c_int

    def call(self, name, *args):
        status = getattr(self.dll, name)(*args)
        if status:
            message = ct.c_char_p()
            self.dll.cuGetErrorName(status, ct.byref(message))
            raise RuntimeError(
                f"{name}: {message.value.decode() if message.value else status}"
            )


class Graph:
    def __init__(self, *, root=None, handle=None):
        if root is None:
            self.driver = Driver()
            self.handle = P()
            self.context = P()
            self.root = self
            self.references = []
            self.executable = P()
            self.driver.call("cuCtxGetCurrent", ct.byref(self.context))
            self.driver.call("cuGraphCreate", ct.byref(self.handle), 0)
        else:
            self.root = root
            self.driver = root.driver
            self.context = root.context
            self.handle = P(handle)
        self.last = None

    def condition(self, default=0):
        out = Q()
        self.driver.call(
            "cuGraphConditionalHandleCreate",
            ct.byref(out),
            self.root.handle,
            self.context,
            default,
            1,
        )
        return out.value

    def add(self, parameters):
        node = P()
        dependencies = (P * 1)(self.last) if self.last is not None else None
        self.driver.call(
            "cuGraphAddNode",
            ct.byref(node),
            self.handle,
            dependencies,
            1 if dependencies else 0,
            ct.byref(parameters),
        )
        self.last = node.value
        return node

    def conditional(self, condition, *, repeat=False):
        parameters = Parameters()
        parameters.kind = 13
        parameters.payload.conditional = Condition(
            condition, int(repeat), 1, None, self.context
        )
        self.add(parameters)
        child = Graph(root=self.root, handle=parameters.payload.conditional.graphs[0])
        self.root.references.append(child)
        return child

    def kernel(self, function, grid, block, args):
        grid = (*grid, 1, 1)[:3]
        block = (*block, 1, 1)[:3]
        values = [
            P(x.data.ptr) if hasattr(x, "data") and hasattr(x.data, "ptr") else x
            for x in args
        ]
        pointers = (P * len(values))(*(ct.cast(ct.pointer(x), P) for x in values))
        self.root.references.extend([function, values, pointers])
        parameters = Parameters()
        parameters.kind = 0
        parameters.payload.kernel = Kernel(
            function.kernel.ptr, *grid, *block, 0, pointers, None, None, None
        )
        self.add(parameters)

    def instantiate(self):
        self.driver.call(
            "cuGraphInstantiateWithFlags", ct.byref(self.executable), self.handle, 0
        )

    def launch(self, stream):
        self.driver.call("cuGraphLaunch", self.executable, P(stream.ptr))

    def close(self):
        if self.root is not self:
            return
        if self.executable.value:
            self.driver.call("cuGraphExecDestroy", self.executable)
            self.executable = P()
        if self.handle.value:
            self.driver.call("cuGraphDestroy", self.handle)
            self.handle = P()

    def __del__(self):
        if getattr(self, "root", None) is self:
            try:
                self.close()
            except Exception:
                pass
