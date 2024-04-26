import base64
import lzma
import os
import pygltflib
import struct
import sys

from collections import namedtuple
from dataclasses import dataclass
from typing import List

if len(sys.argv) != 2:
    print("usage: python unpack_kwsm.py [.kwsm file]")
    sys.exit(1)

C = ""
with open(sys.argv[1], "rb") as f:
    C = f.read()

def decode_utf16(x: bytes) -> str:
    w = x.split(b"\x00\x00")[0]
    if len(w) % 2 == 1:
        w += b"\x00"
    return w.decode("utf-16")

@dataclass
class KwarHeader:
    size_maybe: int
    version_maybe: any
    unknown1: any
    archive_type: any
    unknown2: any
    archive_name: any
    n_files: int

    def __post_init__(self):
        self.version_maybe = decode_utf16(self.version_maybe)
        self.archive_type = decode_utf16(self.archive_type)
        self.archive_name = decode_utf16(self.archive_name)

header_bin = C[:104]
unpacked = struct.unpack("=I 12s 16s 20s 12s 36s I", header_bin)
header = KwarHeader(*unpacked)
print(header)

@dataclass
class KwarCatalog:
    file_name: any
    size: int
    offset: int

    def __post_init__(self):
        self.file_name = decode_utf16(self.file_name)

catalogs = []
for i in range(header.n_files):
    start = 104 + i*32
    unpacked = struct.unpack("=24s I I", C[start:start+32])
    catalog = KwarCatalog(*unpacked)
    catalogs.append(catalog)

# https://stackoverflow.com/a/37400585
def decompress_lzma(data):
    results = []
    while True:
        decomp = lzma.LZMADecompressor(lzma.FORMAT_AUTO, None, None)
        try:
            res = decomp.decompress(data)
        except lzma.LZMAError:
            if results:
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it.
            else:
                raise  # Error on the first iteration; bail out.
        results.append(res)
        data = decomp.unused_data
        if not data:
            break
        if not decomp.eof:
            raise lzma.LZMAError("Compressed data ended before the end-of-stream marker was reached")
    return b"".join(results)

@dataclass
class KwarFile:
    header: KwarCatalog
    contents: any

    def __post_init__(self):
        self.contents = decompress_lzma(self.contents)

files = []
for cat in catalogs:
    file = KwarFile(cat, C[cat.offset+1:cat.offset+cat.size+1])
    files.append(file)

Vertex = namedtuple("Vertex", "z x y i j k")
Attr = namedtuple("Attr", "index unk1 u v unk4 unk5")
Triangle = namedtuple("Triangle", "unk1 v1 v2 v3 i j k")
AnimVertex = namedtuple("AnimVertex", "z x y")
Keyframe = namedtuple("Keyframe", "time_ms anim_vertices")

class KwarAnimMesh:
    archive_name: str
    file_name: str
    unk1: int
    n_matls: int
    matl_names: List[str]
    n_vertices: int
    vertices: List[Vertex]
    n_attrs: int
    attrs: List[Attr]
    n_triangles: int
    triangles: List[Triangle]
    unk_floats: List[float]
    n_keyframes: int
    n_vertices_per_keyframe: int
    keyframes: List[Keyframe]
    footer_str: str

    off = 0
    def _unpack(self, fmt: str, data: bytes, size: int):
        out = struct.unpack(fmt, data[self.off:self.off+size])
        self.off += size
        if len(out) == 1:
            return out[0]
        return out

    def _unpack_kwstring(self, data) -> str:
        size = self._unpack("=I", data, 4)
        s = self._unpack("=%ds" % size, data, size)
        return decode_utf16(s)

    @staticmethod
    def from_bytes(data: bytes):
        out = KwarAnimMesh()
        out.archive_name = out._unpack_kwstring(data)
        out.file_name = out._unpack_kwstring(data)
        out.unk1, out.n_matls = out._unpack("=I I", data, 8)
        assert out.n_matls < 64  # arbitrary size, just don't blow up
        out.matl_names = []
        for _ in range(out.n_matls):
            out.matl_names.append(out._unpack_kwstring(data))
        out.n_vertices = out._unpack("=I", data, 4)
        out.vertices = []
        for _ in range(out.n_vertices):
            v = Vertex(*out._unpack("=f f f f f f", data, 24))
            out.vertices.append(v)
        out.n_attrs = out._unpack("=I", data, 4)
        out.attrs = []
        for _ in range(out.n_attrs):
            a = Attr(*out._unpack("=H f f f f f", data, 22))
            out.attrs.append(a)
        out.n_triangles = out._unpack("=I", data, 4)
        out.triangles = []
        for _ in range(out.n_triangles):
            t = Triangle(*out._unpack("=H H H H f f f", data, 20))
            out.triangles.append(t)
        out.unk_floats = list(out._unpack("=12f", data, 48))
        out.n_keyframes, out.n_vertices_per_keyframe = list(out._unpack("=I I", data, 8))
        out.keyframes = []
        for _ in range(out.n_keyframes):
            verts = []
            index = out._unpack("=I", data, 4)
            for __ in range(out.n_vertices_per_keyframe):
                av = AnimVertex(*out._unpack("=f f f", data, 12))
                verts.append(av)
            kf = Keyframe(index, list(verts))
            out.keyframes.append(kf)
        out.footer_str = out._unpack_kwstring(data)
        return out

def write_as_stl(sm: KwarAnimMesh) -> None:
    path = os.path.join(sm.archive_name, sm.file_name + ".stl")
    f = open(path, "w")
    f.write("solid %s\n" % sm.file_name)
    for t in sm.triangles:
        f.write("facet normal %f %f %f\n" % (t.i, t.j, t.k))
        f.write("outer loop\n")
        v = sm.vertices[sm.attrs[t.v1].index]
        f.write("vertex %f %f %f\n" % (v.x, v.y, v.z))
        v = sm.vertices[sm.attrs[t.v2].index]
        f.write("vertex %f %f %f\n" % (v.x, v.y, v.z))
        v = sm.vertices[sm.attrs[t.v3].index]
        f.write("vertex %f %f %f\n" % (v.x, v.y, v.z))
        f.write("endloop\n")
        f.write("endfacet\n")
    f.close()

def write_as_gltf(am: KwarAnimMesh) -> None:
    vertices = []
    uvs = []
    for attr in am.attrs:
        v = am.vertices[attr.index]
        vertices.append([v.x, v.y, v.z])
        uvs.append([attr.u, attr.v])

    triangles = []
    for trgl in am.triangles:
        triangles.append([trgl.v1, trgl.v2, trgl.v3])

    def pack(what: List, fmt: str) -> bytes:
        out = bytes()
        for x in what:
            out += struct.pack(fmt, *x)
        return out

    triangles_bin = pack(triangles, "H H H")
    vertices_bin = pack(vertices, "f f f")
    uvs_bin = pack(uvs, "f f")

    # AnimMeshes are difficult to export as gltf: gltf's support for
    # morph target animation is a bit weird, so each keyframe becomes
    # a different morph target, and the weight interpolation becomes
    # one-hot encoded. This seems to be the only way to do it...

    # create gltf objects for a scene with a primitive triangle with indexed geometry
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=0,
                            TEXCOORD_0=2,
                        ),
                        targets=[],
                        indices=1,
                        material=0
                    )
                ],
                weights=[]
            )
        ],
        animations=[],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0),
                    metallicFactor=0,
                    roughnessFactor=1
                )
            )
        ],
        textures=[
            pygltflib.Texture(sampler=0, source=0)
        ],
        images=[
            pygltflib.Image(uri=os.path.join(*am.matl_names[0].split(".")) + ".png")
        ],
        samplers=[
            pygltflib.Sampler()
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=len(triangles) * 3,
                type=pygltflib.SCALAR,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.FLOAT,
                count=len(uvs),
                type=pygltflib.VEC2,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(vertices_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=1,
                byteLength=len(triangles_bin),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=2,
                byteLength=len(uvs_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(vertices_bin).decode("utf-8"),
                byteLength=len(vertices_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(triangles_bin).decode("utf-8"),
                byteLength=len(triangles_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(uvs_bin).decode("utf-8"),
                byteLength=len(uvs_bin)
            ),
        ],
    )

    # Add animation. The input is the time values and the output is the
    # weight values for each time period.
    # (Actually they are both inputs but gltf decided to name them this way.)
    gltf.animations.append(pygltflib.Animation(
        samplers=[
            pygltflib.AnimationSampler(
                input=len(gltf.accessors),
                output=len(gltf.accessors)+1
            )
        ],
        channels=[
            pygltflib.AnimationChannel(
                sampler=0,
                target=pygltflib.AnimationChannelTarget(node=0, path="weights")
            )
        ]
    ))

    # Add buffers, buffer views and accessors for animation.
    # First buffer: time values (float, seconds)
    times = [[kf.time_ms / 1000] for kf in am.keyframes]
    times_bin = pack(times, "f")
    gltf.buffers.append(pygltflib.Buffer(
        uri="data:application/octet-stream;base64," + base64.b64encode(times_bin).decode("utf-8"),
        byteLength=len(times_bin)
    ))
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=len(gltf.buffers)-1,
            byteLength=len(times_bin)
        )
    )
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=len(gltf.bufferViews)-1,
            componentType=pygltflib.FLOAT,
            count=len(times),
            type=pygltflib.SCALAR
        )
    )
    # Second buffer: weight values (one-hot encoded vecs)
    weights = []
    for i in range(len(am.keyframes)):
        ws = [0] * len(am.keyframes)
        ws[i] = 1
        weights.append(ws)
    weights_bin = pack(weights, "%df" % len(am.keyframes))
    gltf.buffers.append(pygltflib.Buffer(
        uri="data:application/octet-stream;base64," + base64.b64encode(weights_bin).decode("utf-8"),
        byteLength=len(weights_bin)
    ))
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=len(gltf.buffers)-1,
            byteLength=len(weights_bin)
        )
    )
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=len(gltf.bufferViews)-1,
            componentType=pygltflib.FLOAT,
            count=len(weights) * len(weights[0]),
            type=pygltflib.SCALAR
        )
    )
    # More buffers: keyframes themselves
    for kf in am.keyframes:
        # Compute displacement from original mesh here
        dp = []
        for attr in am.attrs:
            v = am.vertices[attr.index]
            kfv = kf.anim_vertices[attr.index]
            dp.append([kfv.x-v.x, kfv.y-v.y, kfv.z-v.z])
        # Add displacements as a buffer and an accessor
        dp_bin = pack(dp, "f f f")
        gltf.buffers.append(pygltflib.Buffer(
            uri="data:application/octet-stream;base64," + base64.b64encode(dp_bin).decode("utf-8"),
            byteLength=len(dp_bin)
        ))
        gltf.bufferViews.append(
            pygltflib.BufferView(
                buffer=len(gltf.buffers)-1,
                byteLength=len(dp_bin),
                target=pygltflib.ARRAY_BUFFER,
            )
        )
        gltf.accessors.append(
            pygltflib.Accessor(
                bufferView=len(gltf.bufferViews)-1,
                componentType=pygltflib.FLOAT,
                count=len(dp),
                type=pygltflib.VEC3
            )
        )
        # Add the created accessor as a new target and weight to the node
        gltf.meshes[0].primitives[0].targets.append(pygltflib.Attributes(
            POSITION=len(gltf.accessors)-1,
        ))
        gltf.meshes[0].weights.append(0)

    gltf.convert_images(pygltflib.ImageFormat.DATAURI)
    path = os.path.join(am.archive_name, am.file_name + ".gltf")
    print("extracting to", path)
    gltf.save(path)

os.makedirs(header.archive_name, exist_ok=True)

for f in files:
    am = KwarAnimMesh.from_bytes(f.contents)
    print(f.header)
    print("references materials: (only the first is used)", ",".join(am.matl_names))
    write_as_gltf(am)
