import base64
import json
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

class KwarStaticMesh:
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
    unknown_str: str
    unk_floats: List[float]
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
        out = KwarStaticMesh()
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
            a = Attr(*out._unpack("=H i f f f f", data, 22))
            out.attrs.append(a)
        out.n_triangles = out._unpack("=I", data, 4)
        out.triangles = []
        for _ in range(out.n_triangles):
            t = Triangle(*out._unpack("=H H H H f f f", data, 20))
            out.triangles.append(t)
        out.unknown_str = out._unpack_kwstring(data)
        out.unk_floats = list(out._unpack("19f", data, 76))
        out.footer_str = out._unpack_kwstring(data)
        return out

def write_as_stl(sm: KwarStaticMesh) -> None:
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

def write_as_gltf(sm: KwarStaticMesh) -> None:
    vertices = []
    uvs = []
    for attr in sm.attrs:
        v = sm.vertices[attr.index]
        vertices.append([v.x, v.y, v.z])
        uvs.append([attr.u, attr.v])

    def pack(what: List, fmt: str) -> bytes:
        out = bytes()
        for x in what:
            out += struct.pack(fmt, *x)
        return out

    vertices_bin = pack(vertices, "f f f")
    uvs_bin = pack(uvs, "f f")

    # create gltf objects for a scene with a primitive triangle with indexed geometry
    gltf = pygltflib.GLTF2(
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    # fill in later
                ]
            )
        ],
        materials=[
            # fill in later
        ],
        textures=[
            # fill in later
        ],
        images=[
            # fill in later
        ],
        samplers=[
            # fill in later
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
                uri="data:application/octet-stream;base64," + base64.b64encode(uvs_bin).decode("utf-8"),
                byteLength=len(uvs_bin)
            ),
        ],
    )

    # Group triangles wrt their materials
    triangle_groups = [[] for _ in sm.matl_names]
    for trgl in sm.triangles:
        idx = trgl.unk1
        if idx >= len(triangle_groups):
            print("[!] invalid material idx %d" % idx)
            raise ValueError
        triangle_groups[idx].append([trgl.v1, trgl.v2, trgl.v3])

    assert sum(len(tg) for tg in triangle_groups) == len(sm.triangles), "couldn't group every triangle"

    # Create a material & mesh for each different matl index
    for i, matl_name in enumerate(sm.matl_names):
        triangles = triangle_groups[i]
        triangles_bin = pack(triangles, "H H H")
        # 1. add index buffer
        gltf.buffers.append(pygltflib.Buffer(
            uri="data:application/octet-stream;base64," + base64.b64encode(triangles_bin).decode("utf-8"),
            byteLength=len(triangles_bin)
        ))
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=len(gltf.buffers) - 1,
            byteLength=len(triangles_bin),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            componentType=pygltflib.UNSIGNED_SHORT,
            count=len(triangles) * 3,
            type=pygltflib.SCALAR,
        ))
        # 2. add material and image
        matl_path = os.path.join(*matl_name.split(".")) + ".png"
        matl_index = None
        if os.path.isfile(matl_path):
            gltf.images.append(pygltflib.Image(uri=matl_path))
            gltf.samplers.append(pygltflib.Sampler())
            gltf.textures.append(pygltflib.Texture(
                sampler=len(gltf.samplers) - 1,
                source=len(gltf.images) - 1),
            )
            gltf.materials.append(pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=len(gltf.textures) - 1),
                    metallicFactor=0,
                    roughnessFactor=1
                ),
                #alphaMode=pygltflib.MASK,
                #alphaCutoff=None
            ))
            matl_index = len(gltf.materials) - 1
        else:
            print("%s not found, omitting texture for its group" % matl_path)
        # 3. add mesh
        gltf.meshes[0].primitives.append(pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=0,
                TEXCOORD_0=1,
                #TEXCOORD_0=2,
            ),
            indices=len(gltf.accessors) - 1,
            material=matl_index
        ))

    gltf.convert_images(pygltflib.ImageFormat.DATAURI)
    path = os.path.join(sm.archive_name, sm.file_name + ".gltf")
    print("extracting to", path)
    gltf.save(path)

os.makedirs(header.archive_name, exist_ok=True)

mesh_to_matl_names = {}
# If meta.json exists update it, otherwise create one
meta_path = os.path.join(header.archive_name, "meta.json")
if os.path.isfile(meta_path):
    with open(meta_path, "r") as f:
        mesh_to_matl_names = json.load(f)

for f in files:
    sm = KwarStaticMesh.from_bytes(f.contents)
    # print("references materials:", ",".join(sm.matl_names))
    write_as_gltf(sm)
    mesh_name = ".".join([sm.archive_name, sm.file_name])
    mesh_to_matl_names[mesh_name] = sm.matl_names

print("writing metadata to", meta_path)
with open(meta_path, "w") as f:
    json.dump(mesh_to_matl_names, f)
