import base64
import lzma
import numpy as np
import os
import pygltflib
import struct
import sys

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

if len(sys.argv) != 2:
    print("usage: python unpack_kwsk.py [.kwsk file]")
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
    contents: bytes

    def __post_init__(self):
        self.contents = decompress_lzma(self.contents)

files = []
for cat in catalogs:
    file = KwarFile(cat, C[cat.offset+1:cat.offset+cat.size+1])
    files.append(file)

Vertex = namedtuple("Vertex", "x y z i j k")
Triangle = namedtuple("Triangle", "v1 v2 v3 unk1 i j k")
Bone = namedtuple("Bone", "name x y z w i j k parent")
ParticleInfo = namedtuple("ParticleInfo", "name unk1 unk2 unk3 unk4 unk5 unk6")
Joint = namedtuple("Joint", "weight vertex bone")
Attr = namedtuple("Attr", "index u v")

# .kw lives in a different space than .gltf
def transform_vertex(v: Vertex) -> Vertex:
    return Vertex(v.x, v.y, v.z, v.i, v.j, v.k)

def transform_triangle(t: Triangle) -> Triangle:
    return Triangle(t.v1, t.v2, t.v3, t.unk1, t.i, t.j, t.k)

def transform_bone(bone: Bone) -> Bone:
    return Bone(bone.name, bone.x, bone.y, bone.z, -bone.w, bone.i, bone.j, bone.k, bone.parent)

def transform_rot(rot: List[float]) -> List[float]:
    return [rot[0], rot[1], rot[2], rot[3]]

def transform_tran(tran: List[float]) -> List[float]:
    return [tran[0], tran[1], tran[2]]

class LodMesh:
    n_unk1: int
    unk1: List[float]
    n_vertices: int
    vertices: List[Vertex]
    n_extra_vertices: int
    extra_vertices: List[Vertex]
    n_twobytes: int
    twobytes: List[int]
    n_joints: int
    joints: List[Joint]
    n_attrs: int
    attrs: List[Attr]
    n_triangles: int
    triangles: List[Triangle]

class KwarSkelMesh:
    archive_name: str
    file_name: str
    unk1: int
    unk2: int
    n_bones: int
    bones: List[Bone]
    n_quats: int
    quats: List[List[float]]
    n_lods: int
    lods: List[LodMesh]
    n_particles: int
    particles: List[any]
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

    def _unpack_lod(self, data) -> LodMesh:
        out = LodMesh()
        out.n_unk1 = self._unpack("=I", data, 4)
        out.unk1 = []
        for _ in range(out.n_unk1):
            out.unk1.append(self._unpack("=f", data, 4))
        out.n_vertices = self._unpack("=I", data, 4)
        out.vertices = []
        for _ in range(out.n_vertices):
            values = self._unpack("=6f", data, 24)
            vertex = transform_vertex(Vertex(*values))
            out.vertices.append(vertex)
        out.n_extra_vertices = self._unpack("=I", data, 4)
        out.extra_vertices = []
        for _ in range(out.n_extra_vertices):
            values = self._unpack("=6f", data, 24)
            out.extra_vertices.append(Vertex(*values))
        out.n_twobytes = self._unpack("=I", data, 4)
        out.twobytes = []
        for _ in range(out.n_twobytes):
            out.twobytes.append(self._unpack("=H", data, 2))
        out.n_joints = self._unpack("=I", data, 4)
        out.joints = []
        for _ in range(out.n_joints):
            values = self._unpack("=f H H", data, 8)
            out.joints.append(Joint(*values))
        out.n_attrs = self._unpack("=I", data, 4)
        out.attrs = []
        for _ in range(out.n_attrs):
            values = self._unpack("=H f f", data, 10)
            out.attrs.append(Attr(*values))
        out.n_triangles = self._unpack("=I", data, 4)
        out.triangles = []
        for _ in range(out.n_triangles):
            values = self._unpack("=H H H H f f f", data, 20)
            triangle = transform_triangle(Triangle(*values))
            out.triangles.append(triangle)
        return out

    @staticmethod
    def from_bytes(data: bytes):
        out = KwarSkelMesh()
        out.archive_name = out._unpack_kwstring(data)
        out.file_name = out._unpack_kwstring(data)
        out.unk1, out.unk2, out.n_bones = out._unpack("=I I I", data, 12)
        assert out.n_bones < 1024  # arbitrary size, just don't blow up
        out.bones = []
        for _ in range(out.n_bones):
            bone_name = out._unpack_kwstring(data)
            bone_values = out._unpack("=7f I", data, 32)
            bone = transform_bone(Bone(bone_name, *bone_values))
            out.bones.append(bone)
        out.n_quats = out._unpack("=I", data, 4)
        out.quats = []
        for _ in range(out.n_quats):
            quat_values = out._unpack("=12f", data, 48)
            out.quats.append(list(quat_values))
        out.n_lods = out._unpack("=I", data, 4)
        out.lods = []
        for _ in range(out.n_lods):
            lod = out._unpack_lod(data)
            out.lods.append(lod)
        out.n_particles = out._unpack("=I", data, 4)
        out.particles = []
        for _ in range(out.n_particles):
            values1 = out._unpack("=B f", data, 5)
            name = out._unpack_kwstring(data)
            values2 = out._unpack("=f f f f", data, 16)
            p = ParticleInfo(name, *values1, *values2)
            out.particles.append(p)
        out.footer_str = out._unpack_kwstring(data)
        return out

def write_as_gltf(skm: KwarSkelMesh) -> None:
    lod0 = skm.lods[0]
    vertices = []
    uvs = []
    for attr in lod0.attrs:
        v = lod0.vertices[attr.index]
        vertices.append([v.x, v.y, v.z])
        uvs.append([attr.u, attr.v])

    triangles = []
    for trgl in lod0.triangles:
        triangles.append([trgl.v1, trgl.v2, trgl.v3])

    # Translate skel mesh joints to gltf joints and weights.
    # Skel mesh joints look like: weight index joint
    # gltf JOINTS_0 look like: for each vertex, 4 bone indices (0 means none)
    index_to_joints = {}
    for joint in lod0.joints:
        if not index_to_joints.get(joint.vertex):
            index_to_joints[joint.vertex] = []
        if len(index_to_joints[joint.vertex]) == 4:
            print("[!] vertex %d is affected by more than 4 bones which is the Unity limit, ignoring bone %d" % (joint.vertex, joint.bone))
        else:
            index_to_joints[joint.vertex].append(joint)

    joints = []
    weights = []
    for attr in lod0.attrs:
        my_joints = index_to_joints.get(attr.index)
        assert my_joints
        js = [j.bone for j in my_joints]
        ws = [j.weight for j in my_joints]
        js += [0]*(4-len(js))
        ws += [0]*(4-len(ws))
        joints.append(js)
        weights.append(ws)

    # Compute IBMs
    ibms = []
    for i in range(len(skm.bones)):
        quat = skm.quats[i]
        loaded_ibm = np.array([
            [quat[3], quat[4], quat[5], quat[0]],
            [quat[6], quat[7], quat[8], quat[1]],
            [quat[9], quat[10], quat[11], quat[2]],
            [0, 0, 0, 1]
        ])
        ibm = loaded_ibm.flatten("F")
        ibms.append(ibm)

    def pack(what: List, fmt: str) -> bytes:
        out = bytes()
        for x in what:
            out += struct.pack(fmt, *x)
        return out

    triangles_bin = pack(triangles, "H H H")
    vertices_bin = pack(vertices, "f f f")
    uvs_bin = pack(uvs, "f f")

    joints_bin = pack(joints, "4H")
    weights_bin = pack(weights, "4f")
    ibms_bin = pack(ibms, "16f")

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(skin=0, mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(
                            POSITION=0,
                            TEXCOORD_0=2,
                            JOINTS_0=3,
                            WEIGHTS_0=4
                        ),
                        indices=1,
                        material=0
                    )
                ]
            )
        ],
        skins=[
            pygltflib.Skin(inverseBindMatrices=5, joints=[])
        ],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    metallicFactor=0,
                    roughnessFactor=1
                )
            )
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
            pygltflib.Accessor(
                bufferView=3,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=len(joints),
                type=pygltflib.VEC4,
            ),
            pygltflib.Accessor(
                bufferView=4,
                componentType=pygltflib.FLOAT,
                count=len(weights),
                type=pygltflib.VEC4,
            ),
            pygltflib.Accessor(
                bufferView=5,
                componentType=pygltflib.FLOAT,
                count=len(ibms),
                type=pygltflib.MAT4,
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
            pygltflib.BufferView(
                buffer=3,
                byteLength=len(joints_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=4,
                byteLength=len(weights_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=5,
                byteLength=len(ibms_bin),
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
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(joints_bin).decode("utf-8"),
                byteLength=len(joints_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(weights_bin).decode("utf-8"),
                byteLength=len(weights_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(ibms_bin).decode("utf-8"),
                byteLength=len(ibms_bin)
            ),
        ],
    )

    # Translate bone transformations to gltf nodes and register them
    for i, bone in enumerate(skm.bones):
        gltf.nodes.append(pygltflib.Node(
            translation=[bone.i, bone.j, bone.k],
            rotation=[bone.x, bone.y, bone.z, bone.w],
            name=bone.name,
            children=[],
        ))
        if bone.parent != i:
            gltf.nodes[bone.parent + 1].children.append(i+1)
        else: # Root bone
            gltf.scenes[0].nodes.append(i+1)
        gltf.skins[0].joints.append(len(gltf.nodes) - 1)

    path = os.path.join(skm.archive_name, skm.file_name + ".gltf")
    print("writing to", path)
    gltf.save(path)

from pprint import pprint

os.makedirs(header.archive_name, exist_ok=True)

for f in files:
    print(f.header)
    skm = KwarSkelMesh.from_bytes(f.contents)
    write_as_gltf(skm)
    #pprint(vars(skm))
    #pprint(vars(skm.lods[0]))

