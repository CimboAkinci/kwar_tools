# Export models from a .kwsk file with .kwan animations baked into the
# model file.

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

if len(sys.argv) != 3:
    print("usage: python unpack_kwan_unified.py [.kwsk file] [.kwan file]")
    sys.exit(1)

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

@dataclass
class KwarCatalog:
    file_name: any
    size: int
    offset: int

    def __post_init__(self):
        self.file_name = decode_utf16(self.file_name)

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
    header: KwarHeader
    catalog: KwarCatalog
    contents: bytes

    def __post_init__(self):
        self.contents = decompress_lzma(self.contents)

def load_archive(path: str) -> List[KwarFile]:
    C = ""
    with open(path, "rb") as f:
        C = f.read()

    header_bin = C[:104]
    unpacked = struct.unpack("=I 12s 16s 20s 12s 36s I", header_bin)
    header = KwarHeader(*unpacked)
    print(header)

    catalogs = []
    for i in range(header.n_files):
        start = 104 + i*32
        unpacked = struct.unpack("=24s I I", C[start:start+32])
        catalog = KwarCatalog(*unpacked)
        catalogs.append(catalog)

    files = []
    for cat in catalogs:
        file = KwarFile(header, cat, C[cat.offset+1:cat.offset+cat.size+1])
        files.append(file)

    return files

Vertex = namedtuple("Vertex", "x y z i j k")
Triangle = namedtuple("Triangle", "v1 v2 v3 unk1 i j k")
Bone = namedtuple("Bone", "name x y z w i j k parent")
ParticleInfo = namedtuple("ParticleInfo", "name unk1 unk2 unk3 unk4 unk5 unk6")
Joint = namedtuple("Joint", "weight vertex bone")
Attr = namedtuple("Attr", "index u v")

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
            out.vertices.append(Vertex(*values))
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
            out.triangles.append(Triangle(*values))
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
            bone = Bone(bone_name, *bone_values)
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

BoneRef = namedtuple("BoneRef", "name parent")
HitFrame = namedtuple("HitFrame", "unk1 unk2 str1 unk3 unk4 unk5 unk6 unk7 unk8 str2 unk9 unk10 unk11 unk12")
ImgData = namedtuple("RemainImageData", "unk1 unk2 unk3 str1 unk4 unk5 unk6 unk7 unk8 unk9 unk10 unk11 str2 unk12 unk13 unk14")

Rot = namedtuple("Rot", "x y z w")
Tran = namedtuple("Tran", "i j k")

class TrackBone:
    n_rots: int
    rots: List[Rot]
    n_trans: int
    trans: List[Tran]
    n_times: int
    times: List[float]

class MotionFrame:
    name: str
    unk1: int
    unk2: float
    unk3: float
    n_track_bones: int
    track_bones: List[TrackBone]
    n_hit_frames: int
    hit_frames: List[HitFrame]
    n_img_datas: int
    img_datas: List[ImgData]

class KwarSkelAnim:
    archive_name: str
    file_name: str
    unk1: int
    n_bone_refs: int
    bone_refs: List[BoneRef]
    n_motion_frames: int
    motion_frames: List[MotionFrame]
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

    def _unpack_motion_frame(self, data) -> MotionFrame:
        out = MotionFrame()
        out.name = self._unpack_kwstring(data)
        out.unk1, out.unk2, out.unk3 = self._unpack("=H f f", data, 10)
        out.n_track_bones = self._unpack("=I", data, 4)
        out.track_bones = []
        for _ in range(out.n_track_bones):
            track_bone = TrackBone()
            track_bone.n_rots = self._unpack("=I", data, 4)
            track_bone.rots = []
            for __ in range(track_bone.n_rots):
                rot = Rot(*self._unpack("=4f", data, 16))
                track_bone.rots.append(rot)
            track_bone.n_trans = self._unpack("=I", data, 4)
            track_bone.trans = []
            for __ in range(track_bone.n_trans):
                tran = Tran(*self._unpack("=3f", data, 12))
                track_bone.trans.append(tran)
            track_bone.n_times = self._unpack("=I", data, 4)
            track_bone.times = []
            for __ in range(track_bone.n_times):
                time = self._unpack("=f", data, 4)
                track_bone.times.append(time)
            out.track_bones.append(track_bone)
        out.n_hit_frames = self._unpack("=I", data, 4)
        out.hit_frames = []
        for _ in range(out.n_hit_frames):
            values1 = self._unpack("=H B", data, 3)
            str1 = self._unpack_kwstring(data)
            values2 = self._unpack("=B B B B B i", data, 9)
            str2 = self._unpack_kwstring(data)
            values3 = self._unpack("=f f f f", data, 16)
            hit_frame = HitFrame(*values1, str1, *values2, str2, *values3)
            out.hit_frames.append(hit_frame)
        out.n_img_datas = self._unpack("=I", data, 4)
        out.img_datas = []
        for _ in range(out.n_img_datas):
            values1 = self._unpack("=I I I", data, 12)
            str1 = self._unpack_kwstring(data)
            values2 = self._unpack("=I I I I B B B B", data, 20)
            str2 = self._unpack_kwstring(data)
            values3 = self._unpack("=I B I", data, 9)
            img_data = ImgData(*values1, str1, *values2, str2, *values3)
            out.img_datas.append(img_data)
        return out

    @staticmethod
    def from_bytes(data: bytes):
        out = KwarSkelAnim()
        out.archive_name = out._unpack_kwstring(data)
        out.file_name = out._unpack_kwstring(data)
        out.unk1, out.n_bone_refs = out._unpack("=I I", data, 8)
        assert out.n_bone_refs < 1024  # arbitrary size, just don't blow up
        out.bone_refs = []
        for _ in range(out.n_bone_refs):
            bone_name = out._unpack_kwstring(data)
            parent = out._unpack("=I", data, 4)
            bone = BoneRef(bone_name, parent)
            out.bone_refs.append(bone)
        out.n_motion_frames = out._unpack("=I", data, 4)
        out.motion_frames = []
        for _ in range(out.n_motion_frames):
            frame = out._unpack_motion_frame(data)
            out.motion_frames.append(frame)
        out.footer_str = out._unpack_kwstring(data)
        return out

def write_as_gltf(skm: KwarSkelMesh, ska: KwarSkelAnim) -> None:
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
        index_to_joints[joint.vertex].append(joint)
        # assert len(index_to_joints[joint.vertex]) <= 4

    joints = []
    weights = []
    for attr in lod0.attrs:
        my_joints = index_to_joints.get(attr.index)
        assert my_joints
        js = [j.bone for j in my_joints]
        ws = [j.weight for j in my_joints]
        if len(js) <= 4:
            js += [0]*(4-len(js))
            ws += [0]*(4-len(ws))
        else:
            js = js[:4]
            ws = ws[:4]
        joints.append(js)
        weights.append(ws)

    # Translate skel mesh inverse bind matrices to gltf
    ibms = []
    for quat in skm.quats:
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
        nodes=[
            pygltflib.Node(skin=0, mesh=0)
            # Fill in later
        ],
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
            pygltflib.Skin(inverseBindMatrices=5, joints=[
                # Fill in later
            ])
        ],
        animations=[
            # Fill in later
        ],
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
            pygltflib.Image(uri=os.path.join(skm.archive_name, skm.file_name)+".png")
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
    bone_name_to_gltf_node = {}
    for i, bone in enumerate(skm.bones):
        if i == 0:
            rot = [bone.x, bone.y, bone.z, bone.w]
            tran = [bone.i, bone.j, bone.k]
        else:
            rot = [bone.x, bone.y, bone.z, -bone.w]
            tran = [bone.i, bone.j, bone.k]
        gltf.nodes.append(pygltflib.Node(
            translation=tran,
            rotation=rot,
            name=bone.name,
            children=[],
        ))
        if bone.parent != i:
            gltf.nodes[bone.parent + 1].children.append(i+1)
        else: # Root bone
            gltf.scenes[0].nodes.append(i+1)
        gltf.skins[0].joints.append(len(gltf.nodes) - 1)
        bone_name_to_gltf_node[bone.name] = len(gltf.nodes) - 1

    def bounds(vecs: List[List[float]]) -> Tuple[List[float], List[float]]:
        max = [float("-inf")] * len(vecs[0])
        min = [float("inf")] * len(vecs[0])
        for vec in vecs:
            for i in range(len(vec)):
                if vec[i] > max[i]:
                    max[i] = vec[i]
                if vec[i] < min[i]:
                    min[i] = vec[i]
        return min, max

    # Utility fn: add a buffer to the file and return accessor index
    def add_accessor_from_buffer(data: bytes, componentType, count, type, min, max) -> int:
        gltf.buffers.append(pygltflib.Buffer(
            uri="data:application/octet-stream;base64," + base64.b64encode(data).decode("utf-8"),
            byteLength=len(data)
        ))
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=len(gltf.buffers) - 1,
            byteLength=len(data),
        ))
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            componentType=componentType,
            count=count,
            type=type,
            min=min,
            max=max
        ))
        return len(gltf.accessors) - 1

    # Translate animations to gltf
    for motion_frame in ska.motion_frames:
        animation = pygltflib.Animation(
            name=motion_frame.name,
            channels=[],
            samplers=[]
        )
        assert motion_frame.n_track_bones == ska.n_bone_refs
        # Translate TrackBones to animation channels
        for i, track_bone in enumerate(motion_frame.track_bones):
            assert track_bone.n_rots in [1, track_bone.n_times]
            assert track_bone.n_trans in [1, track_bone.n_times]

            # Copy the remaining channel if it's a rotation or translation-only track
            if track_bone.n_rots == 1:
                assert track_bone.n_trans == track_bone.n_times
                track_bone.rots *= track_bone.n_times
                track_bone.n_rots = track_bone.n_times
            elif track_bone.n_trans == 1:
                assert track_bone.n_rots == track_bone.n_times
                track_bone.trans *= track_bone.n_times
                track_bone.n_trans = track_bone.n_times

            target_node = bone_name_to_gltf_node[ska.bone_refs[i].name]
            # 1. Serialize time, rotation and translation tracks
            times = [[t/10] for t in track_bone.times]
            if i == 0:
                rots = [[rot.x, rot.y, rot.z, rot.w] for rot in track_bone.rots]
                trans = [[tran.i, tran.j, tran.k] for tran in track_bone.trans]
            else:
                rots = [[rot.x, rot.y, rot.z, -rot.w] for rot in track_bone.rots]
                trans = [[tran.i, tran.j, tran.k] for tran in track_bone.trans]

            times_bin = pack(times, "f")
            rots_bin = pack(rots, "4f")
            trans_bin = pack(trans, "3f")
            min_t, max_t = bounds(times)
            time_accessor = add_accessor_from_buffer(times_bin, pygltflib.FLOAT, len(times), pygltflib.SCALAR, min_t, max_t)

            min_rot, max_rot = bounds(rots)
            rot_accessor = add_accessor_from_buffer(rots_bin, pygltflib.FLOAT, len(rots), pygltflib.VEC4, min_rot, max_rot)

            min_tran, max_tran = bounds(trans)
            tran_accessor = add_accessor_from_buffer(trans_bin, pygltflib.FLOAT, len(trans), pygltflib.VEC3, min_tran, max_tran)

            rot_sampler = pygltflib.AnimationSampler(
                input=time_accessor,
                output=rot_accessor,
            )
            animation.samplers.append(rot_sampler)
            rot_channel = pygltflib.AnimationChannel(
                sampler=len(animation.samplers) - 1,
                target=pygltflib.AnimationChannelTarget(
                    node=target_node,
                    path="rotation"
                )
            )
            animation.channels.append(rot_channel)

            tran_sampler = pygltflib.AnimationSampler(
                input=time_accessor,
                output=tran_accessor,
            )
            animation.samplers.append(tran_sampler)
            tran_channel = pygltflib.AnimationChannel(
                sampler=len(animation.samplers) - 1,
                target=pygltflib.AnimationChannelTarget(
                    node=target_node,
                    path="translation"
                )
            )
            animation.channels.append(tran_channel)
        gltf.animations.append(animation)

    gltf.convert_images(pygltflib.ImageFormat.DATAURI)
    path = os.path.join(skm.archive_name, skm.file_name + ".gltf")
    print("writing to", path)
    gltf.save(path)

from pprint import pprint

skm_files = load_archive(sys.argv[1])
ska_files = load_archive(sys.argv[2])

os.makedirs(skm_files[0].header.archive_name, exist_ok=True)

# Single animation vs. multiple skeletal meshes
if len(ska_files) == 1:
    ska = KwarSkelAnim.from_bytes(ska_files[0].contents)
    print("animation file:", ska_files[0].catalog)
    for f in skm_files:
        print("skeleton file:", f.catalog)
        skm = KwarSkelMesh.from_bytes(f.contents)
        #pprint(vars(ska))
        #pprint(vars(ska.motion_frames[1]))
        #pprint(vars(ska.motion_frames[1].track_bones[0]))
        write_as_gltf(skm, ska)
else:
    assert len(skm_files) == len(ska_files), "# of animation files is different than # of skeleton files: don't know how to extract this"
    for f1, f2 in zip(skm_files, ska_files):
        print("animation file:", f2.catalog)
        print("skeleton file:", f1.catalog)
        skm = KwarSkelMesh.from_bytes(f1.contents)
        ska = KwarSkelAnim.from_bytes(f2.contents)
        #pprint(vars(ska))
        #pprint(vars(ska.motion_frames[1]))
        #pprint(vars(ska.motion_frames[1].track_bones[0]))
        write_as_gltf(skm, ska)
