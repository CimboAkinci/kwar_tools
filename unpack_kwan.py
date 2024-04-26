import base64
import lzma
import os
import pygltflib
import struct
import subprocess
import sys

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

if len(sys.argv) != 2:
    print("usage: python unpack_kwan.py [.kwan file]")
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

def write_as_gltf(ska: KwarSkelAnim) -> None:
    def pack(what: List, fmt: str) -> bytes:
        out = bytes()
        for x in what:
            out += struct.pack(fmt, *x)
        return out

    # Prepare a placeholder mesh, otherwise import tools fk up
    vertices = [[0,0,0], [0,0,1], [0,1,0]]
    triangles = [[0,1,2]]
    joints = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    weights = [[1,0,0,0], [1,0,0,0], [1,0,0,0]]

    # Pack into base64 data for writing to gltf
    vertices_b64 = "data:application/octet-stream;base64," + base64.b64encode(pack(vertices, "3f")).decode("utf-8")
    triangles_b64 = "data:application/octet-stream;base64," + base64.b64encode(pack(triangles, "3H")).decode("utf-8")
    joints_b64 = "data:application/octet-stream;base64," + base64.b64encode(pack(joints, "4H")).decode("utf-8")
    weights_b64 = "data:application/octet-stream;base64," + base64.b64encode(pack(weights, "4f")).decode("utf-8")

    # Write scene with single triangle + animations
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(skin=0, mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=0,
                JOINTS_0=2,
                WEIGHTS_0=3
            ),
            indices=1,
        )])],
        skins=[
            pygltflib.Skin(joints=[
                # Fill in later
            ])
        ],
        animations=[],
        samplers=[],
        accessors=[
            pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT, count=len(vertices), type=pygltflib.VEC3, min=[0,0,0], max=[0,1,1]),
            pygltflib.Accessor(bufferView=1, componentType=pygltflib.UNSIGNED_SHORT, count=len(triangles) * 3, type=pygltflib.SCALAR),
            pygltflib.Accessor(bufferView=2, componentType=pygltflib.UNSIGNED_SHORT, count=len(joints), type=pygltflib.VEC4),
            pygltflib.Accessor(bufferView=3, componentType=pygltflib.FLOAT, count=len(weights), type=pygltflib.VEC4),
        ],
        bufferViews=[
            pygltflib.BufferView(buffer=0, byteLength=12*len(vertices), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=1, byteLength=6*len(triangles), target=pygltflib.ELEMENT_ARRAY_BUFFER),
            pygltflib.BufferView(buffer=2, byteLength=8*len(joints), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=3, byteLength=16*len(weights), target=pygltflib.ARRAY_BUFFER),
        ],
        buffers=[
            pygltflib.Buffer(uri=vertices_b64, byteLength=12*len(vertices)),
            pygltflib.Buffer(uri=triangles_b64, byteLength=6*len(triangles)),
            pygltflib.Buffer(uri=joints_b64, byteLength=8*len(joints)),
            pygltflib.Buffer(uri=weights_b64, byteLength=16*len(weights)),
        ],
    )

    # Translate bone transformations to gltf nodes and register them
    bone_name_to_gltf_node = {}
    for i, bone in enumerate(ska.bone_refs):
        gltf.nodes.append(pygltflib.Node(
            name=bone.name,
            children=[],
        ))
        if bone.parent != i:
            gltf.nodes[bone.parent+1].children.append(i+1)
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

    path = os.path.join(ska.archive_name, ska.file_name + "_anims.gltf")
    print("writing to", path)
    gltf.save(path)

    compacted_path = os.path.join(ska.archive_name, ska.file_name + "_anims.glb")
    print("calling gltfpack.exe to optimize into", compacted_path)
    my_path = os.path.realpath(__file__)
    gltfpack_path = os.path.join(os.path.dirname(my_path), "gltfpack.exe")
    subprocess.run([gltfpack_path, "-i", path, "-o", compacted_path])

files = load_archive(sys.argv[1])

os.makedirs(files[0].header.archive_name, exist_ok=True)

for f in files:
    ska = KwarSkelAnim.from_bytes(f.contents)
    write_as_gltf(ska)
