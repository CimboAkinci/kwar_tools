import base64
import json
import lzma
import os
import pygltflib
import struct
import sys

from collections import namedtuple
from typing import List, Tuple

# index to terrain texture copied over from TERRAIN.txt
TERRAINS = [
    "TERRAIN.000", "TERR000.000", "TERR000.001", "TERR000.002", "TERR000.003", "TERR000.004", "TERR001.001", "TERR001.002", "TERR001.003",
    "TERR001.004", "TERR000.005", "TERR000.006", "TERR000.007", "TERR000.008", "TERR000.009", "TERR000.010", "TERR000.011", "TERR000.016",
    "TERR000.017", "TERR000.013", "TERR000.014", "TERR000.018", "TERR000.019", "TERR000.044", "TERR000.020", "TERR000.022", "TERR000.023",
    "TERR000.024", "TERR000.025", "TERR000.026", "TERR000.027", "TERR000.028", "TERR000.029", "TERR000.030", "TERR000.031", "TERR000.032",
    "TERR000.034", "TERR000.035", "TERR000.036", "TERR001.005", "TERR001.006", "TERR001.007", "TERR001.010", "TERR001.008", "TERR001.009",
    "TERR002.001", "TERR002.003", "TERR002.002", "TERR002.004", "TERR002.005", "TERR002.006", "TERR002.007", "TERR002.008", "TERR002.009",
    "TERR002.010", "TERR003.001", "TERR003.002", "TERR003.003", "TERR000.037", "TERR003.004", "TERR003.005", "TERR003.006", "TERR003.007",
    "TERR003.008", "TERR000.038", "TERR003.009", "TERR003.010", "TERR000.039", "TERR004.001", "TERR004.002", "TERR004.003", "TERR004.004",
    "TERR004.005", "TERR004.006", "TERR004.007", "TERR004.008", "TERR004.009", "TERR004.010", "TERR005.001", "TERR005.002", "TERR005.003",
    "TERR005.004", "TERR005.005", "TERR005.006", "TERR005.007", "TERR005.008", "TERR005.009", "TERR005.010", "TERR006.001", "TERR006.002",
    "TERR006.003", "TERR006.004", "TERR006.005", "TERR006.006", "TERR006.007", "TERR006.008", "TERR006.009", "TERR006.010", "TERR007.001",
    "TERR007.002", "TERR007.003", "TERR007.004", "TERR007.005", "TERR007.006", "TERR007.007", "TERR007.008", "TERR007.009", "TERR007.010",
    "TERR008.001", "TERR008.002", "TERR008.003", "TERR008.004", "TERR008.005", "TERR008.006", "TERR008.007", "TERR008.008", "TERR008.009",
    "TERR008.010", "TERR009.001", "TERR009.002", "TERR009.003", "TERR009.004", "TERR009.005", "TERR009.006", "TERR009.007", "TERR009.008",
    "TERR009.009", "TERR009.010", "TERR010.001", "TERR010.002", "TERR010.003", "TERR010.004", "TERR010.005", "TERR010.006", "TERR010.007",
    "TERR010.008", "TERR010.009", "TERR010.010", "TERR011.001", "TERR011.002", "TERR011.003", "TERR011.004", "TERR011.005", "TERR011.006",
    "TERR011.007", "TERR011.008", "TERR011.009", "TERR011.010", "TERR012.001", "TERR012.002", "TERR012.003"
]

if len(sys.argv) != 2:
    print("usage: python unpack_kwsc.py [.kwsc file]")
    sys.exit(1)

def decode_utf16(x: bytes) -> str:
    w = x.split(b"\x00\x00")[0]
    if len(w) % 2 == 1:
        w += b"\x00"
    return w.decode("utf-16")

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

RuntimeClass = namedtuple("RuntimeClass", "name unk1")
LevelData = namedtuple("LevelData", "unk1 unk2 unk3 unk4")

unk_2b = namedtuple("unk_2b", "unk1 unk2")
unk_8b = namedtuple("unk_8b", "unk1 unk2")
Attr = namedtuple("unk_20b", "index u v u2 v2")
Triangle = namedtuple("Triangle", "v1 v2 v3 i j k")
Vertex = namedtuple("Vertex", "z x y i j k unk1")
MatlInfo = namedtuple("MatlInfo", "t1 t2 unk3 unk4 unk5 matl_idx unk7 unk8 unk9 unk10")

class SectionData:
    material: str
    unks1: List[int]
    n_unk_8bs_1: int
    unk_8bs_1: List[unk_8b]
    unk2: int
    unk3: int
    unk4: int
    n_vertices: int
    vertices: List[Vertex]
    n_attr: int
    attrs: List[Attr]
    n_triangles: int
    triangles: List[Triangle]
    n_matl_infos: int
    matl_infos: List[MatlInfo]
    n_unk_8bs_2: int
    unk_8bs_2: List[unk_8b]
    n_material_indices: int
    material_indices: List[int]

class TerrainData:
    unk1: int
    unk2: int
    unk3: int
    str1: str
    n_unk4s: int
    unk4s: List[int]
    unk5: int
    unk6: int
    unk7: int
    unk8: int
    section_datas: List[SectionData]
    unk9s: List[int]
    unk10s: List[int]

WeatherSlotData = namedtuple("WeatherSlotData", "unks")
WeatherEffectSlot = namedtuple("WeatherEffectSlot", "str1 str2 unks")

class WeatherEffectData:
    str1: str
    unk1: int
    unk2: int
    unk3: int
    n_weather_effect_slots: int
    weather_effect_slots: List[WeatherEffectSlot]

class WeatherData:
    unk1: int
    unk2: int
    unk3: int
    unk4: int
    unk5: int
    n_slot_datas_1: int
    slot_datas_1: List[WeatherSlotData]
    n_slot_datas_2: int
    slot_datas_2: List[WeatherSlotData]
    n_effect_datas: int
    effect_datas: List[WeatherEffectData]

class SkyboxData:
    unk1: int
    unk2: int
    unk3: int
    unk4: int
    unk5: int
    unk6: int
    unk7: int
    unk8: int
    star_mesh: str
    atm_mesh: str
    scene_mesh: str
    star_matl: str
    atm_matl: str
    sun_matl: str
    moon_matl: str
    sunshine_matl: str
    moonshine_matl: str
    unk9: int
    unk10: int
    unk11: int
    unk12: int
    unk13: int
    unk14: int
    unk15: int
    unk16: int
    unk17: int
    n_unk_2bs: int
    unk_2bs: List[unk_2b]
    unk18: int
    unk19: int
    unk20: int
    unk21: int

OptMaterialData = namedtuple("OptMaterialData", "unk1 unk2 name")
WaterPropData = namedtuple("WaterPropData", "unks name")

class KwarScene:
    n_runtime_classes: int
    runtime_classes: List[RuntimeClass]
    str1: str
    n_actors: int
    level_data: LevelData
    width: int
    height: int
    terrain_data: TerrainData
    weather_data: WeatherData
    skybox_data: SkyboxData
    actors: list

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

    def _unpack_section_data(self, data) -> SectionData:
        out = SectionData()
        out.material = self._unpack_kwstring(data)
        out.unks1 = list(self._unpack("=2i 2I 8f", data, 48))
        out.n_unk_8bs_1 = self._unpack("=I", data, 4)
        out.unk_8bs_1 = []
        for _ in range(out.n_unk_8bs_1):
            x = unk_8b(*self._unpack("=2f", data, 8))
            out.unk_8bs_1.append(x)
        out.unk2, out.unk3, out.unk4 = self._unpack("=I I I", data, 12)
        out.n_vertices = self._unpack("=I", data, 4)
        out.vertices = []
        for _ in range(out.n_vertices):
            x = Vertex(*self._unpack("=6f i", data, 28))
            out.vertices.append(x)
        out.n_attr = self._unpack("=I", data, 4)
        out.attrs = []
        for _ in range(out.n_attr):
            x = Attr(*self._unpack("=I 4f", data, 20))
            out.attrs.append(x)
        out.n_triangles = self._unpack("=I", data, 4)
        out.triangles = []
        for _ in range(out.n_triangles):
            x = Triangle(*self._unpack("=3I 3f", data, 24))
            out.triangles.append(x)
        out.n_matl_infos = self._unpack("=I", data, 4)
        out.matl_infos = []
        for _ in range(out.n_matl_infos):
            x = MatlInfo(*self._unpack("=I I I B H H H H H H", data, 25))
            out.matl_infos.append(x)
        out.n_unk_8bs_2 = self._unpack("=I", data, 4)
        out.unk_8bs_2 = []
        for _ in range(out.n_unk_8bs_2):
            x = unk_8b(*self._unpack("=I I", data, 8))
            out.unk_8bs_2.append(x)
        out.n_material_indices = self._unpack("=I", data, 4)
        out.material_indices = []
        for _ in range(out.n_material_indices):
            out.material_indices.append(self._unpack("=H", data, 2))
        return out

    def _unpack_terrain_data(self, data) -> TerrainData:
        out = TerrainData()
        out.unk1, out.unk2, out.unk3 = self._unpack("=I B B", data, 6)
        out.str1 = self._unpack_kwstring(data)
        out.n_unk4s = self._unpack("=I", data, 4)
        out.unk4s = []
        for _ in range(out.n_unk4s):
            out.unk4s.append(self._unpack("=I", data, 4))
        out.unk5, out.unk6, out.unk7, out.unk8 = self._unpack("=I I I I", data, 16)
        out.section_datas = []
        for _ in range(64):
            section_data = self._unpack_section_data(data)
            out.section_datas.append(section_data)
        out.unk9s = []
        #for _ in range(self.width * self.height):
        for _ in range(self.width * self.height // 16):
            out.unk9s.append(self._unpack("=B B", data, 2))
        out.unk10s = []
        for _ in range(self.width * self.height // 256):
            out.unk10s.append(self._unpack("=B B", data, 2))
        return out

    def _unpack_weather_slot_data(self, data) -> WeatherSlotData:
        values = self._unpack("=4I 4B f 17B I 4B 2f 4B 5f I", data, 85)
        return WeatherSlotData(list(values))

    def _unpack_weather_effect_slot(self, data) -> WeatherEffectSlot:
        str1 = self._unpack_kwstring(data)
        str2 = self._unpack_kwstring(data)
        values = self._unpack("=i 4B f 4B f f 4B 2f", data, 36)
        return WeatherEffectSlot(str1, str2, list(values))

    def _unpack_weather_effect_data(self, data) -> WeatherEffectData:
        out = WeatherEffectData()
        out.str1 = self._unpack_kwstring(data)
        out.unk1, out.unk2, out.unk3 = self._unpack("=3I", data, 12)
        out.n_weather_effect_slots = self._unpack("=I", data, 4)
        out.weather_effect_slots = []
        for _ in range(out.n_weather_effect_slots):
            weather_effect_slot = self._unpack_weather_effect_slot(data)
            out.weather_effect_slots.append(weather_effect_slot)
        return out

    def _unpack_weather_data(self, data) -> WeatherData:
        out = WeatherData()
        out.unk1, out.unk2, out.unk3, out.unk4, out.unk5 = self._unpack("=I B f f f", data, 17)
        out.n_slot_datas_1 = self._unpack("=I", data, 4)
        out.slot_datas_1 = []
        for _ in range(out.n_slot_datas_1):
            slot_data = self._unpack_weather_slot_data(data)
            out.slot_datas_1.append(slot_data)
        out.n_slot_datas_2 = self._unpack("=I", data, 4)
        out.slot_datas_2 = []
        for _ in range(out.n_slot_datas_2):
            slot_data = self._unpack_weather_slot_data(data)
            out.slot_datas_2.append(slot_data)
        out.n_effect_datas = self._unpack("=I", data, 4)
        out.effect_datas = []
        for _ in range(out.n_effect_datas):
            effect_data = self._unpack_weather_effect_data(data)
            out.effect_datas.append(effect_data)
        return out

    def _unpack_skybox_data(self, data) -> SkyboxData:
        out = SkyboxData()
        values = self._unpack("=4f 4I", data, 32)
        out.unk1, out.unk2, out.unk3, out.unk4, out.unk5, out.unk6, out.unk7, out.unk8 = values
        out.star_mesh = self._unpack_kwstring(data)
        out.atm_mesh = self._unpack_kwstring(data)
        out.scene_mesh = self._unpack_kwstring(data)
        out.star_matl = self._unpack_kwstring(data)
        out.atm_matl = self._unpack_kwstring(data)
        out.sun_matl = self._unpack_kwstring(data)
        out.moon_matl = self._unpack_kwstring(data)
        out.sunshine_matl = self._unpack_kwstring(data)
        out.moonshine_matl = self._unpack_kwstring(data)
        values = self._unpack("=9f", data, 36)
        out.unk9, out.unk10, out.unk11, out.unk12, out.unk13, out.unk14, out.unk15, out.unk16, out.unk17 = values
        out.n_unk_2bs = self._unpack("=I", data, 4)
        out.unk_2bs = []
        for _ in range(out.n_unk_2bs):
            x = unk_2b(*self._unpack("=B B", data, 2))
            out.unk_2bs.append(x)
        values = self._unpack("=4f", data, 16)
        out.unk18, out.unk19, out.unk20, out.unk21 = values
        return out

    def _unpack_actor(self, data) -> dict:
        out = {}
        out["actor_name"] = self._unpack_kwstring(data)
        out["actor_unk1"] = self._unpack("=f", data, 4)
        out["actor_unk2"] = self._unpack("=f", data, 4)
        out["actor_unk3"] = self._unpack("=f", data, 4)
        out["actor_unk4"] = self._unpack("=B", data, 1)
        out["actor_unk5"] = self._unpack("=B", data, 1)
        n_triggers = self._unpack("=I", data, 4)
        triggers = []
        for _ in range(n_triggers):
            triggers.append(self._unpack_kwstring(data))
        out["actor_triggers"] = triggers
        out["actor_unk6"] = self._unpack("=B", data, 1)
        out["actor_unk7"] = self._unpack("=B", data, 1)
        out["actor_unk8"] = self._unpack("=B", data, 1)
        out["actor_unk9"] = self._unpack("=B", data, 1)
        out["actor_unk10"] = self._unpack("=B", data, 1)
        out["actor_unk11"] = self._unpack("=B", data, 1)
        out["actor_unk12"] = self._unpack("=B", data, 1)
        out["actor_unk13"] = self._unpack("=f", data, 4)
        return out

    def _unpack_carrier_point_base(self, data) -> dict:
        out = self._unpack_actor(data)
        out["cpbase_str1"] = self._unpack_kwstring(data)
        out["cpbase_unk1"] = self._unpack("=I", data, 4)
        out["cpbase_str2"] = self._unpack_kwstring(data)
        out["cpbase_str3"] = self._unpack_kwstring(data)
        out["cpbase_unk2"] = self._unpack("=I", data, 4)
        out["cpbase_unk3"] = self._unpack("=I", data, 4)
        out["cpbase_unk4"] = self._unpack("=I", data, 4)
        out["cpbase_unk5"] = self._unpack("=I", data, 4)
        out["cpbase_unk6"] = self._unpack("=B", data, 1)
        out["cpbase_unk7"] = self._unpack("=B", data, 1)
        out["cpbase_unk8"] = self._unpack("=B", data, 1)
        return out

    def _unpack_carrier_point(self, data) -> dict:
        out = self._unpack_carrier_point_base(data)
        out["cp_str1"] = self._unpack_kwstring(data)
        out["cp_unk1"] = self._unpack("=I", data, 4)
        out["cp_unk2"] = self._unpack("=I", data, 4)
        out["cp_unk3"] = self._unpack("=I", data, 4)
        out["cp_unk4"] = self._unpack("=I", data, 4)
        out["cp_unk5"] = self._unpack("=I", data, 4)
        out["cp_unk6"] = self._unpack("=I", data, 4)
        return out

    def _unpack_mesh_actor(self, data) -> dict:
        out = self._unpack_actor(data)
        out["mactor_str1"] = self._unpack_kwstring(data)
        n_materials = self._unpack("=I", data, 4)
        materials = []
        for _ in range(n_materials):
            materials.append(self._unpack_kwstring(data))
        out["mactor_materials"] = materials
        n_opt_materials = self._unpack("=I", data, 4)
        opt_materials = []
        for _ in range(n_opt_materials):
            values = self._unpack("=I I", data, 8)
            name = self._unpack_kwstring(data)
            opt_material = OptMaterialData(*values, name)
            opt_materials.append(opt_material)
        out["mactor_opt_materials"] = opt_materials
        n_water_props = self._unpack("=I", data, 4)
        water_props = []
        for _ in range(n_water_props):
            values = self._unpack("=I 4B 4I 3I 3I B", data, 49)
            name = self._unpack_kwstring(data)
            water_prop = WaterPropData(list(values), name)
            water_props.append(water_prop)
        out["mactor_water_props"] = water_props
        out["mactor_unk1"] = self._unpack("=I", data, 4)
        out["mactor_unk2"] = self._unpack("=I", data, 4)
        out["mactor_unk3"] = self._unpack("=f", data, 4)
        out["mactor_unk4"] = self._unpack("=f", data, 4)
        out["mactor_unk5"] = self._unpack("=f", data, 4)
        out["mactor_unk6"] = self._unpack("=f", data, 4)
        out["mactor_unk7"] = self._unpack("=f", data, 4)
        out["mactor_unk8"] = self._unpack("=f", data, 4)
        return out

    def _unpack_static_mesh_actor(self, data) -> dict:
        out = self._unpack_mesh_actor(data)
        out["smactor_unk1"] = self._unpack("=B", data, 1)
        out["smactor_unk2"] = self._unpack("=B", data, 1)
        out["smactor_unk3"] = self._unpack("=B", data, 1)
        out["smactor_str1"] = self._unpack_kwstring(data)
        out["smactor_unk4"] = self._unpack("=B", data, 1)
        out["smactor_unk5"] = self._unpack("=B", data, 1)
        out["smactor_str2"] = self._unpack_kwstring(data)
        out["smactor_unk6"] = self._unpack("=B", data, 1)
        out["smactor_unk7"] = self._unpack("=f", data, 4)
        out["smactor_unk8"] = self._unpack("=f", data, 4)
        out["smactor_unk9"] = self._unpack("=f", data, 4)
        out["smactor_str3"] = self._unpack_kwstring(data)
        out["smactor_str4"] = self._unpack_kwstring(data)
        return out

    def _unpack_light(self, data) -> dict:
        out = self._unpack_actor(data)
        out["light_unks"] = list(self._unpack("=I 4f 3I 5f 3I f 2I 2f 2I", data, 92))
        n_effectors = self._unpack("=I", data, 4)
        effectors = []
        for _ in range(n_effectors):
            effector = {}
            effector_type = self._unpack("=I", data, 4)
            if effector_type == 1:
                effector["unk1"] = self._unpack("=I", data, 4)
                effector["unk2"] = self._unpack("=I", data, 4)
                effector["unk3"] = self._unpack("=I", data, 4)
            elif effector_type == 2:
                effector["unk1"] = self._unpack("=I", data, 4)
                effector["unk2"] = self._unpack("=I", data, 4)
                effector["unk4"] = self._unpack("=I", data, 4)
                effector["unk3"] = self._unpack("=I", data, 4)
            elif effector_type == 3:
                effector["unk1"] = self._unpack("=I", data, 4)
                n_color_scales = self._unpack("=I", data, 4)
                color_scales = []
                for __ in range(n_color_scales):
                    color_scale = list(self._unpack("I 4B", data, 8))
                    color_scales.append(color_scale)
                effector["color_scales"] = color_scales
            else:
                print("unknown light effector type", effector_type)
                sys.exit(-1)
            effector["type"] = effector_type
            effectors.append(effector)
        out["light_effectors"] = effectors
        return out

    def _unpack_mobile_static_mesh(self, data) -> dict:
        out = self._unpack_static_mesh_actor(data)
        values = self._unpack("=2I 3B 4I 3B 3I 25I 2B 3I f I B", data, 165)
        out["msm_values"] = list(values)
        return out

    def _unpack_anim_mesh_actor(self, data) -> dict:
        out = self._unpack_mesh_actor(data)
        out["amactor_unk1"] = self._unpack("=f", data, 4)
        out["amactor_unk2"] = self._unpack("=H", data, 2)
        out["amactor_unk3"] = self._unpack("=I", data, 4)
        out["amactor_unk4"] = self._unpack("=I", data, 4)
        out["amactor_unk5"] = self._unpack("=f", data, 4)
        out["amactor_unk6"] = self._unpack("=f", data, 4)
        out["amactor_unk7"] = self._unpack("=f", data, 4)
        out["amactor_unk8"] = self._unpack("=I", data, 4)
        out["amactor_unk9"] = self._unpack("=I", data, 4)
        out["amactor_unk10"] = self._unpack("=I", data, 4)
        out["amactor_unk11"] = self._unpack("=f", data, 4)
        n_unks = self._unpack("=I", data, 4)
        unks = []
        for _ in range(n_unks):
            unk = list(self._unpack("=f f f", data, 12))
            unks.append(unk)
        out["amactor_unks"] = unks
        out["amactor_unk12"] = self._unpack("=I", data, 4)
        out["amactor_unk13"] = self._unpack("=I", data, 4)
        out["amactor_unk14"] = self._unpack("=d", data, 8)
        out["amactor_unk15"] = self._unpack("=d", data, 8)
        out["amactor_unk16"] = self._unpack("=d", data, 8)
        return out

    def _unpack_scene_path(self, data) -> dict:
        out = self._unpack_actor(data)
        out["scp_unk1"] = self._unpack("=I", data, 4)
        out["scp_unk2"] = self._unpack("=I", data, 4)
        out["scp_str1"] = self._unpack_kwstring(data)
        out["scp_unk3"] = self._unpack("=I", data, 4)
        n_control_nodes = self._unpack("=I", data, 4)
        control_nodes = []
        for _ in range(n_control_nodes):
            control_node = {}
            control_node["values"] = list(self._unpack("=3f I 2i f 2I", data, 36))
            control_node["name"] = self._unpack_kwstring(data)
            control_nodes.append(control_node)
        out["scp_control_nodes"] = control_nodes
        return out

    def _unpack_audio_actor(self, data) -> dict:
        out = self._unpack_actor(data)
        strs = []
        for _ in range(6):
            strs.append(self._unpack_kwstring(data))
        out["aactor_strs"] = strs
        out["aactor_unk1"] = self._unpack("=I", data, 4)
        out["aactor_unk2"] = self._unpack("=d", data, 8)
        out["aactor_unk3"] = self._unpack("=B", data, 1)
        out["aactor_unk4"] = self._unpack("=I", data, 4)
        out["aactor_unk5"] = self._unpack("=I", data, 4)
        out["aactor_unk_block"] = list(self._unpack("=28B", data, 28))
        return out

    def _unpack_emitter_actor(self, data) -> dict:
        out = self._unpack_actor(data)
        out["eactor_str1"] = self._unpack_kwstring(data)
        return out

    actor_loaders = {
        "KCarrierPoint": _unpack_carrier_point,
        "KCarrierPointBase": _unpack_carrier_point_base,
        "KActor": _unpack_actor,
        "KMeshActor": _unpack_mesh_actor,
        "KStaticMeshActor": _unpack_static_mesh_actor,
        "KDynamicLight": _unpack_light,
        "KMobileStaticMesh": _unpack_mobile_static_mesh,
        "KAnimMeshActor": _unpack_anim_mesh_actor,
        "KScenePath": _unpack_scene_path,
        "KAudioActor": _unpack_audio_actor,
        "KEmitterActor": _unpack_emitter_actor,
    }

    def _unpack_any_actor(self, data) -> dict:
        actor = {}
        obj_name = self._unpack_kwstring(data)
        actor["type"] = obj_name
        actor["unk1"] = self._unpack("=f", data, 4)
        actor["unk2"] = self._unpack("=f", data, 4)
        actor["unk3"] = self._unpack("=f", data, 4)
        actor["unk4"] = self._unpack("=I", data, 4)
        actor["unk5"] = self._unpack("=I", data, 4)
        actor["unk6"] = self._unpack("=I", data, 4)
        actor["unk7"] = self._unpack("=B", data, 1)
        load_fn = self.actor_loaders.get(obj_name)
        if load_fn is None:
            print("unknown load fn for actor: ", obj_name)
            sys.exit(-1)
        actor = actor | load_fn(self, data)
        return actor

    @staticmethod
    def from_bytes(data):
        out = KwarScene()
        out.n_runtime_classes = out._unpack("=I", data, 4)
        out.runtime_classes = []
        for _ in range(out.n_runtime_classes):
            name = out._unpack_kwstring(data)
            unk1 = out._unpack("=I", data, 4)
            out.runtime_classes.append(RuntimeClass(name, unk1))
        out.str1 = out._unpack_kwstring(data)
        out.n_actors = out._unpack("=I", data, 4)
        level_data_values = out._unpack("=I B f I", data, 13)
        out.level_data = LevelData(*level_data_values)
        out.width = out._unpack("=I", data, 4)
        out.height = out._unpack("=I", data, 4)
        out.terrain_data = out._unpack_terrain_data(data)
        out.weather_data = out._unpack_weather_data(data)
        out.skybox_data = out._unpack_skybox_data(data)
        out.actors = []
        for _ in range(out.n_actors):
            actor = out._unpack_any_actor(data)
            out.actors.append(actor)
        # what does the remaining data look like?
        #with open("x.bin", "wb") as f:
        #    f.write(data[out.off:])
        return out

def write_section_as_gltf(sd: SectionData, path: str) -> None:
    vertices = []
    uvs1 = []
    uvs2 = []
    for attr in sd.attrs:
        v = sd.vertices[attr.index]
        vertices.append([v.x, v.y, v.z])
        uvs1.append([attr.u, attr.v])
        uvs2.append([attr.u2, attr.v2])

    def pack(what: List, fmt: str) -> bytes:
        out = bytes()
        for x in what:
            out += struct.pack(fmt, *x)
        return out

    vertices_bin = pack(vertices, "f f f")
    uvs1_bin = pack(uvs1, "f f")
    uvs2_bin = pack(uvs2, "f f")

    # create gltf objects for a scene with a primitive triangle with indexed geometry
    gltf = pygltflib.GLTF2(
        scene=0,
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
                count=len(uvs1),
                type=pygltflib.VEC2,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.FLOAT,
                count=len(uvs2),
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
                byteLength=len(uvs1_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=2,
                byteLength=len(uvs2_bin),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(vertices_bin).decode("utf-8"),
                byteLength=len(vertices_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(uvs1_bin).decode("utf-8"),
                byteLength=len(uvs1_bin)
            ),
            pygltflib.Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(uvs2_bin).decode("utf-8"),
                byteLength=len(uvs2_bin)
            ),
        ],
    )

    triangles = []
    for trgl in sd.triangles:
        triangles.append([trgl.v1, trgl.v2, trgl.v3])

    # Group triangles wrt their materials
    triangle_groups = [[] for _ in sd.material_indices]
    for matl_info in sd.matl_infos:
        idx = matl_info.unk3
        if idx >= sd.n_material_indices:
            print("[!] invalid material idx %d, assigning blank material" % idx)
            idx = sd.n_material_indices - 1
        triangle_groups[idx].append(triangles[matl_info.t1])
        triangle_groups[idx].append(triangles[matl_info.t2])

    assert sum(len(tg) for tg in triangle_groups) == len(triangles), "couldn't group every triangle"

    # Create a material & mesh for each different matl index
    for i, matl_idx in enumerate(sd.material_indices):
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
        matl_name = TERRAINS[matl_idx]
        gltf.images.append(pygltflib.Image(uri=os.path.join(*matl_name.split(".")) + ".png"))
        gltf.samplers.append(pygltflib.Sampler())
        gltf.textures.append(pygltflib.Texture(sampler=len(gltf.samplers) - 1, source=len(gltf.images) - 1))
        gltf.materials.append(pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=len(gltf.textures) - 1),
                metallicFactor=0,
                roughnessFactor=1
            )
        ))
        # 3. add mesh
        gltf.meshes[0].primitives.append(pygltflib.Primitive(
            attributes=pygltflib.Attributes(
                POSITION=0,
                TEXCOORD_0=1,
                #TEXCOORD_0=2,
            ),
            indices=len(gltf.accessors) - 1,
            material=len(gltf.materials) - 1
        ))

    gltf.convert_images(pygltflib.ImageFormat.DATAURI)
    gltf.save(path)

# Monkey patch json module's float so that things get rounded
class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, '.10f'))
# json.encoder.float = RoundingFloat

def write_metadata_as_json(ksc: KwarScene, path: str) -> None:
    out = {}
    out = out | vars(ksc)
    out["terrain_data"] = vars(ksc.terrain_data)
    out["terrain_data"]["section_datas"] = [vars(sd) for sd in ksc.terrain_data.section_datas]
    out["weather_data"] = vars(ksc.weather_data)
    out["weather_data"]["effect_datas"] = [vars(we) for we in ksc.weather_data.effect_datas]
    out["skybox_data"] = vars(ksc.skybox_data)

    with open(path, "w") as f:
        json.dump(out, f)

C = ""
with open(sys.argv[1], "rb") as f:
    C = f.read()

print("decompressing", sys.argv[1])
version_key = struct.unpack("=I", C[:4])[0]
data = decompress_lzma(C[5:])

print("decoding", sys.argv[1])
ksc = KwarScene.from_bytes(data)

basename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
os.makedirs(basename, exist_ok=True)
"""
for i, sd in enumerate(ksc.terrain_data.section_datas):
    mesh_path = os.path.join(basename, "terrain_%d.gltf" % i)
    print("writing terrain mesh to", mesh_path)
    write_section_as_gltf(sd, mesh_path)
"""

json_path = os.path.join(basename, "scene.json")
print("writing scene data to", json_path)
write_metadata_as_json(ksc, json_path)
