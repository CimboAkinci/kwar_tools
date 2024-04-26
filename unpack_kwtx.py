import json
import lzma
import os
import struct
import sys

from dataclasses import dataclass

if len(sys.argv) != 2:
    print("usage: python unpack.py [.kwtx file]")
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

#
# Textures
#


class DXTBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.block_countx = self.width // 4
        self.block_county = self.height// 4

        self.decompressed_buffer = ["X"] * ((width * height) * 2) # Dont ask me why

    def _dxt_unpack(self, _bytes):
        STRUCT_SIGNS = {
        1 : 'B',
        2 : 'H',
        4 : 'I',
        8 : 'Q'
        }
        return struct.unpack('<' + STRUCT_SIGNS[len(_bytes)], _bytes)[0]

    # This function converts RGB565 format to raw pixels
    def _unpackRGB(self, packed):
        R = (packed >> 11) & 0x1F
        G = (packed >> 5) & 0x3F
        B = (packed) & 0x1F

        R = (R << 3) | (R >> 2)
        G = (G << 2) | (G >> 4)
        B = (B << 3) | (B >> 2)

        return (R, G, B, 255)

    def DXT1Decompress(self, data: bytes):
        # Loop through each block and decompress it
        for row in range(self.block_county):
            for col in range(self.block_countx):
                # Color 1 color 2, color look up table
                x = (col + row*self.block_countx) * 8
                c0 = self._dxt_unpack(data[x:x+2])
                c1 = self._dxt_unpack(data[x+2:x+4])
                ctable = self._dxt_unpack(data[x+4:x+8])

                # The 4x4 Lookup table loop
                for j in range(4):
                    for i in range(4):
                        self.getColors(col * 4, row * 4, i, j, ctable, self._unpackRGB(c0) ,self._unpackRGB(c1), 255) # Set the color for the current pixel

        return b''.join([_ for _ in self.decompressed_buffer if _ != 'X'])

    def DXT5Decompress(self, data: bytes):
        # Loop through each block and decompress it
        for row in range(self.block_county):
            for col in range(self.block_countx):
                x = (col + row*self.block_countx) * 16

                # Get the alpha values
                a0 = self._dxt_unpack(data[x:x+1])
                a1 = self._dxt_unpack(data[x+1:x+2])
                atable = data[x+2:x+8]

                acode0 = atable[2] | (atable[3] << 8) | (atable[4] << 16) | (atable[5] << 24)
                acode1 = atable[0] | (atable[1] << 8)

                # Color 1 color 2, color look up table
                c0 = self._dxt_unpack(data[x+8:x+10])
                c1 = self._dxt_unpack(data[x+10:x+12])
                ctable = self._dxt_unpack(data[x+12:x+16])

                # The 4x4 Lookup table loop
                for j in range(4):
                    for i in range(4):
                        alpha = self.getAlpha(j, i, a0, a1, atable, acode0, acode1)
                        self.getColors(col * 4, row * 4, i, j, ctable, self._unpackRGB(c0) ,self._unpackRGB(c1), alpha) # Set the color for the current pixel

        return b''.join([_ for _ in self.decompressed_buffer if _ != 'X'])

    def getColors(self, x, y, i, j, ctable, c0, c1, alpha):
        code = (ctable >> ( 2 * (4 * i + j))) & 0x03 # Get the color of the current pixel
        pixel_color = None

        r0 = c0[0]
        g0 = c0[1]
        b0 = c0[2]

        r1 = c1[0]
        g1 = c1[1]
        b1 = c1[2]

        # Main two colors
        if code == 0:
            pixel_color = (r0, g0, b0, alpha)
        if code == 1:
            pixel_color = (r1, g1, b1, alpha)

        # Use the lookup table to determine the other two colors
        if c0 > c1:
            if code == 2:
                pixel_color = ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3, alpha)
            if code == 3:
                pixel_color = ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3, alpha)
        else:
            if code == 2:
                pixel_color = ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2, alpha)
            if code == 3:
                pixel_color = (0, 0, 0, alpha)

        # While not surpassing the image dimensions, assign pixels the colors
        if (x + j) < self.width:
            self.decompressed_buffer[(y + i) * self.width + (x + j)] = struct.pack('<B', pixel_color[0]) + \
                struct.pack('<B', pixel_color[1]) + struct.pack('<B', pixel_color[2]) + struct.pack('<B', pixel_color[3])

    def getAlpha(self, i, j, a0, a1, atable, acode0, acode1):
        alpha = 255
        alpha_index = 3 * (4 * j+i)
        alpha_code = None

        if alpha_index <= 12:
            alpha_code = (acode1 >> alpha_index) & 0x07
        elif alpha_index == 15:
            alpha_code = (acode1 >> 15) | ((acode0 << 1) & 0x06)
        else:
            alpha_code = (acode0 >> (alpha_index - 16)) & 0x07

        if alpha_code == 0:
            alpha = a0
        elif alpha_code == 1:
            alpha = a1
        else:
            if a0 > a1:
                alpha = ((8-alpha_code) * a0 + (alpha_code-1) * a1) // 7
            else:
                if alpha_code == 6:
                    alpha = 0
                elif alpha_code == 7:
                    alpha = 255
                elif alpha_code == 5:
                    alpha = (1 * a0 + 4 * a1) // 5
                elif alpha_code == 4:
                    alpha = (2 * a0 + 3 * a1) // 5
                elif alpha_code == 3:
                    alpha = (3 * a0 + 2 * a1) // 5
                elif alpha_code == 2:
                    alpha = (4 * a0 + 1 * a1) // 5
                else:
                    alpha = 0 # For safety
        return alpha

from PIL import Image
import numpy as np

class KwarTexture:
    unk1: int
    archive_name_len: int
    archive_name: any
    file_name_len: int
    file_name: any
    file_size_maybe: int
    unk2: any
    compression: int
    width: int
    height: int
    data_len: int
    unk3: any
    mips: int
    data: any
    unk4: any

    @staticmethod
    def from_bytes(x: bytes):
        out = KwarTexture()
        out.unk1, a = struct.unpack("=I I", x[:8])
        out.archive_name_len = a
        out.archive_name = decode_utf16(struct.unpack("=%ds" % a, x[8:a+8])[0])
        (b, ) = struct.unpack("I", x[a+8:a+12])
        out.file_name_len = b
        out.file_name = decode_utf16(struct.unpack("=%ds" % b, x[a+12:a+b+12])[0])
        out.file_size_maybe = struct.unpack("=I", x[a+b+12:a+b+16])
        out.unk2, out.compression, out.width, out.height, c, out.unk3, out.mips = struct.unpack("=5s B I I I 28s I", x[a+b+16:a+b+66])
        out.data_len = c
        (out.data, ) = struct.unpack("=%ds" % c, x[a+b+66:a+b+c+66])
        out.unk4 = x[a+b+c+66:]
        return out

    def _decompress(self, data: bytes, width, height) -> bytes:
        if self.compression in [1, 5]: # No compression
            return data
        elif self.compression == 3: # DXT1
            B = DXTBuffer(width, height)
            return B.DXT1Decompress(data)
        elif self.compression in [7, 8]: # DXT5
            B = DXTBuffer(width, height)
            return B.DXT5Decompress(data)
        else:
            print(self.file_name, self.unk1, self.unk2, self.compression, self.width, self.height, self.unk3, self.mips, self.data_len, self.unk4)
            raise NotImplementedError("Unknown compression method in texture: %d", self.compression)

    def _get_mipped_size(self):
        w = self.width
        out = 0
        for m in range(self.mips):
            out += w
            w = w // 2
        return (out, self.height)

    def get_image(self) -> Image:
        # print(self.file_name, self.unk1, self.unk2, self.compression, self.width, self.height, self.unk3, self.mips, self.data_len, self.unk4)
        w, h = self.width, self.height
        BPPS = {1: 32, 3: 4, 5: 32, 7: 8, 8: 8}
        bpp = BPPS[self.compression]
        off = 0
        x = 0
        self.mips = 1 # XXX Don't extract mipmaps
        out_im = Image.new("RGBA", self._get_mipped_size())
        for m in range(self.mips):
            size = (w * h * bpp) // 8
            inflated = self._decompress(self.data[off:off+size], w, h)
            # print(size, w, h)
            # print(len(inflated))
            if len(inflated) == 0:
                print("something's off with mipmap, but cutting off here")
                break
            im = Image.frombytes('RGBA', (w, h), inflated, 'raw')
            if self.compression == 5: # This is a BGR image, rotate colors
                np_im = np.array(im)
                np_im[:,:,[0,1,2]] = np_im[:,:,[2,1,0]]
                im = Image.fromarray(np_im) # Convert to BGR
            out_im.paste(im, (x, 0))
            x += w
            off += size
            w = w // 2
            h = h // 2
        return out_im

class KwarCombiner:
    magic: int
    archive_name: str
    file_name: str
    unk1: int
    unk2: int
    unk3: int
    texture1: str
    texture2: str
    unk4: int
    unk5: int
    unk6: int
    unk7: int

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
        out = KwarCombiner()
        out.magic = out._unpack("=I", data, 4)
        out.archive_name = out._unpack_kwstring(data)
        out.file_name = out._unpack_kwstring(data)
        out.unk1, out.unk2, out.unk3 = out._unpack("=I I B", data, 9)
        out.texture1 = out._unpack_kwstring(data)
        out.texture2 = out._unpack_kwstring(data)
        out.unk4, out.unk5, out.unk6, out.unk7 = out._unpack("=4I", data, 16)
        return out

os.makedirs(header.archive_name, exist_ok=True)

for f in files:
    print(f.header)
    (magic,) = struct.unpack("I", f.contents[:4])
    if magic == 13:
        tex = KwarTexture.from_bytes(f.contents)
        im = tex.get_image()
        path = os.path.join(header.archive_name, f.header.file_name + ".png")
        print("writing to", path)
        im.save(path)
    elif magic == 10:
        kc = KwarCombiner.from_bytes(f.contents)
        path = os.path.join(header.archive_name, f.header.file_name + ".json")
        print("writing to", path)
        with open(path, "w") as f:
            json.dump(vars(kc), f)
    else: # 4, 9, 11, 12, 33
        print("Unknown magic: ", magic)
        path = os.path.join(header.archive_name, f.header.file_name)
        x = open(path, "wb")
        x.write(f.contents)
        x.close()
        continue
