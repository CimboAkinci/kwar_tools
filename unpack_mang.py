import lzma
import os
import struct
import sys

from dataclasses import dataclass

from utils import get_xor_mask

if len(sys.argv) != 3:
    print("usage: python unpack_mang.py [version (tw2010, tr2013, tw2020)] [MangData_pak file]")
    sys.exit(1)

# XOR mask for "decryption"
XOR_MASK = get_xor_mask(sys.argv[1])

C = ""
with open(sys.argv[2], "rb") as f:
    C = f.read()

off = 0
def unpack(fmt: str, data: bytes, size: int):
    global off
    out = struct.unpack(fmt, data[off:off+size])
    off += size
    if len(out) == 1:
        return out[0]
    return out

n_catalogs = unpack("=Q", C, 8)

def decrypt1(word, key) -> int:
    key = key & 0xff  # byte
    out = word >> (key & 0x1f) | word << 0x20 - (key & 0x1f)
    return out & 0xffffffff  # word

def decrypt2(word, key) -> int:
    word = word & 0xffffffff
    return word ^ XOR_MASK[key % 2048]

print("blob1 size:", n_catalogs*268)

decrypted_blob1 = bytes()
for key in range(n_catalogs * 268 // 4):
    x = unpack("=I", C, 4)
    x1 = decrypt1(x, key)
    x2 = decrypt2(x1, key)
    u = struct.pack("=I", x2)
    decrypted_blob1 += u

@dataclass
class MangDataCatalog:
    file_name: any
    key1: int
    key2: int
    offset: int
    uncompressed_size: int
    size: int

    def __post_init__(self):
        self.file_name = self.file_name.split(b"\x00")[0].decode("utf-8")

catalogs = []
for i in range(n_catalogs):
    unpacked = struct.unpack("=253s B H I I I", decrypted_blob1[i*268:(i+1)*268])
    catalog = MangDataCatalog(*unpacked)
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

for catalog in catalogs:
    blob_size = (catalog.size // 4 + 1) * 4
    blob = C[catalog.offset:catalog.offset+blob_size]
    decrypted = bytes()
    for i in range(len(blob) // 4):
        x = struct.unpack("=I", blob[i*4:(i+1)*4])[0]
        x1 = decrypt1(x, catalog.key1 + i)
        x2 = decrypt2(x1, catalog.key2 + i)
        u = struct.pack("=I", x2)
        decrypted += u
    decompressed = decompress_lzma(decrypted[1:])
    raw_path = catalog.file_name.replace("..\\", "")
    split_path = raw_path.split("\\")
    os.makedirs(os.path.join(*split_path[:-1]), exist_ok=True)
    file_path = os.path.join(*split_path)
    print("extracting", file_path)
    with open(file_path, "wb") as f:
        f.write(decompressed)


