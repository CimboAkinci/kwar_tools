import struct
import sys

from utils import get_xor_mask

if len(sys.argv) != 3 or not sys.argv[2].endswith(".inc"):
    print("usage: python decrypt_inc.py [version (tw2010, tr2013, tw2020)] [.inc file]")
    sys.exit(1)

# XOR mask for "decryption"
XOR_MASK = get_xor_mask(sys.argv[1])

C = ""
with open(sys.argv[2], "rb") as f:
    C = f.read()

def decode_utf16(x: bytes) -> str:
    w = x.split(b"\x00\x00")[0]
    if len(w) % 2 == 1:
        w += b"\x00"
    return w.decode("utf-16")

off = 0
def unpack(fmt: str, data: bytes, size: int):
    global off
    out = struct.unpack(fmt, data[off:off+size])
    off += size
    if len(out) == 1:
        return out[0]
    return out

size = unpack("=I", C, 4)
key1 = unpack("=B", C, 1)
key2 = unpack("=H", C, 2)

def decrypt1(word, key) -> int:
    key = key & 0xff  # byte
    out = word >> (key & 0x1f) | word << 0x20 - (key & 0x1f)
    return out & 0xffffffff  # word

def decrypt2(word, key) -> int:
    word = word & 0xffffffff
    return word ^ XOR_MASK[key % 2048]

read_bytes = 2 # no idea what to put here

decrypted = bytes()
while read_bytes < size:
    x = unpack("=I", C, 4)

    k1 = key1 + (read_bytes + (read_bytes >> 0x1f & 3) >> 2)
    x1 = decrypt1(x, k1)

    k2 = key2 + (read_bytes + (read_bytes >> 0x1f & 3) >> 2)
    x2 = decrypt2(x1, k2)

    read_bytes += 4
    u = struct.pack("=I", x2)
    decrypted += u

with open(sys.argv[2].replace(".inc", ".ini"), "w", encoding="utf-8") as f:
    str = decrypted.decode("utf-16")
    f.write(str)
