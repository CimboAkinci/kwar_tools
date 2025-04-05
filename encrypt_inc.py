import os
import struct
import sys

from utils import get_xor_mask

if len(sys.argv) != 3 or not sys.argv[2].endswith(".ini"):
    print("usage: python encrypt_inc.py [version (tw2010, tr2013, tw2020)] [.ini file]")
    sys.exit(1)

# XOR mask for "decryption"
XOR_MASK = get_xor_mask(sys.argv[1])

out = bytes()

# Read .ini and re-encode as UTF16
contents = None
with open(sys.argv[2], "r", encoding="utf-8") as f:
    contents = f.read()

contents_bin = contents.encode("utf-16")

# Do bit shuffle + xor

def encrypt1(word, key) -> int:
    key = key & 0xff  # byte
    out = word << (key & 0x1f) | word >> 0x20 - (key & 0x1f)
    return out & 0xffffffff  # word

def encrypt2(word, key) -> int:
    word = word & 0xffffffff
    return word ^ XOR_MASK[key % 2048]

size = len(contents_bin)
key1 = 0
key2 = 0

out += struct.pack("=I", size)
out += struct.pack("=B", key1)
out += struct.pack("=H", key2)

written_bytes = 2 # no idea what to put here
off = 0

decrypted = bytes()
while written_bytes < size:
    x = struct.unpack("=I", contents_bin[off:off+4])[0]
    off += 4

    k2 = key2 + (written_bytes + (written_bytes >> 0x1f & 3) >> 2)
    x1 = encrypt2(x, k2)

    k1 = key1 + (written_bytes + (written_bytes >> 0x1f & 3) >> 2)
    x2 = encrypt1(x1, k1)

    written_bytes += 4
    u = struct.pack("=I", x2)
    decrypted += u

out += decrypted

filename = os.path.basename(sys.argv[2]).replace(".ini", ".inc")

with open(filename, "wb") as f:
    f.write(out)
