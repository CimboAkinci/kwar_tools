import lzma
import os
import struct
import sys
from dataclasses import dataclass

from utils import get_xor_mask

XOR_MASK = None

def encrypt1(word, key) -> int:
    """反向 decrypt1"""
    key = key & 0xff
    rotated = (word << (key & 0x1f)) | (word >> (0x20 - (key & 0x1f)))
    return rotated & 0xffffffff

def encrypt2(word, key) -> int:
    """反向 decrypt2"""
    return (word & 0xffffffff) ^ XOR_MASK[key % 2048]

@dataclass
class MangDataCatalog:
    file_name: str
    key1: int
    key2: int
    offset: int
    uncompressed_size: int
    size: int

def compress_lzma(data):
    """Use external lzma.exe to compress data with proper parameters."""
    with open("temp_input", "wb") as f:
        f.write(data)
    my_path = os.path.realpath(__file__)
    lzmaexe_path = os.path.join(os.path.dirname(my_path), "lzma.exe")
    os.system(f'"{lzmaexe_path}" e temp_input temp_output.lzma -d23 >NUL 2>&1')
    with open("temp_output.lzma", "rb") as f:
        compressed = f.read()
    os.unlink("temp_input")
    os.unlink("temp_output.lzma")
    return b'\x00' + compressed

def pack_files(input_dir, output_file):
    """將文件打包成 MangData_pak"""
    # 收集所有需要打包的文件
    catalogs = []
    current_offset = 0
    data_blocks = b''

    # 遍歷目錄
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(file_path, input_dir).replace('/', '\\')
            full_file_name = f"..\\MangData\\{rel_path}"  # 恢復原版路徑格式
            print("packing", full_file_name)

            # 讀取文件數據
            with open(file_path, "rb") as f:
                raw_data = f.read()
            
            uncompressed_size = len(raw_data)
            compressed_data = compress_lzma(raw_data)
            key1 = 0
            key2 = 0

            # 加密數據塊
            encrypted_data = b''
            for i in range(0, len(compressed_data), 4):
                chunk = compressed_data[i:i+4].ljust(4, b'\x00')  # 填充至 4 字節對齊
                x = struct.unpack("=I", chunk)[0]
                x1 = encrypt2(x, key2 + i // 4)
                x2 = encrypt1(x1, key1 + i // 4)
                encrypted_data += struct.pack("=I", x2)

            # 計算偏移量（稍後更新）
            size = len(compressed_data)
            catalog = MangDataCatalog(full_file_name, key1, key2, current_offset, uncompressed_size, size)
            catalogs.append(catalog)

            data_blocks += encrypted_data
            current_offset += len(encrypted_data)

    # 更新偏移量
    offset_base = 8 + len(catalogs) * 268  # 頭部 8 字節 + blob1 大小
    for catalog in catalogs:
        catalog.offset += offset_base

    # 構建 blob1（目錄）
    blob1 = b''
    for i, catalog in enumerate(catalogs):
        file_name_bytes = catalog.file_name.encode('utf-8').ljust(253, b'\x00')
        blob1 += struct.pack("=253s B H I I I", file_name_bytes, catalog.key1, catalog.key2, 
                            catalog.offset, catalog.uncompressed_size, catalog.size)
    
    # 加密 blob1
    encrypted_blob1 = b''
    for i in range(0, len(blob1), 4):
        x = struct.unpack("=I", blob1[i:i+4])[0]
        x1 = encrypt2(x, i // 4)
        x2 = encrypt1(x1, i // 4)
        encrypted_blob1 += struct.pack("=I", x2)

    # 構建最終文件
    output_data = struct.pack("=Q", len(catalogs)) + encrypted_blob1 + data_blocks

    # 寫入文件
    with open(output_file, "wb") as f:
        f.write(output_data)
    print(f"打包完成，已生成: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python pack_mang.py [version (tw2010, tr2013, tw2020)] [input directory] [output file]")
        sys.exit(1)

    XOR_MASK = get_xor_mask(sys.argv[1])
    
    input_dir = sys.argv[2]
    output_file = sys.argv[3]
    pack_files(input_dir, output_file)
