# various utils

def get_xor_mask(type: str) -> str:
    with open("xor_mask_%s.txt" % type, "r") as f:
        return [int(x.strip(), 16) for x in f.read().split(",")]
