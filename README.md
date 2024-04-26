# kwar_tools

Tools to work with the ancient archives.

Courtesy of **Project Kehanet Online**. Join our [Discord](https://discord.com/invite/Yjt6YyAxsf) for help and discussion.

These scripts are designed to work with the latest known Eternal81 client. You know where to find it...

Please contact us if you have the files for another version of the game.

# Installation

Just try to run the scripts and install missing packages with `pip`. (TODO: add a `requirements.txt`)

# Usage

Run the script for the file format you want to unpack, i.e. `unpack_kwtx.py` for `.kwtx` files.

Note that mesh files such as `.kwsm`, `.kwam` and `.kwsk` require the corresponding `.kwtx` to be unpacked first.

`.kwan` files can be extracted with a dummy mesh using `unpack_kwan.py`. It's possible to export with mesh using `unpack_kwan_unified.py` if the matching `.kwtx` is given. This doesn't work with every pair.

Scene files get unpacked into 64 terrain meshes and a `.json` file for the metadata.

`unpack_kwan.py` uses the included `gltfpack.exe` (from [meshoptimizer](https://github.com/zeux/meshoptimizer)) to compress the animations.

# Known issues

* There are a bunch of issues with texture alpha, blended textures, cube maps etc. which I'm too lazy to list here.
* Scene files have too many unknown parameters, although it's possible to extract what's needed to recreate a map.
