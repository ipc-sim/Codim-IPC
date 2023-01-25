# merges shell.obj (3d shell) and vol.obj (3d object) to shell.obj file
import os
import sys


def merge_file(file1, file2, out):
    with open(file1, "r") as f:
        data1 = f.read()
    with open(file2, "r") as f:
        data2 = f.read()
    merged = ""
    nvertices = 0
    # add file1 vertices
    merged += f"#file1 vertices\n"
    for line in data1.splitlines():
        if line[0] == 'v':
            merged += f"{line}\n"
            nvertices += 1

    # add file2 vertices
    merged += f"#file2 vertices\n"
    for line in data2.splitlines():
        if line[0] == 'v':
            merged += f"{line}\n"

    # add file1 faces
    merged += f"#file1 faces\n"
    for line in data1.splitlines():
        if line[0] == 'f':
            merged += f"{line}\n"

    # add file2 faces
    merged += f"#file2 faces\n"
    for line in data2.splitlines():
        if line[0] == 'f':
            v1, v2, v3 = line[2:].split()
            v1 = int(v1) + nvertices
            v2 = int(v2) + nvertices
            v3 = int(v3) + nvertices
            merged += f"f {v1} {v2} {v3}\n"
    with open(out, "w") as f:
        f.write(merged)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Expected file1 file2 output")
        exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    out = sys.argv[3]
    merge_file(file1, file2, out)
