#calls combinefile for all obj files
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected folder")
        exit(1)
    folder = sys.argv[1]
    for i in range(10000):
        if os.path.exists(f"{folder}/shell{i}.obj") and os.path.exists(f"{folder}/vol{i}.obj"):
            os.system(f"python3 combinefile.py {folder}/shell{i}.obj {folder}/vol{i}.obj {folder}/shell{i}.obj")
        else:
            break