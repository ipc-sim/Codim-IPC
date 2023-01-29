#converts obj files to png by capturing from vertain angle and converts the png images to video
import sys
import os
import cv2
import subprocess
import time

def run_cmds(cmds):
    #cmd = "echo hi && timeout 6 && e bye"
    handles = []
    for cmd in cmds:
        handle = subprocess.Popen(
            cmd, shell=True, stderr=subprocess.PIPE)
        handles.append(handle)
    while 1:
        tobreak = True
        for handle in handles:
            if handle.poll() is None:
                tobreak = False
                break
        if tobreak:
            break
        time.sleep(.2)
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Expected input_folder output_path")
        exit(1)
    os.system("mkdir -p .temp")
    os.system("rm .temp/*")
    ncores = 15
    folder = sys.argv[1]
    files = os.listdir(folder)
    obj_files = []
    curridx = 0
    left = True
    while left:
        cmds = []
        for _ in range(ncores):
            path = os.path.join(folder, f"shell{curridx}.obj")
            if not(os.path.exists(path)):
                left = False
                break
            else:
                cmds.append(f"python obj2png.py -i {folder}/shell{curridx}.obj -a -90 -e 100 -o ./.temp/shell{curridx}.png")
            curridx += 1
        if len(cmds) != 0:
            run_cmds(cmds)
    image_folder = '.temp'
    video_name = sys.argv[2]

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()