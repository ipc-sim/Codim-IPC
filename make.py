import subprocess

runCommand = 'cd build\nmake -j 15'
subprocess.call([runCommand], shell=True)
