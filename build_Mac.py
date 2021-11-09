import subprocess

runCommand = 'mkdir build\ncd build\nrm -rf CMakeCache.txt\ncmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-11 ..\nmake -j 15'
subprocess.call([runCommand], shell=True)