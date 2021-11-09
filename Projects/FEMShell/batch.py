import subprocess

# Intel MKL number of threads
numThreads = '16'
baseCommand = 'export MKL_NUM_THREADS=' + numThreads + '\nexport OMP_NUM_THREADS=' + numThreads + '\nexport VECLIB_MAXIMUM_THREADS=' + numThreads + '\n'

# run
for script in ['1_table_cloth_pull.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for size in ['default']:
                for membEMult in ['0.1']:
                    for bendEMult in ['1']:
                        for v in ['1', '2', '4', '8']: # different pulling speed
                            runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + size + ' ' + membEMult + ' ' + bendEMult + ' ' + v
                            if subprocess.call([runCommand], shell=True):
                                continue

for script in ['2_twisted_cylinder.py']:
    for algI in ['1']:
        for clothI in ['0']:
            for size in ['88K']:
                for membEMult in ['0.1']:
                    for bendEMult in ['0.1']:
                        for thickness in ['0', '1.5e-3']: # without or with inelastic thickness (offset)
                            runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + size + ' ' + membEMult + ' ' + bendEMult + ' ' + thickness
                            if subprocess.call([runCommand], shell=True):
                                continue

for script in ['3a_braids.py']:
    for polyline_seg in ['60']:
        for strand_size in ['30']:
            runCommand = baseCommand + 'python3 ' + script + ' ' + polyline_seg + ' ' + strand_size
            if subprocess.call([runCommand], shell=True):
                continue

for script in ['3b_hairs.py']:
    for polyline_seg in ['50']:
        for strand_size in ['35']:
            runCommand = baseCommand + 'python3 ' + script + ' ' + polyline_seg + ' ' + strand_size
            if subprocess.call([runCommand], shell=True):
                continue

for script in ['4_reef_knot.py']:
    for algI in ['0']:
        for clothI in ['2']:
            for size in ['100K']:
                for membEMult in ['0.1']:
                    for bendEMult in ['0.01']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + size + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['5_garments.py']:
    for algI in ['0']:
        for clothI in ['2']:
            for garment in ['multilayer']:
                for membEMult in ['0.01']:
                    for bendEMult in ['0.1']:
                        for anim_seq in ['Kick']:
                            runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult + ' ' + anim_seq
                            if subprocess.call([runCommand], shell=True):
                                continue

for script in ['5_garments.py']:
    for algI in ['0']:
        for clothI in ['2']:
            for garment in ['dress_knife']:
                for membEMult in ['0.01']:
                    for bendEMult in ['0.1']:
                        for anim_seq in ['Rumba_Dancing']:
                            runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult + ' ' + anim_seq
                            if subprocess.call([runCommand], shell=True):
                                continue

for script in ['6_cloth_on_rotating_sphere.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['85K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['0.1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['6_cloth_on_rotating_sphere.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['246K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['0.01']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['7_funnel.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['26K']:
                for membEMult in ['0.05']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['8_precision_card_shuffle.py']:
    for algI in ['1']:
        for clothI in ['4']:
            for garment in ['15x7']:
                for membEMult in ['1']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['9_cloth_over_codimensional_needles.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['26K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['10_noodles.py']:
    for polyline_seg in ['200']:
        for strand_size in ['25']:
            runCommand = baseCommand + 'python3 ' + script + ' ' + polyline_seg + ' ' + strand_size
            if subprocess.call([runCommand], shell=True):
                continue

for script in ['11_membrane_locking.py']:
    for algI in ['1']: # no strain limiting
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['1', '0.1', '0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['11_membrane_locking.py']:
    for algI in ['0']: # with strain limiting
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['12a_extreme_strain_limit.py']:
    for SL in ['1.01', '1.001']: # different extreme strain limits
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + SL + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['12b_extreme_strain_limit.py']:
    for SL in ['1.01', '1.001']: # different extreme strain limits
        for clothI in ['0']:
            for garment in ['2K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + SL + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['13_anisotropic_strain_limiting.py']:
    for algI in ['2']:
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['1', '0.1', '0.01']:
                    for bendEMult in ['1']:
                        runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult
                        if subprocess.call([runCommand], shell=True):
                            continue

for script in ['18_sphere_on_cloth_stack.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['0.01']:
                    for bendEMult in ['1']:
                        for n_cloth in ['10']:
                            for elastic_thickness in ['1e-2', '5e-3', '1e-3']:
                                runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult + ' ' + n_cloth + ' ' + elastic_thickness
                                if subprocess.call([runCommand], shell=True):
                                    continue

for script in ['21_sprinkles.py']:
    for vertical_size in ['40']:
        for horizontal_size in ['25']:
            runCommand = baseCommand + 'python3 ' + script + ' ' + vertical_size + ' ' + horizontal_size
            if subprocess.call([runCommand], shell=True):
                continue

for script in ['22_granules_on_cloth.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['8K']:
                for membEMult in ['0.05']:
                    for bendEMult in ['0.1']:
                        for vertical_size in ['500']:
                            for horizontal_size in ['10']:
                                for mu in ['0.05']:
                                    runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult + ' ' + vertical_size + ' ' + horizontal_size + ' ' + mu
                                    if subprocess.call([runCommand], shell=True):
                                        continue

for script in ['23_all-in.py']:
    for algI in ['0']:
        for clothI in ['0']:
            for garment in ['26K']:
                for membEMult in ['0.05']:
                    for bendEMult in ['0.1']:
                        for rod_size in ['50']:
                            runCommand = baseCommand + 'python3 ' + script + ' ' + algI + ' ' + clothI + ' ' + garment + ' ' + membEMult + ' ' + bendEMult + ' ' + rod_size
                            if subprocess.call([runCommand], shell=True):
                                continue