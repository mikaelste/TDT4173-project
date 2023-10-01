import subprocess

def install(packages):
    for package in packages:
        subprocess.call(['conda', 'install', '--yes', package])

install(['numpy', 'pandas', 'scikit-learn'])