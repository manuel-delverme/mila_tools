from setuptools import setup

install_requires = ['GitPython', 'tensorboardX', 'matplotlib', 'wandb', 'fabric']

setup(
    name='mila_tools',
    version='0.0.1',
    packages=["mila_tools", "slurm_scripts"],
    package_data={'': ['*.sh']},
    url='github.com/manuel-delverme/mila_tools/',
    license='',
    author='Manuel Del Verme',
    author_email='',
    description='',
    install_requires=install_requires,
)
