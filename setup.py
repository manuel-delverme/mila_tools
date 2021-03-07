from setuptools import setup

install_requires = [
    'GitPython', 'tensorboardX', 'matplotlib', 'wandb', 'fabric', 'cloudpickle'
]

setup(
    name='experiment_buddy',
    version='0.0.1',
    packages=["mila_tools", "slurm_scripts"],
    package_data={'': ['*.sh']},
    url='github.com/ministry-of-silly-code/experiment_buddy/',
    license='',
    author='Manuel Del Verme, Ionelia Buzatu',
    maintainer='Ionelia Buzatu',
    author_email='',
    description='',
    install_requires=install_requires,
)
