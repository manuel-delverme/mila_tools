from setuptools import setup

install_requires = [
    'GitPython', 'tensorboardX', 'matplotlib', 'wandb', 'fabric', 'cloudpickle', 'PyYaml', 'paramiko', 'tqdm', 'logging'
]

setup(
    name='experiment_buddy',
    version='0.0.1',
    packages=["experiment_buddy", "scripts"],
    package_data={'scripts': ['*/*.sh']},
    url='github.com/ministry-of-silly-code/experiment_buddy/',
    license='',
    author='Manuel Del Verme, Ionelia Buzatu',
    maintainer='Ionelia Buzatu',
    author_email='',
    description='',
    install_requires=install_requires,
)
