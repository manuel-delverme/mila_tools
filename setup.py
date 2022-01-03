from setuptools import setup

install_requires = [
    'GitPython',
    'tensorboardX',
    'matplotlib',
    'wandb',
    'fabric',
    'cloudpickle',
    'PyYaml',
    'paramiko',
    'tqdm',
    'aiohttp',
]

setup(
    name='experiment_buddy',
    version='0.0.9',
    packages=["experiment_buddy", "experiment_buddy.buddy_init", "scripts"],
    package_data={'scripts': ['*/*.sh']},
    url='https://github.com/ministry-of-silly-code/experiment_buddy/',
    license='',
    author='Manuel Del Verme, Ionelia Buzatu, Simone Totaro',
    maintainer='Manuel Del Verme',
    author_email='',
    description='',
    install_requires=install_requires,
    entry_points={"console_scripts": ["buddy-init=experiment_buddy.buddy_init.init_project:sys_main"]}
)
