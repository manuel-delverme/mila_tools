from setuptools import setup, find_namespace_packages

install_requires = ['GitPython', 'tensorboardX', 'matplotlib', 'wandb', 'fabric']

setup(
    name='mila_tools',
    version='0.0.1',
    packages=find_namespace_packages(
        where='mila_tools/',
        include=['mila_tools.*'],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    url='github.com/manuel-delverme/mila_tools/',
    license='',
    author='Manuel Del Verme',
    author_email='',
    description='',
    install_requires=install_requires,
)
