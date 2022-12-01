from setuptools import setup

setup(
    name='gym_grid',
    version='1.0',
    install_requires=['gym', 'numpy', 'matplotlib'],
    author='Sagar',
    packages=['gym_grid', 'gym_grid.envs']
)