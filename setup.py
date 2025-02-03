from setuptools import setup, find_packages

setup(
    name='lr2',
    version='1.0',
    author='Tianyu Ren',
    packages=find_packages(),
    install_requires=[
        'wandb',
        'pettingzoo',
        'tensorboardX',
        'stable_baselines3',
        'torchdata',
        'pydantic',
        'torchinfo',
        'tabulate',
        'imageio',
      ])