from setuptools import find_packages, setup

with open('2048-ai/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

setup(
    name='2048-ai',
    version=__version__,
    description='AI learns to play 2048!',
    long_description=readme,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
    ]
)
