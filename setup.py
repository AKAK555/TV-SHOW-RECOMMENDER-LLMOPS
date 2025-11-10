from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="TVSHOW-RECOMMENDER",
    version="0.1",
    author="Akshay",
    packages=find_packages(),
    install_requires = requirements,
)