from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="memoria-pytorch",
    version="1.0.0",
    description="Memoria is a Hebbian memory architecture for neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["torch"],
    url="https://github.com/cosmoquester/memoria.git",
    author="Park Sangjun",
    keywords=["memoria", "hebbian", "memory", "transformer"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "experiment"]),
)
