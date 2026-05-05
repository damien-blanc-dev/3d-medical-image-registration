from setuptools import setup, find_packages

setup(
    name="med3dreg",
    version="0.1.0",
    author="Damien Blanc",
    description="3D medical image registration: rigid and deep learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "SimpleITK>=2.3.0",
        "nibabel>=5.2.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "torch>=2.2.0",
        "matplotlib>=3.8.0",
        "PyYAML>=6.0.1",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
    ],
)
