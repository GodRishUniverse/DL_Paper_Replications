from setuptools import setup, find_packages
import os


print(f"Current directory: {os.getcwd()}")
# Debug: list all files and folders
print(f"Files and folders: {os.listdir('.')}")
# Debug: print what find_packages found
packages = find_packages()
print(f"Found packages: {packages}")


setup(
    name="paperreplications",
    version="0.1",
    packages=find_packages(),
    author="Rishabh Agarwal",
    author_email="agarwalrishabh2005@gmail.com",
    install_requires=[
        "torch",
        "einops",
        "tqdm",
        "wandb",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "jax",
        # Add other dependencies
    ],
)
