from setuptools import setup, find_packages

setup(
    name="cratrcountr",
    version="0.1.0",
    description="For analyzing crater count data",
    author="Sam Bell",
    author_email="swbell11@gmail.com",
    url="https://github.com/samwbell/cratrcountr",
    packages=find_packages(where='src'),
    install_requires=[
        "numpy>=1.26.4",
        "scipy>=1.14.0",
        "pandas>=2.2.2",
        "ash @ git+https://github.com/ajdittmann/ash.git@master#egg=ash"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',  # Minimum Python version required
)
