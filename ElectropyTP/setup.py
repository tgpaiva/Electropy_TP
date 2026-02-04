from setuptools import setup, find_packages

setup(  
    name="ElectropyTP",
    version="0.3.0",
    description="Python package for electrochemistry data analysis",
    author="Tiago Paiva",
    author_email="tgspaiva@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'pandas', 'natsort'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)