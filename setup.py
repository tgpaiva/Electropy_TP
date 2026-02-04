from setuptools import setup, find_packages

setup(
    name="ElectropyTP",  # Replace with your package name
    version="0.3.2",
    author="Tiago Paiva",
    author_email="tgspaiva@gmail.com.com",
    description="A python package for electrochemistry data analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tgpaiva/Electropy_TP",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy", 
        "pandas",
        "scipy",
        "natsort",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
