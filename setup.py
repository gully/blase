import setuptools


setuptools.setup(
    name="blase",
    version="0.3",
    author="gully",
    author_email="igully@gmail.com",
    description="Forward Modeling echelle spectra with PyTorch",
    long_description="Hybrid data/model-driven forward modeling of echelle spectra",
    long_description_content_type="text/markdown",
    url="https://github.com/gully/blase",
    install_requires=["numpy", "scipy", "torch"],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
