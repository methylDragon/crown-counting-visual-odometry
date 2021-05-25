from setuptools import setup, find_namespace_packages

# Package metadata.
name = "ccvo"
description = "CCVO: Crown Counting Visual Odometry"
version = "0.0.1"
release_status = "Development Status :: 2 - Pre-Alpha"
dependencies = [
    "opencv-contrib-python",
    "numpy",

    # Github Private Repository
    'methyl-opencv-suite @ git+ssh://git@github.com/methylDragon/methyl-opencv-suite.git#egg=methyl-opencv-suite#subdirectory=methyl-opencv-suite'
]
extras = {}

packages = [] #find_namespace_packages(where='src', include=['mos.*'])

setuptools.setup(
    name=name,
    version=version,
    description=description,
    author="methylDragon",
    author_email="methylDragon@gmail.com",
    license="Apache License 2.0",
    url="https://github.com/methylDragon/methyl-opencv-suite",
    classifiers=[
        release_status,
        "Intended Audience :: Developers",
        "License :: Apache License 2.0",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Linux",
    ],
    platforms="Linux",
    package_dir= {'': 'src'},
    packages=packages,
    install_requires=dependencies,
    extras_require=extras,
    python_requires=">=3.6",
    scripts=[],
    zip_safe=True,
)
