import setuptools

with open("README.md", "r") as ReadmeFile:
    long_description = ReadmeFile.read()

with open("requirements.txt", "r") as requirFile:
    requirements = requirFile.read().splitlines()

setuptools.setup(
    name="BanditsSimulator",
    version="0.0.0",
    author="aziz jegham",
    author_email="jeghammedaziz@gmail.Com",
    description="Module simulating bandits for Reinforcement Learning",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="git@github.com:MAJegham/RL_BanditsSimulator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU GPL",
    ],
    python_requires=">=3.6",
    install_requires=requirements
)

