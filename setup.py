from setuptools import find_packages, setup

with open("README.md", "r") as fin:
    long_description = fin.read()


setup(
    name="test_your_embedding",
    version="0.0.1",
    url="",
    license="MIT",
    author="Yhua Liao",
    author_email="ml.liaoyuhua@gmail.com",
    description="Standard and diverse tests for gnn embedding algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.4.3",
        "requests>=2.28.1",
        "scikit-learn>=1.1.1",
        "torch>=1.10.0",
    ],
    extras_require={
        "test": [
            "flake8",
            "isort",
            "pylint",
            "pytest",
        ],
    },
    packages=find_packages(),
)
