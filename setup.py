from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Fashion_MNIST_DVC"
AUTHOR_USER_NAME = "Vinayakam19"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ["dvc","tensorflow","matplotlib","numpy","pandas","tqdm","PyYAML","boto3"]


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="Package for Deep learning based Fashion MNIST classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="vinayakam19@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS   
)
