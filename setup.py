from xml.etree.ElementInclude import include
import setuptools

setuptools.setup(
    name="ct_update",
    version="0.0.0",
    author="Jan Emily Mangulabnan",
    author_email="jmangul1@jhu.edu",
    description="ct_update ",
    long_description="CT Update",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "rich",
        "click",
    ],
    include_package_data=True,
    python_requires=">=3.8",
)