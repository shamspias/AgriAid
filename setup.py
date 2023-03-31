from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="agriaid",
    version="0.1.0",
    author="Shamsuddin Ahmed",
    author_email="shamspiasai@gmail.com",
    description="A package for early detection and diagnosis of plant diseases in Bangladesh",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shamspias/agriaid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.8',
    install_requires=[
        "Flask>=2.0.1",
        "numpy>=1.21.2",
        "Pillow>=8.3.1",
        "scikit-learn>=0.24.2",
        "tensorflow>=2.6.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
        ],
    },
)
