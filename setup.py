"""
Setup script for image alignment package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="image-alignment-dual-camera",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Автоматическое совмещение изображений с двух камер через гомографию",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-alignment",

    packages=find_packages(),

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.10",

    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "scipy>=1.11.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "align-images=main:main",
            "test-alignment=test_single_pair:main",
        ],
    },

    include_package_data=True,

    project_urls={
        "Bug Reports": "https://github.com/yourusername/image-alignment/issues",
        "Source": "https://github.com/yourusername/image-alignment",
        "Documentation": "https://github.com/yourusername/image-alignment/blob/main/README.md",
    },
)
