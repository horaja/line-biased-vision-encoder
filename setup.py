from setuptools import setup, find_packages

setup(
    name="selective-magno-vit",
    version="0.1.0",
    description="Selective patch processing for Vision Transformers using line drawing guidance",
    author="Your Name",
    author_email="horaja@cs.cmu.edu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "timm>=0.6.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.9.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "fvcore>=0.1.5",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "black>=21.0", "flake8>=3.9.0"],
    },
)