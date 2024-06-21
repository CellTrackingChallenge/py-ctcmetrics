from setuptools import setup, find_packages

setup(
    name="ctc-metrics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.2",
        "opencv-python>=4.8.0.76",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.2",
        "tifffile>=2023.8.30",
        "imagecodecs",
        "pandas>=2.2.1"
    ],
    author="Timo Kaiser",
    author_email="kaiser@tnt.uni-hannover.de",
    description="Metrics for Cell Tracking Challenges",
    entry_points={
        'console_scripts': [
            'ctc_evaluate = ctc_metrics.scripts.evaluate:main',
            'ctc_validate = ctc_metrics.scripts.validate:main',
            'ctc_noise = ctc_metrics.scripts.noise:main',
            # Add more scripts here if needed
        ],
    },
)
