from setuptools import setup, find_packages

setup(
    name="py-ctcmetrics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scipy",
        "tifffile",
        "imagecodecs",
        "pandas"
    ],
    author="Timo Kaiser",
    author_email="kaiser@tnt.uni-hannover.de",
    description="Metrics for Cell Tracking Challenges",
    long_description="Metrics for Cell Tracking Challenges",
    entry_points={
        'console_scripts': [
            'ctc_evaluate = ctc_metrics.scripts.evaluate:main',
            'ctc_validate = ctc_metrics.scripts.validate:main',
            'ctc_noise = ctc_metrics.scripts.noise:main',
            'ctc_visualize = ctc_metrics.scripts.visualize:main',
            # Add more scripts here if needed
        ],
    },
)
