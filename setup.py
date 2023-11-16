from setuptools import setup, find_packages

setup(
    name="ctc-metrics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scipy",
        "tifffile",

    ],
    author="Timo Kaiser",
    author_email="kaiser@tnt.uni-hannover.de",
    description="Metrics for Cell Tracking Challenges",
    entry_points={
        'console_scripts': [
            'ctc_evaluate = scripts.evaluate:main',
            'ctc_validate = scripts.validate:main',
            'ctc_eval = scripts.evaluate:main',
            # Add more scripts here if needed
        ],
    },
)
