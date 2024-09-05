# setup.py
from setuptools import setup

setup(
    name="segment_profiles",
    version="0.1",
    packages=["segment_profiles"],
    entry_points={
        "console_scripts": [
            "segment_profiles = segment_profiles.__init__:main",
        ],
    },
)
