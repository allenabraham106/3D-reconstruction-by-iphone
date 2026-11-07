from glob import glob
from setuptools import setup

setup(
    name="pioneer_map",
    version="0.1.0",
    packages=["pioneer_map"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/pioneer_map"]),
        ("share/pioneer_map", ["package.xml"]),
        ("share/pioneer_map/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    entry_points={"console_scripts": ["voxel_publisher = pioneer_map.voxel_publisher:main"]},
)
