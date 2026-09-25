from setuptools import find_packages, setup

package_name = "livox_detection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="Livox LiDAR 3D object detection using VoxelNeXt inference.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "livox_detection_node = livox_detection.livox_detection_node:main",
        ],
    },
)
