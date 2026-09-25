from setuptools import find_packages, setup

package_name = "g1_perception"

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
    description="G1 LiDAR bridge (TF re-framing) and LiDAR-HMR SMPL body-mesh node.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lidar_bridge = g1_perception.lidar_bridge:main",
            "smpl_hmr_node = g1_perception.smpl_hmr_node:main",
        ],
    },
)
