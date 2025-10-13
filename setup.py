from setuptools import find_packages, setup

package_name = 'scannet_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Christian Rauch',
    maintainer_email='Christian.Rauch@unileoben.ac.at',
    description='publisher for ScanNet RGB-D sequences',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scannet_publisher = scannet_publisher.scannet_publisher:main'
        ],
    },
)
