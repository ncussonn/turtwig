import os
from glob import glob
from setuptools import setup

package_name = 'refine_cbf'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nate',
    maintainer_email='ncussonn@gmail.com',
    description='Executes core Refine CBF nodes.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [   
            'state_estimator = refine_cbf.state_estimator:main',
            'safety_filter = refine_cbf.safety_filter:main',
            'dynamic_programming = refine_cbf.dynamic_programming:main',
            'diff_drive = refine_cbf.differential_drive:main',
            'nominal_policy = refine_cbf.nominal_policy:main',
            'low_level_control = refine_cbf.low_level_controller:main',	
        ],
    },
)
