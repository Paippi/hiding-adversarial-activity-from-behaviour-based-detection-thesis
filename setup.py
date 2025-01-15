from setuptools import setup, find_packages
import os
import sys

package_name = "explain_nids"

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, "src")

about = {}
with open(os.path.join(src_dir, package_name, "__about__.py")) as f:
    exec(f.read(), about)

with open(os.path.join(base_dir, "README.md")) as f:
    long_description = f.read()

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__summary__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=about["__license__"],
    url=about["__uri__"],
    author=about["__author__"],
    author_email=about["__email__"],
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
)
