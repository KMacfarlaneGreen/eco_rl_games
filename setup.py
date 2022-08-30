from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="KMacfarlaneGreen",
    author_email="author@example.com",
    description="Building simulations of ecological games through multi agent reinforcement learning",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
