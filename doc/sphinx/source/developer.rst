.. _developer:


=========================
 Developer documentation
=========================

This documentation summarizes tools and documentation for developers.


Releasing
#########

checklist:
- fill up changelog
- change version in setup.py
- create **annotated** tag on git repository
- build tar.gz and upload it to pypi::
    $ python setup.py sdist
- use ``twine`` to upload source build package to pypi::
    $ twine upload dist/*
