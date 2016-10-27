.. _developer:


=======================
Developer documentation
=======================

This documentation summarizes tools and documentation for developers.

Developing PyHRF package
########################

Some guidelines are provided in the
`github's wiki <https://github.com/pyhrf/pyhrf/wiki>`_. Before writing code, be
sure to read at least
`the git workflow page <https://github.com/pyhrf/pyhrf/wiki/Git-workflow>`_ and
the `Code Quality page <https://github.com/pyhrf/pyhrf/wiki/Code-Quality>`_.

Updating Website Documentation
##############################

The documentation is written using
`RestructuredText <http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html>`_
markup language and is compiled into a website using
`Sphinx <http://www.sphinx-doc.org/en/stable/>`_.

**Remark**: You'll need at least the version 1.3.1 of Sphinx to compile the code
into website (see :doc:`../installation` for installation details).

To modify the website documentation, follow these steps:

- update documentation code in ``doc/sphinx/source`` folder
- compile documentation with ``make html``
- check that the local website looks like you want it (open
  ``doc/sphinx/build/html/index.html`` file)
- commit your changes in your local repository and create a pull-request on
  `github pyhrf repository <https://github.com/pyhrf/pyhrf>`_ (see
  `github's wiki to see how <https://github.com/pyhrf/pyhrf/wiki/Git-workflow>`_)
- copy the content of the ``doc/sphinx/build/html/`` folder into the website
  folder on Inria server (only accessible from Inria internal network)
- check the `pyhrf website <http://pyhrf.org>`_

TODO
----

Add website folder access from outside Inria Grenoble to upload website from
outside Inria.


Releasing
#########

A **release** is a stable version of PyHRF package which is available as a
tagged version on github repository and installable from
`the Python Package Index (PyPI) <https://pypi.python.org/pypi>`_ using ``pip``
command.

To upload the package to PyPI, you need to install ``twine`` package.
On most GNU/Linux systems, you can use your package manager to install ``twine``
package or use the ``pip`` command to install it.

To make a new PyHRF release follow these steps:

- fill up ``CHANGELOG.rst`` file with latest modifications
- change version in setup.py
- create **annotated** tag on git repository with ``git tag -a x.y.z`` command
  (x.y.z being the version number)
- build tar.gz package:

.. code:: bash

    $ python setup.py sdist

- to upload the source build package to PyPI, run:

.. code:: bash

    $ twine upload dist/*

The last command will ask for username and password the first time you use it
(it asks for savings credentials in ``$HOME/.pypirc`` file for later uploads).

TODO
----

Remove anything that break a wheel build and make a wheel build instead of
source build.
