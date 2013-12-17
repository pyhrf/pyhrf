#!/bin/sh

cp MANIFEST.default MANIFEST.in && python setup.py egg_info --tag-date sdist
