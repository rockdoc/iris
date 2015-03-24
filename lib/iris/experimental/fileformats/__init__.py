# (C) British Crown Copyright 2010 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Experimental file formats. These formats are not automatically registered when
Iris is started. To use a format of interest call the registration function
provided in this module.

For example, to register the experimental NcML file format loader you would do
something along these lines:

>>> import iris.experimental.fileformats as xfmts
>>> xfmts.register_format_handler('NcML')
>>> cubes = iris.load('cooldata.ncml')

"""

from __future__ import (absolute_import, division, print_function)

from iris.exceptions import TranslationError
from iris.fileformats import FORMAT_AGENT
from iris.io.format_picker import (FileExtension, FormatSpecification)

from . import ncml2

# List of names of experimental file formats.
FILE_FORMATS = ('ncml',)


def register_format_handler(format):
    """Register an i/o handler for the specified file format."""
    if format.lower() not in [f.lower() for f in FILE_FORMATS]:
        raise ValueError("Unrecognised experimental file format: " + format)

    try:
        if format.lower() == 'ncml':
            fmt_spec = FormatSpecification(
                'NetCDF Markup Language (NcML)',
                FileExtension(), ".ncml",
                ncml2.load_cubes, priority=3)
            FORMAT_AGENT.add_spec(fmt_spec)
        print("Registered handler for file format: " + format)

    except:
        msg = "Problem trying to register handler for file format: " + format
        raise TranslationError(msg)
