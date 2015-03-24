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
Iris file format module for loading NcML v2.2 files.

The following components of the NcML 2 model are supported:

* aggregation of datasets encoded in any of the formats recognised by Iris
* union-type aggregation
* joinExisting-type aggregation (for simple 1D aggregation dimensions only)
* joinNew-type aggregation (for simple 1D aggregation dimensions only)
* addition, removal and renaming of data variables
* addition, removal and renaming of global or variable attributes
* support for timeUnitsChange attribute on joinExisting aggregations

The following components are NOT currently supported:

* these attributes of the <netcdf> element: enhance, addRecords, fmrcDefinition
* these attributes of the <dimension> element: isShared, isVariableLength
* these attributes of the <aggregation> element: recheckEvery
* these attributes of the <scan> element: dateFormatMark
* modification or removal of existing dimensions
* nested aggregations
* nested <variable> elements
* <group> elements
* <scanFmrc> elements
* the Structure data type

"""
import sys
from ncml2_helpers import NcmlDataset, NcmlContentError, NcmlSyntaxError


def load_cubes(filenames, callback=None, **kwargs):
    """
    Generator function returning a sequence of cube objects associated with the
    netCDF files specified within an NcML file.

    NOTE: The current implementation can only handle a single NcML file.
    """

    if isinstance(filenames, (list, tuple)):
        if len(filenames) > 1:
            errmsg = "Iris currently can only read a single NcML file.\n" \
                     "{0} filenames were specified.".format(len(filenames))
            raise IOError(errmsg)
        ncml_file = filenames[0]
    elif isinstance(filenames, basestring):
        ncml_file = filenames
    else:
        raise ValueError("Invalid file-like argument passed to "
            "ncml2.load_cubes() function")

    # Create ncml dataset object
    try:
        ncml_dataset = NcmlDataset(ncml_file, **kwargs)
    except (NcmlContentError, NcmlSyntaxError):
        print >>sys.stderr, "Error trying to load NcML dataset"
        raise

    for cube in ncml_dataset.get_cubes():
        if callback is not None:
            cube = iris.io.run_callback(callback, cube, None, ncml_file)
        if cube is not None:
            yield cube


def main():
    """Rudimentary function for test-loading a NcML file."""
    ncml_filename = sys.argv[1]
    kwargs = dict([arg.split('=') for arg in sys.argv[2:] if '=' in arg])
    cubes = list(load_cubes(ncml_filename, **kwargs))
    print "Resultant Cubes:"
    for cube in cubes:
        print cube


if __name__ == '__main__':
    usage = "usage: python ncml2.py <ncml_file> [log_level=<level>]"
    if len(sys.argv) < 2:
        print usage
        sys.exit(1)
    main()
