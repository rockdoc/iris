# (C) British Crown Copyright 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)

import os
import tempfile
import iris
import iris.tests as tests
from iris.experimental.fileformats import register_format_handler
import numpy as np
import numpy.testing as npt


class TestSelfContained(tests.IrisTest):
    """
    Test self-contained NcML files; that is, ones with no references to
    external data.
    """
    def setUp(self):
        register_format_handler('ncml')

    def test_2d_variable(self):
        """Test a NcML file containing a simple 2D data variable."""

        ncml_doc = """<?xml version="1.0" encoding="UTF-8"?>
    <netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2"
    id="root" title="2D data test">
    
    <!-- define dimensions -->
    <dimension name="lat" length="3"/>
    <dimension name="lon" length="6"/>
    
    <!-- define coordinate variables for each of the above dimensions -->
    <variable name="lat" type="float" shape="lat">
       <attribute name="standard_name" type="string" value="latitude"/>
       <attribute name="units" type="string" value="degrees_north"/>
       <values start="0.0" increment="10.0" npts="3"/>
    </variable>
    
    <variable name="lon" type="float" shape="lon">
       <attribute name="standard_name" type="string" value="longitude"/>
       <attribute name="units" type="string" value="degrees_east"/>
       <values start="0.0" increment="10.0" npts="6"/>
    </variable>
    
    <!-- define a temperature variable dimensioned (time, lat, lon) -->
    <variable name="tas" type="float" shape="lat lon">
       <attribute name="standard_name" type="string" value="air_temperature"/>
       <attribute name="long_name" type="string" value="10m Air Temperature"/>
       <attribute name="units" type="string" value="degc"/>
       <values start="0.0" increment="1.0" npts="18"/>
    </variable>
    </netcdf>"""

        try:
            ncml_file = _create_ncml_file_from_text(ncml_doc)
            cubes = iris.load(ncml_file)
            self.assertTrue(len(cubes)==1)
            cube = cubes[0]
            self.assertEqual(cube.ndim, 2)
            self.assertEqual(cube.shape, (3, 6))
            self.assertDictEqual({'id': 'root', 'title': '2D data test'},
                cube.attributes)
            npt.assert_array_equal(cube.data.flatten(), np.arange(0.0, 18.0))
            lat_crd, lon_crd = cube.coords()
            npt.assert_array_equal(lat_crd.points, np.arange(0, 30, 10))
            npt.assert_array_equal(lon_crd.points, np.arange(0, 60, 10))
        finally:
            if os.path.exists(ncml_file): os.remove(ncml_file)

    def test_3d_variable(self):
        """Test a NcML file containing a simple 3D data variable."""

        ncml_doc = """<?xml version="1.0" encoding="UTF-8"?>
    <netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2"
    id="root" title="3D data test">
    
    <!-- define dimensions -->
    <dimension name="time" length="2"/>
    <dimension name="lat" length="3"/>
    <dimension name="lon" length="6"/>
    
    <!-- define coordinate variables for each of the above dimensions -->
    <variable name="time" type="int" shape="time">
       <attribute name="standard_name" type="string" value="time"/>
       <attribute name="units" type="string" value="days since 2000-01-01 0:0:0"/>
       <attribute name="calendar" type="string" value="360_day"/>
       <values>15 45</values>
    </variable>
    
    <variable name="lat" type="float" shape="lat">
       <attribute name="standard_name" type="string" value="latitude"/>
       <attribute name="units" type="string" value="degrees_north"/>
       <values start="0.0" increment="10.0" npts="3"/>
    </variable>
    
    <variable name="lon" type="float" shape="lon">
       <attribute name="standard_name" type="string" value="longitude"/>
       <attribute name="units" type="string" value="degrees_east"/>
       <values start="0.0" increment="10.0" npts="6"/>
    </variable>
    
    <!-- define a temperature variable dimensioned (time, lat, lon) -->
    <variable name="tas" type="float" shape="time lat lon">
       <attribute name="standard_name" type="string" value="air_temperature"/>
       <attribute name="long_name" type="string" value="10m Air Temperature"/>
       <attribute name="units" type="string" value="degc"/>
       <values start="0.0" increment="1.0" npts="36"/>
    </variable>
    </netcdf>"""

        try:
            ncml_file = _create_ncml_file_from_text(ncml_doc)
            cube = iris.load_cube(ncml_file)
            self.assertEqual(cube.ndim, 3)
            self.assertEqual(cube.shape, (2, 3, 6))
            self.assertDictEqual({'id': 'root', 'title': '3D data test'},
                cube.attributes)
            npt.assert_array_equal(cube.data.flatten(), np.arange(0.0, 36.0))
            time_crd, lat_crd, lon_crd = cube.coords()
            npt.assert_array_equal(time_crd.points, np.array([15, 45]))
            npt.assert_array_equal(lat_crd.points, np.arange(0, 30, 10))
            npt.assert_array_equal(lon_crd.points, np.arange(0, 60, 10))
        finally:
            if os.path.exists(ncml_file): os.remove(ncml_file)

    def test_ncml_data_types(self):
        """
        Test data variables using each of the recognised NcML data types, with
        the exception of char/string, which Iris does not currently support.
        """

        ncml_doc = """<?xml version="1.0" encoding="UTF-8"?>
    <netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2">
    
    <!-- define dimensions -->
    <dimension name="dim1" length="3"/>
    
    <!-- define a coordinate variable for dim1 -->
    <variable name="dim1" type="int" shape="dim1">
       <attribute name="long_name" type="string" value="dimension 1"/>
       <attribute name="units" type="string" value="1"/>
       <values>0 1 2</values>
    </variable>
    
    <!-- define a byte-type variable -->
    <variable name="bvar" type="byte" shape="dim1">
       <attribute name="long_name" type="string" value="bvar"/>
       <values>1 2 3</values>
    </variable>
    
    <!-- define a short-type variable -->
    <variable name="svar" type="short" shape="dim1">
       <attribute name="long_name" type="string" value="svar"/>
       <values>1 2 3</values>
    </variable>
    
    <!-- define an int-type variable -->
    <variable name="ivar" type="int" shape="dim1">
       <attribute name="long_name" type="string" value="ivar"/>
       <values>1 2 3</values>
    </variable>
    
    <!-- define a long-type variable -->
    <variable name="lvar" type="long" shape="dim1">
       <attribute name="long_name" type="string" value="lvar"/>
       <values>1 2 3</values>
    </variable>
    
    <!-- define a 32-bit float-type variable -->
    <variable name="fvar" type="float" shape="dim1">
       <attribute name="long_name" type="string" value="fvar"/>
       <values>1 2 3</values>
    </variable>

    <!-- define a 64-bit float-type variable -->
    <variable name="dvar" type="double" shape="dim1">
       <attribute name="long_name" type="string" value="dvar"/>
       <values>1 2 3</values>
    </variable>
    </netcdf>"""

        try:
            ncml_file = _create_ncml_file_from_text(ncml_doc)
            cubes = iris.load(ncml_file)
            vname_dtype = [
                ('bvar', np.int8), ('svar', np.int16),
                ('ivar', np.int32), ('lvar', np.int32),
                ('fvar', np.float32), ('dvar', np.float64)]
            for vname, dtype in vname_dtype:
                cube = cubes.extract_strict(vname)[0]
                self.assertEqual(cube.data.dtype, dtype)
        finally:
            if os.path.exists(ncml_file): os.remove(ncml_file)


def _create_ncml_file_from_text(ncmltext):
    """Create a temporary NcML file containing the contents of `ncmltext`."""
    fd, tmpfile = tempfile.mkstemp('.ncml', text=True)
    fhandle = os.fdopen(fd, 'w')
    fhandle.write(ncmltext)
    fhandle.close()
    return tmpfile


if __name__ == "__main__":
    tests.main()
