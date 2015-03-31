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
Helper classes and functions for loading NcML v2.2 files.

"""

# TODO: handle dateFormatMark attribute for join-type aggregations

import sys
import os
import re
import logging
import numpy as np
import iris
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import ConcatenateError, MergeError
from xml.dom.minidom import parse
from datetime import datetime

# Set default logging options.
DEFAULT_LOG_NAME = 'ncml2'
DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_LOG_FORMAT = '%(levelname)s: %(message)s'

# Define constants for NcML aggregation types.
AGG_UNION = 'union'
AGG_JOIN_NEW = 'joinNew'
AGG_JOIN_EXISTING = 'joinExisting'

# Define lists of the permissible attributes for various NcML 2.2 tags
ATT_NAMELISTS = {
    'aggregation': ('type', 'dimName', 'recheckEvery', 'timeUnitsChange'),
    'attribute': ('name', 'type', 'separator', 'value', 'orgName'),
    'dimension': ('name', 'length', 'isUnlimited', 'isVariableLength',
                  'isShared', 'orgName'),
    'netcdf': ('location', 'id', 'title', 'enhance', 'addRecords', 'ncoords',
               'coordValue'),
    'remove': ('name', 'type'),
    'scan': ('location', 'suffix', 'regExp', 'subdirs', 'olderThan',
             'dateFormatMark', 'enhance'),
    'values': ('start', 'increment', 'npts', 'separator'),
    'variable': ('name', 'type', 'shape', 'orgName'),
    'variableAgg': ('name',)
}

# Mapping from NcML data types to NumPy data types.
NCML_TO_NUMPY_TYPE_MAP = {
    'byte': 'int8',
    'short': 'int16',
    'int': 'int32',
    'long': 'int32',
    'float': 'float32',
    'double': 'float64',
    'char': 'S1'
}


class NcmlSyntaxError(Exception):
    """Exception class for NcML syntax errors."""
    pass


class NcmlContentError(Exception):
    """Exception class for NcML content errors."""
    pass


class NcmlElement(object):
    """
    A lightweight class for creating an object representation of an NcML
    element with equivalent attributes. Avoids the need to create discrete
    classes for each NcML element type, which would be a viable alternative
    approach.
    """
    def __init__(self, node, **kwargs):
        self.node = node
        for att_name in ATT_NAMELISTS.get(node.tagName, []):
            att_value = node.getAttribute(att_name)
            if att_value == '': att_value = None
            setattr(self, att_name, att_value)
        if kwargs: self.__dict__.update(kwargs)

    @property
    def elem_type(self):
        """Return the element type name."""
        return self.node.tagName


class NcmlDataset(object):
    """
    Class for representing an aggregate dataset as defined in an NcML v2.2
    document. Full details of the NcML specification can be found at
    http://www.unidata.ucar.edu/software/netcdf/ncml/

    A DOM XML parser is used here as (i) this is easier to work with compared
    with SAX parsers, and (ii) most NcML documents are likely to be small
    enough to fit into main memory.
    """

    def __init__(self, file_like, **kwargs):
        """
        Initialise an NcML dataset instance from a file or file-like object.

        :param file_like: The pathname or file object containing the NcML
            document to parse.
        """
        self.filename = None
        self.dim_nodes = list()
        self.dim_coords = list()
        self.cubelist = iris.cube.CubeList()
        self.removelist = list()
        self.explicit = False
        self.logger = logging.getLogger(DEFAULT_LOG_NAME)

        if 'log_level' in kwargs:
            log_level = getattr(logging, kwargs['log_level'].upper(),
                DEFAULT_LOG_LEVEL)
            _update_logger(level=log_level)
            self.logger.info("Set logging level to {0}".format(kwargs['log_level']))

        # Check to see whether the caller supplied a filename or file-like
        # object.
        close_after_parse = False
        if isinstance(file_like, basestring):
            self.filename = os.path.expandvars(os.path.expanduser(file_like))
            ncml_file = open(self.filename)
            close_after_parse = True
        elif isinstance(file_like, file):
            ncml_file = file_like
        else:
            raise ValueError("Invalid file-like argument passed to "
                "NcmlDataset constructor.")

        # Parse the NcML file
        try:
            self.logger.info("Parsing NcML source document...")
            try:
                ncml_doc = parse(ncml_file)
            except Exception, exc:
                self.logger.error("Error trying to parse NcML document.")
                raise exc
            self._handle_doc(ncml_doc)
            self.logger.info("Finished parsing. Loaded a total of %d cube%s",
                self.ncubes, 's'[self.ncubes==1:])
        finally:
            if close_after_parse: ncml_file.close()

    @property
    def ncubes(self):
        """Returns the number of distinct cubes comprising the dataset."""
        return len(self.cubelist)

    @property
    def basedir(self):
        """Returns the base directory of the input NcML file."""
        return os.path.dirname(self.filename)

    def get_cubes(self):
        """Return a list of the cubes comprising the dataset."""
        return self.cubelist

    def get_cube_names(self):
        """Return a list of names of the cubes comprising the dataset."""
        return [c.name() for c in self.cubelist]

    ### Private methods only below this point ###

    def _handle_doc(self, doc):
        """Process the document element."""
        self.logger.debug("Document element name: %s",
            doc.documentElement.tagName)

        # Process the <readMetadata> or <explicit> nodes, if present.
        self.explicit = doc.getElementsByTagName('explicit')

        # Create a list of items to remove during or after cube loading.
        remove_nodelist = doc.getElementsByTagName('remove')
        if len(remove_nodelist): self._make_remove_list(remove_nodelist)

        # Process any <dimension> nodes.
        dim_nodelist = doc.getElementsByTagName('dimension')
        if len(dim_nodelist): self._handle_dimensions(dim_nodelist)

        # Process any coordinate <variable> nodes - this needs to be done
        # in advance of processing any aggregation or data variable nodes.
        var_nodelist = doc.getElementsByTagName('variable')
        if len(var_nodelist): self._handle_coord_variables(var_nodelist)

        # Process any <aggregation> nodes.
        agg_nodelist = doc.getElementsByTagName('aggregation')
        if len(agg_nodelist): self._handle_aggregation(agg_nodelist[0])

        # Process any global <attribute> nodes.
        att_nodelist = [n for n in doc.getElementsByTagName('attribute')
            if n.parentNode.tagName == 'netcdf']
        if len(att_nodelist): self._handle_global_attributes(att_nodelist)

        # Process any data <variable> nodes.
        if len(var_nodelist): self._handle_data_variables(var_nodelist)

        # Process any items in the removals list.
        self._process_remove_list()

    def _handle_dimensions(self, dim_nodelist):
        """Process all dimension nodes"""
        self.logger.debug("Processing <dimension> elements...")
        for dim_node in dim_nodelist:
            self._handle_dimension(dim_node)

    def _handle_dimension(self, dim_node):
        """Process a single dimension node"""
        dim = NcmlElement(dim_node)
        if not dim.name:
            raise NcmlSyntaxError(
                "<dimension> elements must include a 'name' attribute.")
        dim.length = 0 if dim.length is None else int(dim.length)
        self.dim_nodes.append(dim)
        self.logger.debug("Added dimension named '%s' with length %d",
            dim.name, dim.length)

    def _handle_global_attributes(self, att_nodelist):
        """Process all attribute nodes"""
        self.logger.debug("Processing global attributes...")
        for att_node in att_nodelist:
            self._handle_attribute(att_node)

    def _handle_attribute(self, att_node, var_name=None):
        """Process a single attribute node"""
        att = NcmlElement(att_node)
        if not att.name:
            raise NcmlSyntaxError(
                "<attribute> elements must include a 'name' attribute.")

        if not att.type: att.type = 'String'
        if not att.value: att.value = _get_node_text(att_node)
        if att.value:
            att_val = _parse_values_from_text(att.value, att.type,
                sep=att.separator)
            if not (isinstance(att_val, basestring) or len(att_val) > 1):
                att_val = att_val[0]
        else:
            att_val = None

        if var_name:
            # local attribute - override current attribute only for the cube
            # with the specified var name
            cubelist = [c for c in self.cubelist if c.var_name == var_name]
            self.logger.debug("Set attribute '%s' on variable '%s'", att.name,
                var_name)
        else:
            # global attribute - override current attribute for all cubes
            cubelist = self.cubelist
            self.logger.debug("Set attribute '%s' on all variables", att.name)

        # rename and/or update the attribute in one or all cubes
        for cube in cubelist:
            try:
                if att.orgName:
                    old_val = cube.attributes.pop(att.orgName, None)
                else:
                    old_val = None
                if att_val is not None:
                    cube.attributes[att.name] = att_val
                elif old_val is not None:
                    cube.attributes[att.name] = old_val
            except:
                self.logger.warn("Unable to set cube attribute with name '%s'",
                    att.name)
                break

    def _handle_data_variables(self, var_nodelist):
        """Process any data variable nodes"""
        self.logger.debug("Processing data <variable> elements...")
        for var_node in var_nodelist:
            self._handle_data_variable(var_node)

    def _handle_data_variable(self, var_node):
        """Process a data variable node"""
        var = NcmlElement(var_node)
        if self._is_coord_variable(var): return
        self.logger.debug("Variable name: '%s', type: '%s'", var.name, var.type)

        # Check to see if the variable contains a nested <values> element
        # (i.e. data).
        val_nodes = var_node.getElementsByTagName('values')
        has_values = len(val_nodes) == 1

        # If data values have been specified then assume that the current
        # <variable> element defines a data variable, possibly a scalar one,
        # which we can reify as a new iris Cube object.
        if has_values:
            cube = self._add_data_variable(var)
            if var.shape:
                self.logger.debug("Added cube for %d-D data variable '%s'",
                    len(cube.shape), var.name)
            else:
                self.logger.debug("Added cube for scalar variable '%s'",
                    var.name)

        # Otherwise assume that the current <variable> element is merely
        # modifying some aspect(s) of an existing cube loaded from a netcdf
        # file.
        else:
            # if necessary, rename the variable from orgName to var.name
            if var.orgName:
                cubes = self._get_cubes_by_var_name(var.orgName)
                if cubes:
                    for cube in cubes: cube.var_name = var.name
                    self.logger.debug("Renamed variable '%s' to '%s'",
                        var.orgName, var.name)

            # update any variable-scope attributes
            att_nodelist = var_node.getElementsByTagName('attribute')
            for att_node in att_nodelist:
                self._handle_attribute(att_node, var.name)

    def _handle_coord_variables(self, var_nodelist):
        """
        Process any coordinate variable nodes. Refer to the _is_coord_variable()
        method to see how coordinate variables are distinguished.
        """
        self.logger.debug("Processing coordinate <variable> elements...")
        for var_node in var_nodelist:
            var = NcmlElement(var_node)
            if self._is_coord_variable(var):
                dim_coord = self._add_coord_variable(var)
                self.logger.debug(
                    "Added coordinate variable '%s' of type: '%s', "
                    "length: %d, units: '%s'", var.name, var.type,
                    len(dim_coord.points), dim_coord.units)

    def _handle_aggregation(self, agg_node):
        """Process a single aggregation node"""
        agg_elem = NcmlElement(agg_node)
        agg_elem.coord_values = list()
        agg_dim_crd = self._find_agg_dim_coord(agg_elem.dimName)
        agg_elem.template_coord = agg_dim_crd
        agg_elem.nctype = agg_dim_crd.nctype if agg_dim_crd else 'double'

        self.logger.debug("Processing <aggregation> element of type '%s'...",
            agg_elem.type)

        if agg_elem.type == AGG_UNION:
            ncubes = self._handle_union(agg_elem)

        elif agg_elem.type == AGG_JOIN_NEW:
            ncubes = self._handle_joinnew(agg_elem)

        elif agg_elem.type == AGG_JOIN_EXISTING:
            ncubes = self._handle_joinexisting(agg_elem)

        else:
            errmsg = "Invalid <aggregation> element type: '%s'" % agg_elem.type
            raise NcmlSyntaxError(errmsg)

        self.logger.debug("Loaded %d cube%s from <aggregation> element",
            ncubes, 's'[ncubes==1:])

    def _handle_union(self, agg_elem):
        """
        Process a union aggregation node. As per the NcML schema specification,
        dimensions and coordinate variables must match exactly across all of the
        nominated files. The current implementation does not check that this
        holds true.
        """

        # Load data from any <netcdf> and/or <scan> child nodes.
        cubes_to_union = self._load_aggregation_data(agg_elem)
        ncubes = len(cubes_to_union)
        if ncubes:
            self._extend_cubelist(cubes_to_union, unique_names=True)

        return ncubes

    def _handle_joinexisting(self, agg_elem):
        """
        Process a joinExisting aggregation element. From the NcML spec::

          A JoinExisting dataset is constructed by transferring objects
          (dimensions, attributes, groups, and variables) from the nested
          datasets in the order the nested datasets are listed. All variables
          that use the aggregation dimension as their outer dimension are
          logically concatenated, in the order of the nested datasets.
          Variables that don't use the aggregation dimension are treated as in
          a Union dataset, i.e. skipped if one with that name already exists.

        """
        if not agg_elem.dimName:
            errmsg = "<aggregation> elements of type 'joinExisting' must " \
                "include a 'dimName' attribute."
            raise NcmlSyntaxError(errmsg)

        # Load data from any <netcdf> and/or <scan> child nodes.
        cubes_to_copy = self._load_aggregation_data(agg_elem)

        # If coordinate values were NOT supplied via coordValue attributes
        # attached to <netcdf> element(s) then check to see if they were
        # specified as part of a coordinate <variable> element.
        if not agg_elem.coord_values:
            agg_dim_crd = self._find_agg_dim_coord(agg_elem.dimName)
            if agg_dim_crd and len(agg_dim_crd.points) > 0:
                agg_elem.coord_values = agg_dim_crd.points

        # Report distinct variable names from the raw loaded cubelist.
        uniq_var_names = sorted(set([c.name() for c in cubes_to_copy]))
        self.logger.debug("Number of cubes loaded: %d", len(cubes_to_copy))
        self.logger.debug("Distinct variable names: %s", ' '.join(uniq_var_names))

        # Extract those cubes which have dimName as the outer dimension.
        # Try to concatenate the resultant cubelist.
        cubes_to_join = _extract_cubes_by_dim_name(cubes_to_copy,
            agg_elem.dimName)
        self.logger.debug("Number of cubes to join: %d", len(cubes_to_join))
        joined_cubes = _join_cubes(cubes_to_join, agg_elem)

        ncubes = len(joined_cubes)
        if ncubes:
            self._extend_cubelist(joined_cubes, unique_names=False)

        if len(cubes_to_copy):
            ncubes += len(cubes_to_copy)
            self._extend_cubelist(cubes_to_copy, unique_names=False)

        return ncubes

    def _handle_joinnew(self, agg_elem):
        """
        Process a joinNew aggregation node. From the NcML spec::

          A JoinNew dataset is constructed by transferring objects (dimensions,
          attributes, groups, and variables) from the nested datasets in the
          order the nested datasets are listed. All variables that are listed
          as aggregation variables are logically concatenated along the new
          dimension, and in the order of the nested datasets. A coordinate
          variable is created for the new dimension. Non-aggregation variables
          are treated as in a Union dataset, i.e. skipped if one of that name
          already exists.

        """
        if not agg_elem.dimName:
            errmsg = "<aggregation> elements of type 'joinNew' must include " \
                "a 'dimName' attribute."
            raise NcmlSyntaxError(errmsg)

        # Get the list of aggregation variables.
        agg_var_nodes = agg_elem.node.getElementsByTagName('variableAgg')
        if not len(agg_var_nodes):
            errmsg = "<aggregation> elements of type 'joinNew' must contain " \
                "one or more <variableAgg> elements."
            raise NcmlSyntaxError(errmsg)
        agg_var_names = [node.getAttribute('name') for node in agg_var_nodes]

        # Load data from any <netcdf> and/or <scan> child nodes.
        cubes_to_copy = self._load_aggregation_data(agg_elem)

        # If coordinate values were NOT supplied via coordValue attributes
        # attached to <netcdf> element(s) then check to see if they were
        # specified as part of a coordinate <variable> element.
        if not agg_elem.coord_values:
            agg_dim_crd = self._find_agg_dim_coord(agg_elem.dimName)
            if agg_dim_crd and len(agg_dim_crd.points) > 0:
                agg_elem.coord_values = agg_dim_crd.points

        # For each named aggregation variable, create a new cube by merging the
        # existing cubes with that name along the new aggregation dimension. It
        # is assumed that the cubelist for each aggregation variable is returned
        # in the order of loading, and thus matches the length and order of the
        # aggregation dimension.

        joined_cubes = iris.cube.CubeList()
        for var_name in agg_var_names:
            self.logger.debug("Attempting to merge cubes for variable '%s'...",
                var_name)
            cubes_to_join = _extract_cubes_by_var_name(cubes_to_copy, var_name)
            if len(cubes_to_join) > 1:
                self.logger.debug("Found %d cubes", len(cubes_to_join))
                try:
                    cubes = _join_cubes(cubes_to_join, agg_elem)
                    if len(cubes): joined_cubes.extend(cubes)
                except:
                    print sys.exc_info()
                    self.logger.warn("Unable to merge cubes")
            elif len(cubes_to_join) == 1:
                self.logger.debug("Only found 1 cube - skipping")
                joined_cubes.extend(cubes_to_join)

        ncubes = len(joined_cubes)
        if ncubes:
            self._extend_cubelist(joined_cubes, unique_names=False)

        if len(cubes_to_copy):
            ncubes += len(cubes_to_copy)
            self._extend_cubelist(cubes_to_copy, unique_names=False)

        return ncubes

    def _handle_agg_netcdf_node(self, netcdf_node, agg_elem):
        """
        Process an aggregation netcdf node. We don't currently handle the
        optional 'ncoords' attribute since Iris's lazy data loading mechanism
        makes this redundant.
        """
        netcdf = NcmlElement(netcdf_node)
        if not netcdf.location:
            raise NcmlSyntaxError(
                "<netcdf> elements must include a 'location' attribute.")

        # Display warnings for any unsupported XML attributes.
        check_unsupported_attributes(netcdf, ['enhance', 'addRecords'])

        ncpath = netcdf.location
        if not ncpath.startswith('/'):
            ncpath = os.path.join(self.basedir, ncpath)

        # Load cubes from the specified netcdf file.
        self.logger.debug("Loading file '%s'...", ncpath)
        cubelist = iris.load(ncpath)

        # If coordinate value(s) were specified via a coordValue attribute,
        # record the values for later use during aggregation.
        if netcdf.coordValue:
            coords = _parse_values_from_text(netcdf.coordValue, agg_elem.nctype)
            agg_elem.coord_values.extend(coords)

        return cubelist

    def _handle_agg_scan_node(self, scan_node, agg_elem):
        """Process an aggregation scan node"""
        scan = NcmlElement(scan_node)
        if not scan.location:
            raise NcmlSyntaxError(
                "<scan> elements must include a 'location' attribute.")

        topdir = scan.location
        self.logger.debug("Scanning for files below directory %s", topdir)
        if not topdir.startswith('/'):
            topdir = os.path.join(self.basedir, topdir)
        if not scan.subdirs:
            scan.subdirs = 'true'
        do_subdirs = scan.subdirs.lower() in ('true', '1')
        min_age = scan.olderThan
        if min_age: min_age = _time_string_to_interval(min_age)

        # if no regex was defined, use a filename suffix if that was defined
        regex = scan.regExp
        if not regex and scan.suffix: regex = '.*' + scan.suffix + '$'

        # walk directory tree from topdir searching for files matching regex
        # load cubes from each matching netcdf file
        cubelist = iris.cube.CubeList()
        for ncpath in _walk_dir_tree(topdir, recurse=do_subdirs, regex=regex,
                min_age=min_age):
            self.logger.debug("Loading file '%s'...", os.path.basename(ncpath))
            cubelist.extend(iris.load(ncpath))

        return cubelist

    def _load_aggregation_data(self, agg_elem):
        """
        Load data from any <netcdf> and <scan> child elements that are contained
        within the aggregation element `agg_elem`.
        """

        # Process any <netcdf> nodes.
        loaded_cubes = iris.cube.CubeList()
        netcdf_nodelist = agg_elem.node.getElementsByTagName('netcdf')
        for node in netcdf_nodelist:
            cubes = self._handle_agg_netcdf_node(node, agg_elem)
            if cubes: loaded_cubes.extend(cubes)

        # Process any <scan> nodes.
        scan_nodelist = agg_elem.node.getElementsByTagName('scan')
        for node in scan_nodelist:
            cubes = self._handle_agg_scan_node(node, agg_elem)
            if cubes: loaded_cubes.extend(cubes)

        return loaded_cubes

    def _make_remove_list(self, remove_nodelist):
        """
        Make a list of objects flagged for removal. At present only variables
        and attributes can be removed.
        """
        if self.explicit:
            self.logger.warn("<remove> elements are not permitted when the "
                "<explicit> element is present")
            return

        for node in remove_nodelist:
            remove = NcmlElement(node)
            if not (remove.name and remove.type):
                self.logger.warn("<remove> elements must include a 'name' and "
                    "'type' attribute.")
                continue
            if remove.type not in ('attribute', 'variable'):
                self.logger.warn("Can only remove elements of type attribute "
                    "or variable")
                continue
            remove.parent_type = node.parentNode.tagName
            if remove.parent_type == 'netcdf':
                remove.parent_name = 'netcdf'
            else:
                remove.parent_name = node.parentNode.getAttribute('name')
            self.removelist.append(remove)

    def _process_remove_list(self):
        """Process items in the remove list."""
        for item in self.removelist:
            if item.type == 'variable':
                # variables should not have made it into the removelist
                continue

            elif item.type == 'attribute':
                att_name = item.name

                # global attribute
                if item.parent_type == 'netcdf':
                    for cube in self.cubelist:
                        cube.attributes.pop(att_name, None)
                    self.logger.debug("Removed attribute '%s' from all cubes",
                        att_name)

                # variable-scope attribute
                elif item.parent_type == 'variable':
                    var_name = item.parent_name
                    cubes = self._get_cubes_by_var_name(var_name)
                    if cubes:
                        for cube in cubes: cube.attributes.pop(att_name, None)
                        self.logger.debug(
                            "Removed attribute '%s' from cube %s",
                            att_name, var_name)
                    else:
                        self.logger.warn(
                            "No cube found corresponding to variable '%s'",
                            var_name)

    def _is_flagged_for_removal(self, node_or_cube):
        """
        Check to see if an NcML element or a cube is flagged for removal via a
        <remove> element.
        """
        if isinstance(node_or_cube, iris.cube.Cube):
            obj_name = node_or_cube.var_name
            obj_type = 'variable'
            parent_type = 'netcdf'
            parent_name = 'netcdf'
        else:
            obj_name = node_or_cube.getAttribute('name')
            obj_type = node_or_cube.getAttribute('type')
            if not (obj_name and obj_type): return False
            parent_type = node_or_cube.parentNode.tagName
            if parent_type == 'netcdf':
                parent_name = 'netcdf'
            else:
                parent_name = node_or_cube.parentNode.getAttribute('name')

        testobj = dict(name=obj_name, type=obj_type, parent_name=parent_name,
            parent_type=parent_type)

        return testobj in self.removelist

    def _is_coord_variable(self, var_elem):
        """
        Return True if var_elem represents a coordinate variable element.
        A coordinate variable is recognised by having its name attribute
        identical to its shape attribute, and by having its name equal to
        the name of a previously defined dimension element, e.g.::

          <dimension name='time' length='30'/>
          <variable name='time' type='int' shape='time'> ... </variable>
        """
        if not var_elem.name:
            raise NcmlSyntaxError(
                "<variable> elements must include a 'name' attribute.")

        if not var_elem.type:
            errmsg = "<variable> element named '%s' does not contain the " \
                "'type' attribute." % var_elem.name
            raise NcmlSyntaxError(errmsg)

        dim_names = [dim.name for dim in self.dim_nodes]

        return (var_elem.name == var_elem.shape and
                var_elem.name in dim_names)

    def _is_data_variable(self, var_elem):
        """
        Return True if var_elem represents a data variable element. A data
        variable is recognised by having its name attribute NOT equal to its
        shape attribute, and by having its name NOT equal to any previously
        defined dimension element, e.g.::

          <variable name='tas' type='float' shape='mydim'> ... </variable>

        Note that the shape attribute is not obligatory for data variables
        (in which case they are treated as scalar variables).
        """
        if not var_elem.name:
            raise NcmlSyntaxError(
                "<variable> elements must include a 'name' attribute.")

        if not var_elem.type:
            errmsg = "<variable> element named '%s' does not contain the " \
                " mandatory 'type' attribute." % var_elem.name
            raise NcmlSyntaxError(errmsg)

        dim_names = [dim.name for dim in self.dim_nodes]

        return (var_elem.name != var_elem.shape and
                var_elem.name not in dim_names)

    def _add_coord_variable(self, var_elem):
        """
        Create a coordinate variable (an iris DimCoord object) from a
        <variable> element.
        """
        nodes = var_elem.node.getElementsByTagName('values')
        if nodes:
            # Find the length of the variable in case the 'npts' attribute is
            # not defined in the <values> element.
            dimlen = None
            for dim in self.dim_nodes:
                if var_elem.shape == dim.name: dimlen = dim.length
            val_node = nodes[0]
            points = _parse_values_node(val_node, var_elem.type, npts=dimlen)

        # No <values> element present: assume that coordinates will be obtained
        # later on from coordValue attributes on <netcdf> elements nested in a
        # join-type aggregation.
        else:
            points = np.array([], dtype=NCML_TO_NUMPY_TYPE_MAP[var_elem.type])
            self.logger.debug(
                "No <values> defined for coordinate variable '%s'",
                var_elem.name)

        # Set any keyword arguments specified by nested <attribute> elements.
        kw = dict(standard_name=None, long_name=None, units='1')
        extra_atts = dict()
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist:
            name, value = _parse_attribute(node)
            if name in kw:
                kw[name] = value
            else:
                extra_atts[name] = value

        # Special handling of calendar attribute: use it to create CF-style
        # time unit object.
        cal = extra_atts.pop('calendar', None)
        if cal: kw['units'] = iris.unit.Unit(kw['units'], calendar=cal)

        # Create and store the DimCoord object for subsequent use.
        dim_coord = DimCoord(points, **kw)
        dim_coord.var_name = var_elem.name
        dim_coord.nctype = var_elem.type
        #if len(dim_coord.points) > 1: dim_coord.guess_bounds()

        # Assign any extra (non-CF) attributes specified in the <variable> element.
        for name, value in extra_atts.items():
            try:
                dim_coord.attributes[name] = value
            except:
                self.logger.warn("Attribute named '%s' is not permitted on a "
                    "DimCoord object", name)

        self.dim_coords.append(dim_coord)
        return dim_coord

    def _add_data_variable(self, var_elem):
        """Create a data variable (an iris Cube) from a <variable> element."""
        nodes = var_elem.node.getElementsByTagName('values')
        if not nodes:
            raise NcmlSyntaxError("A <values> element is required to create a "
                "data variable")

        val_node = nodes[0]
        data = _parse_values_node(val_node, var_elem.type)

        # Set any keyword arguments from nested <attribute> elements.
        kw = dict(standard_name=None, long_name=None, units='1',
            cell_methods=None)
        extra_atts = dict()
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist:
            name, value = _parse_attribute(node)
            if name in kw:
                kw[name] = value
            else:
                extra_atts[name] = value

        # For a dimensioned (non-scalar) variable
        if var_elem.shape:
            shape = []
            coords = []
            # create a coordinate list for this variable
            # FIXME: this may be a somewhat brittle approach
            all_coords = self.dim_coords[:]
            all_coords.extend(_get_coord_list(self.cubelist))
            dimnum = 0
            # for each dimension named in the shape attribute, find the
            # corresponding coord object
            for i, dim_name in enumerate(var_elem.shape.split()):
                for dim_coord in all_coords:
                    if dim_name in [dim_coord.var_name, dim_coord.standard_name]:
                        coords.append([dim_coord, dimnum])
                        shape.append(len(dim_coord.points))
                        dimnum += 1
                        break

        # For a scalar variable we just need to set the array shape.
        else:
            shape = [1]
            coords = []

        # Create a cube object and append to the dataset's cubelist.
        data.shape = shape
        cube = iris.cube.Cube(data, dim_coords_and_dims=coords, **kw)
        cube.var_name = var_elem.name
        cube.nctype = var_elem.type

        # Assign any extra (non-CF) attributes specified in the <variable>
        # element.
        for name, value in extra_atts.items():
            try:
                cube.attributes[name] = value
            except:
                self.logger.warn(
                    "Attribute named '%s' is not permitted on a cube", name)

        self.cubelist.append(cube)
        self.logger.debug("  cube shape: %s", str(data.shape))
        self.logger.debug("  cube min value: %s", cube.data.min())
        self.logger.debug("  cube max value: %s", cube.data.max())

        return cube

    def _extend_cubelist(self, cubelist, unique_names=False):
        """
        Append cubes from cubelist to this NcMLDataset's current cubelist.
        If the unique_names argument is set to true then only append cubes
        with distinct names, which is the behaviour required by union-type
        aggregations.
        """
        for cube in cubelist:
            # skip variable if it's flagged for removal
            if self._is_flagged_for_removal(cube):
                self.logger.debug("Skipped netCDF variable '%s' "
                    "(reason: target of <remove> element)", cube.var_name)

            # skip variable if it's already present
            elif cube.name() in self.get_cube_names() and unique_names:
                self.logger.debug("Skipped netCDF variable '%s' "
                    "(reason: loaded from earlier file)", cube.var_name)

            else:
                self.cubelist.append(cube)
                self.logger.debug("Added cube for variable '%s'", cube.name())

    def _find_agg_dim_coord(self, dim_name):
        """
        Find the aggregation dimension coordinate object associated with name
        dim_name.
        """
        for crd in self.dim_coords:
            if crd.var_name == dim_name: return crd
        return None

    def _update_cube_attributes(self, attrs):
        """
        Override any cube attributes with those specified in the `attrs`
        argument.
        """
        for cube in self.get_cubes():
            if not hasattr(cube, 'attributes'): cube.attributes = dict()
            cube.attributes.update(attrs)

    def _get_cubes_by_var_name(self, var_name):
        """Return the cube or cubes having the specified variable name."""
        return [cube for cube in self.cubelist if cube.var_name == var_name]


# TODO: the following two functions could be merged into one.
def _extract_cubes_by_var_name(cubelist, var_name):
    """
    Extract from `cubelist` those cubes whose variable name matches dim_name.
    """
    cubes = iris.cube.CubeList()
    removelist = []

    for i, cube in enumerate(cubelist):
        if cube.var_name == var_name:
            cubes.append(cube)
            removelist.append(i)

    if removelist:
        for i in removelist[::-1]: cubelist.pop(i)

    return cubes


def _extract_cubes_by_dim_name(cubelist, dim_name):
    """
    Extract from `cubelist` those cubes whose outer dimension name matches
    dim_name.
    """
    cubes = iris.cube.CubeList()
    removelist = []

    for i, cube in enumerate(cubelist):
        try:
            coord0 = cube.dim_coords[0]
            if coord0.name() == dim_name:
                cubes.append(cube)
                removelist.append(i)
        except:
            pass

    if removelist:
        for i in removelist[::-1]: cubelist.pop(i)

    return cubes


def _join_cubes(cubes_to_join, agg_elem):
    """
    Apply an Iris merge or concatenation operation to the specified cubelist.
    """

    joined_cubes = iris.cube.CubeList()
    ncubes = len(cubes_to_join)
    if ncubes < 2: return joined_cubes

    # Remove any cube attributes which might impede the join operation.
    _apply_prejoin_cube_fixes(cubes_to_join)

    # If the timeUnitsChange attribute was specified, then rebase to a common
    # time datum all the coordinate objects that match the nominated aggregation
    # dimension. We have to assume that this attribute is only included if the
    # agg. dimension is indeed a temporal one.
    if agg_elem.timeUnitsChange == "true":
        # iris.util.unify_time_units(cubes_to_join)
        time_coords = _get_coord_list(cubes_to_join, dim_name=agg_elem.dimName)
        if time_coords: _rebase_time_coords(time_coords)

    # For joinExisting aggregations try to concatenate the list of cubes.
    if agg_elem.type == AGG_JOIN_EXISTING:
        logger.debug("Applying cube concatenation operation...")
        _update_agg_dim_coords(cubes_to_join, agg_elem)
        try:
            joined_cubes = cubes_to_join.concatenate()
            logger.debug("Concatenated %d cubes into %d", ncubes,
                len(joined_cubes))
        except ConcatenateError, exc:
            logger.debug("Cube concatenation error:\n%s", str(exc))

    # For joinNew aggregations try to merge the list of cubes.
    elif agg_elem.type == AGG_JOIN_NEW:
        logger.debug("Applying cube merge operation...")
        _assign_new_agg_coords(cubes_to_join, agg_elem)
        try:
            cube = cubes_to_join.merge_cube()
            joined_cubes = iris.cube.CubeList([cube])
            logger.debug("Merged %d cubes into %d", ncubes, len(joined_cubes))
        except MergeError, exc:
            logger.debug("Cube merge error:\n%s", str(exc))

    return joined_cubes


def _apply_prejoin_cube_fixes(cubelist):
    """
    Apply modifications to a list of cubes prior to performing a concatenation
    or merge operation.
    """
    # Remove any cube attributes which might impede the join operation.
    attnames_to_remove = ['history', 'title']
    for cube in cubelist:
        for attname in attnames_to_remove:
            cube.attributes.pop(attname, None)


def _update_agg_dim_coords(cubes_to_join, agg_elem):
    """
    For each cube in the `cubes_to_join` cubelist, overwrite the values in the
    aggregation dimension using the correct value (or values - there may be
    more than one) from the `agg_elem.coord_values` array. It is assumed that
    the order of cubes in `cubes_to_join` matches the order of coordinates in
    the `agg_elem.coord_values` array.
    """
    agg_dim_len = len(agg_elem.coord_values)
    if not agg_dim_len: return   # no coordinates specified in ncml file

    start = stop = 0
    for cube in cubes_to_join:
        dimcrd = cube.coord(agg_elem.dimName)
        stop = start + len(dimcrd.points)
        if stop > agg_dim_len:
            raise NcmlContentError("Insufficient coordinates defined to assign "
                "to aggregation dimension %s", agg_elem.dimName)
        dimcrd.points = np.array(agg_elem.coord_values[start:stop])

        # Remove coordinate bounds, if present, since they will likely not match
        # the new coordinate values generated for the aggregated dataset.
        if dimcrd.has_bounds():
            dimcrd.bounds = None

        # Update coordinate metadata if a template coordinate is provided.
        if agg_elem.template_coord:
            for attname in ('standard_name', 'long_name', 'units'):
                attval = getattr(agg_elem.template_coord, attname, None)
                if attval: setattr(dimcrd, attname, attval)
            dimcrd.attributes.update(agg_elem.template_coord.attributes)

        # Move start index forward ready for next chunk of coordinates.
        start = stop

    if stop != agg_dim_len:
        logger.warn("Mismatch between number of coordinate values specified "
            "for the aggregation dimension (%d)\nand the aggregate length of "
            "the corresponding cubes (%d)", agg_elem.dimName, stop)


def _assign_new_agg_coords(cubes_to_join, agg_elem):
    """
    For each cube in the `cubes_to_join` cubelist, create a new auxiliary coord
    object representing the aggregation dimension, and assign it the correct
    value from the `agg_elem.coord_values` array. It is assumed that the order
    of cubes in `cubes_to_join` matches the order of coordinates in the
    `agg_elem.coord_values` array.
    """
    agg_dim_len = len(agg_elem.coord_values)
    if agg_dim_len == 0:
        logger.warn("Unable to add auxiliary coordinates for dimension %s\n"
            "No coordinate values defined.", agg_elem.dimName)
        return
    elif agg_dim_len != len(cubes_to_join):
        logger.warn("Length of aggregation dimension (%d) does not match"
            " number of cubes to aggregate over (%d)",
            agg_dim_len, len(cubes_to_join))
        return

    for i, cube in enumerate(cubes_to_join):
        coords = agg_elem.coord_values[i:i+1]
        aux_coord = _create_aux_coord(agg_elem.dimName, agg_elem.nctype,
            coords, agg_elem.template_coord)
        cube.add_aux_coord(aux_coord, None)
        logger.debug("Created auxiliary coordinate '%s' with value %s",
            agg_elem.dimName, coords)


def _create_aux_coord(dim_name, nctype, coord_values, template_coord=None):
    """
    Create an iris AuxCoord object from the specified arguments. If a template
    coordinate object is passed in then it is used as the source of additional
    attributes to assign to the new aux coord object.
    """
    try:
        #npoints = len(coord_values)
        points = np.array(coord_values)
        aux_coord = AuxCoord(points, long_name=dim_name, var_name=dim_name)
        aux_coord.nctype = nctype
        #if npoints > 1: aux_coord.guess_bounds()
        if template_coord:
            for attname in ('standard_name', 'long_name', 'units'):
                attval = getattr(template_coord, attname, None)
                if attval: setattr(aux_coord, attname, attval)
            aux_coord.attributes.update(template_coord.attributes)
        return aux_coord

    except:
        logger.error("Error trying to create auxiliary coordinate for dimension %s",
            dim_name)
        raise


def _get_coord_list(cubelist, dim_name=None):
    """
    Return a list of coordinate objects (DimCoords, AuxCoords) associated with
    the specified cubelist. The dim_name argument may be used to select only
    those coordinates with the specified name.
    """
    coord_list = []
    for cube in cubelist:
        for coord in cube.coords(name_or_coord=dim_name):
            if coord not in coord_list: coord_list.append(coord)
    return coord_list


def _get_node_att_dict(node, att_namelist):
    """
    Return a dictionary of the attributes named in att_namelist for the
    specified node.
    """
    att_dict = dict()
    for name in att_namelist:
        value = node.getAttribute(name)
        if value == '': value = None
        att_dict[name] = value
    return att_dict


def _get_node_text(node):
    """Retrieve the character data from an XML node."""
    text = []
    for child in node.childNodes:
        if child.nodeType == child.TEXT_NODE:
            text.append(child.data)
    text = ''.join(text)
    if re.match(r'\s+$', text):   # test for all whitespace
        return ''
    else:
        return text


def _parse_values_node(val_node, nctype, npts=None):
    """Construct an array of points as read from a <values> NcML element."""

    values = NcmlElement(val_node)

    # Read coordinate values from the child text node if defined there.
    if values.start is None:
        text_values = _get_node_text(val_node)
        points = _parse_values_from_text(text_values, nctype,
            sep=values.separator)
        if not len(points):
            raise NcmlContentError(
                "No values defined within a <values> element")
        points = np.array(points, dtype=NCML_TO_NUMPY_TYPE_MAP[nctype])

    # Compute coordinate values from the start, increment and, optionally,
    # npts attributes defined in the <values> element.
    else:
        if values.npts:
            npts = int(values.npts)
        elif not npts:
            raise NcmlContentError("Unable to determine the number of values "
                "defined within a <values> element")

        try:
            start = int(values.start)
            inc = int(values.increment)
        except:
            start = float(values.start)
            inc = float(values.increment)

        points = np.arange(start, start+inc*(npts-0.5), inc,
            dtype=NCML_TO_NUMPY_TYPE_MAP[nctype])

    return points


def _parse_attribute(att_node):
    """
    Parse the name and value (suitably type-converted) as specified in an NcML
    <attribute> node.
    """
    name = att_node.getAttribute('name')
    nctype = att_node.getAttribute('type') or 'String'
    txtval = att_node.getAttribute('value')
    sep = att_node.getAttribute('separator') or None
    value = _parse_values_from_text(txtval, nctype, sep)
    return (name, value)


def _parse_values_from_text(text, nctype='String', sep=None):
    """
    Parse the specified text string into a (possibly length-one) array of values
    of type nctype, being one of the recognised NcML/NetCDF data types ('byte',
    'int', etc). As per the NcML default, values in lists are separated by
    arbitrary amounts of whitespace. An alternative separator may be set using
    the sep argument, in which case values are separated by single occurrences
    of the specified separator.

    NOTE: there is no support at present for the 'Structure' data type described
    in the NcML spec.
    """
    if nctype in ('char', 'string', 'String'):
        return text
    elif nctype == 'byte':
        values = [np.int8(x) for x in text.split(sep)]
    elif nctype == 'short':
        values = [np.int16(x) for x in text.split(sep)]
    elif nctype == 'int' or nctype == 'long':
        values = [np.int32(x) for x in text.split(sep)]
    elif nctype == 'float':
        values = [np.float32(x) for x in text.split(sep)]
    elif nctype == 'double':
        values = [np.float64(x) for x in text.split(sep)]
    else:
        raise NcmlSyntaxError("Unsupported NcML attribute data type: %s" %
            nctype)
    return values


def _rebase_time_coords(coord_list, target_unit=None):
    """
    Rebase a list of CF-style time coordinate objects so that they all reference
    the same time datum. The new time datum may be specified via the target_unit
    argument, which should be a string of the form 'time-units since time-datum'
    or a Unit object which provides the same info. In the latter case, the
    calendar property must match the same property in each coord object. If the
    target_unit argument is not specified then it is set equal to the earliest
    datum in the list of input coordinate objects. All of the input coordinate
    objects must use the same base time units, e.g. seconds, days, hours. If a
    coordinate object contains bounds then those values are also rebased.
    """
    coord0 = coord_list[0]
    base_unit, base_datum = map(str.strip, coord0.units.origin.split('since'))
    base_cal = coord0.units.calendar
    if target_unit:
        if isinstance(target_unit, basestring):
            target_unit = iris.unit.Unit(target_unit, calendar=base_cal)
        elif isinstance(target_unit, iris.unit.Unit):
            if base_cal != target_unit.calendar:
                raise ValueError("Source calendar (%s) and target calendar (%s)"
                    " do not match" % (base_cal, target_unit.calendar))
        else:
            raise ValueError("target_unit argument must be of type string or "
                "iris.unit.Unit")
    else:
        # Find the earliest time datum in the input coordinate list
        min_datum = _get_earliest_time_datum(coord_list)
        target_unit = iris.unit.Unit(base_unit+' since '+min_datum,
            calendar=base_cal)

    # Convert all time coordinates (and bounds, if present) to the new target
    # unit.
    for crd in coord_list:
        if not crd.units.is_convertible(target_unit):
            logger.warn("Cannot convert unit '%s' to '%s'", crd.units,
                target_unit)
        else:
            crd.points = _convert_time_coords(crd.points, crd.units,
                target_unit)
            if crd.has_bounds():
                crd.bounds = _convert_time_coords(crd.bounds, crd.units,
                    target_unit)
            crd.units = target_unit

    logger.debug("Rebased time coordinates to '%s'", target_unit)


def _convert_time_coords(coords, src_units, dest_units):
    """Convert an array of time coordinates from src_units to dest_units."""
    # In Iris, coordinate arrays are immutable, therefore we have to do
    # the conversion on a copy, which gets returned.
    points = coords.copy()
    for idx, x in np.ndenumerate(points):
        points[idx] = dest_units.date2num(src_units.num2date(x))
    return points


def _get_earliest_time_datum(coord_list):
    """
    Return the earliest time datum used by a collection of time coordinate
    objects.
    """
    ref_origin = coord_list[0].units.origin
    ref_cal = coord_list[0].units.calendar
    min_offset = 0
    min_datum = ref_origin.split('since')[1]
    for crd in coord_list[1:]:
        crd_date = iris.unit.num2date(0, crd.units.origin, crd.units.calendar)
        crd_offset = iris.unit.date2num(crd_date, ref_origin, ref_cal)
        if crd_offset < min_offset:
            min_offset = crd_offset
            min_datum = crd.units.origin.split('since')[1]
    return min_datum.strip()


def _time_string_to_interval(timestr):
    """
    Convert an NcML time period defined as a string to an interval in whole
    seconds. The main use for this is to convert the value specified in the
    olderThan attribute to a useable interval. The time string must be in the
    format "value timeunit", e.g. "5 min", "1 hour", and so on.
    """
    try:
        ival, iunit = timestr.split()
        iunit = iris.unit.Unit(iunit)
        sunit = iris.unit.Unit('second')
        interval = iunit.convert(float(ival), sunit)
        return long(interval)
    except:
        raise ValueError("Error trying to decode time period: %s" % timestr)


def _history_timestamp():
    """
    Return a timestamp suitable for use at the beginning of a history attribute.
    """
    now = datetime.now().replace(microsecond=0)
    return now.isoformat()


def _is_older_than(filename, interval):
    """
    Return true if the latest modification time of filename is older than the
    specified interval (in seconds) before the current time.
    """
    # get modification time of filename
    mtime = datetime.fromtimestamp(os.path.getmtime(filename))

    # see if file modification time is older than interval
    age = datetime.now() - mtime
    return (age.total_seconds() > interval)


def _walk_dir_tree(topdir, recurse=True, regex=None, min_age=None, sort=False):
    """
    Walk the directory tree rooted at topdir, yielding filenames which match the
    optional regular expression pattern specified in the regex argument, and
    which are older than min_age, if specified.
    """
    if regex: reobj = re.compile(regex)
    for curr_path, subdirs, filenames in os.walk(topdir):
        if sort: filenames.sort()
        for fn in filenames:
            curr_file = os.path.join(curr_path, fn)
            if (regex and not reobj.match(fn)) or \
               (min_age and not _is_older_than(curr_file, min_age)):
                continue
            yield curr_file
        if not recurse: break


def check_unsupported_attributes(element, att_list):
    """Display a warning message for any unsupported XML attributes."""
    for att_name in att_list:
        if getattr(element, att_name) is None: continue
        msg = "The '{0}' attribute on a <{1}> element is not " \
            "currently supported".format(att_name, element.elem_type)
        logger.warn(msg)


def _init_logger(level=DEFAULT_LOG_LEVEL, fmt=DEFAULT_LOG_FORMAT):
    """Initialise a logger object using default or user-supplied settings."""
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    fmtr = logging.Formatter(fmt)
    console.setFormatter(fmtr)
    lggr = logging.getLogger(DEFAULT_LOG_NAME)
    lggr.addHandler(console)
    lggr.setLevel(level)
    return lggr


def _update_logger(level=None, fmt=None):
    """Update logging level and/or format properties."""
    lggr = logging.getLogger(DEFAULT_LOG_NAME)
    if level: lggr.setLevel(level)
    for handler in lggr.handlers:
        if level: handler.setLevel(level)
        if fmt: handler.setFormatter(logging.Formatter(fmt))


# Initialise a logger object.
logger = _init_logger()
