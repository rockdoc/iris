"""
WARNING: THIS CODE IS EXPERIMENTAL. IT DOESN'T WORK FULLY YET AND IS LIABLE TO CHANGE.

Iris i/o module for loading NcML v2.2 files. Current functionality is limited to the following:

* aggregations of netcdf files only
* union-type aggregation
* joinExisting-type aggregation (only for simple 1D aggregation dimensions)
* joinNew-type aggregation (only for simple 1D aggregation dimensions)
* rename or remove data variables (= cubes)
* add, override, rename or remove attributes (global or per-variable)
* reification of 1D coordinate variables as new DimCoord objects
* handle timeUnitsChange attribute for joinExisting aggregations

Functionality that is excluded (and is unlikely to be implemented):

* modification/removal of dimensions (this would interfere with Iris' handling of dimensions)
* NcML group elements
* forecastModelRun* elements
* the Structure data type
* nested variables
* nested aggregations

TO DO:

* handle dateFormatMark attribute for join-type aggregations
* support loading of non-netcdf data formats?
* add sphinx markup throughout
"""
import sys, os, re, fnmatch, logging
from datetime import datetime
import iris
from iris.coords import AuxCoord, DimCoord
import numpy as np
import netCDF4 as nc4
from xml.dom.minidom import parse

# default logging options
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_FORMAT = "[%(name)s] %(levelname)s: %(message)s"

# Define lists of permissible attributes for various NcML 2.2 tags
att_namelists = {
    'aggregation': ('type', 'dimName', 'recheckEvery', 'timeUnitsChange'),
    'attribute': ('name', 'type', 'separator', 'value', 'orgName'),
    'dimension': ('name', 'length', 'isUnlimited', 'isVariableLength', 'isShared', 'orgName'),
    'netcdf': ('location', 'id', 'title', 'enhance', 'addRecords', 'ncoords', 'coordValue'),
    'remove': ('name', 'type'),
    'scan': ('location', 'suffix', 'regExp', 'subdirs', 'olderThan', 'dateFormatMark', 'enhance'),
    'values': ('start', 'increment', 'npts', 'separator'),
    'variable': ('name', 'type', 'shape', 'orgName'),
    'variableAgg': ('name',),
}

class NcmlSyntaxError(iris.exceptions.IrisError) :
    """Exception class for NcML syntax errors."""
    pass

class NcmlContentError(iris.exceptions.IrisError) :
    """Exception class for NcML content errors."""
    pass

class NcmlElement(object) :
    """
    A lightweight class for creating an object representation of an NcML element with equivalent
    attributes. Avoids the need to create discrete classes for each NcML element type, which would
    be a viable alternative approach."""
    def __init__(self, node, **kwargs) :
        self.node = node
        for att_name in att_namelists.get(node.tagName,[]) :
            att_value = node.getAttribute(att_name)
            if att_value == '' : att_value = None
            setattr(self, att_name, att_value)
        if kwargs : self.__dict__.update(kwargs)

    @property
    def elem_type(self) :
        """Return the element type name."""
        return self.node.tagName

# Might want to rename this to Ncml2Dataset since it's tied to v2.2 of the NcML schema.
class NcmlDataset(object) :
    """
    Class for representing an aggregate dataset defined in an NcML v2.2 document. Full details of
    the NcML specification can be found at http://www.unidata.ucar.edu/software/netcdf/ncml/
    
    NcML documents are parsed using a DOM XML parser as (i) this is easier to work with compared
    with SAX parsers, and (ii) most NcML documents are likely to be small enough to fit into RAM.  
    """

    def __init__(self, file_like, **kwargs) :
        """
        Initialise an NcML dataset instance from a pathname or file-like object.

        :param file_like: The pathname or file object containing the NcML document
        :param log_level: Optional keyword for setting the logging level. Set to one of the levels
            defined by the standard logging module.
        """
        self.filename = None
        self.dimensions = list()
        self.dim_coords = list()
        self.aux_coords = list()
        self.part_filenames = list()
        self.cubelist = iris.cube.CubeList()
        self.removelist = list()
        self.explicit = False
        self.logger = logger

        # Check to see whether caller supplied a filename or file-like object.
        close_after_parse = False
        if isinstance(file_like, basestring) :
            self.filename = os.path.expandvars(os.path.expanduser(file_like))
            ncml_file = open(self.filename)
            close_after_parse = True
        elif isinstance(file_like, file) :
            ncml_file = file_like
        else :
            raise AttributeError("Invalid file-like argument passed to NcmlDataset constructor.")
            
        # Parse the NcML file
        try :
            self.logger.info("Parsing NcML source document...")
            try :
                ncml_doc = parse(ncml_file)
            except Exception, exc :
                self.logger.error("Error trying to parse NcML document.")
                raise exc
            self._handle_doc(ncml_doc)
            self.logger.info("Finished parsing. Loaded %d cube%s from %d files" % \
                (self.ncubes, 's'[self.ncubes==1:], self.nparts))
        finally :
            if close_after_parse : ncml_file.close()

    def __del__(self) :
        """Perform any clean-up operations, e.g. closing open file handles."""
        # TODO
        pass

    @property
    def nparts(self) :
        """Returns the number of file-parts comprising the dataset.""" 
        return len(self.part_filenames)

    @property
    def ncubes(self) :
        """Returns the number of distinct cubes comprising the dataset.""" 
        return len(self.cubelist)

    @property
    def basedir(self) :
        """Returns the base directory of the input NcML file."""
        return os.path.dirname(self.filename)

    def get_cubes(self) :
        """Return a list of the distinct cubes referenced by the NcML file."""
        return self.cubelist

    def get_cube_names(self) :
        """Return a list of names of the distinct cubes referenced by the NcML file."""
        return [c.name() for c in self.cubelist]

    ### Private methods only below this point ###
    
    def _handle_doc(self, doc) :
        self.logger.info("Document element name: "+doc.documentElement.tagName)

        # process <readMetadata> or <explicit> node, if present
        explicit = doc.getElementsByTagName('explicit')
        if explicit : self.explicit == True

        # create a list of items to remove during or after cube loading
        remove_nodelist= doc.getElementsByTagName('remove')
        if len(remove_nodelist) : self._make_remove_list(remove_nodelist)
        
        # process any <dimension> nodes
        dim_nodelist = doc.getElementsByTagName('dimension')
        if len(dim_nodelist) : self._handle_dimensions(dim_nodelist)

        # process any coordinate <variable> nodes - these need to be created in advance of any
        # aggregation or data variable nodes
        var_nodelist = doc.getElementsByTagName('variable')
        if len(var_nodelist) : self._handle_coord_variables(var_nodelist)

        # process an <aggregation> node, if present
        agg_nodelist= doc.getElementsByTagName('aggregation')
        if len(agg_nodelist) : self._handle_aggregation(agg_nodelist[0])

        # process any global <attribute> nodes
        att_nodelist = [n for n in doc.getElementsByTagName('attribute') if n.parentNode.tagName == 'netcdf']
        if len(att_nodelist) : self._handle_global_attributes(att_nodelist)

        # process any data <variable> nodes
        if len(var_nodelist) : self._handle_data_variables(var_nodelist)

        # process any items in the remove list
        self._process_remove_list()
        
    def _handle_dimensions(self, dim_nodelist) :
        """Process all dimension nodes"""
        self.logger.info("Processing dimension elements...")
        for dim_node in dim_nodelist :
            self._handle_dimension(dim_node)

    def _handle_dimension(self, dim_node) :
        """Process a single dimension node"""
        dim = NcmlElement(dim_node)
        if not dim.name :
            raise NcmlSyntaxError("<dimension> elements must include a 'name' attribute.")
        dim.length = 0 if dim.length is None else int(dim.length)
        self.dimensions.append(dim)
        self.logger.info("Added dimension %s with length %d" % (dim.name, dim.length))

    def _handle_global_attributes(self, att_nodelist) :
        """Process all attribute nodes"""
        self.logger.info("Processing global attributes...")
        for att_node in att_nodelist :
            self._handle_attribute(att_node)
    
    def _handle_attribute(self, att_node, var_name=None) :
        """Process a single attribute node"""
        att = NcmlElement(att_node)
        if not att.name :
            raise NcmlSyntaxError("<attribute> elements must include a 'name' attribute.")

        if not att.type : att.type = 'String'
        if not att.value : att.value = _get_node_text(att_node)

        if var_name :
            # local - override current attribute only for the cube with the specified var name
            cubelist = [c for c in self.cubelist if c.var_name == var_name]
            self.logger.info("Set attribute %s on variable %s" % (att.name, var_name))
        else :
            # global - override current attribute for all cubes
            cubelist = self.cubelist
            self.logger.info("Set attribute %s on all variables" % att.name)

        # rename and/or update the attribute in one or all cubes
        for cube in cubelist :
            if att.name in cube.attributes._forbidden_keys :
                self.logger.warn("Unable to set cube attribute with reserved name '%s'" % att.name)
                return
            if att.orgName :
                old_val = cube.attributes.pop(att.orgName, None)
            att_val = _parse_text_string(att.value, att.type, sep=att.separator)
            if isinstance(att_val, basestring) or len(att_val) > 1 :
                cube.attributes[att.name] = att_val     # string or number array
            else :
                cube.attributes[att.name] = att_val[0]  # single number

    def _handle_data_variables(self, var_nodelist) :
        """Process any data variable nodes"""
        self.logger.info("Processing data variable elements...")
        for var_node in var_nodelist :
            self._handle_data_variable(var_node)

    def _handle_data_variable(self, var_node) :
        """Process a data variable node"""
        var = NcmlElement(var_node)
        if self._is_coord_variable(var) : return
        self.logger.debug("Variable element: name: '%s', type: '%s'" % (var.name, var.type))

        # Check to see if the variable contains a nested <values> element (i.e. data).
        val_nodes = var_node.getElementsByTagName('values')
        has_values = len(val_nodes) == 1
        
        # If data values have been specified then assume that the current <variable> element defines
        # a data variable, possibly a scalar one, which we can reify as a new iris Cube object.
        if has_values :
            cube = self._add_data_variable(var)
            if var.shape :
                self.logger.info("Added %dD data variable %s as new cube" % (len(cube.shape), var.name))
            else :
                self.logger.info("Added scalar variable %s as new cube" % var.name)

        # Otherwise assume that the current <variable> element is merely modifying some aspect(s)
        # of an existing cube loaded from a netcdf file. 
        else :
            # if necessary, rename the variable from orgName to var.name
            if var.orgName :
                cubes = self._get_cubes_by_var_name(var.orgName)
                if cubes :
                    for cube in cubes : cube.var_name = var.name
                    self.logger.info("Renamed variable %s to %s" % (var.orgName, var.name))
    
            # update any variable-scope attributes
            att_nodelist = var_node.getElementsByTagName('attribute')
            for att_node in att_nodelist :
                self._handle_attribute(att_node, var.name)
        
    def _handle_coord_variables(self, var_nodelist) :
        """
        Process any coordinate variable nodes. Refer to the _is_coord_variable() method to see how
        coordinate variables are distinguished.
        """
        self.logger.info("Processing coordinate variable elements...")
        for var_node in var_nodelist :
            var = NcmlElement(var_node)
            if self._is_coord_variable(var) :
                dim_coord = self._add_coord_variable(var)
                self.logger.info("Added coordinate variable %s of type %s, length %d" \
                    % (var.name, var.type, len(dim_coord.points)))

    def _handle_aggregation(self, agg_node) :
        """Process a single aggregation node"""
        agg = NcmlElement(agg_node)
        agg.uses_coord_value_att = False   # gets set to T if coordValue attributes are defined
        if agg.type == 'union' :
            self._handle_union(agg)
        elif agg.type == 'joinNew' :
            self._handle_joinnew(agg)
        elif agg.type == 'joinExisting' :
            self._handle_joinexisting(agg)
        else :
            errmsg = "Invalid type value '%s' specified in <aggregation> element." % agg.type
            raise NcmlSyntaxError(errmsg)

    def _handle_union(self, agg_elem) :
        """
        Process a union aggregation node. As per the NcML schema specification, dimensions and
        coordinate variables must match exactly across all of the nominated files. The current
        implementation does not check that this holds true.
        """
        self.logger.info("Processing union aggregation element...")

        # Process any <netcdf> nodes.
        nc_nodes = agg_elem.node.getElementsByTagName('netcdf')
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_agg_netcdf_node(node, agg_elem)

        # Process any <scan> nodes.
        scan_nodes = agg_elem.node.getElementsByTagName('scan')
        if len(scan_nodes) :
            for node in scan_nodes :
                self._handle_agg_scan_node(node, agg_elem)

        self.logger.info("Loaded %d cubes" % len(self.cubelist))

    def _handle_joinexisting(self, agg_elem) :
        """
        Process a joinExisting aggregation node. From the NcML spec:

            A JoinExisting dataset is constructed by transferring objects (dimensions, attributes,
            groups, and variables) from the nested datasets in the order the nested datasets are
            listed. All variables that use the aggregation dimension as their outer dimension are
            logically concatenated, in the order of the nested datasets. Variables that don't use
            the aggregation dimension are treated as in a Union dataset, i.e. skipped if one with
            that name already exists.

        Processing logic:
        
        case 1: no <variable> element used to redefine coordinates for aggregation dimension, so
                simply aggregate over existing dimension called dimName
            - build cubelist from netcdf files specified in <netcdf> and/or <scan> elements
            - identify cubes which have dimension dimName as their outer dimension, then...
            - foreach cubelist:
                - sort these cubes and their dimName DimCoord objects into asc/desc order
                - create a new aggregation DimCoord object from dimName DimCoord objects
                - create a new aggregation cube by concatenating the selected cubes
                - attach the new aggregation DimCoord and any other Coord objects to this new cube
                - discard original cubes
            - copy over any other unjoinable cubes as-is (simple union behaviour)
        
        case 2: a <variable> element is used to redefine coordinates for aggregation dimension
            - create a new aggregation DimCoord object from the <values> sub-element within <variable>
            - build cubelist from netcdf files specified in <netcdf> and/or <scan> elements
            - identify cubes which have dimension dimName as their outer dimension, then...
            - foreach cubelist:
                - sort these cubes and their dimName DimCoord objects into asc/desc order
                - create a new aggregation cube by concatenating the selected cubes
                - attach the new aggregation DimCoord and any other Coord objects to this new cube
                - discard original cubes
            - copy over any other unjoinable cubes as-is (simple union behaviour)

        case 3: coordValue attributes within <netcdf> elements are used to define coordinates for
                the aggregation dimension; <variable> element may define extra attributes such as
                standard_name, long_name, units, and so on 
            - foreach <netcdf> and/or <scan> element:
                - load cubelist from specified netcdf file(s)
                - create a new AuxCoord object to store coord values from coordValue attribute
                - name each such AuxCoord object as "#dimName#", e.g. "time" => "#time#"
                - attach the AuxCoord object to each loaded cube (i.e. the AuxCoord is shared)
            - create a new aggregation DimCoord object named dimName from the sorted #dimName#
                  AuxCoord objects
            - foreach distinct set of named cubes/variables:
                - sort the cubes and their #dimName# AuxCoord objects into asc/desc order
                - create a new aggregation cube by concatenating the selected cubes
                - attach the new aggregation DimCoord and any other Coord objects to this new cube
                - discard original cubes
            - copy over any other unjoinable cubes as-is (simple union behaviour)
        
        In the current implementation below, these cases have NOT been coded up separately, but it
        might aid understanding if they were.
        """
        self.logger.info("Processing joinExisting aggregation element...")
        if not agg_elem.dimName :
            errmsg = "<aggregation> elements of type 'joinExisting' must include a 'dimName' attribute."
            raise NcmlSyntaxError(errmsg)

        # Process any <netcdf> nodes.
        nc_nodes = agg_elem.node.getElementsByTagName('netcdf')
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_agg_netcdf_node(node, agg_elem)

        # Process any <scan> nodes.
        scan_nodes = agg_elem.node.getElementsByTagName('scan')
        if len(scan_nodes) :
            for node in scan_nodes :
                self._handle_agg_scan_node(node, agg_elem)

        # If coordValue attributes were used, create a new DimCoord by concatenating the
        # set of AuxCoord objects created in the _handle_agg_netcdf_node() method. If a zero-length
        # DimCoord with the same name exists in the self.dim_coords cache, replace it with the
        # new one created here.
        if agg_elem.uses_coord_value_att :
            agg_dim_coord = _create_agg_dim_coord(self.aux_coords)
            old_dim_coord = self._find_agg_dim_coord(agg_elem.dimName)
            if old_dim_coord and len(old_dim_coord.points) == 0:
                self.dim_coords.remove(old_dim_coord)
            self.dim_coords.append(agg_dim_coord)
            self.logger.info("Created aggregation coordinate %s from coordValue elements" %
                agg_dim_coord.var_name)
        
        # If the timeUnitsChange attribute was specified, then rebase to a common time datum all
        # the coordinate objects that match the nominated aggregation dimension. We have to assume
        # that this attribute is only included if the agg. dimension is indeed a temporal one. 
        if agg_elem.timeUnitsChange == "true" :
            time_coords = _get_coord_list(self.cubelist, dim_name=agg_elem.dimName)
            if time_coords : _rebase_time_coords(time_coords)
        
        # Get distinct variable names from the raw loaded cubelist.
        distinct_var_names = sorted(set([c.name() for c in self.cubelist]))
        self.logger.debug("Distinct cube names: %s"%distinct_var_names)

        # Create an empty cubelist to hold the joined cubes.
        joined_cubes = iris.cube.CubeList()
        before = len(self.cubelist)
        
        # For each distinct variable, attempt to perform a join operation on those variables which
        # are (i) associated with 2 or more cubes, and (ii) reference the aggregation dimension.
        # All other variables are simply copied over as-is.
        for var_name in distinct_var_names :
            self.logger.info("Attempting to merge cubes with name %s..." % var_name)
            var_cubes = self.cubelist.extract(var_name)
            if len(var_cubes) > 1 :
                if _has_same_outer_dim(var_cubes, agg_elem.dimName) :
                    try :
                        agg_dim_coord = self._find_agg_dim_coord(agg_elem.dimName)
                        tmp_cube = _concat_cubes(var_cubes, agg_elem.dimName, agg_dim_coord=agg_dim_coord)
                        joined_cubes.append(tmp_cube)
                        for cube in var_cubes : del cube
                        self.logger.info("...merged %d cubes" % len(var_cubes))
                    except Exception, exc :
                        self.logger.error("Unable to merge cubes associated with variable "+var_name)
                        self.logger.error(str(exc))
                        raise exc
                else :
                    self.logger.info("...unable to merge cubes - inconsistent dimensions")
                    joined_cubes.extend(var_cubes)
            # copy over single cube unmodified
            else :
                joined_cubes.extend(var_cubes)
                self.logger.info("...only 1 cube found so merge not possible")

        # Advertise the results of the merge.
        after = len(joined_cubes)
        if before != after :
            self.cubelist = joined_cubes
            msg = "Merged %d input cubes into %d after join along %s dimension" \
                % (before, after, agg_elem.dimName)
            self.logger.info(msg)
        else :
            msg = "Unable to join any cubes along %s dimension" % agg_elem.dimName
            self.logger.info(msg)

    def _handle_joinnew(self, agg_elem) :
        """
        Process a joinNew aggregation node. From the NcML spec:

            A JoinNew dataset is constructed by transferring objects (dimensions, attributes,
            groups, and variables) from the nested datasets in the order the nested datasets are
            listed. All variables that are listed as aggregation variables are logically concaten-
            ated along the new dimension, and in the order of the nested datasets. A coordinate
            variable is created for the new dimension. Non-aggregation variables are treated as in
            a Union dataset, i.e. skipped if one of that name already exists.

        Processing logic:
        
        case 1: a <variable> element is used to define coordinates for the aggregation dimension
            - create a new aggregation DimCoord object from the <values> sub-element within <variable>
            - build a cubelist from the netcdf files specified in <netcdf> and/or <scan> elements
            - foreach variable defined in a <variableAgg> element:
                - subset (i.e. extract) the cubes corresponding to the variable
                - create a new cube by concatenating the selected cubes
                - attach the new aggregation DimCoord and any other Coord objects to this new cube
                - discard the original cubes
            - copy over any other unjoinable cubes as-is (simple union behaviour)

        case 2: coordValue attributes within <netcdf> elements are used to define coordinates for
                the aggregation dimension; a <variable> element defines extra attributes such as
                standard_name, long_name, units, and so on 
            - build a cubelist from the netcdf files specified in <netcdf> and/or <scan> elements
            - create a new aggregation DimCoord object from the individual coordValue attributes,
              which are assumed to be occur in the correct order in the ncml file
            - foreach variable defined in a <variableAgg> element:
                - subset (i.e. extract) the cubes corresponding to the variable
                - create a new cube by concatenating the selected cubes
                - attach the new aggregation DimCoord and any other Coord objects to this new cube
                - discard the original cubes
            - copy over any other unjoinable cubes as-is (simple union behaviour)
        """
        self.logger.info("Processing joinNew aggregation element...")
        if not agg_elem.dimName :
            errmsg = "<aggregation> elements of type 'joinNew' must include a 'dimName' attribute."
            raise NcmlSyntaxError(errmsg)

        # Get the list of aggregation variables.
        agg_var_nodes = agg_elem.node.getElementsByTagName('variableAgg')
        if not len(agg_var_nodes) :
            errmsg = "<aggregation> elements of type 'joinNew' must contain one or more " + \
                "<variableAgg> elements."
            raise NcmlSyntaxError(errmsg)
        agg_var_names = []
        for node in agg_var_nodes :
            agg_var_names.append(node.getAttribute('name'))

        # Process any <netcdf> nodes.
        nc_nodes = agg_elem.node.getElementsByTagName('netcdf')
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_agg_netcdf_node(node, agg_elem)

        # Process any <scan> nodes.
        scan_nodes = agg_elem.node.getElementsByTagName('scan')
        if len(scan_nodes) :
            for node in scan_nodes :
                self._handle_agg_scan_node(node, agg_elem)

        # If coordValue attributes were used, create a new DimCoord by concatenating the
        # set of AuxCoord objects created in the _handle_agg_netcdf_node() method. If a zero-length
        # DimCoord with the same name exists in the self.dim_coords cache, replace it with the
        # new one created here.
        if agg_elem.uses_coord_value_att :
            agg_dim_coord = _create_agg_dim_coord(self.aux_coords)
            old_dim_coord = self._find_agg_dim_coord(agg_elem.dimName)
            if old_dim_coord and len(old_dim_coord.points) == 0:
                self.dim_coords.remove(old_dim_coord)
            self.dim_coords.append(agg_dim_coord)
            self.logger.info("Created aggregation coordinate %s from coordValue elements" %
                agg_dim_coord.var_name)

        # To proceed, a new aggregation dimension must have been created earlier, either from
        # <values> in a <variable> element, or from coordValue attributes on <netcdf> elements.  
        agg_dim_coord = self._find_agg_dim_coord(agg_elem.dimName)
        if not agg_dim_coord :
            errmsg = "Unable to find or build aggregation dimension with name " + agg_elem.dimName
            raise NcmlContentError(errmsg)

        # Create an empty cubelist to hold the joined cubes.
        joined_cubes = iris.cube.CubeList()
        before = len(self.cubelist)
        
        # For each named aggregation variable, create a new cube by concatenating the existing cubes
        # with that name along the new aggregation dimension. It is assumed that the cubelist for
        # each aggregation variable is returned in the order of loading, and thus matches the length
        # and order of the aggregation dimension.
        for var_name in agg_var_names :
            self.logger.info("Attempting to aggregate cubes with name %s..." % var_name)
            var_cubes = self._get_cubes_by_var_name(var_name)
            if len(var_cubes) > 1 :
                try :
                    tmp_cube = _aggregate_cubes(var_cubes, agg_dim_coord)
                    joined_cubes.append(tmp_cube)
                    for cube in var_cubes : del cube
                    self.logger.info("...joined %d cubes" % len(var_cubes))
                except Exception, exc :
                    self.logger.error("Unable to join cubes associated with variable "+var_name)
                    self.logger.error(str(exc))
                    raise exc
            else :
                joined_cubes.extend(var_cubes)
                self.logger.info("...only 1 cube found so join not possible")

        # Advertise the results of the merge.
        after = len(joined_cubes)
        if before != after :
            self.cubelist = joined_cubes
            msg = "Aggregated %d input cubes into %d after join along %s dimension" \
                % (before, after, agg_elem.dimName)
            self.logger.info(msg)
        else :
            msg = "Unable to join any cubes along new %s dimension" % agg_elem.dimName
            self.logger.info(msg)

    def _handle_agg_netcdf_node(self, netcdf_node, agg_elem) :
        """Process an aggregation netcdf node"""
        netcdf = NcmlElement(netcdf_node)
        if not netcdf.location :
            raise NcmlSyntaxError("<netcdf> elements must include a 'location' attribute.")
        ncpath = netcdf.location
        if not ncpath.startswith('/') : ncpath = os.path.join(self.basedir, ncpath)

        # Load cubes from the specified netcdf file.
        self.logger.info("Scanning netCDF file '%s'..." % ncpath)
        cubelist = iris.cube.CubeList()
        for cube in iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback) :
            cubelist.append(cube)
        
        # If coordinate values were specified via a coordValue attribute, use them to create a
        # new AuxCoord object which will be used later to apply a join operation to the cubelist.
        if netcdf.coordValue and agg_elem.type.startswith('join') :
            nctype = 'double'
            agg_dim_crd = self._find_agg_dim_coord(agg_elem.dimName)
            if agg_dim_crd : nctype = agg_dim_crd.nctype
            coord_values = _parse_text_string(netcdf.coordValue, nctype)
            dim_name = '#' + agg_elem.dimName + '#'
            # TODO: in the case of a joinExisting aggregation we somehow need to copy over any
            # attributes from the original netcdf coordinate variable associated with dimName.
            aux_coord = _create_aux_coord(dim_name, nctype, coord_values, template_coord=agg_dim_crd)
            self.aux_coords.append(aux_coord)
            dims = 0 if agg_elem.type == 'joinExisting' else None
            for cube in cubelist :
                cube.add_aux_coord(aux_coord, dims)
            agg_elem.uses_coord_value_att = True

        # Store the filename and cubelist associated with this <netcdf> node
        self.part_filenames.append(ncpath)
        self._extend_cubelist(cubelist, unique_names=(agg_elem.type=="union"))

    def _handle_agg_scan_node(self, scan_node, agg_elem) :
        """Process an aggregation scan node"""
        scan = NcmlElement(scan_node)
        if not scan.location :
            raise NcmlSyntaxError("<scan> elements must include a 'location' attribute.")

        topdir = scan.location
        if not topdir.startswith('/') : topdir = os.path.join(self.basedir, topdir)
        if not scan.subdirs : scan.subdirs = 'true'
        subdirs = scan.subdirs in ('true', 'True', '1') 
        regex = scan.regExp
        min_age = scan.olderThan
        if min_age : min_age = _time_string_to_interval(min_age)

        # if no regex was defined, use suffix if that was defined
        if not regex and scan.suffix : regex = '.*' + scan.suffix + '$'

        # walk directory tree from topdir searching for files matching regex
        # load cubes from each matching netcdf file
        for ncpath in _walk_dir_tree(topdir, recurse=subdirs, regex=regex, min_age=min_age) :
            self.logger.info("Scanning netCDF file '%s'..." % ncpath)
            self.part_filenames.append(ncpath)
            cubelist = iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback)
            self._extend_cubelist(cubelist, unique_names=(agg_elem.type=="union"))

    # NOT CURRENTLY USED
    def _handle_remove_nodes(self, remove_nodelist) :
        """Process any remove nodes. At present only variables and attributes can be removed."""
        if self.explicit :
            self.logger.warn("<remove> elements are not permitted when <explicit> element is present")
            return
        self.logger.info("Processing remove nodes...")
        for remove_node in remove_nodelist :
            self._handle_remove_node(remove_node)

    # NOT CURRENTLY USED
    def _handle_remove_node(self, remove_node) :
        """
        Process a single remove node. At present only variable (=cube) and attribute objects can
        be removed.
        """
        obj_name = remove_node.getAttribute('name')
        obj_type = remove_node.getAttribute('type')
        if not (obj_name and obj_type) :
            self.logger.warn("<remove> elements must include a 'name' and 'type' attribute.")
            return
        parent_tagname = remove_node.parentNode.tagName
        
        # for variables remove the corresponding cube(s), if present
        if obj_type == 'variable' :
            cubes = self._get_cubes_by_var_name(obj_name)
            if cubes :
                for cube in cubes : self.cubelist.remove(cube)
                self.logger.info("Removed variable named %s" % obj_name)
            else :
                self.logger.warn("No cube found corresponding to variable named %s" % obj_name)

        # remove an attribute, either from all cubes or from a named cube 
        elif obj_type == 'attribute' :
            if parent_tagname == 'netcdf' :
                for cube in self.cubelist :
                    cube.attributes.pop(obj_name, None)
                self.logger.info("Removed attribute %s from all variables" % obj_name)
            elif parent_tagname == 'variable' :
                var_name = remove_node.parentNode.getAttribute('name')
                cubes = self._get_cubes_by_var_name(var_name)
                if cubes :
                    for cube in cubes : cube.attributes.pop(obj_name, None)
                    self.logger.info("Removed attribute %s from variable %s" % (obj_name, var_name))
                else :
                    self.logger.warn("No cube found corresponding to variable named %s" % var_name)

    def _make_remove_list(self, remove_nodelist) :
        """
        Make a list of objects flagged for removal. At present only variables and attributes can
        be removed
        """
        if self.explicit :
            self.logger.warn("<remove> elements are not permitted when <explicit> element is present")
            return

        for node in remove_nodelist :
            remove = NcmlElement(node)
            if not (remove.name and remove.type) :
                self.logger.warn("<remove> elements must include a 'name' and 'type' attribute.")
                continue
            if remove.type not in ('attribute', 'variable') :
                self.logger.warn("Can only remove elements of type attribute or variable")
                continue
            remove.parent_type = node.parentNode.tagName
            if remove.parent_type == 'netcdf' :
                remove.parent_name = 'netcdf'
            else :
                remove.parent_name = node.parentNode.getAttribute('name')
            self.removelist.append(remove)

    def _process_remove_list(self) :
        """Process items in the remove list."""
        for item in self.removelist :
            if item.type == 'variable' :   # variables should not have made it into the cubelist
                continue
            elif item.type == 'attribute' :
                att_name = item.name
                if item.parent_type == 'netcdf' :   # global attribute
                    for cube in self.cubelist :
                        cube.attributes.pop(att_name, None)
                    self.logger.info("Removed attribute %s from all cubes" % att_name)
                elif item.parent_type == 'variable' :   # variable-scope attribute
                    var_name = item.parent_name
                    cubes = self._get_cubes_by_var_name(var_name)
                    if cubes :
                        for cube in cubes : cube.attributes.pop(att_name, None)
                        self.logger.info("Removed attribute %s from cube %s" % (att_name, var_name))
                    else :
                        self.logger.warn("No cube found corresponding to variable %s" % var_name)

    def _is_flagged_for_removal(self, node_or_cube) :
        """Check to see if an element or cube is flagged for removal via a <remove> element."""
        if isinstance(node_or_cube, iris.cube.Cube) :
            obj_name = node_or_cube.var_name
            obj_type = 'variable'
            parent_type = 'netcdf'
            parent_name = 'netcdf'
        else :
            obj_name = node_or_cube.getAttribute('name')
            obj_type = node_or_cube.getAttribute('type')
            if not (obj_name and obj_type) : return False
            parent_type = node_or_cube.parentNode.tagName
            if parent_type == 'netcdf' :
                parent_name = 'netcdf'
            else :
                parent_name = node_or_cube.parentNode.getAttribute('name')
        d = dict(name=obj_name, type=obj_type, parent_name=parent_name, parent_type=parent_type)
        return d in self.removelist

    def _is_coord_variable(self, var_elem) :
        """
        Return True if var_elem represents a coordinate variable element. A coordinate variable is
        recognised by having its name attribute identical to its shape attribute, and by having its
        name equal to the name of a previously defined dimension element, e.g.
            <dimension name='time' length='30'/>
            <variable name='time' type='int' shape='time'> ... </variable> 
        """
        if not var_elem.name :
            raise NcmlSyntaxError("<variable> elements must include a 'name' attribute.")
        if not var_elem.type :
            errmsg = "<variable> element named %s does not contain a 'type' attribute." \
                % var_elem.name
            self.logger.error(errmsg)
            raise NcmlSyntaxError("<variable> elements must include a 'type' attribute.")
        dim_names = [dim.name for dim in self.dimensions]
        return (var_elem.name == var_elem.shape and var_elem.name in dim_names)

    def _is_data_variable(self, var_elem) :
        """
        Return True if var_elem represents a data variable element. A data variable is recognised by
        having its name attribute NOT equal to its shape attribute, and by having its name NOT equal
        to any if defined previously defined dimension element, e.g.
            <variable name='tas' type='float' shape='mydim'> ... </variable>
        This definition is essentially the inverse of the _is_coord_variable method (which we could
        invoke as a proxy). Note that the shape attribute is not obligatory for data variables
        (in which case they are treated as scalar variables). 
        """
        if not var_elem.name :
            raise NcmlSyntaxError("<variable> elements must include a 'name' attribute.")
        if not var_elem.type :
            errmsg = "<variable> element named %s does not contain a 'type' attribute." \
                % var_elem.name
            self.logger.error(errmsg)
            raise NcmlSyntaxError("<variable> elements must include a 'type' attribute.")
        dim_names = [dim.name for dim in self.dimensions]
        return (var_elem.name != var_elem.shape and var_elem.name not in dim_names)

    def _add_coord_variable(self, var_elem) :
        """Create a coordinate variable (an iris DimCoord object) from a <variable> element."""
        nodes = var_elem.node.getElementsByTagName('values')
        if nodes :
            val_node = nodes[0]
            values = NcmlElement(val_node)
            # Compute coordinate values from start/increment/npts attributes, else read them from the
            # child text node.
            if values.start :
                if values.npts :
                    npts = int(values.npts)
                else :
                    for dim in self.dimensions :
                        if var_elem.shape == dim.name : npts = dim.length
                try :
                    start = int(values.start)
                    inc = int(values.increment)
                except :
                    start = float(values.start)
                    inc = float(values.increment)
                points = []
                for i in range(npts) :
                    points.append(start + i*inc)
            else :
                text_values = _get_node_text(val_node)
                points = _parse_text_string(text_values, var_elem.type, sep=values.separator)
                if not points : points = []
            self.logger.debug("First few coordinate <values>: %s" % points[:10])

        # No <values> element present: assume that coordinates will be obtained later on from
        # coordValue attributes on <netcdf> elements nested in a join-type aggregation.
        else :
            points = []
            self.logger.debug("No <values> defined for coordinate variable "+var_elem.name)

        # Set any keyword arguments specified by nested <attribute> elements.
        kw = dict(standard_name=None, long_name=None, units='1')
        extra_atts = dict()
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist :
            name, value = _parse_attribute(node)
            if name in kw :
                kw[name] = value
            else :
                extra_atts[name] = value
        
        # Special handling of calendar attribute: use it to create CF-style time unit object
        cal = extra_atts.pop('calendar', None)
        if cal :
            tunits = iris.unit.Unit(kw['units'], calendar=cal)
            kw['units'] = tunits

        # Create and store the DimCoord object for subsequent use.
        dim_coord = DimCoord(np.array(points, dtype=var_elem.type), **kw)
        dim_coord.var_name = var_elem.name
        dim_coord.nctype = var_elem.type
        if len(dim_coord.points) > 1 : dim_coord.guess_bounds()
        
        # Assign any extra (non-CF) attributes specified in the <variable> element.
        for name,value in extra_atts.items() :
            try :
                dim_coord.attributes[name] = value
            except :
                self.logger.warn("Attribute named %s is not permitted on a DimCoord object" % name)

        self.dim_coords.append(dim_coord)
        return dim_coord

    def _add_data_variable(self, var_elem) :
        """Create a data variable (an iris Cube) from a <variable> element."""
        nodes = var_elem.node.getElementsByTagName('values')
        if not nodes :
            raise NcmlSyntaxError("A <values> element is required to create a data variable")

        val_node = nodes[0]
        values = NcmlElement(val_node)

        # Compute or retrieve coordinate values.
        if values.start :
            try :
                start = int(values.start)
                inc = int(values.increment)
            except :
                start = float(values.start)
                inc = float(values.increment)
            points = []
            for i in range(int(values.npts)) :
                points.append(start + i*inc)
        else :
            text_values = _get_node_text(val_node)
            points = _parse_text_string(text_values, var_elem.type, sep=values.separator)

        # Set any keyword arguments from nested <attribute> elements.
        kw = dict(standard_name=None, long_name=None, units='1', cell_methods=None)
        extra_atts = dict()
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist :
            name, value = _parse_attribute(node)
            if name in kw :
                kw[name] = value
            else :
                extra_atts[name] = value

        # For a dimensioned (non-scalar) variable
        if var_elem.shape :
            shape = []
            coords = []
            # create coordinate list for this variable
            # FIXME: I reckon this may be a somewhat brittle approach
            all_coords = self.dim_coords[:]
            all_coords.extend(_get_coord_list(self.cubelist))
            dimnum = 0
            # for each dimension named in the shape attribute, find the corresponding coord object
            for i,dim_name in enumerate(var_elem.shape.split()) :
                for dim_coord in all_coords :
                    if dim_name == dim_coord.var_name or dim_name == dim_coord.standard_name :
                        coords.append([dim_coord,dimnum])
                        shape.append(len(dim_coord.points))
                        dimnum += 1
                        break

        # For a scalar variable we just need to set the array shape.
        else :
            shape = [1]
            coords = []
            
        # Create a cube object and append to the dataset's cubelist.
        data = np.array(points)
        data.shape = shape
        cube = iris.cube.Cube(data, dim_coords_and_dims=coords, **kw)
        cube.var_name = var_elem.name
        cube.nctype = var_elem.type

        # Assign any extra (non-CF) attributes specified in the <variable> element.
        for name,value in extra_atts.items() :
            try :
                cube.attributes[name] = value
            except :
                self.logger.warn("Attribute named %s is not permitted on a cube" % name)

        self.cubelist.append(cube)
        self.logger.debug("- cube shape: %s" % str(data.shape))
        self.logger.debug("- cube min value: %s" % cube.data.min())
        self.logger.debug("- cube max value: %s" % cube.data.max())

        return cube

    def _extend_cubelist(self, cubelist, unique_names=False) :
        """
        Append cubes from cubelist to the NcMLDataset's current cubelist. If the unique_names
        argument is set to true then only append cubes with distinct names, which is the behaviour
        required by union-type aggregations.
        """
        for cube in cubelist :
            # skip variable if it's flagged for removal
            if self._is_flagged_for_removal(cube) :
                self.logger.info("Skipped netCDF variable %s (reason: target of <remove> element)" % cube.var_name)
            # skip variable if it's already present
            elif cube.name() in self.get_cube_names() and unique_names :
                self.logger.info("Skipped netCDF variable %s (reason: loaded from earlier file)" % cube.var_name)
            else :
                self.cubelist.append(cube)
                self.logger.info("Added cube for netCDF variable %s" % cube.var_name)

    def _find_agg_dim_coord(self, dim_name) :
        """Find the aggregation dimension Coord object associated with name dim_name."""
        for crd in self.dim_coords :
            if crd.var_name == dim_name : return crd
        return None

    def _update_cube_attributes(self, attrs) :
        """Override any cubes attributes with those specified in the attrs dict argument."""
        for cube in self.get_cubes() :
            if not hasattr(cube, 'attributes') : cube.attributes = dict()
            cube.attributes.update(attrs)

    def _get_cubes_by_var_name(self, var_name) :
        """Return the loaded cube with the specified variable name."""
        return [cube for cube in self.cubelist if cube.var_name == var_name]

def load_cubes(filenames, callback=None, **kwargs) :
    """
    Generator function returning a sequence of cube objects associated with the netCDF files
    specified within an NcML file. The current implementation can only handle a single NcML file.
    """
    # Update logger object if necessary
    log_level = kwargs.pop('log_level', None)
    if log_level : _update_logger(log_level)

    if isinstance(filenames, (list,tuple)) :
        if len(filenames) > 1 :
            errmsg = "Iris can only read a single NcML file; %d filenames were specified."%len(filenames)
            raise iris.exceptions.DataLoadError(errmsg)
        ncml_file = filenames[0]
    elif isinstance(filenames, basestring) :
        ncml_file = filenames
    elif isinstance(filenames, file) :
        ncml_file = filenames
    else :
        raise AttributeError("Invalid file-like argument passed to ncml.load_cubes")

    # Create ncml dataset object
    try :
        ncml_dataset = NcmlDataset(ncml_file, **kwargs)
    except Exception, exc :
        print >>sys.stderr, str(exc)
        raise iris.exceptions.DataLoadError("Error trying to load NcML dataset")
    
    for cube in ncml_dataset.get_cubes() :
        yield cube

def _init_logger(log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT) :
    """Initialise a logger object using default or user-supplied settings."""
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(log_level)
    fmtr = logging.Formatter(log_format)
    console.setFormatter(fmtr)
    logger = logging.getLogger('iris.fileformats.ncml')
    logger.addHandler(console)
    logger.setLevel(log_level)
    return logger

def _update_logger(level=None, format=None) :
    """Update logging level and/or format properties."""
    if level : logger.setLevel(level)
    for handler in logger.handlers :
        if level : handler.setLevel(level)
        if format : handler.setFormatter(logging.Formatter(format))

logger = _init_logger()

def _nc_load_callback(cube, field, filename) :
    """Callback for adding netCDF variable name as cube attribute."""
    # for netcdf files, field is of type iris.fileformats.cf.CFDataVariable
    if hasattr(field, '_name') :
        cube.var_name = field._name   # how robust is this?
    else :
        cube.var_name = 'undefined'

def _get_nc_var_atts(filename, varname) :
    """Return a dictionary of attribute values for variable varname in netcdf filename.""" 
    try :
        ds = nc4.Dataset(filename,  'r')
        var = ds.variables[varname]
        att_dict = dict(var.__dict__)
        ds.close()
    except :
        att_dict = {}
    return att_dict

def _get_node_att_dict(node, att_namelist) :
    """Return a dictionary of the attributes named in att_namelist for the specified node."""
    att_dict = dict()
    for name in att_namelist :
        value = node.getAttribute(name)
        if value == '' : value = None
        att_dict[name] = value
    return att_dict

def _get_node_text(node) :
    """Retrieve the character data from an XML node."""
    text = []
    for child in node.childNodes :
        if child.nodeType == child.TEXT_NODE :
            text.append(child.data)
    text = ''.join(text)
    if re.match('\s+$', text) :   # test for all whitespace
        return ''
    else :
        return text

def _get_coord_list(cubelist, dim_name=None) :
    """
    Return a list of coordinate objects (DimCoords, AuxCoords) associated with the specified
    cubelist. The dim_name argument may be used to select only those coordinates with the
    specified name.
    """
    # FIXME: this function would probably be better recast as a generator
    coord_list = []
    for cube in cubelist :
        for coord in cube.coords(name=dim_name) :
            if coord not in coord_list : coord_list.append(coord)
    return coord_list

def _has_same_outer_dim(cubelist, dim_name) :
    """Tests to see if the cubes in cubelist all have the same outer dimension name."""
    for cube in cubelist :
        if not cube.coords(name=dim_name, dim_coords=True, dimensions=0) : return False
    return True

def _aggregate_cubes(cubelist, agg_dim_coord) :
    """
    Aggregate the specified cubelist along the aggregation dimension defined by agg_dim_coord,
    which will form the outer dimension of the newly created and returned cube.
    
    TODO: much overlap with _concat_cubes method below - consider combining into one method
    """
    # Check that the number of input cubes equals the length of the aggregation dimension.
    agg_dim_len = len(agg_dim_coord.points)
    if agg_dim_len != len(cubelist) :
        errmsg = "Length of aggregation dimension (%d) does not match number of cubes (%d)" \
            % (agg_dim_len, len(cubelist))
        raise iris.exceptions.DataLoadError(errmsg)
    logger.debug("Shape of aggregation dimension: %s" % str(agg_dim_coord.shape))

    # Check that all input cubes have the same shape.
    cube0 = cubelist[0]
    shape = cube0.shape
    for cube in cubelist[1:] :
        if cube.shape != shape :
            errmsg = "Cubes for variable %s have inconsistent dimensions" % cube.var_name
            raise iris.exceptions.DataLoadError(errmsg)

    # Concatenate all the data arrays from the input cubes.
    data = cube0.data
    for cube in cubelist[1:] :
        data = np.append(data, cube.data, axis=0)
    data.shape = (agg_dim_len,) + shape
    logger.debug("Shape of aggregated data array: %s" % str(data.shape))

    # Create a new cube from the aggregated data array and by copying metadata attributes from
    # the first input cube.
    newcube = iris.cube.Cube(data, standard_name=cube0.standard_name, long_name=cube0.long_name,
        units=cube0.units, attributes=cube0.attributes.copy(), cell_methods=cube0.cell_methods)

    # Add aggregation coordinate as new outer dimension.
    newcube.add_dim_coord(agg_dim_coord, 0)
    logger.debug("Added dim coord %s at dimension 0" % agg_dim_coord.name())

    # Copy over all original dimension coordinates, incrementing dimension numbers.
    for coord in cube0.dim_coords :
        dim = cube0.coord_dims(coord)[0] + 1
        newcube.add_dim_coord(coord, dim)
        logger.debug("Added dim coord %s at dimension %d" % (coord.name(), dim))

    # Copy over all auxiliary coordinates, incrementing dimension numbers and skipping any temporary
    # dimensions named '#...#'
    for coord in cube0.aux_coords :
        if getattr(coord, 'var_name', ' ')[0] == '#' : continue
        dims = cube0.coord_dims(coord)
        newcube.add_aux_coord(coord, map(lambda x:x+1, dims))
        logger.debug("Added aux coord %s" % coord.name())

    # Record a summary of the join operation in the history attribute.
    newcube.attributes['history'] = _history_timestamp() + ": iris: variable created by " + \
        "NcML-type aggregation along %s dimension" % agg_dim_coord.var_name

    return newcube

def _concat_cubes(cubelist, dim_name, agg_dim_coord=None) :
    """
    Concatenate the passed in cubelist along the specified dimension. The cube data must be concat-
    enated in the same order as the corresponding dimension coordinates, be they increasing or
    decreasing. The current implementation copies any cube attributes from the first input cube.
    """
    # Retrieve the list of DimCoord objects associated with the input cubes.
    dim_coord_list = []
    for cube in cubelist :
        dim_coord_list.extend(cube.coords(name=dim_name, dim_coords=True))

    # Since cubes don't share coord objects, the lengths of the cube and coord lists should be
    # the same.
    if len(dim_coord_list) != len(cubelist) :
        errmsg = "Number of coordinate objects (%d) does not match number of cubes (%d)" \
            % (len(dim_coord_list), len(cubelist))
        raise iris.exceptions.DataLoadError(errmsg)

    # Sort the cube list and dim coord list in tandem. Do a reverse sort if the first DimCoord
    # object is monotonic decreasing.
    reverse = dim_coord_list[0].points[0] > dim_coord_list[0].points[-1]
    sort_key = lambda i: i[0].points[0]
    dim_coord_list, cubelist = zip(*sorted(zip(dim_coord_list, cubelist), key=sort_key,
        reverse=reverse))

    # Create a new DimCoord object from dim_coord_list, which was sorted earlier.
    if not agg_dim_coord :
        agg_dim_coord = _concat_coords(dim_coord_list, sort=False)
    logger.debug("Shape of aggregated coordinate data: %s" % str(agg_dim_coord.shape))

    # Concatenate all the data arrays from the input cubes.
    # Is this the most efficient idiom? Preferable to np.concatenate I think.
    data = cubelist[0].data
    for cube in cubelist[1:] :
        data = np.append(data, cube.data, axis=0)
    logger.debug("Shape of aggregated data array: %s" % str(data.shape))

    # Create a new cube from the aggregated data array and by copying metadata attributes from
    # the first input cube. Associate the cube with the new aggregated DimCoord object. Because
    # the data and coordinate array shapes have changed we can't use the cube.copy() method.
    cube0 = cubelist[0]
    newcube = iris.cube.Cube(data, standard_name=cube0.standard_name, long_name=cube0.long_name,
        units=cube0.units, attributes=cube0.attributes.copy(), cell_methods=cube0.cell_methods)

    # copy over all dimension coordinates
    for coord in cube0.dim_coords :
        dim = cube0.coord_dims(coord)[0]
        if coord.name() == dim_name :
            newcube.add_dim_coord(agg_dim_coord, dim)
        else :
            newcube.add_dim_coord(coord, dim)
        logger.debug("Added dim coord %s at dimension %d" % (coord.name(), dim))

    # copy over all auxiliary coordinates, skipping any temporary ones named '#...#'
    for coord in cube0.aux_coords :
        if getattr(coord, 'var_name', ' ')[0] == '#' : continue
        dims = cube0.coord_dims(coord)
        newcube.add_aux_coord(coord, dims)
        logger.debug("Added aux coord %s" % coord.name())

    # Record a summary of the join operation in the history attribute.
    newcube.attributes['history'] = _history_timestamp() + ": iris: variable created by " + \
        "NcML-type aggregation along %s dimension" % dim_name

    return newcube

def _concat_coords(coord_list, sort=True, reverse=False) :
    """
    Concatenate a list of DimCoord or AuxCoord objects into a single DimCoord object. Assumes all
    of the input coord objects are identical except for their points and bounds arrays. By default
    the DimCoords are concatenated in ascending order of their first point. Set the reverse keyword
    to True to concatenate in descending order. Set the sort keyword to False to disable sorting.
    """
    # TODO: check that coord object metadata (std name, units, calendar) are equal across input list 
    if sort : coord_list.sort(_coord_sorter, reverse=reverse)
    points = np.concatenate([c.points for c in coord_list], axis=0)
    
    coord0 = coord_list[0]
    if coord0.has_bounds() :
        bounds = np.concatenate([c.bounds for c in coord_list], axis=0)
    else :
        bounds = None
    
    if isinstance(coord0, DimCoord) :
        dcoord = coord0.copy(points, bounds=bounds)
    elif isinstance(coord0, AuxCoord) :
        # create the new DimCoord the long-hand way
        dcoord = DimCoord(points, bounds=bounds, standard_name=coord0.standard_name,
            long_name=coord0.long_name, units=coord0.units, attributes=coord0.attributes)
        dcoord.var_name = coord0.var_name
        dcoord.nctype = coord0.nctype
    else :
        raise TypeError("Unrecognised coordinate type: %s" % type(coord0))

    return dcoord

def _create_aux_coord(dim_name, nctype, coord_values, template_coord=None) :
    """
    Create an auxiliary coordinate from the specified arguments. If a template coordinate object
    is passed in it is used as the source of additional attributes to assign to the new aux coord.
    """
    try :
        npoints = len(coord_values)
        points = np.array(coord_values)
        aux_coord = AuxCoord(points, long_name=dim_name)
        aux_coord.var_name = dim_name
        aux_coord.nctype = nctype
        if npoints > 1 : aux_coord.guess_bounds()
        if template_coord :
            for attname in ('standard_name', 'units', 'attributes') :
                attval = getattr(template_coord, attname, None)
                if attval : setattr(aux_coord, attname, attval)
        logger.debug("Created auxiliary coordinate %s with %d value%s" \
            % (dim_name, npoints, 's'[npoints==1:]))
    except Exception, exc :
        errmsg = "Error trying to create auxiliary coordinate for dimension %s" % dim_name
        logger.error(errmsg)
        logger.error(str(exc))
        raise exc
    return aux_coord

def _create_agg_dim_coord(aux_coord_list) :
    """Create an aggregation DimCoord object by concatenating a list of AuxCoord objects."""
    reverse = aux_coord_list[0].points[0] > aux_coord_list[0].points[-1]
    dcoord = _concat_coords(aux_coord_list, reverse=reverse)
    dcoord.var_name = dcoord.var_name.strip('#')   # remove '#' chars from dim name
    dcoord.long_name = dcoord.var_name
    return dcoord

def _extend_agg_dim_coord(dim_coord, coord_values) :
    """
    Extend the specified aggregation dimension DimCoord object with coord_values. Since iris Coord
    objects are immutable this function returns an extended copy of the original.
    """
    try :
        npoints = len(coord_values)
        points = np.array(coord_values, dtype=dim_coord.points.dtype)
        points = np.append(dim_coord.points, points)
        ext_coord = dim_coord.copy(points)
        if len(ext_coord.points) > 1 : ext_coord.guess_bounds()
        logger.debug("Extended aggregation dimension '%s' with %d value%s" \
            % (ext_coord.name(), npoints, 's'[npoints==1:]))
        return ext_coord
    except Exception, exc :
        errmsg = "Error trying to extend aggregation dimension " + dim_coord.name()
        logger.error(errmsg)
        logger.error(str(exc))
        raise exc

def _rebase_time_coords(coord_list, target_unit=None) :
    """
    Rebase a list of CF-style time coordinate objects so that they all reference the same time
    datum. The new time datum may be specified via the target_unit argument, which should be a
    string of the form 'time-units since time-datum' or a Unit object which provides the same info.
    In the latter case, the calendar property must match the same property in each coord object.
    If the target_unit argument is not specified then it is set equal to the earliest datum in the
    list of input coordinate objects. All of the input coordinate objects must use the same base
    time units, e.g. seconds, days, hours. If a coordinate object contains bounds then those values
    are also rebased.
    """
    coord0 = coord_list[0]
    base_unit, base_datum = coord0.units.origin.split('since')
    base_cal = coord0.units.calendar
    if target_unit :
        if isinstance(target_unit, basestring) :
            target_unit = iris.unit.Unit(target_unit, calendar=base_cal)
        elif isinstance(target_unit, iris.unit.Unit) :
            if base_cal != target_unit.calendar :
                errmsg = "source calendar (%s) and target calendar (%s) do not match" % \
                    (base_cal, target_unit.calendar)
                raise iris.exceptions.DataLoadError(errmsg)
        else :
            errmsg = "target_unit argument must be of type string or iris.unit.Unit"
            raise iris.exceptions.DataLoadError(errmsg)
    else :
        # Find the earliest time datum in the input coordinate list
        min_datum = _get_earliest_time_datum(coord_list)
        target_unit = iris.unit.Unit(base_unit+'since'+min_datum, calendar=base_cal)
    
    # Convert all time coordinates (and bounds, if present) to the new target unit.
    for crd in coord_list :
        if not crd.units.convertible(target_unit) :
            logger.warn("Cannot convert unit '%s' to '%s'" % (crd.units, target_unit))
        else :
            crd.points = crd.units.convert(crd.points, target_unit)
            if crd.has_bounds() : crd.bounds = crd.units.convert(crd.bounds, target_unit)
            crd.units = target_unit

    logger.debug("Rebased time coordinates to '%s'" % target_unit)

def _get_earliest_time_datum(coord_list) :
    """Return the earliest time datum used by a collection of time coordinate objects."""
    ref_origin = coord_list[0].units.origin
    ref_cal = coord_list[0].units.calendar
    min_offset = 0
    min_datum = ref_origin.split('since')[1]
    for crd in coord_list[1:] :
        crd_date = iris.unit.num2date(0, crd.units.origin, crd.units.calendar)
        crd_offset = iris.unit.date2num(crd_date, ref_origin, ref_cal)
        if crd_offset < min_offset :
            min_offset = crd_offset
            min_datum = crd.units.origin.split('since')[1]
    return min_datum

def _coord_sorter(coord, other) :
    """Sort two coord objects by comparing the value of their first point."""
    return cmp(coord.points[0], other.points[0])

def _time_string_to_interval(timestr) :
    """
    Convert an NcML time period defined as a string to an interval in whole seconds. The main use
    for this is to convert the value specified in the olderThan attribute to a useable interval.
    The time string must be in the format "value timeunit", e.g. "5 min", "1 hour", and so on.
    """
    try :
        ival,iunit = timestr.split()
        iunit = iris.unit.Unit(iunit)
        sunit = iris.unit.Unit('second')
        interval = iunit.convert(float(ival), sunit)
        return long(interval)
    except :
        raise ValueError("Error trying to decode time period: %s" % timestr)

def _history_timestamp() :
    """Return a timestamp suitable for use at the beginning of a history attribute."""
    from datetime import datetime
    now = datetime.now()
    return now.isoformat().split('.')[0]

def _is_older_than(filename, interval) :
    """
    Return true if the latest modification time of filename is older than the specified interval
    (in seconds) before present."""
    # get modification time of filename
    mtime = datetime.fromtimestamp(os.path.getmtime(filename))
    
    # see if file modification time is older than interval
    age = datetime.now() - mtime
    return (age.total_seconds() > interval)
    
def _walk_dir_tree(topdir, recurse=True, regex=None, min_age=None, sort=False) :
    """
    Walk the directory tree rooted at topdir, yielding filenames which match the optional
    regular expression pattern specified in the regex argument, and which are older than min_age,
    if specified.
    """ 
    if regex : reobj = re.compile(regex)
    for curr_path, subdirs, filenames in os.walk(topdir) :
        if sort : filenames.sort()
        for fn in filenames :
            curr_file = os.path.join(curr_path, fn)
            if (regex and not reobj.match(fn)) or \
               (min_age and not _is_older_than(curr_file, min_age)) :
                continue
            yield curr_file
        if not recurse : break

def _parse_attribute(att_node) :
    """Parse the name and value (type converted) specified in an NcML <attribute> node."""
    name = att_node.getAttribute('name')
    nctype = att_node.getAttribute('type') or 'String'
    txtval = att_node.getAttribute('value')
    sep = att_node.getAttribute('separator') or None
    value = _parse_text_string(txtval, nctype, sep)
    return (name, value)
     
def _parse_text_string(text, nctype='String', sep=None) :
    """
    Parse the specified text string into a (possibly length-one) array of values of type nctype,
    being one of the recognised NcML/NetCDF data types ('byte', 'int', etc). As per the NcML default,
    values in lists are separated by arbitrary amounts of whitespace. An alternative separator may
    be set using the sep argument, in which case values are separated by single occurrences of the
    specified separator.
    
    NOTE: there is no support at present for the 'Structure' data type described in the NcML spec.
    """
    if nctype in ('char', 'string', 'String') :
        return text
    elif nctype == 'byte' :
        vlist = [np.int8(x) for x in text.split(sep)]
    elif nctype == 'short' :
        vlist = [np.int16(x) for x in text.split(sep)]
    elif nctype == 'int' or nctype == 'long' :
        vlist = [np.int32(x) for x in text.split(sep)]
    elif nctype == 'float' :
        vlist = [np.float32(x) for x in text.split(sep)]
    elif nctype == 'double' :
        vlist = [np.float64(x) for x in text.split(sep)]
    else :
        raise iris.exceptions.DataLoadError("Unsupported NcML attribute data type: %s" % nctype)
    return vlist

def test(ncml_filename) :
    """Rudimentary test function"""
    cubes = load_cubes(ncml_filename, log_level=logging.DEBUG)
    for i,cube in enumerate(cubes) :
        dmin = cube.data.min()
        dmax = cube.data.max()
        print "Cube #%d: { std name: %s, shape: %s }" % (i, cube.name(), cube.shape)

if __name__ == "__main__" :
    usage = "usage: python ncml.py <ncml_file>"
    if len(sys.argv) < 2 :
        print usage
        sys.exit(1)
    test(sys.argv[1])
