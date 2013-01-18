"""
WARNING: THIS CODE IS A WORK IN PROGRESS. IT DOESN'T WORK FULLY YET AND IS LIABLE TO CHANGE.

Iris i/o module for loading NcML v2.2 files. Current functionality is limited to the following:

* only handles aggregations of netcdf files
* union-type aggregation based on <netcdf> or <scan> elements
* joinExisting-type aggregation for simple 1D aggregation dimensions only
* rename or remove variables (= cubes)
* add, override, rename or remove attributes (global or per-variable)
* create new 1D coordinate variables as DimCoord objects

Functionality that is expressly excluded:

* modification/removal of dimensions (this would interfere with Iris' handling of dimensions)
* NcML group elements
* forecastModelRun* elements
* the Structure data type
* nested variables

TO DO:

* joinExisting aggregation
* joinNew aggregation
* handle dateFormatMark for join-type aggregations
* handle coordValue attribute for join-type aggregations
* support loading of non-netcdf data formats?
* add sphinx markup throughout
"""
import sys, os, re, fnmatch, logging
from datetime import datetime
import iris
import numpy as np
import netCDF4 as nc4
from xml.dom.minidom import parse

# default logging options
DEFAULT_LOG_LEVEL  = logging.WARN
DEFAULT_LOG_FORMAT = "[%(name)s.%(funcName)s] %(levelname)s: %(message)s"

# Define lists of permissible attributes for various NcML 2.2 tags
att_namelists = {
    'aggregation': ('type', 'dimName', 'recheckEvery'),
    'attribute': ('name', 'type', 'separator', 'value', 'orgName'),
    'dimension': ('name', 'length', 'isUnlimited', 'isVariableLength', 'isShared', 'orgName'),
    'netcdf': ('location', 'id', 'title', 'enhance', 'addRecords', 'ncoords', 'coordValue'),
    'scan': ('location', 'suffix', 'regExp', 'subdirs', 'olderThan', 'dateFormatMark', 'enhance'),
    'values': ('start', 'increment', 'npts', 'separator'),
    'variable': ('name', 'type', 'shape', 'orgName'),
    'variableAgg': ('name',),
}

class NcmlSyntaxError(iris.exceptions.IrisError) :
    """Exception class for NcML syntax errors."""
    pass

class NcmlElement(object) :
    """
    A lightweight class for creating an object representation of an NcML element with equivalent
    attributes. Avoids the need to create discrete classes for each NcML element type, which would
    be a viable alternative approach."""
    def __init__(self, node) :
        self.node = node
        for att_name in att_namelists.get(node.tagName,[]) :
            att_value = node.getAttribute(att_name)
            if att_value == '' : att_value = None
            setattr(self, att_name, att_value)

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
        self.part_filenames = list()
        self.part_cubelists = list()   # FIXME: redundant?
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

        # process an <aggregation> node, if present
        agg_nodelist= doc.getElementsByTagName('aggregation')
        if len(agg_nodelist) : self._handle_aggregation(agg_nodelist[0])

        # process any global <attribute> nodes
        att_nodelist = [n for n in doc.getElementsByTagName('attribute') if n.parentNode.tagName == 'netcdf']
        if len(att_nodelist) : self._handle_global_attributes(att_nodelist)

        # process any <variable> nodes
        var_nodelist = doc.getElementsByTagName('variable')
        if len(var_nodelist) : self._handle_variables(var_nodelist)

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
        self.dimensions.append(dim)
        self.logger.info("Added dimension %s" % dim.name)

    def _handle_global_attributes(self, att_nodelist) :
        """Process all attribute nodes"""
        self.logger.info("Processing global attributes...")
        for att_node in att_nodelist :
            self._handle_attribute(att_node)
    
    # TODO: certain attributes are treated as special by iris and cannot be assigned to a cube's
    # attributes dictionary. The list of such attributes is defined in the LimitedAttributeDict
    # class in module iris._cube_coord_common.py. We'll need to skip these.
    
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
            if att.orgName :
                old_val = cube.attributes.pop(att.orgName, None)
            cube.attributes[att.name] = _cast_data_value(att.value, att.type, sep=att.separator)

    def _handle_variables(self, var_nodelist) :
        """Process all variable nodes"""
        self.logger.info("Processing variable nodes...")
        for var_node in var_nodelist :
            self._handle_variable(var_node)

    def _handle_variable(self, var_node) :
        """Process a single variable node"""
        var = NcmlElement(var_node)
        if not var.name :
            raise NcmlSyntaxError("<variable> elements must include a 'name' attribute.")
        if not var.type :
            raise NcmlSyntaxError("<variable> elements must include a 'type' attribute.")
        self.logger.info("Variable element: name: '%s', type: '%s'" % (var.name, var.type))

        # check to see if the variable contains a nested <values> element (i.e. data)
        val_nodes = var_node.getElementsByTagName('values')
        has_values = len(val_nodes) == 1
        
        if has_values :
            dim_names = [dim.name for dim in self.dimensions]
            # If the variable's name and shape match a single dimension name, then assume that the
            # variable element defines a coordinate variable which we can reify as an iris DimCoord
            # object.
            if var.name == var.shape and var.name in dim_names :
                self._add_coord_variable(var)
                self.logger.info("Added coordinate variable %s" % var.name)

            # If variable name is not a dimension name, but shape name/s is/are, then assume that
            # the variable element defines a data variable, possibly a scalar one, which we can
            # reify as an iris Cube object.
            elif var.name != var.shape : # and set(var.shape.split()) <= set(dim_names) :
                self._add_data_variable(var)
                if var.shape :
                    self.logger.info("Added data variable %s as new cube" % var.name)
                else :
                    self.logger.info("Added scalar data variable %s as new cube" % var.name)
            
        else :
            # if necessary, rename the variable from orgName to var.name
            if var.orgName :
                cube = self._get_cube_by_var_name(var.orgName)
                if cube :
                    cube.var_name = var.name
                    self.logger.info("Renamed variable %s to %s" % (var.orgName, var.name))
    
            # update any variable-scope attributes
            att_nodelist = var_node.getElementsByTagName('attribute')
            for att_node in att_nodelist :
                self._handle_attribute(att_node, var.name)
        
    def _handle_aggregation(self, agg_node) :
        """Process a single aggregation node"""
        agg = NcmlElement(agg_node)
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
        self.logger.info("Processing union aggregation node...")
        nc_nodes = agg_elem.node.getElementsByTagName('netcdf')
        scan_nodes = agg_elem.node.getElementsByTagName('scan')

        # Process any <netcdf> nodes.
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_agg_netcdf_node(node)

        # Process any <scan> nodes.
        if len(scan_nodes) :
            for node in scan_nodes :
                self._handle_agg_scan_node(node)

    def _handle_joinexisting(self, agg_elem) :
        """
        Process a joinExisting aggregation node. From the NcML spec:
            A JoinExisting dataset is constructed by transferring objects (dimensions, attributes,
            groups, and variables) from the nested datasets in the order the nested datasets are
            listed. All variables that use the aggregation dimension as their outer dimension are
            logically concatenated, in the order of the nested datasets. Variables that don't use
            the aggregation dimension are treated as in a Union dataset, i.e. skipped if one with
            that name already exists.
        """
        self.logger.info("Processing joinExisting aggregation node...")
        if not agg_elem.dimName :
            errmsg = "<aggregation> elements of type 'joinExisting' must include a 'dimName' attribute."
            raise NcmlSyntaxError(errmsg)

        nc_nodes = agg_elem.node.getElementsByTagName('netcdf')
        scan_nodes = agg_elem.node.getElementsByTagName('scan')

        # Process any <netcdf> nodes.
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_agg_netcdf_node(node, agg_type="joinExisting")

        # Process any <scan> nodes.
        if len(scan_nodes) :
            for node in scan_nodes :
                self._handle_agg_scan_node(node, agg_type="joinExisting")

        # Get distinct variable names from the raw loaded cubelist.
        distinct_var_names = set([c.name() for c in self.cubelist])

        # Create an empty cubelist to hold the joined cubes.
        joined_cubes = iris.cube.CubeList()
        before = len(self.cubelist)
        
        # For each distinct variable, attempt to perform a join operation on those variables which
        # are (i) associated with 2 or more cubes, and (ii) reference the aggregation dimension.
        # All other variables are simply copied over as-is.
        for var_name in distinct_var_names :
            var_cubes = [c for c in self.cubelist if c.name() == var_name]
            if len(var_cubes) > 1 and var_cubes[0].coords(name=agg_elem.dimName, dim_coords=True) :
                try :
                    tmp_cube = _concat_cubes(var_cubes, agg_elem.dimName)
                    joined_cubes.append(tmp_cube)
                    for cube in var_cubes : del cube
                except Exception, exc :
                    self.logger.error("Unable to join cubes associated with variable "+var_name)
                    self.logger.error(str(exc))
                    raise exc
            else :
                joined_cubes.extend(var_cubes)

        # Advertise the results.
        after = len(joined_cubes)
        if before != after :
            self.cubelist = joined_cubes
            msg = "Reduced %d input cubes down to %d after join along %s dimension" \
                % (before, after, agg_elem.dimName)
            self.logger.info(msg)
        else :
            msg = "Unable to join any cubes along %s dimension" % agg_elem.dimName
            self.logger.info(msg)

    def _handle_joinnew(self, agg_elem) :
        """Process a joinNew aggregation node"""
        raise iris.exceptions.NotYetImplementedError()

    def _handle_agg_netcdf_node(self, node, agg_type="union") :
        """Process an aggregation netcdf node"""
        netcdf = NcmlElement(node)
        if not netcdf.location :
            raise NcmlSyntaxError("<netcdf> elements must include a 'location' attribute.")
        ncpath = netcdf.location
        if not ncpath.startswith('/') : ncpath = os.path.join(self.basedir, ncpath)

        # load cubes from the specified netcdf file
        self.logger.info("Scanning netCDF file '%s'..." % ncpath)
        cubelist = iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback)

        # store the filename and cubelist associated with this <netcdf> node
        self.part_filenames.append(ncpath)
        #self.part_cubelists.append(cubelist)
        self._extend_cubelist(cubelist, unique_names=(agg_type=="union"))

    def _handle_agg_scan_node(self, node, agg_type="union") :
        """Process an aggregation scan node"""
        scan = NcmlElement(node)
        if not scan.location :
            raise NcmlSyntaxError("<scan> elements must include a 'location' attribute.")

        topdir = scan.location
        if not topdir.startswith('/') : topdir = os.path.join(self.basedir, topdir)
        if not scan.subdirs : scan.subdirs = 'true'
        subdirs = scan.subdirs in ('true', 'True', '1') 
        regex = scan.regExp
        min_age = scan.olderThan
        if min_age : min_age = _conv_time_string_to_interval(min_age)

        # if no regex was defined, use suffix if that was defined
        if not regex and scan.suffix : regex = '.*' + scan.suffix + '$'

        # walk directory tree from topdir searching for files matching regex
        # load cubes from each matching netcdf file
        for ncpath in _walk_dir_tree(topdir, recurse=subdirs, regex=regex, min_age=min_age) :
            self.logger.info("Scanning netCDF file '%s'..." % ncpath)
            self.part_filenames.append(ncpath)
            cubelist = iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback)
            #self.part_cubelists.append(cubelist)
            self._extend_cubelist(cubelist, unique_names=(agg_type=="union"))

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
        
        # for variables remove the corresponding cube, if present
        if obj_type == 'variable' :
            cube = self._get_cube_by_var_name(obj_name)
            if cube:
                self.cubelist.remove(cube)
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
                cube = self._get_cube_by_var_name(var_name)
                if cube:
                    cube.attributes.pop(obj_name, None)
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
            obj_name = node.getAttribute('name')
            obj_type = node.getAttribute('type')
            if not (obj_name and obj_type) :
                self.logger.warn("<remove> elements must include a 'name' and 'type' attribute.")
                continue
            if obj_type not in ('attribute', 'variable') :
                self.logger.warn("Can only remove elements of type attribute or variable")
                continue
            parent_type = node.parentNode.tagName
            if parent_type == 'netcdf' :
                parent_name = 'netcdf'
            else :
                parent_name = node.parentNode.getAttribute('name')
            # TODO: consider storing <remove> elements using NcmlElement class
            d = dict(name=obj_name, type=obj_type, parent_name=parent_name, parent_type=parent_type)
            self.removelist.append(d)

    def _process_remove_list(self) :
        """Process items in the remove list."""
        for item in self.removelist :
            if item['type'] == 'variable' :   # variables should not have made it into the cubelist
                continue
            elif item['type'] == 'attribute' :
                att_name = item['name']
                if item['parent_type'] == 'netcdf' :   # global attribute
                    for cube in self.cubelist :
                        cube.attributes.pop(att_name, None)
                    self.logger.info("Removed attribute %s from all cubes" % att_name)
                elif item['parent_type'] == 'variable' :   # variable-scope attribute
                    var_name = item['parent_name']
                    cube = self._get_cube_by_var_name(var_name)
                    if cube:
                        cube.attributes.pop(att_name, None)
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

    def _add_coord_variable(self, var_elem) :
        """Create a coordinate variable (an iris DimCoord object) from a <variable> element."""
        nodes = var_elem.node.getElementsByTagName('values')
        if not nodes :
            raise NcmlSyntaxError("A <values> element is required to create a coordinate variable")

        val_node = nodes[0]
        values = NcmlElement(val_node)

        # compute or retrieve coordinate values
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
            points = _cast_data_value(text_values, var_elem.type, sep=values.separator)

        # set any keyword arguments from nested <attribute> elements
        kw = dict(standard_name=None, long_name=None, units='1')
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist :
            name = node.getAttribute('name')
            if name in kw : kw[name] = node.getAttribute('value')
        
        # create and store the DimCoord object for subsequent use
        dim_coord = iris.coords.DimCoord(np.array(points), **kw)
        dim_coord.var_name = var_elem.name
        self.dim_coords.append(dim_coord)

    def _add_data_variable(self, var_elem) :
        """Create a data variable (an iris Cube) from a <variable> element."""
        nodes = var_elem.node.getElementsByTagName('values')
        if not nodes :
            raise NcmlSyntaxError("A <values> element is required to create a coordinate variable")

        val_node = nodes[0]
        values = NcmlElement(val_node)

        # compute or retrieve coordinate values
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
            points = _cast_data_value(text_values, var_elem.type, sep=values.separator)

        # set any keyword arguments from nested <attribute> elements
        kw = dict(standard_name=None, long_name=None, units='1', cell_methods=None)
        att_nodelist = var_elem.node.getElementsByTagName('attribute')
        for node in att_nodelist :
            name = node.getAttribute('name')
            if name in kw : kw[name] = node.getAttribute('value')

        # for a dimensioned (non-scalar) variable
        if var_elem.shape :
            shape = []
            coords = []
            # create coordinate list for this variable
            # FIXME: I reckon this may be a somewhat brittle approach
            all_coords = self.dim_coords[:]
            all_coords.extend(_get_coord_list(self.cubelist))
            dimnum = 0
            for i,dim_name in enumerate(var_elem.shape.split()) :
                for dim_coord in all_coords :
                    if dim_name == dim_coord.var_name or dim_name == dim_coord.standard_name :
                        coords.append([dim_coord,dimnum])
                        shape.append(len(dim_coord.points))
                        dimnum += 1
                        break

        # for a scalar variable we just need to set the array shape
        else :
            shape = [1]
            coords = []
            
        # create a cube object and add to self's cubelist
        data = np.array(points)
        data.shape = shape
        cube = iris.cube.Cube(data, dim_coords_and_dims=coords, **kw)
        cube.var_name = var_elem.name
        self.cubelist.append(cube)
        self.logger.debug("- cube min value: %s" % cube.data.min())
        self.logger.debug("- cube max value: %s" % cube.data.max())

    def _extend_cubelist(self, cubelist, unique_names=False) :
        """Append distinct cubes from cubelist to the NcMLDataset's current cubelist."""
        for cube in cubelist :
            # skip variable if it's flagged for removal
            if self._is_flagged_for_removal(cube) :
                self.logger.info("Skipped netCDF variable %s (reason: target of <remove> element)" % cube.var_name)
            # skip variable if it's already present
            elif cube.name() in self.get_cube_names() and unique_names :
                self.logger.info("Skipped netCDF variable %s (reason: loaded from earlier file)" % cube.var_name)
            else :
                self.cubelist.append(cube)
                self.logger.info("Added netCDF variable %s" % cube.var_name)

    def _update_cube_attributes(self, attrs) :
        """Override any cubes attributes with those specified in the attrs dict argument."""
        for cube in self.get_cubes() :
            if not hasattr(cube, 'attributes') : cube.attributes = dict()
            cube.attributes.update(attrs)

    def _get_cube_by_var_name(self, var_name) :
        """Return the loaded cube with the specified variable name."""
        for cube in self.cubelist :
            if cube.var_name == var_name : return cube
        return None

def load_cubes(filenames, callback=None, **kwargs) :
    """
    Generator function returning a sequence of cube objects associated with the netCDF files
    specified within an NcML file. The current implementation can only handle a single NcML file.
    """
    # Update logger object if necessary
    log_level = kwargs.pop('log_level')
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
        print str(exc)
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
    return ''.join(text)

def _get_coord_list(cubelist) :
    """
    Return a list of coordinate objects (DimCoords, AuxCoords) associated with the specified
    cubelist.
    """
    # FIXME: this function would probably be better recast as a generator
    coord_list = []
    for cube in cubelist :
        for coord in cube.coords() :
            if coord not in coord_list : coord_list.append(coord)
    return coord_list

def _concat_cubes(cubelist, dim_name) :
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

    # Create a new DimCoord object from dim_coord_list, which has just been sorted.
    agg_dim_coord = _concat_dim_coords(dim_coord_list, sort=False)
    logger.debug("Shape of aggregated coordinate data: %s" % str(agg_dim_coord.points.shape))

    # Concatenate all the data arrays from the input cubes.
    # Is this the most efficient idiom? Preferable to np.concatenate I think.
    data = cubelist[0].data
    for c in cubelist[1:] :
        data = np.append(data, c.data, axis=0)
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
    # copy over all auxiliary coordinates
    for coord in cube0.aux_coords :
        dims = cube0.coord_dims(coord)
        newcube.add_aux_coord(coord, dims)

    return newcube

def _concat_dim_coords(coord_list, sort=True, reverse=False) :
    """
    Concatenate a set of similar DimCoord objects into a single DimCoord object. Assumes that all
    of the input coord objects are identical except for their points and bounds arrays. By default
    the DimCoords are concatenated in ascending order of their first point. Set the reverse keyword
    to True to concatenate in descending order. Set the sort keyword to False to disable sorting.
    """
    if sort : coord_list.sort(_coord_sorter, reverse=reverse)
    points = np.concatenate([c.points for c in coord_list], axis=0)
    if coord_list[0].has_bounds() :
        bounds = np.concatenate([c.bounds for c in coord_list], axis=0)
    else :
        bounds = None
    dcoord = coord_list[0].copy(points, bounds)
    return dcoord

def _coord_sorter(coord, other) :
    """Sort two coord objects by comparing the value of their first point."""
    return cmp(coord.points[0], other.points[0])

def _conv_time_string_to_interval(timestr) :
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
        #print "interval (secs):", interval
        return long(interval)
    except :
        raise ValueError("Error trying to decode time period: %s" % timestr)

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
 
def _cast_data_value(value, nctype, sep=None) :
    """
    Cast the specified text value to the NcML/NetCDF data type given by nctype. By default, in
    NcML documents, values in lists are separated by whitespace.
    NOTE: there is no support at present for the 'Structure' data type described in the NcML spec.
    """
    if nctype in ('char', 'string', 'String') :
        return value
    elif nctype == 'byte' :
        vlist = [np.int8(x) for x in value.split(sep)]
    elif nctype == 'short' :
        vlist = [np.int16(x) for x in value.split(sep)]
    elif nctype == 'int' :
        vlist = [np.int32(x) for x in value.split(sep)]
    elif nctype == 'long' :   # synonym for int in NcML/CDL grammar
        vlist = [np.int32(x) for x in value.split(sep)]
    elif nctype == 'float' :
        vlist = [np.float32(x) for x in value.split(sep)]
    elif nctype == 'double' :
        vlist = [np.float64(x) for x in value.split(sep)]
    else :
        raise iris.exceptions.DataLoadError("Unsupported NcML attribute data type: %s" % nctype)

    if len(vlist) == 1 :
        return vlist[0]
    else :
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
