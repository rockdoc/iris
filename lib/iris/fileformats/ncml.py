"""
WARNING: THIS CODE IS A WORK IN PROGRESS. IT DOESN'T WORK YET AND IS LIABLE TO CHANGE.

Iris i/o module for reading NcML files. Current functionality is limited to the following:

* only handles aggregations of netcdf files
* union-type aggregation based on <netcdf> elements
* union-type aggregation based on <scan> elements
* rename or remove variables (= cubes)
* add, override, rename or remove attributes (global or per-variable)
* handle the olderThan attribute on a scan element

Functionality that is expressly excluded:

* modification or removal of dimensions (this would interfere with Iris' handling of dimensions)
* NcML group elements
* forecastModelRun* elements
* the Structure data type
* nested variables

TO DO:

* add a new dimension (as a DimCoord?)
* add a new data variable?
* joinExisting aggregation
* joinNew aggregation
* handle dateFormatMark for joinNew aggregation

"""
import sys, os, re, fnmatch, logging
from datetime import datetime
from collections import OrderedDict
import iris
import netCDF4 as nc4
from xml.dom.minidom import parse

# default logging options
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_FORMAT = "[%(name)s.%(funcName)s] %(levelname)s: %(message)s"

# Define lists of permissible attributes for various NcML tags
att_namelists = {
    'variable': ('name', 'type', 'shape', 'orgName'),
    'attribute': ('name', 'type', 'separator', 'value', 'orgName'),
    'scan': ('location', 'suffix', 'regExp', 'subdirs', 'olderThan', 'dateFormatMark', 'enhance'),
}

class NcmlSyntaxError(iris.exceptions.IrisError) :
    pass

class NcmlDataset(object) :
    """
    Class for representing an aggregate dataset defined in an NcML file. Full details of the
    NcML specification can be found at http://www.unidata.ucar.edu/software/netcdf/ncml/ 
    """

    def __init__(self, file_like, **kwargs) :
        """
        Initialise an NcML dataset instance from a pathname or file-like object.

        :param file_like: The pathname or file object containing the NcML document
        :param log_level: Optional keyword for setting the logging level. Set to one of the levels
            defined by the standard logging module.
        """
        self.filename = None
        self.attributes = dict()
        self.part_filenames = list()
        self.part_cubelists = list()
        self.cubelist = iris.cube.CubeList()
        self.removelist = list()
        self.explicit = False

        # Initialise logger object
        self.log_level = kwargs.pop('log_level', DEFAULT_LOG_LEVEL)
        self._init_logger()

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
            self.logger.info("Finished parsing. Loaded %d cubes from %d files" % (self.ncubes, self.nparts))
        finally :
            if close_after_parse : ncml_file.close()

    def __del__(self) :
        """Perform any clean-up operations, e.g. closing open file handles."""
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
        explicit = doc.getElementsByTagName("explicit")
        if explicit : self.explicit == True

        # create a list of items to remove during or after cube loading
        remove_nodelist= doc.getElementsByTagName("remove")
        if len(remove_nodelist) : self._make_remove_list(remove_nodelist)
        
        # process an <aggregation> node, if present
        agg_nodelist= doc.getElementsByTagName("aggregation")
        if len(agg_nodelist) : self._handle_aggregation(agg_nodelist[0])

        # process any global <attribute> nodes
        att_nodelist = [n for n in doc.getElementsByTagName("attribute") if n.parentNode.tagName == 'netcdf']
        if len(att_nodelist) : self._handle_global_attributes(att_nodelist)

        # process any <variable> nodes
        var_nodelist = doc.getElementsByTagName("variable")
        if len(var_nodelist) : self._handle_variables(var_nodelist)

        # process any items in the remove list
        self._process_remove_list()
        
    def _handle_global_attributes(self, att_nodelist) :
        """Process all attribute nodes"""
        self.logger.info("Processing global attributes...")
        for att_node in att_nodelist :
            self._handle_attribute(att_node)
    
    # TODO: certain attributes are treated as special by iris and cannot be assigned to a cube's
    # attributes dictionary. The list of such attributes is defined in the LimitedAttributeDict
    # class in module iris._cube_coord_common.py. We'll need to skip these.
    
    def _handle_attribute(self, att_node, var_name=None) :
        """Process single attribute node"""
        att_dict = self._get_node_att_dict(att_node, att_namelists['attribute'])
        if not att_dict.get("name") :
            raise NcmlSyntaxError("<attribute> elements must include a 'name' attribute.")

        att_name = att_dict.get("name")
        att_type = att_dict.get("type") or 'String'
        att_sep  = att_dict.get("separator") or ','
        att_val  = att_dict.get("value") or _get_node_text(att_node)

        if var_name :
            # local - override current attribute only for the cube with the specified var name
            cubelist = [c for c in self.cubelist if c.var_name == var_name]
            self.logger.info("Setting attribute %s for variable %s" % (att_name, var_name))
        else :
            # global - override current attribute for all cubes
            cubelist = self.cubelist
            self.logger.info("Setting attribute %s for all variables" % att_name)

        # rename and/or update the attribute in one or all cubes
        for cube in cubelist :
            if att_dict.get('orgName') :
                old_val = cube.attributes.pop(att_dict['orgName'], None)
            cube.attributes[att_name] = _cast_data_value(att_val, att_type, sep=att_sep)

    def _handle_variables(self, var_nodelist) :
        """Process all variable nodes"""
        self.logger.info("Processing variable nodes...")
        for var_node in var_nodelist :
            self._handle_variable(var_node)

    def _handle_variable(self, var_node) :
        """Process a single variable node"""
        att_dict = self._get_node_att_dict(var_node, att_namelists['variable'])
        if not att_dict.get("name") :
            raise NcmlSyntaxError("<variable> elements must include a 'name' attribute.")
        if not att_dict.get("type") :
            raise NcmlSyntaxError("<variable> elements must include a 'type' attribute.")

        var_name = att_dict.get("name")
        var_type = att_dict.get("type")
        var_shape  = att_dict.get("shape")
        self.logger.info("Variable node: name: '%s', type: '%s'" % (var_name, var_type))

        # if necessary, first rename the variable from orgName to name
        if att_dict.get('orgName') :
            cube = self._get_cube_by_var_name(att_dict['orgName'])
            if cube :
                cube.var_name = var_name
                self.logger.info("Renamed variable %s to %s" % (att_dict['orgName'], var_name))

        # update any variable-scope attributes
        att_nodelist = var_node.getElementsByTagName("attribute")
        for att_node in att_nodelist :
            self._handle_attribute(att_node, var_name)
        
    def _handle_aggregation(self, agg_node) :
        """Process a single aggregation node"""
        agg_type = agg_node.getAttribute("type")
        if agg_type == "union" :
            self._handle_union(agg_node)
        elif agg_type == "joinNew" :
            self._handle_joinnew(agg_node)
        elif agg_type == "joinExisting" :
            self._handle_joinexisting(agg_node)

    def _handle_union(self, agg_node) :
        """Process a union aggregation node"""
        self.logger.info("Processing union aggregation node...")
        nc_nodes = agg_node.getElementsByTagName("netcdf")
        scan_nodes = agg_node.getElementsByTagName("scan")

        # process <netcdf> nodes
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_union_netcdf_node(node)

        # process <scan> nodes
        elif len(scan_nodes) :
            for node in scan_nodes :
                self._handle_union_scan_node(node)

    def _handle_union_netcdf_node(self, node) :
        """Process a union aggregation netcdf node"""
        ncpath = node.getAttribute("location")
        if not ncpath.startswith('/') : ncpath = os.path.join(self.basedir, ncpath)

        self.logger.info("Scanning netCDF file '%s'..." % ncpath)
        cubelist = iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback)

        # store the filename and cubelist associated with this <netcdf> node
        self.part_filenames.append(ncpath)
        self.part_cubelists.append(cubelist)
        self._merge_cubelist(cubelist)

    def _handle_union_scan_node(self, node) :
        """Process a union aggregation scan node"""
        att_dict = self._get_node_att_dict(node, att_namelists['scan'])
        if not att_dict.get('location') :
            raise NcmlSyntaxError("The <scan> element must include a 'location' attribute.")

        topdir = att_dict.get('location')
        if not topdir.startswith('/') : topdir = os.path.join(self.basedir, topdir)
        subdirs = att_dict.get('subdirs', 'true') in ('true', 'True', '1') 
        suffix = att_dict.get('suffix')
        regex = att_dict.get('regExp')
        min_age = att_dict.get('olderThan')

        # if no regex was defined, use suffix if that was defined
        if not regex and suffix : regex = '.*' + suffix + "$"

        # walk directory tree from topdir searching for files matching regex
        for ncpath in _walk_dir_tree(topdir, recurse=subdirs, regex=regex, min_age=min_age) :
            self.logger.info("Scanning netCDF file '%s'..." % ncpath)
            self.part_filenames.append(ncpath)
            cubelist = iris.fileformats.netcdf.load_cubes(ncpath, callback=_nc_load_callback)
            self.part_cubelists.append(cubelist)
            self._merge_cubelist(cubelist)

    def _handle_joinnew(self, agg_node) :
        """Process a joinNew aggregation node"""
        # raise iris.exceptions.NotYetImplementedError()
        pass

    def _handle_joinexisting(self, agg_node) :
        """Process a joinExisting aggregation node"""
        # raise iris.exceptions.NotYetImplementedError()
        pass

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
        obj_name = remove_node.getAttribute("name")
        obj_type = remove_node.getAttribute("type")
        if not (obj_name and obj_type) :
            self.logger.warn("<remove> elements must include a 'name' and 'type' attribute.")
            return
        parent_tagname = remove_node.parentNode.tagName
        
        # for variables remove the corresponding cube, if present
        if obj_type == "variable" :
            cube = self._get_cube_by_var_name(obj_name)
            if cube:
                self.cubelist.remove(cube)
                self.logger.info("Removed variable named %s" % obj_name)
            else :
                self.logger.warn("No cube found corresponding to variable named %s" % obj_name)

        # remove an attribute, either from all cubes or from a named cube 
        elif obj_type == "attribute" :
            if parent_tagname == "netcdf" :
                for cube in self.cubelist :
                    cube.attributes.pop(obj_name, None)
                self.logger.info("Removed attribute %s from all variables" % obj_name)
            elif parent_tagname == "variable" :
                var_name = remove_node.parentNode.getAttribute("name")
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
            obj_name = node.getAttribute("name")
            obj_type = node.getAttribute("type")
            if not (obj_name and obj_type) :
                self.logger.warn("<remove> elements must include a 'name' and 'type' attribute.")
                return
            if obj_type not in ("attribute", "variable") :
                self.logger.warn("Can only remove elements of type attribute or variable")
                return
            parent_type = node.parentNode.tagName
            if parent_type == "netcdf" :
                parent_name = "netcdf"
            else :
                parent_name = node.parentNode.getAttribute("name")
            d = dict(name=obj_name, type=obj_type, parent_name=parent_name, parent_type=parent_type)
            self.removelist.append(d)

    def _process_remove_list(self) :
        """Process items in the remove list."""
        for item in self.removelist :
            if item['type'] == "variable" :   # variables should not have made it into the cubelist
                continue
            elif item['type'] == "attribute" :
                att_name = item['name']
                if item['parent_type'] == "netcdf" :   # global attribute
                    for cube in self.cubelist :
                        cube.attributes.pop(att_name, None)
                    self.logger.info("Removed attribute %s from all cubes" % att_name)
                elif item['parent_type'] == "variable" :   # variable-scope attribute
                    var_name = item['parent_name']
                    cube = self._get_cube_by_var_name(var_name)
                    if cube:
                        cube.attributes.pop(att_name, None)
                        self.logger.info("Removed attribute %s from variable %s" % (att_name, var_name))
                    else :
                        self.logger.warn("No cube found corresponding to variable %s" % var_name)

    def _is_flagged_for_removal(self, node_or_cube) :
        """Check to see if an element or cube is flagged for removal via a <remove> element."""
        if isinstance(node_or_cube, iris.cube.Cube) :
            obj_name = node_or_cube.var_name
            obj_type = "variable"
            parent_type = "netcdf"
            parent_name = "netcdf"
        else :
            obj_name = node_or_cube.getAttribute("name")
            obj_type = node_or_cube.getAttribute("type")
            if not (obj_name and obj_type) : return False
            parent_type = node_or_cube.parentNode.tagName
            if parent_type == "netcdf" :
                parent_name = "netcdf"
            else :
                parent_name = node_or_cube.parentNode.getAttribute("name")
        d = dict(name=obj_name, type=obj_type, parent_name=parent_name, parent_type=parent_type)
        return d in self.removelist

    def _merge_cubelist(self, cubelist) :
        """Append distinct cubes from cubelist to the ncml dataset cubelist."""
        for cube in cubelist :
            # skip this variable if it's flagged for removal
            if self._is_flagged_for_removal(cube) :
                self.logger.info("Removed netCDF variable %s" % cube.var_name)
                continue
            # ignore this variable if it's already present
            elif cube.name() in self.get_cube_names() :
                continue
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

    def _get_node_att_dict(self, node, att_namelist) :
        """Return a dictionary of the attributes named in att_namelist for the specified node."""
        att_dict = dict()
        for name in att_namelist :
            value = node.getAttribute(name)
            if value == '' : value = None
            att_dict[name] = value
        return att_dict

    def _init_logger(self) :
        """Initialise a logger object using default or user-supplied settings."""
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(self.log_level)
        fmtr = logging.Formatter(DEFAULT_LOG_FORMAT)
        console.setFormatter(fmtr)
        #self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(console)
        self.logger.setLevel(self.log_level)

def load_cubes(filenames, callback=None, **kwargs) :
    """
    Generator function returning a sequence of cube objects associated with the netCDF files
    specified within an NcML file. The current implementation can only handle a single NcML file.
    """
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

def _nc_load_callback(cube, field, filename) :
    """Callback for adding netCDF variable name as cube attribute."""
    # for netcdf files, field is of type iris.fileformats.cf.CFDataVariable
    if hasattr(field, '_name') :
        cube.var_name = field._name
    else :
        cube.var_name = 'undefined'

def _get_node_text(node) :
    """Retrieve the character data from an XML node."""
    rc = []
    for child in node.childNodes :
        if child.nodeType == child.TEXT_NODE :
            rc.append(child.data)
    return ''.join(rc)

def _is_older_than(filename, interval) :
    """
    Return true if the latest modification time of filename is older than the specified interval
    before present. As per the NcML spec, interval must be a udunits2-compliant time value/unit pair
    e.g. '5 min'."""
    # get modification time of filename
    mtime = datetime.fromtimestamp(os.path.getmtime(filename))
    
    # convert the specified time interval to seconds
    ival,iunit = interval.split()
    iunit = iris.unit.Unit(iunit)
    sunit = iris.unit.Unit('second')
    interval_sec = iunit.convert(float(ival), sunit)
#    print "interval (secs):", interval_sec

    # see if file modification time is older than interval_sec
    age = datetime.now() - mtime
#    print "age (secs):", age.total_seconds()
    return (age.total_seconds() > interval_sec)
    
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
 
def _cast_data_value(value, nctype, sep=",") :
    """
    Cast the specified text value to the NcML/NetCDF data type given by nctype.
    NOTE: there is no support at present for the 'Structure' data type described in the NcML spec.
    """
    # TODO: consider casting return values to numpy data types rather than the python data
    #       types currently used below
    if nctype == 'byte' :
        if sep in value :
            return [int(x) for x in value.split(sep)]
        else :
            return int(value)
    elif nctype == 'short' :
        if sep in value :
            return [int(x) for x in value.split(sep)]
        else :
            return int(value)
    elif nctype == 'int' :
        if sep in value :
            return [int(x) for x in value.split(sep)]
        else :
            return int(value)
    elif nctype == 'long' :   # synonym for int in NcML/CDL grammar
        if sep in value :
            return [int(x) for x in value.split(sep)]
        else :
            return long(value)
    elif nctype == 'float' :
        if sep in value :
            return [float(x) for x in value.split(sep)]
        else :
            return float(value)
    elif nctype == 'double' :
        if sep in value :
            return [float(x) for x in value.split(sep)]
        else :
            return float(value)
    elif nctype in ('char','string','String') :
        return value
    else :
        raise iris.exceptions.DataLoadError("Unsupported NcML attribute data type: %s" % nctype)

def test(ncml_filename) :
    """Rudimentary test function"""
    cubes = load_cubes(ncml_filename, log_level=logging.DEBUG)
    for i,cube in enumerate(cubes) :
        dmin = cube.data.min()
        dmax = cube.data.max()
        print "Cube #%d, std name: %s, shape: %s, min: %f, max: %f" % (i, cube.name(), cube.shape, dmin, dmax)

if __name__ == "__main__" :
    usage = "usage: python ncml.py <ncml_file>"
    if len(sys.argv) < 2 :
        print usage
        sys.exit(1)
    test(sys.argv[1])
