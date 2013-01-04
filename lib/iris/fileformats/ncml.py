"""
Iris i/o module for reading NcML files. Current functionality is limited to the following:

* only handles netcdf files at present
* union-type aggregation based on <netcdf> elements
* union-type aggregation based on <scan> elements

WARNING: THIS CODE IS A WORK IN PROGRESS. IT DOESN'T WORK YET AND IS LIABLE TO CHANGE.
"""
import sys, os, re, fnmatch, logging
from collections import OrderedDict
import iris
import netCDF4 as nc4
from xml.dom.minidom import parse

# default logging options
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_FORMAT = "[%(name)s.%(funcName)s] %(levelname)s: %(message)s"

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
            self.logger.debug("Parsing NcML source document...")
            try :
                ncml_doc = parse(ncml_file)
            except Exception, exc :
                self.logger.error("Error trying to parse NcML document.")
                raise exc
            self._handle_doc(ncml_doc)
            self.logger.debug("Finished parsing. Loaded %d cubes from %d files" % (self.ncubes, self.nparts))
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
        self.logger.debug("Document element name: "+doc.documentElement.tagName)
        # process an aggregation node, if present
        aggregations = doc.getElementsByTagName("aggregation")
        if len(aggregations) : self._handle_aggregation(aggregations[0])
        # process any attribute nodes
        # FIXME: next statement retrieves attribute nodes embedded within variable nodes
        attributes = doc.getElementsByTagName("attribute")
        self._handle_attributes(attributes)
        # process any variable nodes
        variables = doc.getElementsByTagName("variable")
        self._handle_variables(variables)

    def _handle_attributes(self, attributes, var=None) :
        """Process attribute nodes"""
        self.logger.debug("Processing global attributes...")
        for att in attributes :
            self._handle_attribute(att, var)
    
    def _handle_attribute(self, attribute, var=None) :
        """Process single attribute node"""
        att_name = attribute.getAttribute("name")
        att_type = attribute.getAttribute("type") or 'String'
        att_sep  = attribute.getAttribute("separator") or ','
        att_val  = attribute.getAttribute("value")
        if not att_val : att_val = _get_node_text(attribute)
        self.logger.debug("Attribute name: '%s', type: '%s', value: '%s'" % (att_name, att_type, att_val))
        # TODO: add attributes to each cube
        if not var :
            self.attributes[att_name] = _cast_data_value(att_val, att_type, sep=att_sep)

    def _handle_variables(self, variables) :
        """Process variable nodes"""
        self.logger.debug("Processing variable nodes...")
        for var in variables :
            self._handle_variable(var)

    def _handle_variable(self, variable) :
        """Process single variable node"""
        var_name = variable.getAttribute("name")
        var_type = variable.getAttribute("type")
        var_shape  = variable.getAttribute("shape")
        self.logger.debug("Variable name: '%s', type: '%s', shape: '%s'" % (var_name, var_type, var_shape))

    def _handle_aggregation(self, aggregation) :
        """Process aggregation node"""
        agg_type = aggregation.getAttribute("type")
        if agg_type == "union" :
            self._handle_union(aggregation)
        elif agg_type == "joinNew" :
            self._handle_joinnew(aggregation)
        elif agg_type == "joinExisting" :
            self._handle_joinexisting(aggregation)

    def _handle_union(self, aggregation) :
        """Process union aggregation node"""
        self.logger.debug("Processing union aggregation node...")
        nc_nodes = aggregation.getElementsByTagName("netcdf")
        scan_nodes = aggregation.getElementsByTagName("scan")
        # process <netcdf> nodes
        if len(nc_nodes) :
            for node in nc_nodes :
                self._handle_union_netcdf_node(node)
        # process <scan> nodes
        elif len(scan_nodes) :
            for node in scan_nodes :
                self._handle_union_scan_node(node)

    def _handle_union_netcdf_node(self, node) :
        """Process union aggregation netcdf node"""
        ncpath = node.getAttribute("location")
        if not ncpath.startswith('/') : ncpath = os.path.join(self.basedir, ncpath)
        self.logger.debug("Scanning netCDF file '%s'..." % ncpath)
        self.part_filenames.append(ncpath)
        cubelist = iris.fileformats.netcdf.load_cubes(ncpath)
        self.part_cubelists.append(cubelist)
        for cube in cubelist :
            if cube.name() in self.get_cube_names() : continue
            self.cubelist.append(cube)
            self.logger.debug("Added netCDF variable '%s'" % cube.name())

    def _handle_union_scan_node(self, node) :
        """Process union aggregation scan node"""
        att_dict = self._get_scan_node_attrs(node)
        topdir = att_dict.get('location', '')
        if not topdir.startswith('/') : topdir = os.path.join(self.basedir, topdir)
        subdirs = att_dict.get('subdirs', 'true') in ('true', 'True', '1') 
        suffix = att_dict.get('suffix', '')
        regex = att_dict.get('regExp', '')   # regExp takes precedence over suffix
        if not regex and suffix : regex = '.*' + suffix + "$"
        # walk directory tree from topdir
        for ncpath in _walk_dir_tree(topdir, recurse=subdirs, regex=regex) :
            self.logger.debug("Scanning netCDF file '%s'..." % ncpath)
            self.part_filenames.append(ncpath)
            cubelist = iris.fileformats.netcdf.load_cubes(ncpath)
            self.part_cubelists.append(cubelist)
            for cube in cubelist :
                if cube.name() in self.get_cube_names() : continue
                self.cubelist.append(cube)
                self.logger.debug("Added netCDF variable '%s'" % cube.name())

    def _handle_joinnew(self, aggregation) :
        """Process joinNew aggregation node"""
        # raise iris.exceptions.NotYetImplementedError()
        pass

    def _handle_joinexisting(self, aggregation) :
        """Process joniExisting aggregation node"""
        # raise iris.exceptions.NotYetImplementedError()
        pass

    def _update_cube_attributes(self) :
        """Override any cubes attributes with those specified in the NcML file."""
        # global attributes 
        for cube in self.get_cubes() :
            if not hasattr(cube, 'attributes') : cube.attributes = dict()
            cube.attributes.update(self.attributes)

    def _get_scan_node_attrs(self, node) :
        att_names = ('location', 'suffix', 'regExp', 'subdirs', 'olderThan', 'dateFormatMark', 'enhance')
        att_dict = dict()
        for name in att_names :
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

def load_cubes(filenames, callback=None) :
    """
    Generator function returning a sequence of cube objects associated with the netCDF files
    specified within an NcML file. The current implementation can only handle a single NcML file.
    """
    if isinstance(filenames, (list,tuple)) :
        if len(filenames) > 1 :
            errmsg = "Iris can only read a single NcML file; %d filenames specified."%len(filenames)
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
        ncml_dataset = NcmlDataset(ncml_file, log_level=logging.DEBUG)
    except Exception, exc :
        print str(exc)
        raise iris.exceptions.DataLoadError("Error trying to load NcML dataset")
    
    for cube in ncml_dataset.get_cubes() :
        yield cube

def _get_node_text(node) :
    """Retrieve the character data from an XML node."""
    rc = []
    for child in node.childNodes :
        if child.nodeType == child.TEXT_NODE :
            rc.append(child.data)
    return ''.join(rc)

def _walk_dir_tree(topdir, recurse=True, regex=None, sort=False) :
    """
    Walk the directory tree rooted at topdir, yielding filenames which match the optional
    regular expression pattern specified in the regex argument.
    """ 
    if regex : reobj = re.compile(regex)
    for curpath, subdirs, filenames in os.walk(topdir) :
        if sort : filenames.sort()
        for fn in filenames :
            if not regex or (regex and reobj.match(fn)) :
                yield os.path.join(curpath, fn)
        if not recurse : break
 
def _cast_data_value(value, nctype, sep=",") :
    """
    Cast the specified text value to the NcML/NetCDF data type given by nctype.
    NOTE: there is no support at present for the 'Structure' data type mentioned in the NcML spec.
    """
    # TODO: we may want to cast the return values to numpy data types rather than the python data
    #       types currently used
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
    elif nctype == 'long' :
        if sep in value :
            return [long(x) for x in value.split(sep)]
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
    cubes = load_cubes(ncml_filename)
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
