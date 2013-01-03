"""
Iris i/o module for reading NcML files.

WARNING: THIS CODE IS A WORK IN PROGRESS. IT DOESN'T WORK YET AND IS LIABLE TO CHANGE.
"""
import sys
import os
import logging
from collections import OrderedDict
#import iris
import netCDF4 as nc4
from xml.dom.minidom import parse

# default logging options
DEFAULT_LOG_LEVEL  = logging.WARNING
DEFAULT_LOG_FORMAT = "[%(levelname)s] %(funcName)s: %(message)s"

class NcmlDataset(object) :
    """
    Class for representing an aggregate dataset defined in an NcML file. Full details of the
    NcML specification can be found at http://www.unidata.ucar.edu/software/netcdf/ncml/ 
    """

    def __init__(self, file_like, **kwargs) :
        """Initialise an NcML dataset instance from a pathname or file-like object."""
        self.filename = None
        self.log_level = kwargs.pop('log_level', DEFAULT_LOG_LEVEL)
        self.init_logger()
        self.attributes = dict()
        self.part_filenames = list()
        self.part_cubelists = list()

        close_after_parse = False
        if isinstance(file_like, basestring) :
            self.filename = os.path.expandvars(os.path.expanduser(file_like))
            ncml_file = open(self.filename)
            close_after_parse = True
        elif isinstance(file_like, file) :
            ncml_file = file_like
        else :
            raise AttributeError("Invalid file-like argument passed to NcmlDataset constructor")
            
        # Parse the NcML file
        try :
            self.logger.debug("Parsing NcML source document...")
            try :
                ncml_doc = parse(ncml_file)
                self.handle_doc(ncml_doc)
            except Exception, exc :
                self.logger.error("Error trying to parse NcML document")
                self.logger.error(str(exc))
                raise
        finally :
            if close_after_parse : ncml_file.close()

    def __del__(self) :
        """Perform any clean-up operations, e.g. closing open file handles."""
        pass

    @property
    def nparts(self) :
        """Returns the number of component parts comprising the dataset.""" 
        return len(self.part_filenames)

    @property
    def basedir(self) :
        """Returns the base directory of the input NcML file."""
        return os.path.dirname(self.filename)

    def handle_doc(self, doc) :
        self.logger.debug("Document element name: "+doc.documentElement.tagName)
        attributes = doc.getElementsByTagName("attribute")
        self.handle_attributes(attributes)
        variables = doc.getElementsByTagName("variable")
        self.handle_variables(variables)
        aggregations = doc.getElementsByTagName("aggregation")
        if len(aggregations) : self.handle_aggregation(aggregations[0])

    def handle_attributes(self, attributes, var=None) :
        self.logger.debug("Processing global attributes...")
        for att in attributes :
            self.handle_attribute(att, var)
    
    def handle_attribute(self, attribute, var=None) :
        att_name = attribute.getAttribute("name")
        att_type = attribute.getAttribute("type")
        att_val  = attribute.getAttribute("value")
        self.logger.debug("Attribute name: '%s', type: '%s', value: '%s'" % (att_name, att_type, att_val))
        if not var :
            self.attributes[att_name] = att_val

    def handle_variables(self, variables) :
        self.logger.debug("Processing variable elements...")
        for var in variables :
            self.handle_variable(var)

    def handle_variable(self, variable) :
        var_name = variable.getAttribute("name")
        var_type = variable.getAttribute("type")
        var_shape  = variable.getAttribute("shape")
        self.logger.debug("Variable name: '%s', type: '%s', shape: '%s'" % (var_name, var_type, var_shape))

    def handle_aggregation(self, aggregation) :
        #self.logger.debug("Processing aggregation element...")
        agg_type = aggregation.getAttribute("type")
        if agg_type == "union" :
            self.handle_union(aggregation)
        elif agg_type == "joinNew" :
            self.handle_joinnew(aggregation)
        elif agg_type == "joinExisting" :
            self.handle_joinexisting(aggregation)

    def handle_union(self, aggregation) :
        self.logger.debug("Processing union aggregation element...")
        nc_nodes = aggregation.getElementsByTagName("netcdf")
        if len(nc_nodes) :
            for node in nc_nodes :
                loc = node.getAttribute("location")
                if not loc.startswith('/') : loc = os.path.join(self.basedir, loc)
                self.part_filenames.append(loc)
#                cubelist = iris.fileformats.netcdf.load_cubes(loc)
#                self.part_cubelists.append(cubelist)
                self.logger.debug("Loaded netCDF file '%s'" % loc)

    def handle_joinnew(self, aggregation) :
        # raise iris.exceptions.NotYetImplementedError()
        pass

    def handle_joinexisting(self, aggregation) :
        # raise iris.exceptions.NotYetImplementedError()
        pass

    def get_cubes(self) :
        """Return a list of cubes"""
        # TODO
        pass

    def get_text(self, nodelist) :
        rc = []
        for node in nodelist :
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)
    
    def init_logger(self) :
        """Configure a logger object."""
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(self.log_level)
        fmtr = logging.Formatter(DEFAULT_LOG_FORMAT)
        console.setFormatter(fmtr)
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.logger.addHandler(console)
        self.logger.setLevel(self.log_level)

def load_cubes(filenames, callback=None) :
    """
    Generator function returning a sequence of cube objects associated with the netCDF files
    specified within an NcML file. The current implementation can only handle a single NcML file.
    """
    if isinstance(filenames, (list,tuple)) :
        if len(filenames) > 1 :
            # TODO: replace with an Iris-specific exception
            raise IOError("Can only read in a single NcML file, %d filenames specified"%len(filenames))
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
        raise IOError("Error trying to create NcML dataset")
    
    #for cube in ncml_dataset.get_cubes() :
    #    yield cube

# TODO: the code in the following function needs to be copied to the fileformats.__init__.py file
def register_ncml_format() :
    """Create and register a format specification for the NcML file format"""
    ncml_spec = iris.io.format_picker.FormatSpecification(
        'NetCDF Markup Language (NcML)',
        format_picker.FILE_EXTENSION,
        "ncml",
        load_cubes)

    # Register the format with iris
    iris.fileformats.FORMAT_AGENT.add_spec(ncml_spec)

def main(ncml_filename) :
    load_cubes(ncml_filename)

if __name__ == "__main__" :
    usage = "usage: python ncml.py <ncml_file>"
    if len(sys.argv) < 2 :
        print usage
        sys.exit(1)
    main(sys.argv[1])
