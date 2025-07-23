"""Support for simplified access to data on nodes of type omni.ocean.OceanDeformer

Mesh deformer modeling ocean waves.
"""

import omni.graph.core as og
import omni.graph.core._omni_graph_core as _og
import omni.graph.tools.ogn as ogn
import numpy
import sys
import traceback
class OceanDeformerDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.ocean.OceanDeformer

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.antiAlias
            inputs.cameraPos
            inputs.direction
            inputs.directionality
            inputs.points
            inputs.scale
            inputs.time
            inputs.waterDepth
            inputs.waveAmplitude
            inputs.windSpeed
        Outputs:
            outputs.points
    """
    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}
    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata,
    #     Is_Required, DefaultValue, Is_Deprecated, DeprecationMsg
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:antiAlias', 'bool', 0, 'Anti aliasing', 'damp distant waves at grazing view angles to reduce spatial aliasing', {ogn.MetadataKeys.DEFAULT: 'false'}, True, False, False, ''),
        ('inputs:cameraPos', 'vector3f', 0, 'Camera position', 'input camera position', {ogn.MetadataKeys.DEFAULT: '[0.0, 0.0, 0.0]'}, True, [0.0, 0.0, 0.0], False, ''),
        ('inputs:direction', 'float', 0, 'Direction', 'wave direction', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('inputs:directionality', 'float', 0, 'directionality', 'isotropic vs directional wave motion', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('inputs:points', 'point3f[]', 0, 'input points', 'input points', {ogn.MetadataKeys.MEMORY_TYPE: 'cuda', ogn.MetadataKeys.DEFAULT: '[]'}, True, [], False, ''),
        ('inputs:scale', 'float', 0, 'Horizontal wave scale', 'Horizontal wave scaling, can be used to match scene units.', {ogn.MetadataKeys.DEFAULT: '1.0'}, True, 1.0, False, ''),
        ('inputs:time', 'double', 0, 'Animation time', 'animation time', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('inputs:waterDepth', 'float', 0, 'Water depth', 'water depth (1..1000 m)', {ogn.MetadataKeys.DEFAULT: '50.0'}, True, 50.0, False, ''),
        ('inputs:waveAmplitude', 'float', 0, 'Wave amplitude', 'Wave amplitude factor.', {ogn.MetadataKeys.DEFAULT: '1.0'}, True, 1.0, False, ''),
        ('inputs:windSpeed', 'float', 0, 'Wind speed', 'wind speed (m/s) 10m above ocean (0..30)', {ogn.MetadataKeys.DEFAULT: '10.0'}, True, 10.0, False, ''),
        ('outputs:points', 'point3f[]', 0, 'output points', 'output points (must match input point array size)', {ogn.MetadataKeys.MEMORY_TYPE: 'cuda', ogn.MetadataKeys.DEFAULT: '[]'}, True, [], False, ''),
    ])
    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.inputs.cameraPos = og.Database.ROLE_VECTOR
        role_data.inputs.points = og.Database.ROLE_POINT
        role_data.outputs.points = og.Database.ROLE_POINT
        return role_data
    class ValuesForInputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"antiAlias", "cameraPos", "direction", "directionality", "scale", "time", "waterDepth", "waveAmplitude", "windSpeed", "_setting_locked", "_batchedReadAttributes", "_batchedReadValues"}
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self._batchedReadAttributes = [self._attributes.antiAlias, self._attributes.cameraPos, self._attributes.direction, self._attributes.directionality, self._attributes.scale, self._attributes.time, self._attributes.waterDepth, self._attributes.waveAmplitude, self._attributes.windSpeed]
            self._batchedReadValues = [False, [0.0, 0.0, 0.0], 0, 0, 1.0, 0, 50.0, 1.0, 10.0]

        @property
        def points(self):
            data_view = og.AttributeValueHelper(self._attributes.points)
            data_view.gpu_ptr_kind = og.PtrToPtrKind.CPU
            return data_view.get(on_gpu=True)

        @points.setter
        def points(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.points)
            data_view = og.AttributeValueHelper(self._attributes.points)
            data_view.gpu_ptr_kind = og.PtrToPtrKind.CPU
            data_view.set(value, on_gpu=True)
            self.points_size = data_view.get_array_size()

        @property
        def antiAlias(self):
            return self._batchedReadValues[0]

        @antiAlias.setter
        def antiAlias(self, value):
            self._batchedReadValues[0] = value

        @property
        def cameraPos(self):
            return self._batchedReadValues[1]

        @cameraPos.setter
        def cameraPos(self, value):
            self._batchedReadValues[1] = value

        @property
        def direction(self):
            return self._batchedReadValues[2]

        @direction.setter
        def direction(self, value):
            self._batchedReadValues[2] = value

        @property
        def directionality(self):
            return self._batchedReadValues[3]

        @directionality.setter
        def directionality(self, value):
            self._batchedReadValues[3] = value

        @property
        def scale(self):
            return self._batchedReadValues[4]

        @scale.setter
        def scale(self, value):
            self._batchedReadValues[4] = value

        @property
        def time(self):
            return self._batchedReadValues[5]

        @time.setter
        def time(self, value):
            self._batchedReadValues[5] = value

        @property
        def waterDepth(self):
            return self._batchedReadValues[6]

        @waterDepth.setter
        def waterDepth(self, value):
            self._batchedReadValues[6] = value

        @property
        def waveAmplitude(self):
            return self._batchedReadValues[7]

        @waveAmplitude.setter
        def waveAmplitude(self, value):
            self._batchedReadValues[7] = value

        @property
        def windSpeed(self):
            return self._batchedReadValues[8]

        @windSpeed.setter
        def windSpeed(self, value):
            self._batchedReadValues[8] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _prefetch(self):
            readAttributes = self._batchedReadAttributes
            newValues = _og._prefetch_input_attributes_data(readAttributes)
            if len(readAttributes) == len(newValues):
                self._batchedReadValues = newValues
    class ValuesForOutputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = { }
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self.points_size = 0
            self._batchedWriteValues = { }

        @property
        def points(self):
            data_view = og.AttributeValueHelper(self._attributes.points)
            data_view.gpu_ptr_kind = og.PtrToPtrKind.CPU
            return data_view.get(reserved_element_count=self.points_size, on_gpu=True)

        @points.setter
        def points(self, value):
            data_view = og.AttributeValueHelper(self._attributes.points)
            data_view.gpu_ptr_kind = og.PtrToPtrKind.CPU
            data_view.set(value, on_gpu=True)
            self.points_size = data_view.get_array_size()

        def _commit(self):
            _og._commit_output_attributes_data(self._batchedWriteValues)
            self._batchedWriteValues = { }
    class ValuesForState(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to state attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
    def __init__(self, node):
        super().__init__(node)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
        self.inputs = OceanDeformerDatabase.ValuesForInputs(node, self.attributes.inputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        self.outputs = OceanDeformerDatabase.ValuesForOutputs(node, self.attributes.outputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE)
        self.state = OceanDeformerDatabase.ValuesForState(node, self.attributes.state, dynamic_attributes)
    class abi:
        """Class defining the ABI interface for the node type"""
        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):
                return get_node_type_function()
            return 'omni.ocean.OceanDeformer'
        @staticmethod
        def compute(context, node):
            try:
                per_node_data = OceanDeformerDatabase.PER_NODE_DATA[node.node_id()]
                db = per_node_data.get('_db')
                if db is None:
                    db = OceanDeformerDatabase(node)
                    per_node_data['_db'] = db
            except:
                db = OceanDeformerDatabase(node)

            try:
                compute_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:
                    return compute_function(context, node)

                db.inputs._prefetch()
                db.inputs._setting_locked = True
                with og.in_compute():
                    return OceanDeformerDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:
                stack_trace = "".join(traceback.format_tb(sys.exc_info()[2].tb_next))
                db.log_error(f'Assertion raised in compute - {error}\n{stack_trace}', add_context=False)
            finally:
                db.inputs._setting_locked = False
                db.outputs._commit()
            return False
        @staticmethod
        def initialize(context, node):
            OceanDeformerDatabase._initialize_per_node_data(node)
            initialize_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):
                initialize_function(context, node)
        @staticmethod
        def release(node):
            release_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):
                release_function(node)
            OceanDeformerDatabase._release_per_node_data(node)
        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):
                return update_node_version_function(context, node, old_version, new_version)
            return False
        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata(ogn.MetadataKeys.EXTENSION, "omni.ocean")
                node_type.set_metadata(ogn.MetadataKeys.UI_NAME, "Ocean Deformer")
                node_type.set_metadata(ogn.MetadataKeys.DESCRIPTION, "Mesh deformer modeling ocean waves.")
                node_type.set_metadata(ogn.MetadataKeys.LANGUAGE, "Python")
                OceanDeformerDatabase.INTERFACE.add_to_node_type(node_type)
                node_type.set_has_state(True)
        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OceanDeformerDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):
                on_connection_type_resolve_function(node)
    NODE_TYPE_CLASS = None
    GENERATOR_VERSION = (1, 17, 2)
    TARGET_VERSION = (2, 65, 4)
    @staticmethod
    def register(node_type_class):
        OceanDeformerDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(OceanDeformerDatabase.abi, 1)
    @staticmethod
    def deregister():
        og.deregister_node_type("omni.ocean.OceanDeformer")
