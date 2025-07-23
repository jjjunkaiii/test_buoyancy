"""
This is the implementation of the OGN node defined in Groundgrid.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import numpy as np
import omni.ext
import math
import warp as wp
import ctypes


@wp.kernel
def update_points(out_points: wp.array(dtype=wp.vec3),
                 in_points: wp.array(dtype=wp.vec3),
                 camPosX: float,
                 camPosY: float,
                 camPosZ: float,
                 minHeight: float,
                 heightBias: float):

    tid = wp.tid()
    p_in = in_points[tid]
    
    # map maximum(||x||,||z||) coordinate to horizon
    #l = wp.max(wp.abs(p_in[0]), wp.abs(p_in[2]))
    #alpha = 0.49 * 3.14159265359 * l
    l = wp.max( 0.001, wp.length( wp.vec2(p_in[0], p_in[2]) ) )
    alpha = 0.499 * 3.14159265359 * l
    
    yDist = wp.max(minHeight, wp.abs(camPosY - heightBias - p_in[1]))
    posX = yDist * wp.tan(alpha) * p_in[0] / l + camPosX
    posY = heightBias
    posZ = yDist * wp.tan(alpha) * p_in[2] / l + camPosZ
    outP = wp.vec3(posX, posY, posZ)
    wp.store(out_points, tid, outP)



class GroundgridInternalState:
    """Convenience class for maintaining per-node state information"""
    def __init__(self):
        self.initialized = False



class Groundgrid:
    """
         Mesh deformer projecting onto the scene ground.
    """
    
    @staticmethod
    def internal_state():
        """Returns an object that will contain per-node state information"""
        return GroundgridInternalState()
     
    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""

        state = db.internal_state

        with wp.ScopedDevice("cuda:0"):

            # initialization state
            if state.initialized is False:
                # we keep a GPU copy of the input points for performance reasons
                state.initialized = True

            # Parameters given by user
            campos = db.inputs.cameraPos
            minHeight = max(0.01, min(100000.0, float(db.inputs.minHeight)))
            heightbias = max(-100000.0, min(100000.0, float(db.inputs.heightBias)))

            # load input and output points and check if they are present
            in_points = omni.warp.from_omni_graph(db.inputs.points, dtype=wp.vec3)
            if (in_points.shape[0] == 0):
                return True
            db.outputs.points_size = in_points.shape[0]
            out_points = omni.warp.from_omni_graph(db.outputs.points, dtype=wp.vec3) 
            # displace point positions by projecting them onto ground plane around the camera
            wp.launch(
                kernel=update_points,
                dim=in_points.shape[0], 
                inputs=[
                    out_points, in_points, 
                    float(campos[0]), float(campos[1]), float(campos[2]),
                    float(minHeight),
                    float(heightbias)], 
                outputs=[],
                device="cuda:0")

            return True
