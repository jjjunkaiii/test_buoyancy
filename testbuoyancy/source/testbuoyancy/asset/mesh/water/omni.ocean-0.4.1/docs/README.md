# Ocean Deformer

### Compute graph node animating mesh vertices according to deep water waves.  Written in WARP language for high performance.    

How to use this compute node in Create:

1. Make sure to enable this node extension in the extension manager (along with the WARP extension).
2. Add a mesh you would like to get deformed to the scene.
3. Duplicate that mesh (right click -> duplicate).
4. In the compute graph editor:
   - Create nodes for both meshes (original and duplicate)
   - Create read/write nodes 
   - Create nodes for this extension and a "time"-node
   - Connect the nodes as follows:
     - Original mesh to "read node" and further to "input points" of this node  
     - Duplicate mesh to "write node" and further to "output points" of this node 
     - time to "animation time" of this node
5. Hit play.
6. Parameters and common values explained
   - Smallest/Largest wavelengths (1 m to 500 m wavelengths for full ocean) 
   - Windspeed: U_10 windspeed (0..32 m/s) 
   - Water depth: determines the amplitude of the largest waves (20 m)
   - Directionality: dimensionless, adjust to taste (5.0)
   - Direction: Angle of directional waves, adjust to taste 
   - Wave amplitude: Factor for wave amplitudes (1.0)
   - Wave scale: Horizontal scaling of the waves (1.0)

A working oceanExample.usd file can be found in the "data" subfolder.

Here is a video showing the basic steps of the above process (using an older version):
https://nvidia.slack.com/files/U2VJFNAAD/F026WLQTCUQ/OceanDeformer.mp4
