import numpy as np
import sys

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    print("angle: ", angle)

    if angle>0 and angle<np.pi/2:
        angle = angle
    elif angle>np.pi/2 and angle<np.pi:
        angle = np.pi-angle
    elif angle>np.pi and angle<3*np.pi/2:
        angle =  angle-np.pi
    elif angle>3*np.pi/2 and angle<2*np.pi:
        angle = 2*np.pi-angle
    elif angle==0:
        angle = 0
    elif angle==np.pi:
        angle = np.pi
    else:
        sys.exit("Error in angle_between function")

    return angle


print(angle_between((1, 1, 0), (-1,-1.5, 0)))
