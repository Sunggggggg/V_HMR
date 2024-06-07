import numpy as np

def build_verts_joints_relation(joints, vertices):
    '''
    get the nearest joints of every vertex
    joints : [17, 3] / vertices[431, 3]
    '''
    vertix_num = vertices.shape[0]
    joints_num = joints.shape[0]
    nearest_relation = np.zeros((vertix_num))
    jv_sets = {}
    for (idx, v) in enumerate(vertices):
        nst_joint = v - joints
        nst_joint = nst_joint ** 2
        nst_joint = nst_joint.sum(1)
        nst_joint = np.argmin(nst_joint)
        nearest_relation[idx] = nst_joint
        if nst_joint not in jv_sets:
            jv_sets[nst_joint] = [idx]
        else:
            jv_sets[nst_joint].append(idx)
            
    return nearest_relation, jv_sets