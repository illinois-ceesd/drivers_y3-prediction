import numpy as np
import sys
import meshio


def check(V, E):
    """Check the orientation of each simplex in the mesh.

    Parameters
    ----------
    V : ndarray
        n x 2 or n x 3 list of coordinates
    E : ndarray
        n x 3 or n x 4 list of vertices
    """

    m = E.shape[1]            # dim = m - 1
    G = np.zeros((m-1, m-1))  # shape matrix
    D = np.zeros(E.shape[0])  # determinants

    print(f'Checking {m-1}D mesh')
    for j, el in enumerate(E):
        G[:,:] = V[el[1:],:m-1] - V[el[0],:m-1]
        D[j] = np.linalg.det(G)

    Ipos = np.where(D > 0)[0]
    Ineg = np.where(D < 0)[0]
    Izer = np.where(np.abs(D) < 1e-14)[0]
    return Ipos, Ineg, Izer


def test_check():
    # 3D (pos)
    V = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    E = np.array([[0, 1, 2, 3]])
    Ipos, Ineg, Izer = check(V, E)
    assert len(Ipos) == 1
    assert len(Ineg) == 0
    assert len(Izer) == 0

    # (neg)
    E = np.array([[1, 0, 2, 3]])
    Ipos, Ineg, Izer = check(V, E)
    assert len(Ipos) == 0
    assert len(Ineg) == 1
    assert len(Izer) == 0

    # 2D
    # CCW (positive)
    V = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
    E = np.array([[0, 1, 2]])
    Ipos, Ineg, Izer = check(V, E)
    assert len(Ipos) == 1
    assert len(Ineg) == 0
    assert len(Izer) == 0

    # CW (negative)
    E = np.array([[0, 2, 1]])
    Ipos, Ineg, Izer = check(V, E)
    assert len(Ipos) == 0
    assert len(Ineg) == 1
    assert len(Izer) == 0

    mesh = meshio.read('testmesh.msh')
    V = mesh.points
    E = mesh.cells_dict['triangle']
    Ipos, Ineg, Izer = check(V, E)
    assert len(Ipos) == 1
    assert len(Ineg) == 1
    assert len(Izer) == 0


if __name__ == '__main__':
    """
    python check.py mymesh.msh
    """
    mname = sys.argv[1]
    mesh = meshio.read(mname)
    V = mesh.points
    if 'tetra' in mesh.cells_dict:
        E = mesh.cells_dict['tetra']
    else:
        E = mesh.cells_dict['triangle']

    Ipos, Ineg, Izer = check(V, E)
    print(f'found {len(Ipos)} positive  elements')
    print(f'found {len(Ineg)} negative  elements')
    print(f'found {len(Izer)} co-planar elements')
