import numpy as np
import logging

from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec
from naslib.utils.encodings import EncodingType

"""
These are the encoding methods for nasbench101.
The plan is to unify encodings across all search spaces.
"""

logger = logging.getLogger(__name__)

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def get_paths(spec):
    """
    return all paths from input to output
    """
    matrix, ops = spec["matrix"], spec["ops"]
    paths = []
    for j in range(0, NUM_VERTICES):
        paths.append([[]]) if matrix[0][j] else paths.append([])

    # create paths sequentially
    for i in range(1, NUM_VERTICES - 1):
        for j in range(1, NUM_VERTICES):
            if matrix[i][j]:
                for path in paths[i]:
                    paths[j].append([*path, ops[i]])
    return paths[-1]


def get_path_indices(spec):
    """
    compute the index of each path
    There are 3^0 + ... + 3^5 paths total.
    (Paths can be length 0 to 5, and for each path, for each node, there
    are three choices for the operation.)
    """
    paths = get_paths(spec)
    mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
    path_indices = []

    for path in paths:
        index = 0
        for i in range(NUM_VERTICES - 1):
            if i == len(path):
                path_indices.append(index)
                break
            else:
                index += len(OPS) ** i * (mapping[path[i]] + 1)

    path_indices.sort()
    return tuple(path_indices)


def encode_paths(spec):
    """output one-hot encoding of paths"""
    path_indices = get_path_indices(spec)
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding


def encode_adj(spec):
    """
    compute adjacency matrix + op list encoding
    """
    matrix, ops = spec["matrix"].copy(), spec["ops"].copy()
    op_dict = {CONV1X1: [0, 0, 1], CONV3X3: [0, 1, 0], MAXPOOL3X3: [1, 0, 0], "MISSING_OP": [0, 0, 0]}
    encoding = []
    # In the encoding of nb101, there are architectures with less than 7 operations. 
    # This is causing problems during the binary encoding. 
    # To fix this, we add a missing op to the matrix and ops list.
    # Since the first or the last op are always input and output.
    if matrix.shape[0] < NUM_VERTICES:
        available_indices = sorted(list(range(2, NUM_VERTICES - 1)))
        num_missing_ops = NUM_VERTICES - matrix.shape[0]
        selected_indices = np.random.choice(available_indices, size=num_missing_ops, replace=False)
    
        for missing_op_idx in selected_indices:
            # Insert a new column of zeros at missing_op_idx
            new_column = np.zeros((matrix.shape[0], 1), dtype=np.int8)
            matrix = np.concatenate((matrix[:, :missing_op_idx], new_column, matrix[:, missing_op_idx:]), axis=1)
        
        for missing_op_idx in selected_indices:
            # Insert a new row of zeros at missing_op_idx
            new_row = np.zeros((1, NUM_VERTICES), dtype=np.int8)
            matrix = np.concatenate((matrix[:missing_op_idx], new_row, matrix[missing_op_idx:]))

            # Insert MISSING_OP into the ops list
            ops.insert(missing_op_idx, "MISSING_OP")
    
    for i in range(NUM_VERTICES - 1):
        for j in range(i + 1, NUM_VERTICES):
            encoding.append(matrix[i][j])
            
    for i in range(1, NUM_VERTICES - 1):
        encoding = [*encoding, *op_dict[ops[i]]]
    
    return encoding


def encode_gcn(spec):
    """
    Input:
    a list of categorical ops starting from 0
    """
    matrix, ops = spec["matrix"], spec["ops"]
    op_map = [OUTPUT, INPUT, *OPS]
    ops_onehot = np.array(
        [[i == op_map.index(op) for i in range(len(op_map))] for op in ops],
        dtype=np.float32,
    )

    dic = {
        "num_vertices": 7,
        "adjacency": matrix,
        "operations": ops_onehot,
        "mask": np.array([i < 7 for i in range(7)], dtype=np.float32),
        "val_acc": 0.0,
    }
    return dic


def encode_bonas(spec):
    """
    Input:
    a list of categorical ops starting from 0
    """
    matrix, ops = spec["matrix"], spec["ops"]
    op_map = [INPUT, *OPS, OUTPUT]
    ops_onehot = np.array(
        [[i == op_map.index(op) for i in range(len(op_map))] for op in ops],
        dtype=np.float32,
    )

    matrix = add_global_node(matrix, True)
    ops_onehot = add_global_node(ops_onehot, False)
    matrix = np.array(matrix, dtype=np.float32)
    ops_onehot = np.array(ops_onehot, dtype=np.float32)
    dic = {"adjacency": matrix, "operations": ops_onehot, "val_acc": 0.0}
    return dic


def add_global_node(mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if ifAdj:
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return mx


def encode_seminas(spec):
    matrix, ops = spec["matrix"], spec["ops"]
    # offset ops list by one, add input and output to ops list
    ops = [OPS_INCLUSIVE.index(op) for op in ops]
    dic = {
        "num_vertices": 7,
        "adjacency": matrix,
        "operations": ops,
        "mask": np.array([i < 7 for i in range(7)], dtype=np.float32),
        "val_acc": 0.0,
    }
    return dic


def encode_101(arch, encoding_type=EncodingType.PATH):
    spec = arch.get_spec()

    if encoding_type == EncodingType.PATH:
        return encode_paths(spec=spec)

    elif encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adj(spec=spec)

    elif encoding_type == EncodingType.GCN:
        return encode_gcn(spec=spec)

    elif encoding_type == EncodingType.SEMINAS:
        return encode_seminas(spec=spec)

    elif encoding_type == EncodingType.BONAS:
        return encode_bonas(spec=spec)

    else:
        print(
            "{} is not yet implemented as an encoding type \
         for nb101".format(
                encoding_type
            )
        )
        raise NotImplementedError()


def encode_101_spec(spec, encoding_type=EncodingType.PATH):
    if encoding_type == EncodingType.PATH:
        return encode_paths(spec=spec)

    elif encoding_type == EncodingType.ADJACENCY_ONE_HOT:
        return encode_adj(spec=spec)

    elif encoding_type == EncodingType.GCN:
        return encode_gcn(spec=spec)

    elif encoding_type == EncodingType.SEMINAS:
        return encode_seminas(spec=spec)

    elif encoding_type == EncodingType.BONAS:
        return encode_bonas(spec=spec)

    else:
        logger.info(f'{encoding_type} is not yet implemented as an encoding type for nb101')
        raise NotImplementedError()


def encode_spec(spec, encoding_type=EncodingType.ADJACENCY_ONE_HOT):
    if isinstance(spec, tuple):
        spec = convert_tuple_to_spec(spec)
        return encode_101_spec(spec, encoding_type=encoding_type)
    else:
        raise NotImplementedError(f'No implementation found for encoding search space nb101 with {encoding_type}')
