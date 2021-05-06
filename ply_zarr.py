#!/usr/bin/env python
# coding: utf-8

"""
Write and read PLY files to a zarr store
here is a basic ply file

    ply
    format ascii 1.0           { ascii/binary, format version number }
    comment made by Greg Turk  { comments keyword specified, like all lines }
    comment this file is a cube
    element vertex 8           { define "vertex" element, 8 of them in file }
    property float x           { vertex contains float "x" coordinate }
    property float y           { y coordinate is also a vertex property }
    property float z           { z coordinate, too }
    property float color       { color is a custom property}
    element face 6             { there are 6 "face" elements in the file }
    property list uchar int vertex_indices { "vertex_indices" is a list of ints }
    end_header                 { delimits the end of the header }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0

The header is translated to a zarr group attrs object (JSON like)

    ply_header = {
        "format": "ascii 1.0",
        "comments": [f"created by ply_zarr v0.0.1, {datetime.now().isoformat()}",],
        "elements": {
            "vertex": {
                "size": 8,
                "properties": [
                    ("float", "x"),
                    ("float", "y"),
                    ("float", "z"),
                    ("float", "intensity")
                ]
            },
            "face": {
                "size": 6,
                "properties": [
                    ("list", "uint8", "int32", "vertex_indices"),
                ]
            }
        }
    }

The array heirarchy of the created zarr group is as follow:

    group
     ├── 4
     │   └── vertex_indices (6, 4) int32
     └── points
         ├── color (8,) float64
         ├── x (8,) int64
         ├── y (8,) int64
         └── z (8,) int64

"""

import io
import sys
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
from meshio import CellBlock, Mesh
from meshio.ply._ply import numpy_to_ply_dtype, cell_type_from_count


def write(group, mesh):
    """Writes a MeshIO mesh to zarr-ply
    """
    # PLY header to mesh
    fh = io.BytesIO()
    header_from_mesh(mesh, fh, binary=False)
    group.attrs.update(parse_ply_header(fh))
    points_grp = group.create_group("points")
    for i, coord in enumerate("xyz"[: mesh.points.shape[1]]):
        points_grp[coord] = mesh.points[:, i]
    for key, val in mesh.point_data.items():
        points_grp[key] = val

    for i, block in enumerate(mesh.cells):
        cnt = block.data.shape[1]
        cnt_group = group.create_group(cnt)
        cnt_group["vertex_indices"] = block.data
        for key, vals in mesh.cell_data.items():
            cnt_group[key] = np.array(vals[i])


def read(group):
    """Reads from a zarr-ply group and returns a Mesh object
    """
    point_props = [p[-1] for p in group.attrs["elements"]["vertex"]["properties"]]
    coords = [c for c in "xyz" if c in point_props]
    points = np.vstack([group.points[c] for c in coords]).T
    point_data = {}
    for key in point_props:
        if key in coords:
            continue
        point_data[key] = np.array(group.points[key])

    cells = []
    cell_data = defaultdict(list)
    cell_props = group.attrs["elements"]["face"]["properties"]
    for key in group.keys():
        if not key.isnumeric():
            continue
        nsides = int(key)
        cell_type = cell_type_from_count(nsides)
        cells.append((cell_type, np.array(group[key]["vertex_indices"])))
        for *_, prop in cell_props:
            if prop == "vertex_indices":
                continue
            cell_data[prop].append(np.array(group[key][prop]),)

    return Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)


def to_ply(group, fh):
    """This is using meshio to write back to ply
    """
    mesh = read(group)
    mesh.write(fh, file_format="ply")


def parse_ply_header(fh):
    """
    fh: file handle on a ply file
    """
    fh.seek(0)
    attrs = {}
    attrs["format"] = ""
    attrs["comments"] = []
    attrs["elements"] = {}
    if not fh.readline().decode("utf-8").startswith("ply"):
        raise IOError("Not a ply file")
    for line in fh:
        line = line.decode("utf-8")
        start, *rest = line.split()
        if start == "format":
            attrs["format"] = " ".join(rest)

        elif start == "comment":
            attrs["comments"].append(" ".join(rest))

        elif start == "element":
            elem = rest[0]
            attrs["elements"][elem] = {"size": int(rest[1]), "properties": []}

        elif start == "property":
            attrs["elements"][elem]["properties"].append(tuple(rest))

        elif start == "end_header":
            break
    return attrs


def attrs_to_ply_header(attrs):
    lines = []
    lines.append("ply")
    lines.append("format " + attrs["format"])

    if attrs.get("comments"):
        lines.extend(["comment " + c for c in attrs["comments"]])
    for elem, elem_dict in attrs["elements"].items():
        lines.append(f"element {elem} {elem_dict['size']}")
        for prop in elem_dict["properties"]:
            lines.append("property " + " ".join(prop))
    lines.append("end_header\n")
    return "\n".join(lines)


def header_from_mesh(mesh, fh, binary=False):
    """Shamelessly stolen from meshio

    https://github.com/nschloe/meshio/blob/0600ac9e9e8d1e1a27d5f3f2f4235414f4482cac/meshio/ply/_ply.py#L392
    """
    # TODO: what to do with binary?

    fh.write(b"ply\n")

    if binary:
        fh.write(f"format binary_{sys.byteorder}_endian 1.0\n".encode("utf-8"))
    else:
        fh.write(b"format ascii 1.0\n")

    fh.write(
        "comment Created by ply-zarr v0.0.1, {}\n".format(
            datetime.now().isoformat()
        ).encode("utf-8")
    )

    # counts
    fh.write("element vertex {:d}\n".format(mesh.points.shape[0]).encode("utf-8"))
    #
    dim_names = ["x", "y", "z"]
    # From <https://en.wikipedia.org/wiki/PLY_(file_format)>:
    #
    # > The type can be specified with one of char uchar short ushort int uint float
    # > double, or one of int8 uint8 int16 uint16 int32 uint32 float32 float64.
    #
    # We're adding [u]int64 here.
    type_name_table = {
        np.dtype(np.int8): "int8",
        np.dtype(np.int16): "int16",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.uint64): "uint64",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    for k in range(mesh.points.shape[1]):
        type_name = type_name_table[mesh.points.dtype]
        fh.write("property {} {}\n".format(type_name, dim_names[k]).encode("utf-8"))

    pd = []
    for key, value in mesh.point_data.items():
        if len(value.shape) > 1:
            warnings.warn(
                "PLY writer doesn't support multidimensional point data yet. Skipping {}.".format(
                    key
                )
            )
            continue
        type_name = type_name_table[value.dtype]
        fh.write(f"property {type_name} {key}\n".encode("utf-8"))
        pd.append(value)

    num_cells = 0
    for cellblock in mesh.cells:
        num_cells += cellblock.data.shape[0]

    if num_cells > 0:
        fh.write(f"element face {num_cells:d}\n".encode("utf-8"))

        # possibly cast down to int32
        has_cast = False
        for k, (cell_type, data) in enumerate(mesh.cells):
            if data.dtype == np.int64:
                has_cast = True
                mesh.cells[k] = CellBlock(cell_type, data.astype(np.int32))

        if has_cast:
            warnings.warn(
                "PLY doesn't support 64-bit integers. Casting down to 32-bit."
            )

        # assert that all cell dtypes are equal
        cell_dtype = None
        for _, cell in mesh.cells:
            if cell_dtype is None:
                cell_dtype = cell.dtype
            if cell.dtype != cell_dtype:
                raise IOError("Could not write ply header from mesh")

        if cell_dtype is not None:
            ply_type = numpy_to_ply_dtype[cell_dtype]
            fh.write(
                "property list {} {} vertex_indices\n".format("uint8", ply_type).encode(
                    "utf-8"
                )
            )
        for key, vals in mesh.cell_data:
            sample = np.asarray(vals[0])
            prop_type = numpy_to_ply_dtype[sample.dtype]
            if sample.dims != 1:
                fh.write(
                    "property list {} {} {}\n".format("uint8", prop_type, key).encode(
                        "utf-8"
                    )
                )
            else:
                fh.write("property {} {}\n".format(prop_type, key).encode("utf-8"))

    # TODO other cell data
    fh.write(b"end_header\n")
