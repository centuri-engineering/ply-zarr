import tempfile

import meshio
import zarr

from ply_zarr import write, to_ply, attrs_to_ply_header, parse_ply_header


def test_parse_header():
    with open("data/test.ply", "rb") as fh:
        attrs = parse_ply_header(fh)
    assert "format" in attrs
    assert "properties" in attrs["elements"]["face"]


def test_write():
    mesh = meshio.read("data/test.ply")
    root = zarr.group()
    write(root, mesh)
    assert "points" in root


def test_write_ply_haeder():
    with open("data/test.ply", "rb") as fh:
        attrs = parse_ply_header(fh)

    with open("data/test.ply", "r") as fh:
        content = []
        for line in fh:
            content.append(line)
            if line.startswith("end_header"):
                break

    header = "".join(content)
    assert header == attrs_to_ply_header(attrs)


def test_to_ply():

    mesh = meshio.read("data/test.ply")
    root = zarr.group()
    write(root, mesh)
    tmp_ply = tempfile.mktemp(suffix=".ply")

    with open(tmp_ply, "wb") as fh:
        to_ply(root, fh)

    mesh_r = meshio.read(tmp_ply)
    assert mesh_r.cells
