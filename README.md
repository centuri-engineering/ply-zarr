# ply-zarr: A PLY format implementation in zarr


PLY is a standard format to store mesh data.

See the [specification](http://paulbourke.net/dataformats/ply/)

[ZARR](https://zarr.readthedocs.io/en/stable/) is a data store format.


Write and read PLY files to a zarr store
here is a basic ply file

```
ply
format ascii 1.0           { ascii/binary, format version number }
comment made by Greg Turk  { comments keyword specified, like all lines }
comment this file is a cube
element vertex 8           { define "vertex" element, 8 of them in file }
property float x           { vertex contains float "x" coordinate }
property float y           { y coordinate is also a vertex property }
property float z           { z coordinate, too }
element face 6             { there are 6 "face" elements in the file }
property list uchar int vertex_index { "vertex_indices" is a list of ints }
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
```

The header is translated to a zarr group attrs object (JSON like)

```python
ply_header = {
    "format": "ascii 1.0",
    "comments": [f"created by ply_zarr v0.0.1, {datetime.now().isoformat()}",],
    "elements": {
        "vertex": {
            "size": 47,
            "properties": [
                ("double", "x"),
                ("double", "y"),
                ("double", "z")
            ]
        },
        "face": {
            "size": 105,
            "properties": [
                ("list", "uint8", "int32", "vertex_indices"),
            ]
        }
    }
}
```
The array heirarchy of the created zarr group is as follows:

```
    group
     ├── 4
     │   └── vertex_indices (6, 4) int32
     └── points
         ├── color (8,) float64
         ├── x (8,) int64
         ├── y (8,) int64
         └── z (8,) int64
```

## Usage

ply-zarr relies on [meshio](https://github.com/nschloe/meshio) and
uses a `meshio.Mesh` object as input to write a zarr group.

```python
from ply_zarr import write

mesh = meshio.read("data/test.ply")
root = zarr.group()
write(root, mesh)

```

## Tests

```sh
pytest tests
```

## TODO

- Due to a [bug](https://github.com/nschloe/meshio/issues/1095), exporting in other formats for non-triangular meshes will not work in general

- Type hints & mypy everything

- Strictier handeling of data formats

- xarray compatibility (if possible?)
