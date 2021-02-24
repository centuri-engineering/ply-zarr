def test_plyzarr():

    with open("test.ply", "rb") as fh:
        attrs = parse_ply_header(fh)

    mesh = meshio.read('test.ply')
    root = zarr.group()
    write_plyzarr(root, mesh)

    with open("test.ply", "r") as fh:
        content = fh.readlines()

    header = "".join(content[:10])

    header == attrs_to_ply_header(attrs)

    with open("test_out.ply", "wb") as fh:
        plyzarr_to_ply(root, fh)

    mesh_r = meshio.read("test_out.ply")
