import os
import dask.array as da
import dask_image.imread
import zarr

from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar  # optional for a terminal progress bar

if __name__ == "__main__":
    # 1. Create/attach a Local Dask Cluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="10GB")
    client = Client(cluster)
    print("Dask dashboard link:", client.dashboard_link)

    # 2. Prepare your data
    tiff_dir = "/home/sean/Documents/GitHub/VC-Surface-Models/custom/normals/preprocess/tests/writing_normals/face_normals_continuous_new"



    # 3. Use dask-image to lazily read all .tif files matching your pattern
    #    This returns a Dask array with shape (N, height, width[, channel]).
    #    * If the TIFFs are RGB or multi-channel, dask-image might produce
    #      a dimension for the color channel: (N, height, width, channels).
    #    * If they're single-channel, shape will be (N, height, width).
    d_stack = dask_image.imread.imread(os.path.join(tiff_dir, "*.tif"))

    print("Dask array shape from imread:", d_stack.shape)
    print("Dask array dtype:", d_stack.dtype)
    # e.g., shape might be (100, 512, 512, 3) if you have 100 RGB TIFFs.

    # 4. (Optional) Rechunk for better write/read performance
    #    You can tune chunk sizes based on your data size and memory constraints.
    #    For instance, if your data shape is (N, Y, X, C) = (100, 1024, 1024, 3),
    #    you might want something like (1, 512, 512, 3) or (10, 256, 256, 3).
    #    This is a performance knob, so adjust as needed.
    desired_chunks = (192, 192, 192, 3)
    d_stack = d_stack.rechunk(desired_chunks)

    # 5. Write to Zarr (with optional progress bar in the terminal)
    # Output Zarr store
    zarr_out = "/home/sean/Documents/GitHub/VC-Surface-Models/custom/normals/normals.zarr"
    with ProgressBar():
        d_stack.to_zarr(zarr_out, component="normals", overwrite=True)

    # 6. Verify the output
    store = zarr.DirectoryStore(zarr_out)
    root = zarr.open(store, mode='r')
    print("Created Zarr array shape:", root["normals"].shape)
    print("Created Zarr array chunks:", root["normals"].chunks)

