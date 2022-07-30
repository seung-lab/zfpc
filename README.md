[![Test Suite](https://github.com/seung-lab/zfpc/actions/workflows/test-suite.yml/badge.svg)](https://github.com/seung-lab/zfpc/actions/workflows/test-suite.yml)

# zfpc: zfp container format

_An unofficial project unaffiliated with the [`zfp`](https://github.com/LLNL/zfp/) project._

An exerimental container format for `zfp` encoded vector fields. As described in the [zfp documentation](https://zfp.readthedocs.io/en/latest/faq.html#q-vfields), datasets such as vector fields are not optimally compressed within a single zfp stream. This is due to the uncorrelated X and Y components. Compress the X and Y components as separate `zfp` arrays and you will yield a higher compression ratio.

However, this method of separate files is cumbersome, must be maintained per a project, and is not compatible with existing data viewers (such as Neuroglancer) that expect to download a single file per an image tile. `zfpc` provides a means for splitting up a 1-4D array based on their (user specified) uncorrelated dimensions, compressing those slices into separate `zfp` streams, and encoding them into a single file. This file can then be decompressed back into its original form seamlessly using `zfpc`. In the future, it may be possible to automatically determine which dimensions are uncorrelated using statistical tests.

In fixed rate mode, it should still be possible to perform random access though this feature is not available yet.

```python
import zfpc

# example shape: (1202, 1240, 64, 2)
vector_field = np.array(...) # dtype must be float or int, 32 or 64-bit

# For data that are arranged as a Z stack of planar XY vectors
# e.g. arr[x,y,z,channel] mostly smoothly vary in the XY plane
# per a channel. Therefore, we set correlated_dims as 
# [True,True,False,False] as the z and channel dimensions
# do not smoothly vary to obtain optimal compression.
#
# tolerance, rate, and precision are supported modes.
# By default, lossless compression is used.
correlated_dims = [True, True, False, False]
binary = zfpc.compress(
	vector_field, 
	tolerance=0.01,
	correlated_dims=correlated_dims,
)
recovered_img = zfpc.decompress(binary)
```

## Container Format

header,index,streams

### Header

The header is 15 bytes long in the following format written in little endian.

| Field             | Type    | Description                                                                                              |
|-------------------|---------|----------------------------------------------------------------------------------------------------------|
| magic             | char[4] | "zfpc" magic number for file format.                                                                     |
| format version    | uint8   | Always version 0 (for now).                                                                              |
| dtype,mode,order  | uint8   | bits 1-3: zfp data type; bits 4-6: zfp mode; bit 7: unused; bit 8: true indicates c order (bits: DDDMMMUC)                          |
| nx                | uint32  | Size of x axis.                                                                                          |
| ny                | uint32  | Size of y axis.                                                                                          |
| nz                | uint32  | Size of z axis.                                                                                          |
| nw                | uint32  | Number of channels.                                                                                      |
| correlated dims   | uint8   | Bottom 4 bits are a bitfield with 1 indicating correlated, 0 uncorrelated. Top 4 bits unused. (xyzw0000) |

### Index

All entries in the index are uint64 (8 bytes) little endian.

Stream offset followed by the size of each stream. The number of streams is calculated by the product of all the uncorrelated dimension sizes.

The stream offset is not a strictly necessary element, but will allow the format to be changed while allowing older decompressors to still function.

### Streams

All zfp streams are concatenated together in Fortran order. The streams are written with a full header so that they can be decompressed independently. 

In the future, it might make sense to get savings by condensing them into a single header and writing headerless streams. However, writing full headers presents the possibility of using different compression settings for each stream which could pay off for different components.
