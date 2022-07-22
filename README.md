# zfpc
Exerimental container format for zfp encoded vector fields. zfpc = zfp container

```python
import zfpc

# shape: (1202, 1240, 64, 2)
vector_field = np.array(...) # dtype must be float or int, 32 or 64-bit

# For data that are arranged as a Z stack of planar XY vectors
# e.g. arr[x,y,z,channel] mostly smoothly vary in the XY plane
# per a channel. Therefore, we set correlated_dims as 
# [True,True,False,False] as the z and channel dimensions
# do not smoothly vary to obtain optimal compression.
correlated_dims = [True, True, False, False]
binary = zfpc.compress(
	vector_field, 
	tolerance=0.01,
	correlated_dims=[True,True,False,False],
)
recovered_img = zfpc.decompress(binary)
```

## Container Format

header,index,streams

### Header

The header is 15 bytes long in the following format:

| Field             | Type    | Description                                                                                              |
|-------------------|---------|----------------------------------------------------------------------------------------------------------|
| magic             | char[4] | "zfpc" magic number for file format.                                                                     |
| format version    | uint8   | Always version 0 (for now).                                                                              |
| data type / order | uint8   | Top bit true indicates c order, bottom bits indicate data type. (bits: DDDDDDC)                          |
| nx                | uint16  | Size of x axis.                                                                                          |
| ny                | uint16  | Size of y axis.                                                                                          |
| nz                | uint16  | Size of z axis.                                                                                          |
| nw                | uint16  | Number of channels.                                                                                      |
| correlated dims   | uint8   | Bottom 4 bits are a bitfield with 1 indicating correlated, 0 uncorrelated. Top 4 bits unused. (xyzw0000) |

### Index

All entries in the index are uint64 (8 bytes).

Stream offset followed by the size of each stream. The number of streams is calculated by the product of all the uncorrelated dimension sizes.

The stream offset is not a strictly necessary element, but will allow the format to be changed while allowing older decompressors to still function.

### Streams

All zfp streams are concatenated together. The streams must be written with a full header so that they can be decompressed independently.
