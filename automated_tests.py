import numpy as np
import zfpc
import pytest

@pytest.fixture
def vector_img():
	print("HIII")
	return np.load("vector_field_sample.npy")

@pytest.mark.parametrize("correlated_dims", [[True,True,True,True], [True,True,False,False]])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_compression_decompression(vector_img, correlated_dims, order):
	tolerance = 0.01

	if order == "C":
		vector_img = np.ascontiguousarray(vector_img)
	else:
		vector_img = np.asfortranarray(vector_img)

	compressed = zfpc.compress(
		vector_img, 
		tolerance=tolerance, 
		correlated_dims=correlated_dims
	)
	recovered = zfpc.decompress(compressed)

	assert np.allclose(recovered, compressed, atol=tolerance)



