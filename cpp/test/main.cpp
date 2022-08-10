#include "zfp.hpp"
#include "zfpc.hpp"
#include <cstdio>


#include <fstream>
#include <sstream>
#include <string>

std::string slurp(const char *filename) {
    std::ifstream in;
    in.open(filename, std::ifstream::in | std::ifstream::binary);
    std::stringstream sstr;
    sstr << in.rdbuf();
    in.close();
    return sstr.str();
}


int main () {
	std::string data = slurp("/Users/wms/code/zfpc/wasm/0-128_0-128_0-10");


	size_t voxels = 128 * 128 * 10 * 2;
	float* out = new float[voxels]();
	const unsigned char* ptr = reinterpret_cast<unsigned char*>(const_cast<char*>(data.c_str()));

	int err = zfpc::decompress(ptr, data.size(), out, voxels * 4);

	std::ofstream f;
	f.open("/Users/wms/code/zfpc/wasm/0-128_0-128_0-10.raw");

	unsigned char* outc = reinterpret_cast<unsigned char*>(out);

	for (int i = 0; i < voxels * 4; i++) {
		f << outc[i];
	}

	f.close();

	return 0;
}