CFLAGS=-O2 -L../cudasymmfft -I../cudasymmfft -L.
GFLAGS=-arch sm_35
LDFLAGS=-lcudasymmfft -lcufft -lcuTranspose
CCFLAGS=$(CFLAGS) $(GFLAGS)
SOLVER=cuda_helmholtz_solver.cu 
CPUSOLVER=hw3crt_wrapper.cpp
CPULDFLAGS=-lfishpack

testgpu:testgpu.cu $(SOLVER)
	nvcc $(CCFLAGS) -o $@ $^ $(LDFLAGS)
testgpurectangle:testgpurectangle.cu $(SOLVER)
	nvcc $(CCFLAGS) -o $@ $^ $(LDFLAGS)

testcpu:testcpu.cpp $(CPUSOLVER)
	g++ $(CFLAGS) -o $@ $^ $(CPULDFLAGS)
testcpurectangle:testcpurectangle.cpp $(CPUSOLVER)
	g++ $(CFLAGS) -o $@ $^ $(CPULDFLAGS)

clean:
	rm -fr testgpu testgpurectangle testcpu testcpurectangle
