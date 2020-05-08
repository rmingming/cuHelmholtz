CFLAGS=-O2 -arch=sm_35

all:dst1funcinplace.o dst2funcinplace.o dst3funcinplace.o \
		dct1funcinplace.o dct2funcinplace.o dct3funcinplace.o \
		funcinterface.o dftfunc.o
%.o:%.cu
	nvcc $(CFLAGS) -c -o $@ $^ -lcufft  -lm

lib:
	ar cr libcudasymmfft.a *.o

.PHONY:clean

clean:cleanobj cleanlib

cleanobj:
	rm *.o

cleanlib:
	rm libcudasymmfft.a
