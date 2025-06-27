CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=-lSDL2 -lm

all: neuralnet

neuralnet: main.o neuralnet.o matrix.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c neuralnet.h matrix.h
neuralnet.o: neuralnet.c neuralnet.h matrix.h
matrix.o: matrix.c matrix.h

clean:
	rm -f *.o neuralnet
