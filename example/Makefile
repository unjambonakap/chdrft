
all: sharedTrampo.so

trampoline.o: trampoline.S
	nasm -f elf trampoline.S
sharedTrampo.so: a.cpp trampoline.o
	g++ -m32 -shared -fPIC a.cpp trampoline.o -o sharedTrampo.so


