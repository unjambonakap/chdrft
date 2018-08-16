# chdrft python libraries

## What is this

same as opa, but for python (past projects that got put in libraries for reuse)

## utils

- stuff that does not deserve it's own project
- mostly misc.py

## tube

Helpers to abstract connections (socket/file/serial)
Inspired from pwntools, back when it was only python2


## gen

- Parsing C/C++ code with libclang.
- Used to interact with C/C++ APIs directly in python, without redefining the interface


## emu

- Interaction with C/C++/ASM (calling functions, packing/unpacking structures)
- ELF helpers using pyelftools
- Running x86/arm ELF using unicorn-engine with some syscalls are emulated. Used for reversing


## dbg

- Controlling gdb from python (using gdb's python api)


## elec

- control of a few microcontrollers from python (cc1110, max31855, bus pirate)

## display

- 2d/3d plotting using vispy and pyqtgraph

## waf

- dependency for the waf build system of the C/C++


## tools

Some random projects that needed to be made as binaries
- tools/elf_mod.py: add symbols to an elf (symbols provided by radare2)
- tools/pin_analyzer.py: draw the execution graph using the Intel PIN framework (for some crackmes)

