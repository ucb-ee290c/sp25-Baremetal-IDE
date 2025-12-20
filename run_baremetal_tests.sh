#!/bin/bash

make build CHIP=bearly25 TARGET=ope-tests

cd ../../sims/vcs

make run-binary CONFIG=SaturnOPEConfigBup LOADMEM=1 BINARY=/tools/C/jonathanji/bring_up_work/sp25-chips/software/sp25-Baremetal-IDE/build/bearly25-tests/ope-tests/ope-tests.elf

cd ../..
cd software/sp25-Baremetal-IDE/