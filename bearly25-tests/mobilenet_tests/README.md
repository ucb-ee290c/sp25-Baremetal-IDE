# Running mobilenet tests: 
1. Run the python script to generate the data (we'll use pytorch as the golden model): 
    `python data_gen.py gen --name p1s1_5x5_c3 --H 224 --W 224 --Cin 6 --stride 1 --padding 1 --out-dir include`
2. Build the tests from `baremetal-ide` root dir: 
    `make build CHIP=bearly25 TARGET=mobilenet_tests VECNN=1 RVV=1`
3. Run spike (or vcs) with the binary: 
    `spike --isa=rv64gcv_zicntr build/bearly25-tests/mobilenet_tests/mobilenet_tests.elf`
