#!/bin/bash

nvcc gemm.cu; time nvprof ./a.out
