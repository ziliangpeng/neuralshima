#!/bin/bash

nvcc gemm_main.cu; time nvprof ./a.out
