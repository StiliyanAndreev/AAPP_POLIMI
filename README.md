Polimi 2025/2026 AAPP masters
```
Primality test on 512bits number
```

- Implement the Primality test described during the Primality testing lecture (see lecture10PrimalityTest.pdf pp. 12-14).

- The data type used for n can represent a number representable with 512 bits.

- Check the asymptotic complexity with Google Benchmarking library

-Provide sufficient tests to validate the implementation

- Groups up to 3 people
- Changing team for the 2nd challenge is ok
- One week of time
-  points for free in the 1st part of the exam
- Skip selected questions
- Valid any time in the whole academic year
 
Material to be delivered on Webeep:
 - Copy of the Colab with code .pynb (no link)
 - Short PDF report (max 2 pages)

```
challenge
```
Challenge 2
A toy virtual screening application scores a set of molecules using MPI. The initial implementation randomly creates an input. Then, only the process with rank 0 will perform the computation of the whole input, generating the output.

The goal of this challenge is to parrallelize the application. There are two main challenges: 1) distribute the molecules between MPI processes, and 2) perform the reduction at the end of the computation.

This is the list of activities requested to complete the challenge:

Download the application skeleton
Compile and execute the application as shown in class with one MPI process to use as reference
e.g. mpirun -np 1 ./build/main 10
Update the main file (src/main.cpp) to implement the parallelization
Upload only the updated version of the main file
