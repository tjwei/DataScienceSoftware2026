# Compute 2^7 = 128: A starts at 1 and is added to itself 7 times
POW2 = """
    MVI A,1
    MVI C,7
    MOV B,A     ; 2: B = A
    ADD B       ;    A = A + A
    DEC C
    JNZ 2
    ST [3],A
    HALT
"""
# 100 divided by 7: keep subtracting 7 and count the subtractions (quotient in D, remainder in A)
DIV = """
    MVI A,100
    MVI B,7
    MVI D,0
    SUB B       ; 3: A -= 7
    JC 7        ;    borrow => subtracted too far
    INC D
    JP 3
    ADD B       ; 7: add back to get the remainder
    ST [4],D
    ST [5],A
    HALT
"""
compare(POW2, 40, "2^7")
compare(DIV, 120, "100/7")
