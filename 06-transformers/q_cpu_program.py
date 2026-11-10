# 計算 2^7 = 128：A 從 1 開始，自己加自己 7 次
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
# 100 除以 7：一直減 7，數減了幾次 (商放在 D，餘數在 A)
DIV = """
    MVI A,100
    MVI B,7
    MVI D,0
    SUB B       ; 3: A -= 7
    JC 7        ;    借位 => 減過頭了
    INC D
    JP 3
    ADD B       ; 7: 加回來得到餘數
    ST [4],D
    ST [5],A
    HALT
"""
compare(POW2, 40, "2^7")
compare(DIV, 120, "100/7")
