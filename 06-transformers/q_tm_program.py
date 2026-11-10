# Unary addition: 111_11 (3 + 2) -> 11111 (5)
ADD = machine({"A1": "1RA",      # walk right over the first number
               "A_": "1RB",      # the blank in the middle becomes 1
               "B1": "1RB",      # walk right over the second number
               "B_": "_LC",      # past the end: step back
               "C1": "_SH"},     # erase the last 1, done
              "ABC")
tape = [0] * TAPE
for i, c in enumerate("111_11"):
    tape[5 + i] = SYMBOLS.index(c)
state, pos = 0, 5
ref = (state, pos, tape)
for got in run_transformer(ADD, state, pos, tape, 50):
    ref = tm_step(ADD, *ref)
    assert got == ref, "the Transformer and the reference disagree"
    if got[0] == HALT:
        break
print("result:", "".join(SYMBOLS[s] for s in got[2]).strip("_"), "  ", show(*got, "ABC"))
