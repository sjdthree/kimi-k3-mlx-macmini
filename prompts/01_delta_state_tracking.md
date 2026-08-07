Act as a deterministic execution runtime. Execute the state mutations defined in the log below. Track memory slots S0 through S7.

Rules:
- OVERWRITE replaces a slot completely.
- MERGE_XOR performs bitwise XOR on slot hex values.
- DECAY_HALF shifts the binary representation of the slot right by 1 bit.
- PURGE sets the slot to 0x00 and flags it invalid.

Execution Log:
1. SET S0 = 0xF4A2, S1 = 0x3B89, S2 = 0x00FF, S3 = 0xAA55
2. OVERWRITE S1 = 0x1234
3. MERGE_XOR S0 WITH S1 -> STORE S4
4. DECAY_HALF S4
5. SET S5 = 0xFFFF
6. MERGE_XOR S5 WITH S4 -> STORE S6
7. PURGE S0
8. OVERWRITE S2 = 0x8888
9. MERGE_XOR S2 WITH S6 -> STORE S7
10. DECAY_HALF S7
11. OVERWRITE S4 = 0xABCD
12. MERGE_XOR S7 WITH S4 -> STORE S3
13. PURGE S1
14. DECAY_HALF S3

Provide a step-by-step state trace table for each operation, followed by a final dictionary output mapping active slots to their final Hex, Binary, and Decimal values. Limit your entire response to under 1,500 tokens.
