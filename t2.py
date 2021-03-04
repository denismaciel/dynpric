
x = [[tuple(-1 if j == i else 0 for j in range(4)), 0] for i in range(4)]
print(x)


assert x == [
        [(-1, 0, 0, 0), 0],
        [(0, -1, 0, 0), 0],
        [(0, 0, -1, 0), 0],
        [(0, 0, 0, -1), 0]
        ]



