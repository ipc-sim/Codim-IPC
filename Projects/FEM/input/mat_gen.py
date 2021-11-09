if __name__ == "__main__":
    n = int(input())
    m = int(input())

    f = open("mat" + str(n) + "x" + str(m) + ".obj", "w")
    s = float(max(n, m))
    for i in range(n):
        for j in range(m):
            f.write("v %.3f %.3f 0\n" % (i / s, j / s))
    index = lambda r, c: r * m + c + 1
    for i in range(n - 1):
        for j in range(m - 1):
            f.write("f %d %d %d\n" % (index(i, j), index(i + 1, j), index(i, j + 1)))
            f.write("f %d %d %d\n" % (index(i + 1, j + 1), index(i, j + 1), index(i + 1, j)))
    f.close()
