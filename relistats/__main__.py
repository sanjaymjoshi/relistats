import sys

from relistats.binomial import assurance

if __name__ == "__main__":
    n = int(sys.argv[1])
    a = assurance(n, 0) or 0
    print(f"Assurance at {n} good samples: {a*100:.1f}%")
