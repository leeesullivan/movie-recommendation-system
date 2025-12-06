#!/usr/bin/env python3
import sys
import math

def main():
    current_pair = None
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        pair, value = line.split("\t", 1)
        r1_str, r2_str = value.split(":", 1)

        r1 = float(r1_str)
        r2 = float(r2_str)

        if current_pair is None:
            current_pair = pair

        if pair != current_pair:
            if sum_x2 > 0 and sum_y2 > 0:
                sim = sum_xy / math.sqrt(sum_x2 * sum_y2)
                print(f"{current_pair}\t{sim}")
            current_pair = pair
            sum_xy = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0

        sum_xy += r1 * r2
        sum_x2 += r1 * r1
        sum_y2 += r2 * r2

    if current_pair is not None and sum_x2 > 0 and sum_y2 > 0:
        sim = sum_xy / math.sqrt(sum_x2 * sum_y2)
        print(f"{current_pair}\t{sim}")

if __name__ == "__main__":
    main()
