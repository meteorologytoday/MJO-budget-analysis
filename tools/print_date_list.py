import pandas as pd
import argparse

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)
parser.add_argument('--date-rng', type=str, nargs=2, help='Input file', required=True)
parser.add_argument('--inclusive', type=str, help='Input file', default="both")
args = parser.parse_args()

time_beg = pd.Timestamp(args.date_rng[0])
time_end = pd.Timestamp(args.date_rng[1])

s = ""
for dt in pd.date_range(time_beg, time_end, freq="D", inclusive=args.inclusive):
    print(dt.strftime("%Y-%m-%d"), end=' ')
