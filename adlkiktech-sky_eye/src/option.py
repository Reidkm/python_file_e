import argparse
import os

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser(description="parser for Lane Segmentation")
		subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

		# demo
		start_arg = subparsers.add_parser("start", help="parser for Lane Segmentation arguments")

		start_arg.add_argument("--record", type=int, default=0,
								help="set it to 1 for recording into video file")

		# demo_arg.add_argument("--video-source", type=str, required=True,
		# 						help="source of video import, if camera using 0 which is the number of camera, if video, give filename")

	def parse(self):
		return self.parser.parse_args()
