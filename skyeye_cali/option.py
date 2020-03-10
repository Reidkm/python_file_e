#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os

class Options():
	def __init__(self):
		#self.parser = argparse.ArgumentParser(description="parser calibration")

		self.parser = argparse.ArgumentParser(description="skysye calibration")

		subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

		# demo
		calibration_arg = subparsers.add_parser("calibration", help="parser for camera index")

		calibration_arg.add_argument("--demo-size", type=int, default=480,
								help="demo window height, default 480")
		#calibration_arg.add_argument("--video-source", type=str, default=0,
		#						help="source of video import, if camera using 0 which is the number of camera, if video, give filename")
   		# calibration_arg.add_argument("--camera-center", type=str, default="230,345",
		# 						help="where camera center should be, like: 230,345")
		# calibration_arg.add_argument("--chessboard-size", type=str, default="7,4,38",
		# 						help="size of checkboard , like 7,4")
		# calibration_arg.add_argument("--chessboard-position", type=str, required=True,
		# 						help="the position of chessboard, like 10,10;50,10;50,40;10,40")
		#calibration_arg.add_argument("--camera-type", type=str, default="right_side_down_camera",required=True,
		#						help="which camera should be calibrated , eg : front_camera/back_camera/right_down_camera/left_down_camera/left_back_camera/right_back_camera")


	def parse(self):
		return self.parser.parse_args()
