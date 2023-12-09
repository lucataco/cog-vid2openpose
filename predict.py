# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
from controlnet_aux import OpenposeDetector


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


    def get_frames(self, video_in):
        frames = []
        #resize the video
        clip = VideoFileClip(video_in)
        #check fps
        if clip.fps > 30:
            print("vide rate is over 30, resetting to 30")
            clip_resized = clip.resize(height=512)
            clip_resized.write_videofile("video_resized.mp4", fps=30)
        else:
            print("video rate is OK")
            clip_resized = clip.resize(height=512)
            clip_resized.write_videofile("video_resized.mp4", fps=clip.fps)
        
        print("video resized to 512 height")
        # Opens the Video file with CV2
        cap= cv2.VideoCapture("video_resized.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("video fps: " + str(fps))
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite('/tmp/kang'+str(i)+'.jpg',frame)
            frames.append('/tmp/kang'+str(i)+'.jpg')
            i+=1
        
        cap.release()
        cv2.destroyAllWindows()
        print("broke the video into frames")
        return frames, fps


    def get_openpose_filter(self, i):
        image = Image.open(i)
        image = self.openpose(image)
        frame = str(i).replace("/tmp/", "")
        image.save("/tmp/openpose_frame_" + frame + ".jpeg")
        return "/tmp/openpose_frame_" + frame + ".jpeg"

    def create_video(self, frames, fps, type):
        print("building video result")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(type + "_result.mp4", fps=fps)
        return type + "_result.mp4"

    def infer(self, video_in):
        # 1. break video into frames and get FPS
        break_vid = self.get_frames(video_in)
        frames_list= break_vid[0]
        fps = break_vid[1]
        n_frame = len(frames_list)
        if n_frame >= len(frames_list):
            print("video is shorter than the cut value")
            n_frame = len(frames_list)
        # 2. prepare frames result arrays
        result_frames = []
        print("set stop frames to: " + str(n_frame))
        for i in frames_list[0:int(n_frame)]:
            openpose_frame = self.get_openpose_filter(i)
            result_frames.append(openpose_frame)
            print("frame " + i + "/" + str(n_frame) + ": done;")
        
        final_vid = self.create_video(result_frames, fps, "/tmp/openpose")
        # files = [final_vid]
        # return final_vid, files
        return final_vid


    def predict(
        self,
        video: Path = Input(description="Input Video"),
    ) -> Path:
        """Run a single prediction on the model"""
        output_video = self.infer(str(video))
        print("output video: " + output_video)
        return Path(output_video)
