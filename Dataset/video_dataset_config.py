import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_file, super_img_rows, super_img_cols):
    """
    Extracts frames from a video and returns a list of sampled frames.
    """
    video = cv2.VideoCapture(video_file)
    frames = []

    # Define the number of frames to sample
    num_frames_to_sample = super_img_rows * super_img_cols

    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the indices of frames to sample
    frame_indices = [int(x) for x in range(0, total_frames, total_frames // num_frames_to_sample)]

    # Loop over sampled frames in video
    for i in frame_indices:
        # Set the current frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()

        # Break loop if end of video is reached
        if not ret:
            break

        frames.append(frame)

    video.release()
    return frames

def resize_frames(frames, super_img_rows, super_img_cols, super_img_size):
    """
    Resizes a list of frames to the given image size.
    """
    # Define the resize image size
    resize_frame_h, resize_frame_w = super_img_size[0]//super_img_rows, super_img_size[0]//super_img_cols
    resize_frame_size = (resize_frame_h, resize_frame_w)
    # print(resize_frame_size)

    resized_frames = []
    for frame in frames:
        # Resize frame to super image size
        resized_frame = cv2.resize(frame, resize_frame_size)
        tensor_frame = torch.from_numpy(resized_frame).float()
        resized_frames.append(tensor_frame)

    return resized_frames

def create_super_img(resized_frames, super_img_rows, super_img_cols, super_img_size):
    """
    Creates a super image from a list of resized frames.
    """
    # Get the resized_frames_height and resized_frames_width
    resized_frames_h, resized_frames_w = resized_frames[0].shape[0], resized_frames[0].shape[1]
    
    # Create empty super image
    super_image = np.zeros((super_img_size[0], super_img_size[1], 3))

    # Loop over resized frames and append to super image
    idx = 0
    for i in range(super_img_rows):
        for j in range(super_img_cols):
            # Get current row and column indices for super image
            row_idx = i * resized_frames_h
            col_idx = j * resized_frames_w

            # Get current resized frame
            resized_frame = resized_frames[idx]
            array_frame = resized_frame.numpy()

            # Append resized frame to super image
            super_image[row_idx:row_idx+resized_frames_h, col_idx:col_idx+resized_frames_w, :] = array_frame
            idx += 1

    tensor_image = torch.from_numpy(super_image).permute(2, 0, 1)
    return tensor_image

def save_super_img(super_image, save_path):
    """
    Saves a super image as an image file.
    """
    # Save the super image as a file
    plt.imsave(save_path, super_image.permute(1,2,0).numpy()/255)


if __name__ == "__main__":
    # Set the path to the video file
    video_file = "sample_5s.mp4"
    # Set the path to save the super image
    save_path = "./super_image4.jpg"
    # Define the super image size
    super_img_size = (224, 224)
    # Define the number of rows and columns in the super image
    super_img_rows = 4
    super_img_cols = 4

    frames = extract_frames(video_file, super_img_rows, super_img_cols)
    resized_frames = resize_frames(frames, super_img_rows, super_img_cols, super_img_size)
    super_image = create_super_img(resized_frames, super_img_rows, super_img_cols, super_img_size)
    save_super_img(super_image, save_path)
