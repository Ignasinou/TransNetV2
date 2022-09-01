from transnetv2 import TransNetV2
import argparse
import subprocess
import cv2
import tqdm


def main():
    parser = argparse.ArgumentParser(description="Face_blur_detection")
    parser.add_argument('--videoFile')
    parser.add_argument('--th', type=float, default=0.5)

    args = parser.parse_args()
    file_extension = "." + args.videoFile.split(".")[-1]
    output_video_filename = args.videoFile.replace(file_extension, '_output.mp4')

    video = cv2.VideoCapture(args.videoFile)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    output_videos = []
    scene_model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = scene_model.predict_video(args.videoFile)
    scenes_int32 = scene_model.predictions_to_scenes(single_frame_predictions, threshold=args.th)
    scenes = [list(map(int, s)) for s in scenes_int32]
    print('='*15)
    for scene in scenes:
        print(f'SCENE: {scene[0]}-{scene[1]}')
        output_video = output_video_filename.replace('_output.mp4', f'_{scene[0]}-{scene[1]}_output.mp4')
        output_videos.append(cv2.VideoWriter(
            output_video
            , cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height)))
    all_frames = []
    print('='*15)

    while True:
        for frame_num in tqdm.tqdm(range(int(num_frames))):
            ret, frame = video.read()
            all_frames.append(frame)
        break

    for idx, scene in enumerate(scenes):
        for fr in range(scene[0], scene[1], 1):
            cv2.putText(all_frames[fr], f"Frame: #{fr}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            output_videos[idx].write(all_frames[fr])

    for outVideo in output_videos:
        outVideo.release()

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
