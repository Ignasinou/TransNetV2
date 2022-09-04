from transnetv2 import TransNetV2
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Face_blur_detection")
    parser.add_argument('--videoFile')
    parser.add_argument('--th', type=float, default=0.5)

    args = parser.parse_args()
    filename = args.videoFile.split('/')[-1]
    output_txt = args.videoFile.replace(filename, f'output_scenes.json')
    print(output_txt)
    scene_model = TransNetV2()
    video_frames, single_frame_predictions, all_frame_predictions = scene_model.predict_video(args.videoFile)
    scenes_int32 = scene_model.predictions_to_scenes(single_frame_predictions, threshold=args.th)
    scenes = [list(map(int, s)) for s in scenes_int32]
    dictionary = {
        "scenes": scenes
    }

    with open(output_txt, 'w') as f:
        json.dump(dictionary, f)

if __name__ == "__main__":
    main()
