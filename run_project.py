"""
Run the motion estimation pipeline on a single video.
"""

import argparse
import os
import cv2

from src.motion_estimation import lucas_kanade, horn_schunck
from src.trajectory import extract_trajectory_from_flow, track_object_centroid, analyze_trajectory
from src.visualization import draw_flow, draw_trajectory, plot_flow_magnitude, plot_trajectory_analysis, save_visualization
from src.utils import load_video_frames, ensure_folder, write_text_report


def main():
    parser = argparse.ArgumentParser(description='Run motion estimation for a single object.')
    parser.add_argument('--video', required=True, help='Input video file path.')
    parser.add_argument('--method', default='lucas_kanade', choices=['lucas_kanade', 'horn_schunck'], help='Optical flow method.')
    parser.add_argument('--output', default='results/run', help='Output folder for results.')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum number of frames to process.')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), help='Resize frames to this resolution.')
    parser.add_argument('--track-method', default='contour', choices=['contour', 'background_subtraction', 'flow'], help='Object tracking method or flow-based trajectory extraction.')
    args = parser.parse_args()

    output_folder = args.output
    ensure_folder(output_folder)

    frames = load_video_frames(args.video, max_frames=args.max_frames, resize=tuple(args.resize) if args.resize else None)
    if len(frames) < 2:
        raise ValueError('Video must contain at least two frames.')

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    flows = []

    for i in range(len(gray_frames) - 1):
        if args.method == 'lucas_kanade':
            flow = lucas_kanade(gray_frames[i], gray_frames[i + 1])
        else:
            flow = horn_schunck(gray_frames[i], gray_frames[i + 1])
        flows.append(flow)

    # Track object and analyze trajectory
    if args.track_method == 'flow':
        trajectory = extract_trajectory_from_flow(flows, threshold=1.0)
    else:
        trajectory = track_object_centroid(frames, method=args.track_method)
    stats = analyze_trajectory(trajectory)

    # Save summary metrics
    summary_lines = [f'Method: {args.method}', f'Track method: {args.track_method}', f'Frames processed: {len(frames)}']
    for key, value in stats.items():
        summary_lines.append(f'{key}: {value}')
    write_text_report(os.path.join(output_folder, 'summary.txt'), summary_lines)

    # Save a trajectory image on the last frame
    trajectory_image = draw_trajectory(frames[-1], trajectory)
    cv2.imwrite(os.path.join(output_folder, 'trajectory.png'), trajectory_image)

    # Save flow visualization for the first and last computed flow
    if flows:
        indices = [0]
        if len(flows) > 1:
            indices.append(len(flows) - 1)

        for index in indices:
            flow = flows[index]
            frame = frames[index + 1]
            flow_image = draw_flow(frame, flow)
            cv2.imwrite(os.path.join(output_folder, f'flow_{index + 1}.png'), flow_image)

        # Save magnitude heatmap for final flow
        heatmap_fig = plot_flow_magnitude(flows[-1], title='Final Flow Magnitude')
        save_visualization(os.path.join(output_folder, 'flow_magnitude.png'), heatmap_fig)
    else:
        print('No optical flow frames were computed; skipping flow visualizations.')

    # Save trajectory analysis plots
    analysis_fig = plot_trajectory_analysis(trajectory, stats=stats)
    save_visualization(os.path.join(output_folder, 'trajectory_analysis.png'), analysis_fig)

    print(f'Results saved to {output_folder}')


if __name__ == '__main__':
    main()
